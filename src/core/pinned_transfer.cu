#include "core/pinned_transfer.h"

#include "core/arena.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace ninfer {
namespace {

std::size_t checked_slot_capacity(std::size_t capacity_bytes) {
    if (capacity_bytes < 2 || capacity_bytes % 2 != 0) {
        throw std::invalid_argument(
            "PinnedTransferBuffer capacity must divide evenly across two slots");
    }
    return capacity_bytes / 2;
}

// Adjacent descriptors that are contiguous on both sides describe one larger copy. Callers build
// these lists per logical element - paged KV emits one descriptor per page per plane, tens of
// thousands for a long session - and the pinned pipeline pays an event synchronize and a launch
// for every descriptor regardless of size. Physical pages are handed out in ascending runs, so
// merging turns that back into a handful of large transfers. The bytes moved are identical.
bool contiguous(const DeviceToHostTransfer& a, const DeviceToHostTransfer& b) noexcept {
    return a.destination == b.destination && a.destination_offset + a.bytes == b.destination_offset &&
           static_cast<const std::uint8_t*>(a.source) + a.bytes ==
               static_cast<const std::uint8_t*>(b.source);
}

bool contiguous(const HostToDeviceTransfer& a, const HostToDeviceTransfer& b) noexcept {
    return a.source == b.source && a.source_offset + a.bytes == b.source_offset &&
           static_cast<std::uint8_t*>(a.destination) + a.bytes ==
               static_cast<std::uint8_t*>(b.destination);
}

} // namespace

struct PinnedTransferBuffer::Impl {
    explicit Impl(std::size_t capacity_bytes)
        : slot_bytes(checked_slot_capacity(capacity_bytes)),
          slots{PinnedHostBuffer(slot_bytes), PinnedHostBuffer(slot_bytes)} {
        try {
            for (cudaEvent_t& event : events) {
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
        } catch (...) {
            for (cudaEvent_t event : events) {
                if (event != nullptr) { (void)cudaEventDestroy(event); }
            }
            throw;
        }
    }

    ~Impl() {
        for (cudaEvent_t event : events) {
            if (event != nullptr) { (void)cudaEventDestroy(event); }
        }
    }

    std::size_t slot_bytes;
    std::array<PinnedHostBuffer, 2> slots;
    std::array<cudaEvent_t, 2> events{};
};

PinnedTransferBuffer::PinnedTransferBuffer(std::size_t capacity_bytes)
    : impl_(std::make_unique<Impl>(capacity_bytes)) {}

PinnedTransferBuffer::~PinnedTransferBuffer() = default;
PinnedTransferBuffer::PinnedTransferBuffer(PinnedTransferBuffer&&) noexcept = default;
PinnedTransferBuffer& PinnedTransferBuffer::operator=(PinnedTransferBuffer&&) noexcept = default;

std::size_t PinnedTransferBuffer::capacity() const noexcept {
    return impl_ == nullptr ? 0 : impl_->slot_bytes * impl_->slots.size();
}

std::size_t PinnedTransferBuffer::slot_capacity() const noexcept {
    return impl_ == nullptr ? 0 : impl_->slot_bytes;
}

void PinnedTransferBuffer::copy_device_to_host(
    std::span<const DeviceToHostTransfer> transfers, cudaStream_t stream) {
    // One slot carries as many descriptors as fit and is synchronized once. Callers describe
    // transfers per logical element, and a shared KV pool in steady use hands out scattered
    // physical pages, so contiguous merging alone cannot reduce that count - a long session still
    // produces tens of thousands of descriptors. Synchronizing per slot instead of per descriptor
    // is what makes the cost track bytes moved rather than element count.
    struct Piece {
        std::vector<std::uint8_t>* destination = nullptr;
        std::size_t destination_offset         = 0;
        std::size_t slot_offset                = 0;
        std::size_t bytes                      = 0;
    };
    std::array<std::vector<Piece>, 2> staged;
    std::array<bool, 2> pending{};
    std::size_t slot = 0;
    std::size_t used = 0;

    const auto finish = [&](std::size_t index) {
        if (!pending[index]) { return; }
        CUDA_CHECK(cudaEventSynchronize(impl_->events[index]));
        for (const Piece& piece : staged[index]) {
            std::memcpy(piece.destination->data() + piece.destination_offset,
                        static_cast<std::uint8_t*>(impl_->slots[index].data()) + piece.slot_offset, piece.bytes);
        }
        staged[index].clear();
        pending[index] = false;
    };

    // Hand the open slot to the device, then take the other one back from it.
    const auto rotate = [&] {
        if (!staged[slot].empty()) {
            CUDA_CHECK(cudaEventRecord(impl_->events[slot], stream));
            pending[slot] = true;
        }
        slot = (slot + 1) % staged.size();
        finish(slot);
        used = 0;
    };

    try {
        finish(slot);
        // Every descriptor is validated as given; only their submission is packed.
        DeviceToHostTransfer run{};
        bool have_run = false;
        const auto submit = [&](const DeviceToHostTransfer& transfer) {
            std::size_t copied = 0;
            while (copied < transfer.bytes) {
                if (used == impl_->slot_bytes) { rotate(); }
                const std::size_t chunk =
                    std::min(impl_->slot_bytes - used, transfer.bytes - copied);
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::uint8_t*>(impl_->slots[slot].data()) + used,
                    static_cast<const std::uint8_t*>(transfer.source) + copied, chunk,
                    cudaMemcpyDeviceToHost, stream));
                staged[slot].push_back(
                    {transfer.destination, transfer.destination_offset + copied, used, chunk});
                used += chunk;
                copied += chunk;
            }
        };
        for (const DeviceToHostTransfer& transfer : transfers) {
            if (transfer.destination == nullptr ||
                (transfer.bytes != 0 && transfer.source == nullptr) ||
                transfer.destination_offset > transfer.destination->size() ||
                transfer.bytes > transfer.destination->size() - transfer.destination_offset) {
                throw std::invalid_argument("device-to-host pinned transfer is out of bounds");
            }
            if (have_run && contiguous(run, transfer)) {
                run.bytes += transfer.bytes;
                continue;
            }
            if (have_run) { submit(run); }
            run      = transfer;
            have_run = true;
        }
        if (have_run) { submit(run); }
        rotate();
        for (std::size_t index = 0; index < staged.size(); ++index) { finish(index); }
    } catch (...) {
        // A failed enqueue/record must not let pinned storage die while prior work still references it.
        (void)cudaStreamSynchronize(stream);
        throw;
    }
}

void PinnedTransferBuffer::copy_host_to_device(
    std::span<const HostToDeviceTransfer> transfers, cudaStream_t stream) {
    std::array<bool, 2> pending{};
    std::size_t slot   = 0;
    std::size_t used   = 0;
    bool slot_has_work = false;

    const auto finish = [&](std::size_t index) {
        if (!pending[index]) { return; }
        CUDA_CHECK(cudaEventSynchronize(impl_->events[index]));
        pending[index] = false;
    };

    const auto rotate = [&] {
        if (slot_has_work) {
            CUDA_CHECK(cudaEventRecord(impl_->events[slot], stream));
            pending[slot] = true;
        }
        slot = (slot + 1) % pending.size();
        finish(slot);
        used          = 0;
        slot_has_work = false;
    };

    try {
        finish(slot);
        // Every descriptor is validated as given; only their submission is packed.
        HostToDeviceTransfer run{};
        bool have_run     = false;
        const auto submit = [&](const HostToDeviceTransfer& transfer) {
            std::size_t copied = 0;
            while (copied < transfer.bytes) {
                if (used == impl_->slot_bytes) { rotate(); }
                const std::size_t chunk =
                    std::min(impl_->slot_bytes - used, transfer.bytes - copied);
                std::memcpy(static_cast<std::uint8_t*>(impl_->slots[slot].data()) + used,
                            transfer.source->data() + transfer.source_offset + copied, chunk);
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::uint8_t*>(transfer.destination) + copied,
                    static_cast<std::uint8_t*>(impl_->slots[slot].data()) + used, chunk, cudaMemcpyHostToDevice, stream));
                used += chunk;
                copied += chunk;
                slot_has_work = true;
            }
        };
        for (const HostToDeviceTransfer& transfer : transfers) {
            if (transfer.source == nullptr ||
                (transfer.bytes != 0 && transfer.destination == nullptr) ||
                transfer.source_offset > transfer.source->size() ||
                transfer.bytes > transfer.source->size() - transfer.source_offset) {
                throw std::invalid_argument("host-to-device pinned transfer is out of bounds");
            }
            if (have_run && contiguous(run, transfer)) {
                run.bytes += transfer.bytes;
                continue;
            }
            if (have_run) { submit(run); }
            run      = transfer;
            have_run = true;
        }
        if (have_run) { submit(run); }
        rotate();
        for (std::size_t index = 0; index < pending.size(); ++index) { finish(index); }
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
}

void PinnedTransferBuffer::copy_device_to_host(std::vector<std::uint8_t>& destination,
                                                const void* source, std::size_t bytes,
                                                cudaStream_t stream,
                                                std::size_t destination_offset) {
    const DeviceToHostTransfer transfer{&destination, destination_offset, source, bytes};
    copy_device_to_host(std::span<const DeviceToHostTransfer>(&transfer, 1), stream);
}

void PinnedTransferBuffer::copy_host_to_device(void* destination,
                                                const std::vector<std::uint8_t>& source,
                                                std::size_t bytes, cudaStream_t stream,
                                                std::size_t source_offset) {
    const HostToDeviceTransfer transfer{destination, &source, source_offset, bytes};
    copy_host_to_device(std::span<const HostToDeviceTransfer>(&transfer, 1), stream);
}

} // namespace ninfer
