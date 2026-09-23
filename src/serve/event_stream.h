#pragma once

// The structured record funnel: one schema instance, two transports.
//
// `request_log.cpp` owns the record format (schema 20). This class owns the live instance of it -
// the per-process `server_instance_id`, the timestamp, and the fan-out - so a record is formatted
// exactly once and delivered to both the optional `--request-log-jsonl` file and every connected
// GET /events subscriber. Adding the stream therefore does not duplicate the schema, and a
// dashboard reading /events sees byte-identical records to the ones a post-hoc reader parses out
// of the file.
//
// Subscribers are bounded and lossy by construction. Records are published from the HTTP request
// threads and the stats reporter, and a browser that stops reading must never apply backpressure
// to them, so a full queue drops its oldest record and counts the drop. A dashboard that sees
// `dropped` advance re-reads GET /telemetry, which is a complete snapshot rather than a delta.

#include "serve/request_log.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::serve {

// One connected /events reader.
class EventSubscriber {
public:
    explicit EventSubscriber(std::size_t capacity) : capacity_(capacity) {}

    // Non-blocking publish. Drops the oldest queued record when full.
    void publish(const std::string& record);

    // Blocks for at most `timeout` waiting for one record. Returns false on timeout or close, so
    // the caller can emit an SSE keepalive comment and re-check that its socket is still writable.
    bool next(std::string& record, std::chrono::milliseconds timeout);

    void close();

    [[nodiscard]] std::uint64_t dropped() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::string> queue_;
    std::size_t capacity_  = 0;
    std::uint64_t dropped_ = 0;
    bool closed_           = false;
};

class EventStream {
public:
    // `replay_capacity` bounds the ring of recent records replayed to a new subscriber, so a
    // dashboard opened mid-run renders immediately instead of waiting for the next 5s interval.
    EventStream(const std::string& jsonl_path, const std::string& protected_artifact_path,
                std::size_t replay_capacity);

    EventStream(const EventStream&)            = delete;
    EventStream& operator=(const EventStream&) = delete;

    [[nodiscard]] const std::string& server_instance_id() const noexcept {
        return server_instance_id_;
    }

    [[nodiscard]] bool jsonl_enabled() const noexcept { return jsonl_.enabled(); }

    void emit_server_start(const ServeOptions& options,
                           const ninfer::ModelSamplingDefaults& sampling_defaults,
                           const std::string& public_model_id, const ninfer::LoadSummary& load,
                           const ninfer::MemorySummary& memory);
    void emit_request_start(const RequestLogContext& context);
    void emit_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void emit_request_error(const RequestLogContext& context, const std::string& message);
    void emit_throughput(const ThroughputReport& report);

    // Registers a reader and hands back the retained `server_start` record followed by the replay
    // ring, so the caller can flush a complete opening state before entering its read loop.
    std::shared_ptr<EventSubscriber> subscribe(std::vector<std::string>& backlog);
    void unsubscribe(const std::shared_ptr<EventSubscriber>& subscriber);

    // Releases every reader blocked in next() so shutdown does not wait out a keepalive period.
    void close_all();

    [[nodiscard]] std::size_t subscriber_count() const;

private:
    void publish(std::string record, bool retain_as_server_start);

    JsonlRequestLog jsonl_;
    std::string server_instance_id_;
    std::size_t replay_capacity_ = 0;

    mutable std::mutex mutex_;
    std::string server_start_record_;
    std::deque<std::string> replay_;
    std::vector<std::shared_ptr<EventSubscriber>> subscribers_;
};

} // namespace ninfer::serve
