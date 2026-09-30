#include "serve/event_stream.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <system_error>
#include <utility>

namespace ninfer::serve {

void EventSubscriber::publish(const std::string& record) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) { return; }
        while (queue_.size() >= capacity_) {
            queue_.pop_front();
            ++dropped_;
        }
        queue_.push_back(record);
    }
    ready_.notify_one();
}

bool EventSubscriber::next(std::string& record, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!ready_.wait_for(lock, timeout, [this] { return closed_ || !queue_.empty(); })) {
        return false;
    }
    if (queue_.empty()) { return false; }
    record = std::move(queue_.front());
    queue_.pop_front();
    return true;
}

void EventSubscriber::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        queue_.clear();
    }
    ready_.notify_all();
}

std::uint64_t EventSubscriber::dropped() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return dropped_;
}

EventStream::EventStream(const std::string& jsonl_path,
                         const std::string& protected_artifact_path, std::size_t replay_capacity)
    : jsonl_(jsonl_path, protected_artifact_path),
      server_instance_id_(new_server_instance_id()),
      replay_capacity_(replay_capacity) {}

void EventStream::publish(std::string record, bool retain_as_server_start) {
    jsonl_.write_record(record);

    std::vector<std::shared_ptr<EventSubscriber>> targets;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (retain_as_server_start) {
            server_start_record_ = record;
        } else if (replay_capacity_ != 0) {
            if (replay_.size() >= replay_capacity_) { replay_.pop_front(); }
            replay_.push_back(record);
        }
        targets = subscribers_;
    }
    // Published outside the registry lock: a subscriber's queue mutex is independent, and holding
    // both would let a slow reader serialize the request threads behind the registry.
    for (const std::shared_ptr<EventSubscriber>& subscriber : targets) {
        subscriber->publish(record);
    }
}

void EventStream::emit_server_start(const ServeOptions& options,
                                    const ninfer::ModelSamplingDefaults& sampling_defaults,
                                    const std::string& public_model_id,
                                    const ninfer::LoadSummary& load,
                                    const ninfer::MemorySummary& memory) {
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(options.artifact_path, error);
    const std::optional<std::uint64_t> artifact_size =
        error ? std::nullopt : std::optional<std::uint64_t>(size);
    publish(format_server_start_json(server_instance_id_, unix_time_ms(), options,
                                     sampling_defaults, public_model_id, load, memory,
                                     query_server_log_environment(options.device), artifact_size),
            true);
}

void EventStream::emit_request_start(const RequestLogContext& context) {
    publish(format_request_start_json(server_instance_id_, unix_time_ms(), context), false);
}

void EventStream::emit_request_done(const RequestLogContext& context,
                                    const GenerationOutcome& outcome) {
    publish(format_request_done_json(server_instance_id_, unix_time_ms(), context, outcome), false);
}

void EventStream::emit_request_error(const RequestLogContext& context,
                                     const std::string& message) {
    publish(format_request_error_json(server_instance_id_, unix_time_ms(), context, message),
            false);
}

void EventStream::emit_decision_start(const DecisionLogContext& context) {
    publish(format_decision_start_json(server_instance_id_, unix_time_ms(), context), false);
}

void EventStream::emit_decision_done(const DecisionLogContext& context,
                                     const ninfer::DecisionResult& result,
                                     std::uint32_t output_tokens) {
    publish(format_decision_done_json(server_instance_id_, unix_time_ms(), context, result,
                                      output_tokens),
            false);
}

void EventStream::emit_decision_error(const DecisionLogContext& context, int status,
                                      const std::string& message) {
    publish(format_decision_error_json(server_instance_id_, unix_time_ms(), context, status,
                                       message),
            false);
}

void EventStream::emit_throughput(const ThroughputReport& report) {
    publish(format_throughput_json(server_instance_id_, unix_time_ms(), report), false);
}

std::shared_ptr<EventSubscriber> EventStream::subscribe(std::vector<std::string>& backlog) {
    // Queue depth covers the replay ring plus a full reporting interval of request records, so a
    // reader that is merely slow to start does not immediately register drops.
    auto subscriber = std::make_shared<EventSubscriber>(replay_capacity_ + 256);
    std::lock_guard<std::mutex> lock(mutex_);
    backlog.clear();
    backlog.reserve(replay_.size() + 1);
    if (!server_start_record_.empty()) { backlog.push_back(server_start_record_); }
    backlog.insert(backlog.end(), replay_.begin(), replay_.end());
    subscribers_.push_back(subscriber);
    return subscriber;
}

void EventStream::unsubscribe(const std::shared_ptr<EventSubscriber>& subscriber) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        subscribers_.erase(std::remove(subscribers_.begin(), subscribers_.end(), subscriber),
                           subscribers_.end());
    }
    subscriber->close();
}

void EventStream::close_all() {
    std::vector<std::shared_ptr<EventSubscriber>> targets;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        targets.swap(subscribers_);
    }
    for (const std::shared_ptr<EventSubscriber>& subscriber : targets) { subscriber->close(); }
}

std::size_t EventStream::subscriber_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return subscribers_.size();
}

} // namespace ninfer::serve
