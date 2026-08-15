#pragma once

#include "NvcParameters.hpp"
#include "SpscRing.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nvc {

class WorkerClient {
public:
    WorkerClient();
    ~WorkerClient();

    WorkerClient(const WorkerClient&) = delete;
    WorkerClient& operator=(const WorkerClient&) = delete;

    void setEnabled(bool enabled) noexcept;
    void setSampleRate(double sampleRate) noexcept;
    void setParameter(ParameterId id, float value) noexcept;
    void setPath(StateId id, const char* value);

    std::size_t pushInput(const float* samples, std::size_t count) noexcept;
    std::size_t popOutput(float* samples, std::size_t count) noexcept;

    bool isReady() const noexcept { return ready_.load(std::memory_order_acquire); }
    int status() const noexcept { return status_.load(std::memory_order_acquire); }
    float inferMs() const noexcept { return inferMs_.load(std::memory_order_relaxed); }
    float droppedBlocks() const noexcept { return static_cast<float>(droppedBlocks_.load(std::memory_order_relaxed)); }
    uint32_t blockFrames() const noexcept { return blockFrames_.load(std::memory_order_relaxed); }
    uint32_t latencyFrames() const noexcept { return latencyFrames_.load(std::memory_order_relaxed); }
    std::string statusText() const;

private:
    struct Paths {
        std::string model;
        std::string index;
        std::string nvcRoot;
        std::string python;
    };

    struct Ipc;

    void threadMain();
    bool launchWorker(const Paths& paths, uint64_t version);
    void stopWorker();
    bool processOneBlock();
    Paths pathsSnapshot() const;
    void setStatus(int status, const std::string& text);
    uint32_t calculateBlockFrames() const noexcept;
    std::string writeWorkerConfig(const Paths& paths, std::string& error) const;

    static constexpr std::size_t kRingCapacity = 1u << 20;
    SpscFloatRing inputRing_ {kRingCapacity};
    SpscFloatRing outputRing_ {kRingCapacity};

    std::array<std::atomic<float>, kParameterCount> parameters_ {};
    std::atomic<bool> enabled_ {false};
    std::atomic<bool> stopRequested_ {false};
    std::atomic<bool> ready_ {false};
    std::atomic<int> status_ {kStatusOff};
    std::atomic<float> inferMs_ {0.0f};
    std::atomic<uint32_t> droppedBlocks_ {0};
    std::atomic<uint32_t> blockFrames_ {6240};
    std::atomic<uint32_t> latencyFrames_ {12480};
    std::atomic<double> sampleRate_ {48000.0};
    std::atomic<uint64_t> configVersion_ {1};

    mutable std::mutex pathsMutex_;
    Paths paths_;
    mutable std::mutex statusTextMutex_;
    std::string statusText_ {"Off"};

    std::unique_ptr<Ipc> ipc_;
    std::vector<float> requestBuffer_;
    std::vector<float> responseBuffer_;
    std::thread thread_;
};

} // namespace nvc
