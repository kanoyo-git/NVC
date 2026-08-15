#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>

namespace nvc {

class SpscFloatRing {
public:
    explicit SpscFloatRing(const std::size_t requestedCapacity)
    {
        std::size_t capacity = 1;
        while (capacity < requestedCapacity)
            capacity <<= 1;
        capacity_ = capacity;
        mask_ = capacity - 1;
        data_.reset(new float[capacity]);
        std::memset(data_.get(), 0, capacity * sizeof(float));
    }

    std::size_t readable() const noexcept
    {
        const auto write = write_.load(std::memory_order_acquire);
        const auto read = read_.load(std::memory_order_acquire);
        return write - read;
    }

    std::size_t writable() const noexcept
    {
        return capacity_ - readable();
    }

    std::size_t push(const float* src, std::size_t count) noexcept
    {
        const auto write = write_.load(std::memory_order_relaxed);
        const auto read = read_.load(std::memory_order_acquire);
        count = std::min(count, capacity_ - (write - read));
        if (count == 0)
            return 0;

        const std::size_t pos = write & mask_;
        const std::size_t first = std::min(count, capacity_ - pos);
        std::memcpy(data_.get() + pos, src, first * sizeof(float));
        if (count > first)
            std::memcpy(data_.get(), src + first, (count - first) * sizeof(float));
        write_.store(write + count, std::memory_order_release);
        return count;
    }

    std::size_t pushZeros(std::size_t count) noexcept
    {
        static constexpr float zeros[1024] = {};
        std::size_t total = 0;
        while (total < count) {
            const std::size_t chunk = std::min<std::size_t>(1024, count - total);
            const std::size_t written = push(zeros, chunk);
            total += written;
            if (written != chunk)
                break;
        }
        return total;
    }

    std::size_t pop(float* dst, std::size_t count) noexcept
    {
        const auto read = read_.load(std::memory_order_relaxed);
        const auto write = write_.load(std::memory_order_acquire);
        count = std::min(count, write - read);
        if (count == 0)
            return 0;

        const std::size_t pos = read & mask_;
        const std::size_t first = std::min(count, capacity_ - pos);
        std::memcpy(dst, data_.get() + pos, first * sizeof(float));
        if (count > first)
            std::memcpy(dst + first, data_.get(), (count - first) * sizeof(float));
        read_.store(read + count, std::memory_order_release);
        return count;
    }

    void resetUnsafe() noexcept
    {
        read_.store(0, std::memory_order_relaxed);
        write_.store(0, std::memory_order_relaxed);
    }

private:
    std::unique_ptr<float[]> data_;
    std::size_t capacity_ = 0;
    std::size_t mask_ = 0;
    alignas(64) std::atomic<std::size_t> write_ {0};
    alignas(64) std::atomic<std::size_t> read_ {0};
};

} // namespace nvc
