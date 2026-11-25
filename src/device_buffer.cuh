#pragma once
#include "cuda_utils.cuh"
#include <cstddef>
#include <utility>
#include <vector>

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            swap(other);
        }
        return *this;
    }

    void reserve(std::size_t count) {
        if (count <= capacity_) return;
        release();
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
        capacity_ = count;
    }

    void upload(const T* host, std::size_t count, cudaStream_t stream) {
        reserve(count);
        if (count) CHECK_CUDA(cudaMemcpyAsync(data_, host, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    void upload(const std::vector<T>& host, cudaStream_t stream) { upload(host.data(), host.size(), stream); }

    void download(T* host, std::size_t count, cudaStream_t stream) const {
        if (count) CHECK_CUDA(cudaMemcpyAsync(host, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    std::vector<T> to_host(std::size_t count, cudaStream_t stream) const {
        std::vector<T> host(count);
        download(host.data(), count, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        return host;
    }

    void fill_bytes(int value, std::size_t count, cudaStream_t stream) {
        if (count) CHECK_CUDA(cudaMemsetAsync(data_, value, count * sizeof(T), stream));
    }

    void swap(DeviceBuffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(capacity_, other.capacity_);
    }

    T* data() const { return data_; }
    std::size_t capacity() const { return capacity_; }

private:
    void release() noexcept {
        if (data_) cudaFree(data_);
        data_ = nullptr;
        capacity_ = 0;
    }

    T* data_ = nullptr;
    std::size_t capacity_ = 0;
};
