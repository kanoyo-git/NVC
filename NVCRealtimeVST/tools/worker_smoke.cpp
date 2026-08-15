#include "WorkerClient.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

namespace {

bool waitForReady(nvc::WorkerClient& worker, const std::chrono::seconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (worker.isReady())
            return true;
        if (worker.status() == nvc::kStatusError)
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

} // namespace

int main(const int argc, char** argv)
{
    constexpr double sampleRate = 48000.0;
    constexpr double frequency = 220.0;
    constexpr double pi = 3.14159265358979323846;

  nvc::WorkerClient worker;
  if (argc < 4) {
      std::cerr << "Usage: nvc-worker-smoke NVC_ROOT PYTHON_EXE MODEL_PTH [INDEX] [BLOCK_MS] [CROSSFADE_MS] [BLOCK_COUNT]\n";
      return EXIT_FAILURE;
  }
  worker.setPath(nvc::kStateNvcRoot, argv[1]);
  worker.setPath(nvc::kStatePythonPath, argv[2]);
  worker.setPath(nvc::kStateModelPath, argv[3]);
  worker.setPath(nvc::kStateIndexPath, argc > 4 ? argv[4] : "");
  if (argc > 5)
      worker.setParameter(nvc::kParamBlockMs, std::stof(argv[5]));
  if (argc > 6)
      worker.setParameter(nvc::kParamCrossfadeMs, std::stof(argv[6]));
  const std::size_t blockCount = argc > 7 ? std::max<std::size_t>(1, std::stoul(argv[7])) : 1;
  worker.setSampleRate(sampleRate);
    worker.setEnabled(true);

    if (!waitForReady(worker, std::chrono::seconds(180))) {
        std::cerr << "Worker did not become ready: " << worker.statusText() << '\n';
        return EXIT_FAILURE;
    }

    const std::size_t frames = worker.blockFrames();
  std::vector<float> input(frames * blockCount);
  for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<float>(0.1 * std::sin(2.0 * pi * frequency * static_cast<double>(i) / sampleRate));

    if (worker.pushInput(input.data(), input.size()) != input.size()) {
        std::cerr << "Input ring rejected the test block\n";
        return EXIT_FAILURE;
    }
  const std::size_t expected = worker.latencyFrames() + input.size();
  std::vector<float> output(expected);
  std::size_t received = 0;
  const auto outputDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  while (received < expected && std::chrono::steady_clock::now() < outputDeadline) {
      received += worker.popOutput(output.data() + received, expected - received);
      if (worker.status() == nvc::kStatusError)
          break;
      if (received < expected)
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (received != expected) {
      std::cerr << "Output ring returned " << received << " of " << expected << " frames\n";
      return EXIT_FAILURE;
    }

    double sumSquares = 0.0;
    for (std::size_t i = expected - frames; i < expected; ++i)
        sumSquares += static_cast<double>(output[i]) * output[i];
    const double rms = std::sqrt(sumSquares / static_cast<double>(frames));

    std::cout << "READY frames=" << frames
              << " latency=" << worker.latencyFrames()
              << " infer_ms=" << worker.inferMs()
              << " output_rms=" << rms
              << " drops=" << worker.droppedBlocks()
              << " blocks=" << blockCount
              << " status=\"" << worker.statusText() << "\"\n";
    return rms > 0.0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
