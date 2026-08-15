#pragma once

#include <cstdint>

namespace nvc {

enum ParameterId : uint32_t {
    kParamEngine = 0,
    kParamPitch,
    kParamFormant,
    kParamIndexRate,
    kParamRmsMix,
    kParamThreshold,
    kParamBlockMs,
    kParamCrossfadeMs,
    kParamExtraMs,
    kParamF0Method,
    kParamDryWet,
    kParamOutputGain,
    kParamStatus,
    kParamInferMs,
    kParamDroppedBlocks,
    kParameterCount
};

enum StateId : uint32_t {
    kStateModelPath = 0,
    kStateIndexPath,
    kStateNvcRoot,
    kStatePythonPath,
    kStateCount
};

enum WorkerStatus : int {
    kStatusOff = 0,
    kStatusStarting = 1,
    kStatusLoading = 2,
    kStatusReady = 3,
    kStatusError = 4
};

namespace ids {
inline constexpr const char* engine = "engine";
inline constexpr const char* pitch = "pitch";
inline constexpr const char* formant = "formant";
inline constexpr const char* indexRate = "index_rate";
inline constexpr const char* rmsMix = "rms_mix";
inline constexpr const char* threshold = "threshold";
inline constexpr const char* blockMs = "block_ms";
inline constexpr const char* crossfadeMs = "crossfade_ms";
inline constexpr const char* extraMs = "extra_ms";
inline constexpr const char* f0Method = "f0_method";
inline constexpr const char* dryWet = "mix";
inline constexpr const char* outputGain = "output_gain";
} // namespace ids

inline const char* stateKey(const StateId id)
{
    switch (id) {
    case kStateModelPath: return "modelPath";
    case kStateIndexPath: return "indexPath";
    case kStateNvcRoot: return "nvcRoot";
    case kStatePythonPath: return "pythonPath";
    default: return "";
    }
}

} // namespace nvc
