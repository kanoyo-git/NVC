#pragma once

#include "IPlug_include_in_plug_hdr.h"

#include "NvcParameters.hpp"
#include "WorkerClient.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

const int kNumPresets = 1;

enum EParams {
  kEngine = 0,
  kPitch,
  kFormant,
  kIndexRate,
  kRmsMix,
  kThreshold,
  kBlockMs,
  kCrossfadeMs,
  kExtraMs,
  kF0Method,
  kDryWet,
  kOutputGain,
  kNumParams
};

enum EControlTags {
  kCtrlStatus = 100,
  kCtrlPerformance,
  kCtrlStatusDetail,
  kCtrlNvcRoot,
  kCtrlPythonPath,
  kCtrlModelName,
  kCtrlIndexName
};

using namespace iplug;
using namespace igraphics;

class NVCRealtime final : public Plugin {
public:
  NVCRealtime(const InstanceInfo& info);

#if IPLUG_DSP
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
  void OnReset() override;
  void OnParamChange(int paramIdx) override;
#endif

  void OnIdle() override;
  void OnUIOpen() override;
  bool SerializeState(IByteChunk& chunk) const override;
  int UnserializeState(const IByteChunk& chunk, int startPos) override;

private:
  void SyncParametersToWorker();
  int CalculateLatencyFrames(double blockMs) const;
  void ResizeBuffers(int blockSize, double sampleRate);
  void ChooseNvcRoot(IGraphics* graphics);
  void ChoosePython(IGraphics* graphics);
  void ChooseModel(IGraphics* graphics);
  void ChooseIndex(IGraphics* graphics);
  void SetNvcRoot(const char* path);
  void SetPythonPath(const char* path);
  void SetModelPath(const char* path);
  void SetIndexPath(const char* path);
  void UpdateFileLabels();
  void StopEngineForPathChange();
  bool ValidateConfiguration(std::string& error) const;
  void LoadUserConfiguration();
  void SaveUserConfiguration() const;

  nvc::WorkerClient mWorker;
  std::vector<float> mMonoInput;
  std::vector<float> mWetOutput;
  std::vector<float> mDryDelay;
  int mDryWritePosition = 0;
  std::atomic<int> mTargetDelayFrames {12480};
  float mActiveBlend = 0.0f;

  mutable std::mutex mStateMutex;
  WDL_String mModelPath {NVC_DEFAULT_MODEL};
  WDL_String mIndexPath {NVC_DEFAULT_INDEX};
  WDL_String mNvcRoot {NVC_DEFAULT_ROOT};
  WDL_String mPythonPath {NVC_DEFAULT_PYTHON};
  WDL_String mModelBrowseDirectory;
  WDL_String mIndexBrowseDirectory;
  WDL_String mValidationMessage;

  NVCRealtime(const NVCRealtime&) = delete;
  NVCRealtime& operator=(const NVCRealtime&) = delete;
};
