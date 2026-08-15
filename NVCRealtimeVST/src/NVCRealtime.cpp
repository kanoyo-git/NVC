#include "NVCRealtime.h"
#include "IPlug_include_in_plug_src.h"
#include "IPlugPaths.h"
#include "IControls.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

const IColor kBackground = IColor(255, 18, 20, 23);
const IColor kPanel = IColor(255, 33, 36, 39);
const IColor kHeaderColor = IColor(255, 235, 234, 228);
const IColor kText = IColor(255, 233, 232, 226);
const IColor kMuted = IColor(255, 152, 155, 155);
const IColor kAccent = IColor(255, 41, 151, 126);
const IColor kAmber = IColor(255, 230, 160, 48);

enum class PathRow { NvcRoot, Python, Model, Index };

std::wstring SettingsFilePath(const bool createDirectory)
{
  WDL_String directory;
  iplug::INIPath(directory, BUNDLE_NAME);
  const UTF8AsUTF16 directoryWide(directory.Get());
  if (createDirectory)
    CreateDirectoryW(directoryWide.Get(), nullptr);
  std::wstring result(directoryWide.Get());
  result += L"\\settings.ini";
  return result;
}

std::string ReadSetting(const wchar_t* key)
{
  const std::wstring settingsPath = SettingsFilePath(false);
  std::vector<wchar_t> value(32768, L'\0');
  GetPrivateProfileStringW(L"Paths", key, L"", value.data(), static_cast<DWORD>(value.size()), settingsPath.c_str());
  return UTF16AsUTF8(value.data()).Get();
}

void WriteSetting(const std::wstring& settingsPath, const wchar_t* key, const std::string& value)
{
  WritePrivateProfileStringW(L"Paths", key, UTF8AsUTF16(value.c_str()).Get(), settingsPath.c_str());
}

bool PathIsFile(const std::string& path)
{
  if (path.empty())
    return false;
  const DWORD attributes = GetFileAttributesW(UTF8AsUTF16(path.c_str()).Get());
  return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

bool PathIsDirectory(const std::string& path)
{
  if (path.empty())
    return false;
  const DWORD attributes = GetFileAttributesW(UTF8AsUTF16(path.c_str()).Get());
  return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

std::string TrimTrailingSeparators(std::string path)
{
  while (path.size() > 3 && (path.back() == '\\' || path.back() == '/'))
    path.pop_back();
  return path;
}

std::string JoinPath(const std::string& root, const char* relative)
{
  if (root.empty())
    return {};
  std::string result = TrimTrailingSeparators(root);
  result += "\\";
  result += relative;
  return result;
}

std::string ParentDirectory(const std::string& filePath)
{
  const std::size_t separator = filePath.find_last_of("\\/");
  return separator == std::string::npos ? std::string() : filePath.substr(0, separator);
}

bool Is64BitExecutable(const std::string& path)
{
  DWORD binaryType = 0;
  return GetBinaryTypeW(UTF8AsUTF16(path.c_str()).Get(), &binaryType) != FALSE
      && binaryType == SCS_64BIT_BINARY;
}

IVStyle MakeStyle()
{
  return DEFAULT_STYLE
    .WithColor(kBG, kPanel)
    .WithColor(kFG, kAccent)
    .WithColor(kPR, kText)
    .WithColor(kFR, IColor(255, 64, 68, 72))
    .WithLabelText(IText(13.f, kMuted, "Roboto-Regular", EAlign::Center))
    .WithValueText(IText(14.f, kText, "Roboto-Regular", EAlign::Center))
    .WithRoundness(0.15f)
    .WithWidgetFrac(0.55f)
    .WithDrawShadows(false)
    .WithDrawFrame(false);
}

class NVCSliderControl final : public IVSliderControl
{
public:
  using IVSliderControl::IVSliderControl;

  void OnMouseDown(float x, float y, const IMouseMod& mod) override
  {
    ISliderControlBase::OnMouseDown(x, y, mod);
  }

  void OnMouseDblClick(float x, float y, const IMouseMod&) override
  {
    if (mStyle.showValue && mValueBounds.Contains(x, y))
      PromptUserInput(mValueBounds);
  }
};

const char* StatusName(const int status)
{
  switch (status) {
    case nvc::kStatusStarting: return "STARTING";
    case nvc::kStatusLoading: return "LOADING MODEL";
    case nvc::kStatusReady: return "READY";
    case nvc::kStatusError: return "ERROR";
    default: return "ENGINE OFF";
  }
}

} // namespace

NVCRealtime::NVCRealtime(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPresets))
{
  GetParam(kEngine)->InitBool("Engine", false);
  GetParam(kPitch)->InitInt("Pitch", 12, -24, 24, "st");
  GetParam(kFormant)->InitDouble("Formant", 0.0, -12.0, 12.0, 0.01, "st");
  GetParam(kIndexRate)->InitDouble("Index", 0.0, 0.0, 1.0, 0.001);
  GetParam(kRmsMix)->InitDouble("RMS Mix", 0.5, 0.0, 1.0, 0.001);
  GetParam(kThreshold)->InitDouble("Gate", -60.0, -60.0, 0.0, 0.1, "dB");
  GetParam(kBlockMs)->InitInt("Block", 130, 20, 1000, "ms");
  GetParam(kCrossfadeMs)->InitInt("Crossfade", 80, 10, 100, "ms");
  GetParam(kExtraMs)->InitInt("Context", 2000, 500, 3000, "ms");
  GetParam(kF0Method)->InitEnum("F0 Method", 0, {"RMVPE", "FCPE", "PM"});
  GetParam(kDryWet)->InitDouble("Mix", 100.0, 0.0, 100.0, 0.1, "%");
  GetParam(kOutputGain)->InitDouble("Output", 0.0, -18.0, 12.0, 0.1, "dB");

  LoadUserConfiguration();
  if (mPythonPath.GetLength() == 0 && mNvcRoot.GetLength() > 0) {
    const std::string detected = JoinPath(mNvcRoot.Get(), "runtime\\python.exe");
    if (PathIsFile(detected))
      mPythonPath.Set(detected.c_str());
  }
  const std::string modelDirectory = ParentDirectory(mModelPath.Get());
  const std::string indexDirectory = ParentDirectory(mIndexPath.Get());
  mModelBrowseDirectory.Set(modelDirectory.empty() ? mNvcRoot.Get() : modelDirectory.c_str());
  mIndexBrowseDirectory.Set(indexDirectory.empty() ? mNvcRoot.Get() : indexDirectory.c_str());

  mWorker.setPath(nvc::kStateModelPath, mModelPath.Get());
  mWorker.setPath(nvc::kStateIndexPath, mIndexPath.Get());
  mWorker.setPath(nvc::kStateNvcRoot, mNvcRoot.Get());
  mWorker.setPath(nvc::kStatePythonPath, mPythonPath.Get());
  SyncParametersToWorker();

#if IPLUG_EDITOR
  mMakeGraphicsFunc = [&]() {
    return MakeGraphics(*this, PLUG_WIDTH, PLUG_HEIGHT, PLUG_FPS,
                        GetScaleForScreen(PLUG_WIDTH, PLUG_HEIGHT));
  };

  mLayoutFunc = [&](IGraphics* graphics) {
    graphics->EnableMouseOver(true);
    graphics->AttachCornerResizer(EUIResizerMode::Scale, false); // drag corner to scale whole UI
    graphics->AttachPanelBackground(kBackground);
    graphics->LoadFont("Roboto-Regular", ROBOTO_FN);
    const IVStyle style = MakeStyle();
    const IRECT bounds = graphics->GetBounds();

    // Header
    graphics->AttachControl(new IPanelControl(bounds.GetFromTop(86.f), kHeaderColor));
    graphics->AttachControl(new ITextControl(IRECT(28, 10, 420, 52), "NVC REALTIME",
                                              IText(28.f, IColor(255, 23, 25, 27), "Roboto-Regular", EAlign::Near)));
    graphics->AttachControl(new ITextControl(IRECT(30, 50, 420, 76), "VOICE CONVERSION / CUDA BRIDGE",
                                              IText(12.f, IColor(255, 85, 89, 90), "Roboto-Regular", EAlign::Near)));
    graphics->AttachControl(new ITextControl(IRECT(560, 16, 750, 42), "ENGINE OFF",
                                              IText(14.f, IColor(255, 80, 84, 86), "Roboto-Regular", EAlign::Far)), kCtrlStatus);
    graphics->AttachControl(new ITextControl(IRECT(560, 42, 750, 66), "0 ms / 0 drop",
                                              IText(12.f, IColor(255, 100, 104, 105), "Roboto-Regular", EAlign::Far)), kCtrlPerformance);

    // Runtime and model paths
    auto attachFileRow = [&](const float y, const char* label, const int textTag, const PathRow pathRow) {
      const float rowHeight = 36.f;
      graphics->AttachControl(new ITextControl(IRECT(30, y, 92, y + rowHeight), label,
                                                IText(13.f, kMuted, "Roboto-Regular", EAlign::Near)));
      graphics->AttachControl(new IPanelControl(IRECT(96, y, 692, y + rowHeight), kPanel));
      const char* initial = "";
      switch (pathRow) {
        case PathRow::NvcRoot: initial = mNvcRoot.Get(); break;
        case PathRow::Python: initial = mPythonPath.Get(); break;
        case PathRow::Model: initial = mModelPath.get_filepart(); break;
        case PathRow::Index: initial = mIndexPath.get_filepart(); break;
      }
      graphics->AttachControl(new ITextControl(IRECT(110, y, 680, y + rowHeight), initial,
                                                IText(12.f, kText, "Roboto-Regular", EAlign::Near)), textTag);
      auto action = [this, graphics, pathRow](IControl*) {
        switch (pathRow) {
          case PathRow::NvcRoot: ChooseNvcRoot(graphics); break;
          case PathRow::Python: ChoosePython(graphics); break;
          case PathRow::Model: ChooseModel(graphics); break;
          case PathRow::Index: ChooseIndex(graphics); break;
        }
      };
      graphics->AttachControl(new IVButtonControl(IRECT(700, y, 750, y + rowHeight), action, "...",
                                                  style.WithLabelText(IText(18.f, kText, "Roboto-Regular", EAlign::Center)), true));
    };
    attachFileRow(100.f, "NVC ROOT", kCtrlNvcRoot, PathRow::NvcRoot);
    attachFileRow(142.f, "PYTHON", kCtrlPythonPath, PathRow::Python);
    attachFileRow(184.f, "MODEL", kCtrlModelName, PathRow::Model);
    attachFileRow(226.f, "INDEX", kCtrlIndexName, PathRow::Index);
    graphics->AttachControl(new ITextControl(IRECT(96, 266, 750, 290), "Select NVC root and Python runtime",
                                              IText(12.f, kAmber, "Roboto-Regular", EAlign::Near)), kCtrlStatusDetail);

    // 3x3 slider grid
    struct SliderLayout { int param; const char* label; };
    const SliderLayout sliders[] = {
      {kPitch, "PITCH"}, {kFormant, "FORMANT"}, {kIndexRate, "INDEX"},
      {kRmsMix, "RMS MIX"}, {kThreshold, "GATE"}, {kDryWet, "MIX"},
      {kBlockMs, "BLOCK"}, {kCrossfadeMs, "CROSSFADE"}, {kExtraMs, "CONTEXT"}
    };
    const float gridLeft = 30.f, gridTop = 300.f;
    const float cellWidth = 226.f, cellHeight = 62.f;
    const float gapX = 21.f, gapY = 10.f;
    for (int i = 0; i < 9; ++i) {
      const int column = i % 3;
      const int row = i / 3;
      const float x = gridLeft + column * (cellWidth + gapX);
      const float y = gridTop + row * (cellHeight + gapY);
      const IRECT cell(x, y, x + cellWidth, y + cellHeight);
      graphics->AttachControl(new IPanelControl(cell, kPanel));
      graphics->AttachControl(new NVCSliderControl(cell.GetPadded(-8.f), sliders[i].param,
                                                   sliders[i].label, style, true, EDirection::Horizontal));
    }

    // Bottom row: F0 method, output gain, engine toggle — one shared baseline
    const float bottomY = 566.f, bottomHeight = 44.f;
    graphics->AttachControl(new ITextControl(IRECT(30, bottomY, 100, bottomY + bottomHeight), "F0 METHOD",
                                              IText(12.f, kMuted, "Roboto-Regular", EAlign::Near)));
    graphics->AttachControl(new IVMenuButtonControl(IRECT(104, bottomY, 236, bottomY + bottomHeight), kF0Method,
                                                    "", style));
    graphics->AttachControl(new IPanelControl(IRECT(258, bottomY, 560, bottomY + bottomHeight), kPanel));
    graphics->AttachControl(new NVCSliderControl(IRECT(266, bottomY + 2, 552, bottomY + bottomHeight - 2), kOutputGain,
                                                 "OUTPUT", style, true, EDirection::Horizontal));
    graphics->AttachControl(new IVToggleControl(IRECT(600, bottomY, 750, bottomY + bottomHeight), kEngine,
                                                "ENGINE", style));
  };
#endif
}

#if IPLUG_DSP
void NVCRealtime::OnReset()
{
  const double sampleRate = GetSampleRate() > 0.0 ? GetSampleRate() : 48000.0;
  const int blockSize = std::max(1, GetBlockSize());
  mWorker.setSampleRate(sampleRate);
  ResizeBuffers(blockSize, sampleRate);
  SyncParametersToWorker();
  const int latency = CalculateLatencyFrames(GetParam(kBlockMs)->Value());
  mTargetDelayFrames.store(latency, std::memory_order_relaxed);
  SetLatency(latency);
}

void NVCRealtime::OnParamChange(const int paramIdx)
{
  if (paramIdx < 0 || paramIdx >= kNumParams)
    return;
  double value = GetParam(paramIdx)->Value();
  if (paramIdx == kDryWet)
    value /= 100.0;
  mWorker.setParameter(static_cast<nvc::ParameterId>(paramIdx), static_cast<float>(value));
  if (paramIdx == kEngine) {
    if (value >= 0.5) {
      std::string error;
      if (!ValidateConfiguration(error)) {
        {
          std::lock_guard<std::mutex> lock(mStateMutex);
          mValidationMessage.Set(error.c_str());
        }
        GetParam(kEngine)->Set(0.0);
        SendParameterValueFromAPI(kEngine, 0.0, false);
        mWorker.setEnabled(false);
        return;
      }
      {
        std::lock_guard<std::mutex> lock(mStateMutex);
        mValidationMessage.Set("");
      }
      SaveUserConfiguration();
      mWorker.setEnabled(true);
    } else {
      mWorker.setEnabled(false);
    }
  }
  if (paramIdx == kBlockMs) {
    const int latency = CalculateLatencyFrames(value);
    mTargetDelayFrames.store(latency, std::memory_order_relaxed);
    SetLatency(latency);
  }
}

void NVCRealtime::ProcessBlock(sample** inputs, sample** outputs, const int nFrames)
{
  const int nIn = NInChansConnected();
  const int nOut = NOutChansConnected();
  if (nFrames <= 0 || nOut <= 0)
    return;

  if (nIn <= 0) {
    for (int channel = 0; channel < nOut; ++channel)
      std::memset(outputs[channel], 0, static_cast<size_t>(nFrames) * sizeof(sample));
    return;
  }

  // Mono-safe passthrough: never leave output buffers unwritten.
  auto passthrough = [&]() {
    for (int channel = 0; channel < nOut; ++channel) {
      const sample* source = inputs[std::min(channel, nIn - 1)];
      if (outputs[channel] != source)
        std::memcpy(outputs[channel], source, static_cast<size_t>(nFrames) * sizeof(sample));
    }
  };

  if (nFrames > static_cast<int>(mMonoInput.size())) {
    passthrough();
    return;
  }

  const bool requested = GetParam(kEngine)->Bool();
  if (!requested || !mWorker.isReady()) {
    mActiveBlend = std::max(0.0f, mActiveBlend - static_cast<float>(nFrames) / 1024.0f);
    passthrough();
    return;
  }

  const sample* inL = inputs[0];
  const sample* inR = inputs[nIn > 1 ? 1 : 0];
  sample* outL = outputs[0];
  sample* outR = nOut > 1 ? outputs[1] : nullptr;

  for (int frame = 0; frame < nFrames; ++frame)
    mMonoInput[static_cast<size_t>(frame)] = static_cast<float>(0.5 * (inL[frame] + inR[frame]));
  mWorker.pushInput(mMonoInput.data(), static_cast<size_t>(nFrames));
  const size_t wetRead = mWorker.popOutput(mWetOutput.data(), static_cast<size_t>(nFrames));
  std::fill(mWetOutput.begin() + static_cast<ptrdiff_t>(wetRead),
            mWetOutput.begin() + nFrames, 0.0f);

  const float mix = static_cast<float>(GetParam(kDryWet)->Value() / 100.0);
  const float gain = static_cast<float>(std::pow(10.0, GetParam(kOutputGain)->Value() / 20.0));
  const int maxDelay = static_cast<int>(mDryDelay.size() / 2);
  const int delay = std::clamp(mTargetDelayFrames.load(std::memory_order_relaxed), 0, maxDelay - 1);
  const int readPosition = (mDryWritePosition + maxDelay - delay) % maxDelay;
  mActiveBlend = std::min(1.0f, mActiveBlend + static_cast<float>(nFrames) / 2048.0f);

  for (int frame = 0; frame < nFrames; ++frame) {
    const int write = (mDryWritePosition + frame) % maxDelay;
    const int read = (readPosition + frame) % maxDelay;
    mDryDelay[static_cast<size_t>(write) * 2] = static_cast<float>(inL[frame]);
    mDryDelay[static_cast<size_t>(write) * 2 + 1] = static_cast<float>(inR[frame]);
    const float wet = mWetOutput[static_cast<size_t>(frame)] * gain;
    const float processedL = mDryDelay[static_cast<size_t>(read) * 2] * (1.0f - mix) + wet * mix;
    const float processedR = mDryDelay[static_cast<size_t>(read) * 2 + 1] * (1.0f - mix) + wet * mix;
    outL[frame] = static_cast<sample>(inL[frame] * (1.0f - mActiveBlend) + processedL * mActiveBlend);
    if (outR)
      outR[frame] = static_cast<sample>(inR[frame] * (1.0f - mActiveBlend) + processedR * mActiveBlend);
  }
  mDryWritePosition = (mDryWritePosition + nFrames) % maxDelay;

  // Copy to any extra connected outputs beyond stereo.
  for (int channel = 2; channel < nOut; ++channel)
    std::memcpy(outputs[channel], outputs[channel & 1], static_cast<size_t>(nFrames) * sizeof(sample));
}
#endif

void NVCRealtime::OnIdle()
{
#if IPLUG_EDITOR
  if (GetUI() == nullptr)
    return;
  std::string validationMessage;
  {
    std::lock_guard<std::mutex> lock(mStateMutex);
    validationMessage = mValidationMessage.Get();
  }
  if (auto* status = GetUI()->GetControlWithTag(kCtrlStatus))
    status->As<ITextControl>()->SetStr(validationMessage.empty() ? StatusName(mWorker.status()) : "CONFIG ERROR");
  if (auto* detail = GetUI()->GetControlWithTag(kCtrlStatusDetail))
    detail->As<ITextControl>()->SetStr(validationMessage.empty() ? mWorker.statusText().c_str() : validationMessage.c_str());
  if (auto* performance = GetUI()->GetControlWithTag(kCtrlPerformance))
    performance->As<ITextControl>()->SetStrFmt(80, "%.0f ms / %.0f drop", mWorker.inferMs(), mWorker.droppedBlocks());
#endif
}

void NVCRealtime::OnUIOpen()
{
  Plugin::OnUIOpen(); // pushes current param values to the UI controls
  UpdateFileLabels();
}

bool NVCRealtime::SerializeState(IByteChunk& chunk) const
{
  std::lock_guard<std::mutex> lock(mStateMutex);
  chunk.PutStr(mModelPath.Get());
  chunk.PutStr(mIndexPath.Get());
  chunk.PutStr(mNvcRoot.Get());
  chunk.PutStr(mPythonPath.Get());
  return SerializeParams(chunk);
}

int NVCRealtime::UnserializeState(const IByteChunk& chunk, int startPos)
{
  WDL_String model, index, root, python;
  startPos = chunk.GetStr(model, startPos);
  startPos = chunk.GetStr(index, startPos);
  startPos = chunk.GetStr(root, startPos);
  startPos = chunk.GetStr(python, startPos);
  if (startPos < 0)
    return startPos;
  {
    std::lock_guard<std::mutex> lock(mStateMutex);
    mModelPath.Set(model.Get());
    mIndexPath.Set(index.Get());
    mNvcRoot.Set(root.Get());
    mPythonPath.Set(python.Get());
  }
  mWorker.setPath(nvc::kStateModelPath, model.Get());
  mWorker.setPath(nvc::kStateIndexPath, index.Get());
  mWorker.setPath(nvc::kStateNvcRoot, root.Get());
  mWorker.setPath(nvc::kStatePythonPath, python.Get());
  startPos = UnserializeParams(chunk, startPos);
  SyncParametersToWorker();
  UpdateFileLabels();
  return startPos;
}

void NVCRealtime::SyncParametersToWorker()
{
  for (int i = 0; i < kNumParams; ++i)
    OnParamChange(i);
}

int NVCRealtime::CalculateLatencyFrames(const double blockMs) const
{
  const double sampleRate = GetSampleRate() > 0.0 ? GetSampleRate() : 48000.0;
  const double zc = std::max(1.0, std::floor(sampleRate / 100.0));
  const int block = static_cast<int>(std::round(blockMs / 1000.0 * sampleRate / zc) * zc);
  return block * 2;
}

void NVCRealtime::ResizeBuffers(const int blockSize, const double sampleRate)
{
  const size_t scratch = static_cast<size_t>(std::max(blockSize, 131072));
  mMonoInput.assign(scratch, 0.0f);
  mWetOutput.assign(scratch, 0.0f);
  mDryDelay.assign(static_cast<size_t>(std::max(1.0, sampleRate * 4.0)) * 2, 0.0f);
  mDryWritePosition = 0;
  mActiveBlend = 0.0f;
}

void NVCRealtime::ChooseNvcRoot(IGraphics* graphics)
{
  WDL_String directory(mNvcRoot.Get());
  graphics->PromptForDirectory(directory, [this](const WDL_String&, const WDL_String& selectedDirectory) {
    if (selectedDirectory.GetLength() > 0)
      SetNvcRoot(selectedDirectory.Get());
  });
}

void NVCRealtime::ChoosePython(IGraphics* graphics)
{
  WDL_String fileName, path;
  const std::string currentDirectory = ParentDirectory(mPythonPath.Get());
  if (!currentDirectory.empty())
    path.Set(currentDirectory.c_str());
  else if (mNvcRoot.GetLength() > 0)
    path.Set(JoinPath(mNvcRoot.Get(), "runtime").c_str());
  graphics->PromptForFile(fileName, path, EFileAction::Open, "exe",
                          [this](const WDL_String& selected, const WDL_String&) {
    if (selected.GetLength() > 0)
      SetPythonPath(selected.Get());
  });
}

void NVCRealtime::ChooseModel(IGraphics* graphics)
{
  WDL_String fileName, path(mModelBrowseDirectory.Get());
  if (path.GetLength() == 0)
    path.Set(mNvcRoot.Get());
  graphics->PromptForFile(fileName, path, EFileAction::Open, "pth",
                          [this](const WDL_String& selected, const WDL_String& selectedDirectory) {
    if (selected.GetLength() > 0) {
      mModelBrowseDirectory.Set(selectedDirectory.Get());
      SetModelPath(selected.Get());
    }
  });
}

void NVCRealtime::ChooseIndex(IGraphics* graphics)
{
  WDL_String fileName, path(mIndexBrowseDirectory.Get());
  if (path.GetLength() == 0)
    path.Set(mNvcRoot.Get());
  graphics->PromptForFile(fileName, path, EFileAction::Open, "index",
                          [this](const WDL_String& selected, const WDL_String& selectedDirectory) {
    if (selected.GetLength() > 0) {
      mIndexBrowseDirectory.Set(selectedDirectory.Get());
      SetIndexPath(selected.Get());
    }
  });
}

void NVCRealtime::SetNvcRoot(const char* path)
{
  const std::string root = TrimTrailingSeparators(path != nullptr ? path : "");
  const std::string detectedPython = JoinPath(root, "runtime\\python.exe");
  const bool pythonDetected = PathIsFile(detectedPython);
  {
    std::lock_guard<std::mutex> lock(mStateMutex);
    mNvcRoot.Set(root.c_str());
    mPythonPath.Set(pythonDetected ? detectedPython.c_str() : "");
    mModelBrowseDirectory.Set(root.c_str());
    mIndexBrowseDirectory.Set(root.c_str());
    mValidationMessage.Set(pythonDetected ? "" : "runtime\\python.exe not found; select Python manually.");
  }
  mWorker.setPath(nvc::kStateNvcRoot, root.c_str());
  mWorker.setPath(nvc::kStatePythonPath, pythonDetected ? detectedPython.c_str() : "");
  StopEngineForPathChange();
  UpdateFileLabels();
}

void NVCRealtime::SetPythonPath(const char* path)
{
  const char* value = path != nullptr ? path : "";
  {
    std::lock_guard<std::mutex> lock(mStateMutex);
    mPythonPath.Set(value);
    mValidationMessage.Set("");
  }
  mWorker.setPath(nvc::kStatePythonPath, value);
  StopEngineForPathChange();
  UpdateFileLabels();
}

void NVCRealtime::SetModelPath(const char* path)
{
  {
    std::lock_guard<std::mutex> lock(mStateMutex);
    mModelPath.Set(path);
  }
  mWorker.setPath(nvc::kStateModelPath, path);
  StopEngineForPathChange();
  UpdateFileLabels();
}

void NVCRealtime::SetIndexPath(const char* path)
{
  {
    std::lock_guard<std::mutex> lock(mStateMutex);
    mIndexPath.Set(path);
  }
  mWorker.setPath(nvc::kStateIndexPath, path);
  StopEngineForPathChange();
  UpdateFileLabels();
}

void NVCRealtime::StopEngineForPathChange()
{
  mWorker.setEnabled(false);
  if (GetParam(kEngine)->Bool()) {
    GetParam(kEngine)->Set(0.0);
    SendParameterValueFromAPI(kEngine, 0.0, false);
  }
}

bool NVCRealtime::ValidateConfiguration(std::string& error) const
{
  std::string root, python, model, index;
  {
    std::lock_guard<std::mutex> lock(mStateMutex);
    root = mNvcRoot.Get();
    python = mPythonPath.Get();
    model = mModelPath.Get();
    index = mIndexPath.Get();
  }
  if (root.empty()) { error = "Select the NVC package root."; return false; }
  if (!PathIsDirectory(root)) { error = "NVC root folder does not exist."; return false; }
  if (!PathIsFile(JoinPath(root, "infer\\rtnvc.py"))) { error = "NVC source not found: infer\\rtnvc.py."; return false; }
  if (!PathIsFile(JoinPath(root, "configs\\config.py"))) { error = "NVC source not found: configs\\config.py."; return false; }
  if (python.empty()) { error = "runtime\\python.exe not found; select Python manually."; return false; }
  if (!PathIsFile(python)) { error = "Selected Python executable does not exist."; return false; }
  if (!Is64BitExecutable(python)) { error = "Selected Python must be a 64-bit executable."; return false; }
  if (model.empty()) { error = "Select an NVC .pth model."; return false; }
  if (!PathIsFile(model)) { error = "Selected model file does not exist."; return false; }
  if (!index.empty() && !PathIsFile(index)) { error = "Selected index file does not exist."; return false; }
  error.clear();
  return true;
}

void NVCRealtime::LoadUserConfiguration()
{
  const std::string root = ReadSetting(L"NvcRoot");
  const std::string python = ReadSetting(L"PythonPath");
  const std::string model = ReadSetting(L"ModelPath");
  const std::string index = ReadSetting(L"IndexPath");
  if (!root.empty()) mNvcRoot.Set(root.c_str());
  if (!python.empty()) mPythonPath.Set(python.c_str());
  if (!model.empty()) mModelPath.Set(model.c_str());
  if (!index.empty()) mIndexPath.Set(index.c_str());
}

void NVCRealtime::SaveUserConfiguration() const
{
  std::string root, python, model, index;
  {
    std::lock_guard<std::mutex> lock(mStateMutex);
    root = mNvcRoot.Get();
    python = mPythonPath.Get();
    model = mModelPath.Get();
    index = mIndexPath.Get();
  }
  const std::wstring settingsPath = SettingsFilePath(true);
  WriteSetting(settingsPath, L"NvcRoot", root);
  WriteSetting(settingsPath, L"PythonPath", python);
  WriteSetting(settingsPath, L"ModelPath", model);
  WriteSetting(settingsPath, L"IndexPath", index);
}

void NVCRealtime::UpdateFileLabels()
{
#if IPLUG_EDITOR
  if (GetUI() == nullptr)
    return;
  std::lock_guard<std::mutex> lock(mStateMutex);
  if (auto* root = GetUI()->GetControlWithTag(kCtrlNvcRoot))
    root->As<ITextControl>()->SetStr(mNvcRoot.Get());
  if (auto* python = GetUI()->GetControlWithTag(kCtrlPythonPath))
    python->As<ITextControl>()->SetStr(mPythonPath.Get());
  if (auto* model = GetUI()->GetControlWithTag(kCtrlModelName))
    model->As<ITextControl>()->SetStr(mModelPath.get_filepart());
  if (auto* index = GetUI()->GetControlWithTag(kCtrlIndexName))
    index->As<ITextControl>()->SetStr(mIndexPath.get_filepart());
#endif
}
