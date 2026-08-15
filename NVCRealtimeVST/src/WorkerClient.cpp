#include "WorkerClient.hpp"
#include "config.h"

#if !defined(_WIN32)
#error NVC Realtime worker bridge currently targets Windows.
#endif

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace nvc {
namespace {

constexpr uint32_t kMagic = 0x50564E43; // NVCP
constexpr uint32_t kProtocolVersion = 1;
constexpr uint32_t kHeaderBytes = 4096;
constexpr uint32_t kMaxFrames = 131072;
constexpr uint32_t kMapBytes = kHeaderBytes + kMaxFrames * sizeof(float) * 2;
constexpr uint32_t kInputOffset = kHeaderBytes;
constexpr uint32_t kOutputOffset = kHeaderBytes + kMaxFrames * sizeof(float);
constexpr uint32_t kStatusTextOffset = 128;
constexpr uint32_t kStatusTextBytes = 512;

template <typename T>
void writeAt(void* const base, const std::size_t offset, const T value)
{
    std::memcpy(static_cast<unsigned char*>(base) + offset, &value, sizeof(T));
}

template <typename T>
T readAt(const void* const base, const std::size_t offset)
{
    T value {};
    std::memcpy(&value, static_cast<const unsigned char*>(base) + offset, sizeof(T));
    return value;
}

std::wstring utf8ToWide(const std::string& text)
{
    if (text.empty())
        return {};
    const int length = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    std::wstring result(static_cast<std::size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, result.data(), length);
    if (!result.empty() && result.back() == L'\0')
        result.pop_back();
    return result;
}

std::string wideToUtf8(const std::wstring& text)
{
    if (text.empty())
        return {};
    const int length = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<std::size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, result.data(), length, nullptr, nullptr);
    if (!result.empty() && result.back() == '\0')
        result.pop_back();
    return result;
}

std::wstring moduleDirectory()
{
    static int moduleAnchor = 0;
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&moduleAnchor), &module))
        return {};
    std::vector<wchar_t> path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length >= path.size())
        return {};
    std::wstring result(path.data(), length);
    const std::size_t separator = result.find_last_of(L"\\/");
    return separator == std::wstring::npos ? std::wstring() : result.substr(0, separator);
}

bool isFile(const std::wstring& path)
{
    const DWORD attributes = GetFileAttributesW(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

std::wstring workerScriptPath()
{
    const std::wstring directory = moduleDirectory();
    if (directory.empty())
        return {};
    const std::wstring relativeWorker = utf8ToWide(NVC_WORKER_RELATIVE_PATH);
    const std::wstring vst2Resources = utf8ToWide(NVC_VST2_RESOURCES_DIR);

    const std::wstring vst3Path = directory + L"\\..\\Resources\\" + relativeWorker;
    if (isFile(vst3Path))
        return vst3Path;

    const std::wstring vst2Path = directory + L"\\" + vst2Resources + L"\\" + relativeWorker;
    if (isFile(vst2Path))
        return vst2Path;

    return {};
}

bool ensureDirectory(const std::wstring& path, std::string& error)
{
    if (CreateDirectoryW(path.c_str(), nullptr) != FALSE)
        return true;
    const DWORD code = GetLastError();
    if (code == ERROR_ALREADY_EXISTS) {
        const DWORD attributes = GetFileAttributesW(path.c_str());
        if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
            return true;
    }
    error = "Cannot create " + wideToUtf8(path) + " (Windows error " + std::to_string(code) + ")";
    return false;
}

std::wstring temporaryLogDirectory(std::string& error)
{
    std::vector<wchar_t> temporaryPath(32768, L'\0');
    const DWORD length = GetTempPathW(static_cast<DWORD>(temporaryPath.size()), temporaryPath.data());
    if (length == 0 || length >= temporaryPath.size()) {
        error = "Windows temporary directory is unavailable (Windows error " + std::to_string(GetLastError()) + ")";
        return {};
    }

    std::wstring productDirectory(temporaryPath.data(), length);
    if (!productDirectory.empty() && productDirectory.back() != L'\\')
        productDirectory += L'\\';
    productDirectory += L"NVCRealtime";
    if (!ensureDirectory(productDirectory, error))
        return {};

    const std::wstring logDirectory = productDirectory + L"\\logs";
    if (!ensureDirectory(logDirectory, error))
        return {};
    return logDirectory;
}

std::string jsonEscape(const std::string& text)
{
    std::ostringstream out;
    for (const unsigned char ch : text) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20)
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
            else
                out << static_cast<char>(ch);
        }
    }
    return out.str();
}

std::string quoteArg(const std::string& text)
{
    return "\"" + text + "\"";
}

} // namespace

struct WorkerClient::Ipc {
    HANDLE mapping = nullptr;
    HANDLE requestEvent = nullptr;
    HANDLE responseEvent = nullptr;
    HANDLE process = nullptr;
    HANDLE processThread = nullptr;
    void* view = nullptr;
    std::string mapName;
    std::string requestName;
    std::string responseName;
    std::string configPath;
    uint32_t sequence = 0;

    ~Ipc()
    {
        if (view != nullptr)
            UnmapViewOfFile(view);
        if (mapping != nullptr)
            CloseHandle(mapping);
        if (requestEvent != nullptr)
            CloseHandle(requestEvent);
        if (responseEvent != nullptr)
            CloseHandle(responseEvent);
        if (processThread != nullptr)
            CloseHandle(processThread);
        if (process != nullptr)
            CloseHandle(process);
    }
};

WorkerClient::WorkerClient()
{
    parameters_[kParamPitch].store(12.0f);
    parameters_[kParamFormant].store(0.0f);
    parameters_[kParamIndexRate].store(0.0f);
    parameters_[kParamRmsMix].store(0.5f);
    parameters_[kParamThreshold].store(-60.0f);
    parameters_[kParamBlockMs].store(130.0f);
    parameters_[kParamCrossfadeMs].store(80.0f);
    parameters_[kParamExtraMs].store(2000.0f);
    parameters_[kParamF0Method].store(0.0f);

    paths_.model = NVC_DEFAULT_MODEL;
    paths_.index = NVC_DEFAULT_INDEX;
    paths_.nvcRoot = NVC_DEFAULT_ROOT;
    paths_.python = NVC_DEFAULT_PYTHON;

    thread_ = std::thread(&WorkerClient::threadMain, this);
}

WorkerClient::~WorkerClient()
{
    stopRequested_.store(true, std::memory_order_release);
    if (thread_.joinable())
        thread_.join();
}

void WorkerClient::setEnabled(const bool enabled) noexcept
{
    if (enabled_.exchange(enabled, std::memory_order_acq_rel) != enabled)
        configVersion_.fetch_add(1, std::memory_order_release);
}

void WorkerClient::setSampleRate(const double sampleRate) noexcept
{
    if (std::abs(sampleRate_.load(std::memory_order_relaxed) - sampleRate) > 0.5) {
        sampleRate_.store(sampleRate, std::memory_order_relaxed);
        configVersion_.fetch_add(1, std::memory_order_release);
    }
}

void WorkerClient::setParameter(const ParameterId id, const float value) noexcept
{
    if (id >= kParameterCount)
        return;
    const float old = parameters_[id].exchange(value, std::memory_order_relaxed);
    if ((id == kParamBlockMs || id == kParamCrossfadeMs || id == kParamExtraMs) && std::abs(old - value) > 0.01f)
        configVersion_.fetch_add(1, std::memory_order_release);
}

void WorkerClient::setPath(const StateId id, const char* const value)
{
    if (value == nullptr)
        return;
    std::lock_guard<std::mutex> lock(pathsMutex_);
    switch (id) {
    case kStateModelPath: paths_.model = value; break;
    case kStateIndexPath: paths_.index = value; break;
    case kStateNvcRoot: paths_.nvcRoot = value; break;
    case kStatePythonPath: paths_.python = value; break;
    default: return;
    }
    configVersion_.fetch_add(1, std::memory_order_release);
}

std::size_t WorkerClient::pushInput(const float* const samples, const std::size_t count) noexcept
{
    if (!isReady())
        return 0;
    const std::size_t pushed = inputRing_.push(samples, count);
    if (pushed != count)
        droppedBlocks_.fetch_add(1, std::memory_order_relaxed);
    return pushed;
}

std::size_t WorkerClient::popOutput(float* const samples, const std::size_t count) noexcept
{
    if (!isReady())
        return 0;
    return outputRing_.pop(samples, count);
}

std::string WorkerClient::statusText() const
{
    std::lock_guard<std::mutex> lock(statusTextMutex_);
    return statusText_;
}

WorkerClient::Paths WorkerClient::pathsSnapshot() const
{
    std::lock_guard<std::mutex> lock(pathsMutex_);
    return paths_;
}

void WorkerClient::setStatus(const int status, const std::string& text)
{
    status_.store(status, std::memory_order_release);
    std::lock_guard<std::mutex> lock(statusTextMutex_);
    statusText_ = text;
}

uint32_t WorkerClient::calculateBlockFrames() const noexcept
{
    const double sampleRate = sampleRate_.load(std::memory_order_relaxed);
    const double zc = std::max(1.0, std::floor(sampleRate / 100.0));
    const double seconds = parameters_[kParamBlockMs].load(std::memory_order_relaxed) / 1000.0;
    const auto frames = static_cast<uint32_t>(std::round(seconds * sampleRate / zc) * zc);
    return std::clamp<uint32_t>(frames, static_cast<uint32_t>(zc), kMaxFrames);
}

std::string WorkerClient::writeWorkerConfig(const Paths& paths, std::string& error) const
{
    const DWORD pid = GetCurrentProcessId();
    const std::wstring logDirectory = temporaryLogDirectory(error);
    if (logDirectory.empty())
        return {};

    std::wostringstream name;
    name << logDirectory << L"\\instance_" << pid << L"_" << GetTickCount64() << L".json";
    const std::wstring configPath = name.str();
    HANDLE file = CreateFileW(configPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                              CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        error = "Cannot create " + wideToUtf8(configPath) + " (Windows error "
            + std::to_string(GetLastError()) + ")";
        return {};
    }

    std::ostringstream json;
    json << "{\n"
         << "  \"nvc_root\": \"" << jsonEscape(paths.nvcRoot) << "\",\n"
         << "  \"model_path\": \"" << jsonEscape(paths.model) << "\",\n"
         << "  \"index_path\": \"" << jsonEscape(paths.index) << "\",\n"
         << "  \"sample_rate\": " << static_cast<uint32_t>(sampleRate_.load()) << ",\n"
         << "  \"block_ms\": " << parameters_[kParamBlockMs].load() << ",\n"
         << "  \"crossfade_ms\": " << parameters_[kParamCrossfadeMs].load() << ",\n"
         << "  \"extra_ms\": " << parameters_[kParamExtraMs].load() << "\n"
         << "}\n";
    const std::string contents = json.str();
    DWORD bytesWritten = 0;
    const BOOL written = WriteFile(file, contents.data(), static_cast<DWORD>(contents.size()), &bytesWritten, nullptr);
    const DWORD writeError = written != FALSE ? ERROR_SUCCESS : GetLastError();
    CloseHandle(file);
    if (written == FALSE || bytesWritten != static_cast<DWORD>(contents.size())) {
        DeleteFileW(configPath.c_str());
        error = "Cannot write " + wideToUtf8(configPath) + " (Windows error "
            + std::to_string(writeError) + ")";
        return {};
    }
    error.clear();
    return wideToUtf8(configPath);
}

bool WorkerClient::launchWorker(const Paths& paths, const uint64_t)
{
    ipc_ = std::make_unique<Ipc>();
    const DWORD pid = GetCurrentProcessId();
    static std::atomic<uint32_t> instanceCounter {0};
    const uint32_t instance = instanceCounter.fetch_add(1);
    const std::string id = std::to_string(pid) + "_" + std::to_string(instance) + "_" + std::to_string(GetTickCount64());
    ipc_->mapName = "Local\\NVCVST_" + id + "_map";
    ipc_->requestName = "Local\\NVCVST_" + id + "_request";
    ipc_->responseName = "Local\\NVCVST_" + id + "_response";

    const std::wstring mapName = utf8ToWide(ipc_->mapName);
    const std::wstring requestName = utf8ToWide(ipc_->requestName);
    const std::wstring responseName = utf8ToWide(ipc_->responseName);
    ipc_->mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, kMapBytes, mapName.c_str());
    ipc_->requestEvent = CreateEventW(nullptr, FALSE, FALSE, requestName.c_str());
    ipc_->responseEvent = CreateEventW(nullptr, FALSE, FALSE, responseName.c_str());
    if (ipc_->mapping == nullptr || ipc_->requestEvent == nullptr || ipc_->responseEvent == nullptr) {
        setStatus(kStatusError, "IPC initialization failed");
        stopWorker();
        return false;
    }

    ipc_->view = MapViewOfFile(ipc_->mapping, FILE_MAP_ALL_ACCESS, 0, 0, kMapBytes);
    if (ipc_->view == nullptr) {
        setStatus(kStatusError, "Shared memory mapping failed");
        stopWorker();
        return false;
    }
    std::memset(ipc_->view, 0, kMapBytes);
    writeAt<uint32_t>(ipc_->view, 0, kMagic);
    writeAt<uint32_t>(ipc_->view, 4, kProtocolVersion);
    writeAt<int32_t>(ipc_->view, 8, kStatusStarting);

    std::string configError;
    ipc_->configPath = writeWorkerConfig(paths, configError);
    if (ipc_->configPath.empty()) {
        setStatus(kStatusError, configError.empty() ? "Could not write worker configuration" : configError);
        stopWorker();
        return false;
    }

    const std::wstring workerScript = workerScriptPath();
    if (workerScript.empty()) {
        setStatus(kStatusError, "Plugin worker resource is missing");
        stopWorker();
        return false;
    }
    std::string command = quoteArg(paths.python) + " -I " + quoteArg(wideToUtf8(workerScript))
        + " --map " + quoteArg(ipc_->mapName)
        + " --request " + quoteArg(ipc_->requestName)
        + " --response " + quoteArg(ipc_->responseName)
        + " --config " + quoteArg(ipc_->configPath);
    std::wstring commandWide = utf8ToWide(command);
    std::vector<wchar_t> commandBuffer(commandWide.begin(), commandWide.end());
    commandBuffer.push_back(L'\0');

    STARTUPINFOW startup {};
    startup.cb = sizeof(startup);
    SECURITY_ATTRIBUTES security {};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;
    const std::wstring logPath = utf8ToWide(ipc_->configPath + ".process.log");
    HANDLE logFile = CreateFileW(logPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                 &security, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (logFile != INVALID_HANDLE_VALUE) {
        startup.dwFlags |= STARTF_USESTDHANDLES;
        startup.hStdOutput = logFile;
        startup.hStdError = logFile;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    }
    PROCESS_INFORMATION process {};
    const std::wstring cwd = utf8ToWide(paths.nvcRoot);
    const BOOL inheritHandles = logFile != INVALID_HANDLE_VALUE;
    const std::wstring application = utf8ToWide(paths.python);
    const BOOL launched = CreateProcessW(application.c_str(), commandBuffer.data(), nullptr, nullptr, inheritHandles,
                                         CREATE_NO_WINDOW, nullptr, cwd.c_str(), &startup, &process);
    if (logFile != INVALID_HANDLE_VALUE)
        CloseHandle(logFile);
    if (!launched) {
        setStatus(kStatusError, "Python worker launch failed");
        stopWorker();
        return false;
    }
    ipc_->process = process.hProcess;
    ipc_->processThread = process.hThread;
    setStatus(kStatusLoading, "Loading NVC model");
    return true;
}

void WorkerClient::stopWorker()
{
    ready_.store(false, std::memory_order_release);
    if (ipc_ && ipc_->view != nullptr && ipc_->requestEvent != nullptr) {
        writeAt<int32_t>(ipc_->view, 8, -2);
        SetEvent(ipc_->requestEvent);
    }
    if (ipc_ && ipc_->process != nullptr) {
        if (WaitForSingleObject(ipc_->process, 1200) == WAIT_TIMEOUT)
            TerminateProcess(ipc_->process, 0);
    }
    ipc_.reset();
}

bool WorkerClient::processOneBlock()
{
    if (!ipc_ || !ipc_->view)
        return false;
    const uint32_t frames = blockFrames_.load(std::memory_order_relaxed);
    if (inputRing_.readable() < frames)
        return true;
    if (requestBuffer_.size() < frames) {
        requestBuffer_.resize(frames);
        responseBuffer_.resize(frames);
    }
    if (inputRing_.pop(requestBuffer_.data(), frames) != frames)
        return true;

    const uint32_t sequence = ++ipc_->sequence;
    writeAt<uint32_t>(ipc_->view, 12, sequence);
    writeAt<uint32_t>(ipc_->view, 20, frames);
    writeAt<uint32_t>(ipc_->view, 24, static_cast<uint32_t>(sampleRate_.load(std::memory_order_relaxed)));
    writeAt<float>(ipc_->view, 32, parameters_[kParamPitch].load(std::memory_order_relaxed));
    writeAt<float>(ipc_->view, 36, parameters_[kParamFormant].load(std::memory_order_relaxed));
    writeAt<float>(ipc_->view, 40, parameters_[kParamIndexRate].load(std::memory_order_relaxed));
    writeAt<float>(ipc_->view, 44, parameters_[kParamRmsMix].load(std::memory_order_relaxed));
    writeAt<float>(ipc_->view, 48, parameters_[kParamThreshold].load(std::memory_order_relaxed));
    writeAt<uint32_t>(ipc_->view, 64, static_cast<uint32_t>(std::round(parameters_[kParamF0Method].load(std::memory_order_relaxed))));
    std::memcpy(static_cast<unsigned char*>(ipc_->view) + kInputOffset, requestBuffer_.data(), frames * sizeof(float));
    SetEvent(ipc_->requestEvent);

    const DWORD timeout = std::max<DWORD>(5000, static_cast<DWORD>(parameters_[kParamBlockMs].load() * 8.0f));
    const DWORD wait = WaitForSingleObject(ipc_->responseEvent, timeout);
    if (wait != WAIT_OBJECT_0) {
        setStatus(kStatusError, "NVC inference timed out");
        return false;
    }
    if (readAt<uint32_t>(ipc_->view, 16) != sequence) {
        droppedBlocks_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    const int workerState = readAt<int32_t>(ipc_->view, 8);
    if (workerState < 0) {
        const char* const text = reinterpret_cast<const char*>(static_cast<unsigned char*>(ipc_->view) + kStatusTextOffset);
        setStatus(kStatusError, text[0] != '\0' ? text : "Worker error");
        return false;
    }

    inferMs_.store(readAt<float>(ipc_->view, 56), std::memory_order_relaxed);
    std::memcpy(responseBuffer_.data(), static_cast<unsigned char*>(ipc_->view) + kOutputOffset, frames * sizeof(float));
    if (outputRing_.push(responseBuffer_.data(), frames) != frames)
        droppedBlocks_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void WorkerClient::threadMain()
{
    uint64_t activeVersion = 0;
    while (!stopRequested_.load(std::memory_order_acquire)) {
        const bool enabled = enabled_.load(std::memory_order_acquire);
        const uint64_t requestedVersion = configVersion_.load(std::memory_order_acquire);
        if (!enabled) {
            if (ipc_)
                stopWorker();
            setStatus(kStatusOff, "Off");
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
            continue;
        }

        if (!ipc_ || activeVersion != requestedVersion) {
            stopWorker();
            const Paths paths = pathsSnapshot();
            if (paths.model.empty() || paths.nvcRoot.empty() || paths.python.empty()) {
                setStatus(kStatusError, "Select a model and NVC runtime");
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            activeVersion = requestedVersion;
            blockFrames_.store(calculateBlockFrames(), std::memory_order_relaxed);
            latencyFrames_.store(blockFrames_.load() * 2, std::memory_order_relaxed);
            droppedBlocks_.store(0, std::memory_order_relaxed);
            inferMs_.store(0.0f, std::memory_order_relaxed);
            if (!launchWorker(paths, activeVersion)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
        }

        if (!ready_.load(std::memory_order_acquire)) {
            const int workerState = readAt<int32_t>(ipc_->view, 8);
            const char* const workerText = reinterpret_cast<const char*>(static_cast<unsigned char*>(ipc_->view) + kStatusTextOffset);
            if (workerState == kStatusReady) {
                inputRing_.resetUnsafe();
                outputRing_.resetUnsafe();
                outputRing_.pushZeros(latencyFrames_.load(std::memory_order_relaxed));
                setStatus(kStatusReady, workerText[0] != '\0' ? workerText : "Ready");
                ready_.store(true, std::memory_order_release);
            } else if (workerState < 0) {
                setStatus(kStatusError, workerText[0] != '\0' ? workerText : "Model load failed");
                stopWorker();
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            } else if (workerText[0] != '\0') {
                setStatus(kStatusLoading, workerText);
            }
            if (WaitForSingleObject(ipc_->process, 0) == WAIT_OBJECT_0) {
                if (workerState >= 0) {
                    DWORD exitCode = 0;
                    GetExitCodeProcess(ipc_->process, &exitCode);
                    setStatus(kStatusError, "Python worker exited with code " + std::to_string(exitCode));
                }
                stopWorker();
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }

        if (!processOneBlock()) {
            ready_.store(false, std::memory_order_release);
            stopWorker();
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }
        if (inputRing_.readable() < blockFrames_.load(std::memory_order_relaxed))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stopWorker();
}

} // namespace nvc
