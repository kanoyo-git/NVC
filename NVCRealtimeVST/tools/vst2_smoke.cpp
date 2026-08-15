#include <windows.h>

#include "aeffectx.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

VstIntPtr VSTCALLBACK hostCallback(AEffect*, VstInt32 opcode, VstInt32, VstIntPtr, void*, float)
{
    if (opcode == audioMasterVersion)
        return kVstVersion;
    return 0;
}

std::wstring widen(const char* text)
{
    if (text == nullptr)
        return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    std::wstring result(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text, -1, result.data(), size);
    if (!result.empty())
        result.pop_back();
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: nvc-vst2-smoke <plugin.dll>\n";
        return EXIT_FAILURE;
    }

    const std::wstring path = widen(argv[1]);
    HMODULE module = LoadLibraryW(path.c_str());
    if (module == nullptr) {
        std::cerr << "LoadLibraryW failed: " << GetLastError() << '\n';
        return EXIT_FAILURE;
    }

    using PluginMain = AEffect* (VSTCALLBACK*)(audioMasterCallback);
    const auto pluginMain = reinterpret_cast<PluginMain>(GetProcAddress(module, "VSTPluginMain"));
    if (pluginMain == nullptr) {
        std::cerr << "VSTPluginMain export is missing\n";
        FreeLibrary(module);
        return EXIT_FAILURE;
    }

    AEffect* effect = pluginMain(hostCallback);
    if (effect == nullptr || effect->magic != kEffectMagic || effect->dispatcher == nullptr
        || effect->processReplacing == nullptr) {
        std::cerr << "Invalid VST2 effect structure\n";
        FreeLibrary(module);
        return EXIT_FAILURE;
    }

    effect->dispatcher(effect, effOpen, 0, 0, nullptr, 0.0f);
    effect->dispatcher(effect, effSetSampleRate, 0, 0, nullptr, 48000.0f);
    constexpr VstInt32 frames = 512;
    effect->dispatcher(effect, effSetBlockSize, 0, frames, nullptr, 0.0f);
    effect->dispatcher(effect, effMainsChanged, 0, 1, nullptr, 0.0f);

    std::vector<float> left(frames);
    std::vector<float> right(frames);
    std::vector<float> outputLeft(frames, -10.0f);
    std::vector<float> outputRight(frames, -10.0f);
    for (VstInt32 i = 0; i < frames; ++i) {
        left[static_cast<std::size_t>(i)] = static_cast<float>(0.1 * std::sin(i * 0.03));
        right[static_cast<std::size_t>(i)] = static_cast<float>(0.1 * std::cos(i * 0.03));
    }
    float* inputs[] = {left.data(), right.data()};
    float* outputs[] = {outputLeft.data(), outputRight.data()};
    effect->processReplacing(effect, inputs, outputs, frames);

    float maxError = 0.0f;
    for (VstInt32 i = 0; i < frames; ++i) {
        const auto index = static_cast<std::size_t>(i);
        maxError = std::max(maxError, std::abs(outputLeft[index] - left[index]));
        maxError = std::max(maxError, std::abs(outputRight[index] - right[index]));
    }

    const VstInt32 inputsCount = effect->numInputs;
    const VstInt32 outputsCount = effect->numOutputs;
    const VstInt32 paramsCount = effect->numParams;
    effect->dispatcher(effect, effMainsChanged, 0, 0, nullptr, 0.0f);
    effect->dispatcher(effect, effClose, 0, 0, nullptr, 0.0f);
    FreeLibrary(module);

    if (inputsCount != 2 || outputsCount != 2 || paramsCount != 12 || maxError > 1e-6f) {
        std::cerr << "Unexpected VST2 behavior: inputs=" << inputsCount
                  << " outputs=" << outputsCount << " params=" << paramsCount
                  << " max_error=" << maxError << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "VST2 OK inputs=" << inputsCount << " outputs=" << outputsCount
              << " params=" << paramsCount << " dry_max_error=" << maxError << '\n';
    return EXIT_SUCCESS;
}
