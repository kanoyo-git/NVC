// Permissive compatibility names for iPlug2's VST2 adapter.
// The ABI values and layout are backed by the BSD-3-Clause clean-room SDK in
// third_party/vst2sdk. This file contains only the subset used by iPlug2.

#pragma once

#include "vst.h"

#include <cstddef>
#include <cstdint>

using VstInt16 = std::int16_t;
using VstInt32 = std::int32_t;
using VstInt64 = std::int64_t;
using VstIntPtr = std::intptr_t;

#define VSTCALLBACK VST_FUNCTION_INTERFACE

struct AEffect;

using audioMasterCallback = VstIntPtr (VSTCALLBACK*)(AEffect*, VstInt32, VstInt32, VstIntPtr, void*, float);
using AEffectDispatcherProc = VstIntPtr (VSTCALLBACK*)(AEffect*, VstInt32, VstInt32, VstIntPtr, void*, float);
using AEffectProcessProc = void (VSTCALLBACK*)(AEffect*, float**, float**, VstInt32);
using AEffectProcessDoubleProc = void (VSTCALLBACK*)(AEffect*, double**, double**, VstInt32);
using AEffectSetParameterProc = void (VSTCALLBACK*)(AEffect*, VstInt32, float);
using AEffectGetParameterProc = float (VSTCALLBACK*)(AEffect*, VstInt32);

#pragma pack(push, 8)

struct ERect {
    VstInt16 top;
    VstInt16 left;
    VstInt16 bottom;
    VstInt16 right;
};

struct AEffect {
    VstInt32 magic;
    AEffectDispatcherProc dispatcher;
    AEffectProcessProc __processDeprecated;
    AEffectSetParameterProc setParameter;
    AEffectGetParameterProc getParameter;
    VstInt32 numPrograms;
    VstInt32 numParams;
    VstInt32 numInputs;
    VstInt32 numOutputs;
    VstInt32 flags;
    VstIntPtr resvd1;
    VstIntPtr resvd2;
    VstInt32 initialDelay;
    VstInt32 __realQualitiesDeprecated;
    VstInt32 __offQualitiesDeprecated;
    float __ioRatioDeprecated;
    void* object;
    void* user;
    VstInt32 uniqueID;
    VstInt32 version;
    AEffectProcessProc processReplacing;
    AEffectProcessDoubleProc processDoubleReplacing;
    char future[56];
};

struct VstEvent {
    VstInt32 type;
    VstInt32 byteSize;
    VstInt32 deltaFrames;
    VstInt32 flags;
    char data[16];
};

struct VstEvents {
    VstInt32 numEvents;
    VstIntPtr reserved;
    VstEvent* events[2];
};

struct VstMidiEvent {
    VstInt32 type;
    VstInt32 byteSize;
    VstInt32 deltaFrames;
    VstInt32 flags;
    VstInt32 noteLength;
    VstInt32 noteOffset;
    char midiData[4];
    char detune;
    char noteOffVelocity;
    char reserved1;
    char reserved2;
};

struct VstMidiSysexEvent {
    VstInt32 type;
    VstInt32 byteSize;
    VstInt32 deltaFrames;
    VstInt32 flags;
    VstInt32 dumpBytes;
    VstIntPtr resvd1;
    char* sysexDump;
    VstIntPtr resvd2;
};

struct VstTimeInfo {
    double samplePos;
    double sampleRate;
    double nanoSeconds;
    double ppqPos;
    double tempo;
    double barStartPos;
    double cycleStartPos;
    double cycleEndPos;
    VstInt32 timeSigNumerator;
    VstInt32 timeSigDenominator;
    VstInt32 smpteOffset;
    VstInt32 smpteFrameRate;
    VstInt32 samplesToNextClock;
    VstInt32 flags;
};

struct VstVariableIo {
    float** inputs;
    float** outputs;
    VstInt32 numSamplesInput;
    VstInt32 numSamplesOutput;
    VstInt32* numSamplesInputProcessed;
    VstInt32* numSamplesOutputProcessed;
};

inline constexpr int kVstMaxLabelLen = 64;
inline constexpr int kVstMaxShortLabelLen = 8;
inline constexpr int kVstMaxCategLabelLen = 24;
inline constexpr int kVstMaxNameLen = 64;

struct VstParameterProperties {
    float stepFloat;
    float smallStepFloat;
    float largeStepFloat;
    char label[kVstMaxLabelLen];
    VstInt32 flags;
    VstInt32 minInteger;
    VstInt32 maxInteger;
    VstInt32 stepInteger;
    VstInt32 largeStepInteger;
    char shortLabel[kVstMaxShortLabelLen];
    VstInt16 displayIndex;
    VstInt16 category;
    VstInt16 numParametersInCategory;
    VstInt16 reserved;
    char categoryLabel[kVstMaxCategLabelLen];
    char future[16];
};

struct VstPinProperties {
    char label[kVstMaxLabelLen];
    VstInt32 flags;
    VstInt32 arrangementType;
    char shortLabel[kVstMaxShortLabelLen];
    char future[48];
};

struct MidiKeyName {
    VstInt32 thisProgramIndex;
    VstInt32 thisKeyNumber;
    char keyName[kVstMaxNameLen];
    VstInt32 reserved;
    VstInt32 flags;
};

struct VstSpeakerProperties {
    float azimuth;
    float elevation;
    float radius;
    float reserved;
    char name[kVstMaxNameLen];
    VstInt32 type;
    char future[28];
};

struct VstSpeakerArrangement {
    VstInt32 type;
    VstInt32 numChannels;
    VstSpeakerProperties speakers[8];
};

#pragma pack(pop)

static_assert(sizeof(AEffect) == sizeof(vst_effect_t), "VST2 effect ABI size mismatch");
static_assert(offsetof(AEffect, dispatcher) == offsetof(vst_effect_t, control), "VST2 dispatcher ABI mismatch");
static_assert(offsetof(AEffect, object) == offsetof(vst_effect_t, effect_internal), "VST2 object ABI mismatch");
static_assert(offsetof(AEffect, processReplacing) == offsetof(vst_effect_t, process_float), "VST2 float process ABI mismatch");
static_assert(offsetof(ERect, right) == offsetof(vst_rect_t, right), "VST2 editor rectangle ABI mismatch");

#define CCONST(a, b, c, d) VST_FOURCC(a, b, c, d)
#define kEffectMagic static_cast<VstInt32>(VST_MAGICNUMBER)
inline constexpr VstInt32 kVstVersion = VST_VERSION_2_4_0_0;

enum VstAEffectFlags : VstInt32 {
    effFlagsHasEditor = VST_EFFECT_FLAG_EDITOR,
    __effFlagsCanMonoDeprecated = 1 << 3,
    effFlagsCanReplacing = VST_EFFECT_FLAG_SUPPORTS_FLOAT,
    effFlagsProgramChunks = VST_EFFECT_FLAG_CHUNKS,
    effFlagsIsSynth = VST_EFFECT_FLAG_INSTRUMENT,
    effFlagsCanDoubleReplacing = VST_EFFECT_FLAG_SUPPORTS_DOUBLE
};

enum AEffectOpcodes : VstInt32 {
    effOpen = 0,
    effClose = 1,
    effSetProgram = 2,
    effGetProgram = 3,
    effSetProgramName = 4,
    effGetProgramName = 5,
    effGetParamLabel = 6,
    effGetParamDisplay = 7,
    effGetParamName = 8,
    effSetSampleRate = 10,
    effSetBlockSize = 11,
    effMainsChanged = 12,
    effEditGetRect = 13,
    effEditOpen = 14,
    effEditClose = 15,
    effEditIdle = 19,
    __effIdentifyDeprecated = 22,
    effGetChunk = 23,
    effSetChunk = 24,
    effProcessEvents = 25,
    effCanBeAutomated = 26,
    effString2Parameter = 27,
    effGetProgramNameIndexed = 29,
    effGetInputProperties = 33,
    effGetOutputProperties = 34,
    effGetPlugCategory = 35,
    effProcessVarIo = 41,
    effSetSpeakerArrangement = 42,
    effSetBypass = 44,
    effGetEffectName = 45,
    effGetVendorString = 47,
    effGetProductString = 48,
    effGetVendorVersion = 49,
    effVendorSpecific = 50,
    effCanDo = 51,
    effGetTailSize = 52,
    __effIdleDeprecated = 53,
    effGetParameterProperties = 56,
    effGetVstVersion = 58,
    effEditKeyDown = 59,
    effEditKeyUp = 60,
    effGetMidiProgramName = 62,
    effGetCurrentMidiProgram = 63,
    effGetMidiProgramCategory = 64,
    effHasMidiProgramsChanged = 65,
    effGetMidiKeyName = 66,
    effBeginSetProgram = 67,
    effEndSetProgram = 68,
    effGetSpeakerArrangement = 69
};

enum AudioMasterOpcodes : VstInt32 {
    audioMasterAutomate = 0,
    audioMasterVersion = 1,
    audioMasterCurrentId = 2,
    audioMasterIdle = 3,
    __audioMasterWantMidiDeprecated = 6,
    audioMasterGetTime = 7,
    audioMasterProcessEvents = 8,
    audioMasterIOChanged = 13,
    audioMasterSizeWindow = 15,
    audioMasterGetCurrentProcessLevel = 23,
    audioMasterGetProductString = 33,
    audioMasterGetVendorVersion = 34,
    audioMasterUpdateDisplay = 42,
    audioMasterBeginEdit = 43,
    audioMasterEndEdit = 44
};

enum VstPlugCategory : VstInt32 {
    kPlugCategUnknown = 0,
    kPlugCategEffect = VST_EFFECT_CATEGORY_EFFECT,
    kPlugCategSynth = VST_EFFECT_CATEGORY_INSTRUMENT
};

enum VstSpeakerArrangementType : VstInt32 {
    kSpeakerArrUserDefined = VST_SPEAKER_ARRANGEMENT_TYPE_CUSTOM,
    kSpeakerArrEmpty = VST_SPEAKER_ARRANGEMENT_TYPE_UNKNOWN,
    kSpeakerArrMono = VST_SPEAKER_ARRANGEMENT_TYPE_MONO,
    kSpeakerArrStereo = VST_SPEAKER_ARRANGEMENT_TYPE_STEREO
};

enum VstEventTypes : VstInt32 {
    kVstMidiType = VST_EVENT_TYPE_MIDI,
    kVstSysExType = VST_EVENT_TYPE_MIDI_SYSEX
};

enum VstParameterFlags : VstInt32 {
    kVstParameterIsSwitch = VST_PARAMETER_FLAG_SWITCH,
    kVstParameterUsesIntegerMinMax = VST_PARAMETER_FLAG_INTEGER_LIMITS,
    kVstParameterUsesFloatStep = VST_PARAMETER_FLAG_STEP_FLOAT,
    kVstParameterUsesIntStep = VST_PARAMETER_FLAG_STEP_INT
};

enum VstPinPropertiesFlags : VstInt32 {
    kVstPinIsActive = 1 << 0,
    kVstPinIsStereo = VST_STREAM_FLAG_STEREO
};

enum VstProcessLevels : VstInt32 {
    kVstProcessLevelOffline = 4
};

enum VstTimeInfoFlags : VstInt32 {
    kVstTransportPlaying = 1 << 1,
    kVstTransportCycleActive = 1 << 2,
    kVstPpqPosValid = 1 << 9,
    kVstTempoValid = 1 << 10,
    kVstBarsValid = 1 << 11,
    kVstCyclePosValid = 1 << 12,
    kVstTimeSigValid = 1 << 13
};

enum VstVirtualKey : VstInt32 {
    VKEY_BACK = 1, VKEY_TAB, VKEY_CLEAR, VKEY_RETURN, VKEY_PAUSE, VKEY_ESCAPE,
    VKEY_SPACE, VKEY_NEXT, VKEY_END, VKEY_HOME, VKEY_LEFT, VKEY_UP, VKEY_RIGHT,
    VKEY_DOWN, VKEY_PAGEUP, VKEY_PAGEDOWN, VKEY_SELECT, VKEY_PRINT, VKEY_ENTER,
    VKEY_SNAPSHOT, VKEY_INSERT, VKEY_DELETE, VKEY_HELP, VKEY_NUMPAD0, VKEY_NUMPAD1,
    VKEY_NUMPAD2, VKEY_NUMPAD3, VKEY_NUMPAD4, VKEY_NUMPAD5, VKEY_NUMPAD6, VKEY_NUMPAD7,
    VKEY_NUMPAD8, VKEY_NUMPAD9, VKEY_MULTIPLY, VKEY_ADD, VKEY_SEPARATOR, VKEY_SUBTRACT,
    VKEY_DECIMAL, VKEY_DIVIDE, VKEY_F1, VKEY_F2, VKEY_F3, VKEY_F4, VKEY_F5, VKEY_F6,
    VKEY_F7, VKEY_F8, VKEY_F9, VKEY_F10, VKEY_F11, VKEY_F12, VKEY_NUMLOCK, VKEY_SCROLL,
    VKEY_SHIFT, VKEY_CONTROL, VKEY_ALT, VKEY_EQUALS
};

enum VstModifierKey : VstInt32 {
    MODIFIER_SHIFT = VST_VKEY_MODIFIER_SHIFT,
    MODIFIER_ALTERNATE = VST_VKEY_MODIFIER_ALT,
    MODIFIER_COMMAND = VST_VKEY_MODIFIER_SYSTEM,
    MODIFIER_CONTROL = VST_VKEY_MODIFIER_CONTROL
};
