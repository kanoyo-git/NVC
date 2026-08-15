# Third-Party Notices

The NVC Realtime VST source tree uses the following third-party components.
Each dependency remains under its own license.

## iPlug2

- Source: https://github.com/iPlug2/iPlug2
- Locked commit: `5c2df9dce3f5258acfeff3846a6a9563f382212c`
- License: zlib-style license
- License file: `third_party/iPlug2/LICENSE.txt`

## Steinberg VST 3 SDK

- Source: https://github.com/steinbergmedia/vst3sdk
- Locked commit: `58f8da7936800732561402d7936584ca4505de07`
- License: MIT
- License file: `third_party/vst3sdk/LICENSE.txt`

The required nested SDK repositories are locked by the VST 3 SDK gitlinks.
The build initializes `base`, `cmake`, `pluginterfaces`, and `public.sdk`.

## Xaymar VST2 SDK

- Source: https://github.com/Xaymar/vst2sdk
- Locked commit: `339d4f31590bf77c0d0d248e09a380ac6285e069`
- License: BSD-3-Clause
- License file: `third_party/vst2sdk/LICENSE`

`third_party/vst2-compat/aeffectx.h` supplies the compatibility names used by
iPlug2 and is backed by the BSD-3-Clause ABI declarations above.

## Roboto

- Font file: `resources/fonts/Roboto-Regular.ttf`
- License: Apache License 2.0
- License copy: `third_party/licenses/Roboto-Apache-2.0.txt`

The font file is byte-identical to the Roboto resource in the locked iPlug2
source tree.
