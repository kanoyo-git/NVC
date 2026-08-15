$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ThirdParty = Join-Path $Root "third_party"
$IPlug2 = Join-Path $ThirdParty "iPlug2"
$Vst3Source = Join-Path $ThirdParty "vst3sdk"
$Vst2Source = Join-Path $ThirdParty "vst2sdk"
$Vst2Compat = Join-Path $ThirdParty "vst2-compat\aeffectx.h"

$ExpectedCommits = [ordered]@{
    $IPlug2 = "5c2df9dce3f5258acfeff3846a6a9563f382212c"
    $Vst3Source = "58f8da7936800732561402d7936584ca4505de07"
    $Vst2Source = "339d4f31590bf77c0d0d248e09a380ac6285e069"
}

foreach ($Entry in $ExpectedCommits.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath (Join-Path $Entry.Key ".git"))) {
        throw "Submodule is missing: $($Entry.Key). Run: git submodule update --init --recursive"
    }
    $ActualCommit = (& git -C $Entry.Key rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $ActualCommit -ne $Entry.Value) {
        throw "Unexpected submodule revision at $($Entry.Key): $ActualCommit (expected $($Entry.Value))"
    }
}

$RequiredVst3Modules = @("base", "cmake", "pluginterfaces", "public.sdk")
$RequiredVst3ModuleFiles = @(
    "base\source\baseiids.cpp",
    "cmake\modules\SMTG_AddVST3Library.cmake",
    "pluginterfaces\base\funknown.cpp",
    "public.sdk\source\main\dllmain.cpp"
)
$MissingVst3Modules = $RequiredVst3ModuleFiles | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $Vst3Source $_) -PathType Leaf)
}
if ($MissingVst3Modules.Count -gt 0) {
    & git -C $Vst3Source submodule update --init @RequiredVst3Modules
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize the required nested VST3 SDK submodules."
    }
}

$Vst3Destination = Join-Path $IPlug2 "Dependencies\IPlug\VST3_SDK"
New-Item -ItemType Directory -Force -Path $Vst3Destination | Out-Null
foreach ($Directory in @("base", "cmake", "pluginterfaces", "public.sdk")) {
    $Source = Join-Path $Vst3Source $Directory
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "VST3 SDK directory is missing: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $Vst3Destination -Recurse -Force
}
foreach ($File in @("CMakeLists.txt", "LICENSE.txt")) {
    Copy-Item -LiteralPath (Join-Path $Vst3Source $File) -Destination (Join-Path $Vst3Destination $File) -Force
}

$Vst2Destination = Join-Path $IPlug2 "Dependencies\IPlug\VST2_SDK"
New-Item -ItemType Directory -Force -Path $Vst2Destination | Out-Null
Get-ChildItem -LiteralPath (Join-Path $Vst2Source "include") -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Vst2Destination $_.Name) -Force
}
Copy-Item -LiteralPath $Vst2Compat -Destination (Join-Path $Vst2Destination "aeffectx.h") -Force
Copy-Item -LiteralPath (Join-Path $Vst2Source "LICENSE") -Destination (Join-Path $Vst2Destination "LICENSE.BSD-3-Clause.txt") -Force

$RequiredFiles = @(
    (Join-Path $Vst3Destination "base\source\baseiids.cpp"),
    (Join-Path $Vst3Destination "public.sdk\source\main\dllmain.cpp"),
    (Join-Path $Vst2Destination "aeffectx.h"),
    (Join-Path $Vst2Destination "vst.h")
)
foreach ($RequiredFile in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Prepared dependency file is missing: $RequiredFile"
    }
}

Write-Host "Prepared locked iPlug2, VST3, and VST2 dependencies."
