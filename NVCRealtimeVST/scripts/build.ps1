$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $Root "build"
$Dist = Join-Path $Root "dist"
$PackageName = "NVCRealtime-Win64"
$Package = Join-Path $Dist $PackageName
$Zip = Join-Path $Dist "$PackageName.zip"

& (Join-Path $PSScriptRoot "prepare-dependencies.ps1")

cmake -S $Root -B $Build -G "Visual Studio 17 2022" -A x64
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed with exit code $LASTEXITCODE"
}
cmake --build $Build --config Release --target NVCRealtime-vst2 NVCRealtime-vst3 --parallel
if ($LASTEXITCODE -ne 0) {
    throw "Plugin build failed with exit code $LASTEXITCODE"
}

New-Item -ItemType Directory -Force -Path $Dist | Out-Null
$Vst2Source = Join-Path $Build "out\NVCRealtime.dll"
$Vst3Source = Join-Path $Build "out\NVCRealtime.vst3"
$LegacyVst3 = Join-Path $Dist "NVC Realtime.vst3"

Copy-Item -Force $Vst2Source (Join-Path $Dist "NVC Realtime.dll")
if (Test-Path $LegacyVst3) {
    Remove-Item -Recurse -Force $LegacyVst3
}
if (Test-Path (Join-Path $Dist "NVCRealtime.vst3")) {
    Remove-Item -Recurse -Force (Join-Path $Dist "NVCRealtime.vst3")
}
Copy-Item -Recurse -Force $Vst3Source (Join-Path $Dist "NVCRealtime.vst3")
$LooseResources = Join-Path $Dist "NVCRealtime.resources"
if (Test-Path $LooseResources) {
    Remove-Item -Recurse -Force $LooseResources
}
New-Item -ItemType Directory -Force -Path (Join-Path $LooseResources "worker") | Out-Null
Copy-Item -Force (Join-Path $Root "worker\nvc_worker.py") (Join-Path $LooseResources "worker\nvc_worker.py")

$LooseVst3Worker = Join-Path $Dist "NVCRealtime.vst3\Contents\Resources\worker"
New-Item -ItemType Directory -Force -Path $LooseVst3Worker | Out-Null
Copy-Item -Force (Join-Path $Root "worker\nvc_worker.py") (Join-Path $LooseVst3Worker "nvc_worker.py")

if (Test-Path $Package) {
    Remove-Item -Recurse -Force $Package
}
if (Test-Path $Zip) {
    Remove-Item -Force $Zip
}
$PackageVst2 = Join-Path $Package "VST2"
$PackageVst3 = Join-Path $Package "VST3"
New-Item -ItemType Directory -Force -Path $PackageVst2, $PackageVst3 | Out-Null
Copy-Item -Force (Join-Path $Dist "NVC Realtime.dll") (Join-Path $PackageVst2 "NVC Realtime.dll")
Copy-Item -Recurse -Force $LooseResources (Join-Path $PackageVst2 "NVCRealtime.resources")
Copy-Item -Recurse -Force (Join-Path $Dist "NVCRealtime.vst3") (Join-Path $PackageVst3 "NVCRealtime.vst3")
$ReadmeSource = Join-Path $Root "resources\README.txt"
$ReadmeDestination = Join-Path $Package "README.txt"
$Utf8WithBom = New-Object System.Text.UTF8Encoding -ArgumentList $true
[System.IO.File]::WriteAllText($ReadmeDestination, [System.IO.File]::ReadAllText($ReadmeSource), $Utf8WithBom)

$RequiredReleaseFiles = @(
    (Join-Path $Package "README.txt"),
    (Join-Path $Package "VST2\NVC Realtime.dll"),
    (Join-Path $Package "VST2\NVCRealtime.resources\worker\nvc_worker.py"),
    (Join-Path $Package "VST3\NVCRealtime.vst3\Contents\x86_64-win\NVCRealtime.vst3"),
    (Join-Path $Package "VST3\NVCRealtime.vst3\Contents\Resources\worker\nvc_worker.py")
)
foreach ($RequiredFile in $RequiredReleaseFiles) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Release package is missing: $RequiredFile"
    }
}
Compress-Archive -Path $Package -DestinationPath $Zip -CompressionLevel Optimal

Write-Host "Built artifacts and release ZIP: $Zip"
