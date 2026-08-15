param(
    [Parameter(Mandatory = $true)]
    [string]$NvcRoot,
    [string]$Python = "",
    [Parameter(Mandatory = $true)]
    [string]$Model,
    [string]$Index = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $Root "build"

& (Join-Path $PSScriptRoot "prepare-dependencies.ps1")

if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $NvcRoot "runtime\python.exe"
}
foreach ($RequiredPath in @($NvcRoot, $Python, $Model)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required worker test path does not exist: $RequiredPath"
    }
}
if (-not [string]::IsNullOrWhiteSpace($Index) -and -not (Test-Path -LiteralPath $Index -PathType Leaf)) {
    throw "Index path does not exist: $Index"
}

cmake -S $Root -B $Build -G "Visual Studio 17 2022" -A x64 -DNVC_BUILD_SMOKE_TEST=ON
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed with exit code $LASTEXITCODE"
}
cmake --build $Build --config Release --target nvc-worker-smoke --parallel
if ($LASTEXITCODE -ne 0) {
    throw "Worker smoke-test build failed with exit code $LASTEXITCODE"
}
$WorkerArgs = @($NvcRoot, $Python, $Model)
if (-not [string]::IsNullOrWhiteSpace($Index)) {
    $WorkerArgs += $Index
}
& (Join-Path $Build "Release\nvc-worker-smoke.exe") @WorkerArgs
if ($LASTEXITCODE -ne 0) {
    throw "Worker smoke test failed with exit code $LASTEXITCODE"
}
