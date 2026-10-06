param(
    [Parameter(Mandatory = $true)]
    [string]$Module,
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScenarioArgs = @()
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repositoryRoot

$cacheRoot = Join-Path $repositoryRoot '.runtime-cache'
$resultRoot = Join-Path $repositoryRoot 'out\full_trajectory_validation'
$temporaryRoot = Join-Path $cacheRoot 'tmp'
New-Item -ItemType Directory -Force -Path $resultRoot, $temporaryRoot | Out-Null

$env:PYTHONUTF8 = '1'
$env:QD_OFFLINE_CACHE_FILE_PATH = Join-Path $cacheRoot 'all-five-q12\quadrants'
$env:GS_CACHE_FILE_PATH = Join-Path $cacheRoot 'genesis'
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot

$python = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
$logPath = Join-Path $resultRoot "$Name.log"
$exitPath = Join-Path $resultRoot "$Name.exitcode"
$startedPath = Join-Path $resultRoot "$Name.started"
$finishedPath = Join-Path $resultRoot "$Name.finished"

Remove-Item -LiteralPath $exitPath, $finishedPath -Force -ErrorAction SilentlyContinue
Set-Content -LiteralPath $startedPath -Value (Get-Date).ToString('o') -Encoding ascii

try {
    & $python -m $Module @ScenarioArgs 2>&1 | Tee-Object -FilePath $logPath
    $processExitCode = $LASTEXITCODE
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $logPath
    $processExitCode = 1
}
finally {
    Set-Content -LiteralPath $exitPath -Value $processExitCode -Encoding ascii
    Set-Content -LiteralPath $finishedPath -Value (Get-Date).ToString('o') -Encoding ascii
}

exit $processExitCode
