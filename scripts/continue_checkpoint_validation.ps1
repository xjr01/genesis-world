param(
    [Parameter(Mandatory = $true)]
    [int]$TrajectoryCoordinatorProcessId
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$resultRoot = Join-Path $repositoryRoot 'out\full_trajectory_validation'
$coordinatorLog = Join-Path $resultRoot 'checkpoint_coordinator.log'
$trajectoryExitPath = Join-Path $resultRoot 'butter_spreading.exitcode'

Set-Location -LiteralPath $repositoryRoot
Set-Content -LiteralPath $coordinatorLog -Value (
    "Waiting for trajectory coordinator $TrajectoryCoordinatorProcessId at $((Get-Date).ToString('o'))"
)
Wait-Process -Id $TrajectoryCoordinatorProcessId -ErrorAction SilentlyContinue

for ($attempt = 0; $attempt -lt 30 -and -not (Test-Path -LiteralPath $trajectoryExitPath); $attempt++) {
    Start-Sleep -Seconds 1
}
if (-not (Test-Path -LiteralPath $trajectoryExitPath)) {
    Add-Content -LiteralPath $coordinatorLog -Value 'Trajectory queue ended without a butter exit code.'
    exit 1
}

$trajectoryExitCode = [int](Get-Content -LiteralPath $trajectoryExitPath -Raw)
if ($trajectoryExitCode -ne 0) {
    Add-Content -LiteralPath $coordinatorLog -Value (
        "Trajectory queue failed with butter exit code $trajectoryExitCode; checkpoint validation was not started."
    )
    exit $trajectoryExitCode
}

$cacheRoot = Join-Path $repositoryRoot '.runtime-cache'
$temporaryRoot = Join-Path $cacheRoot 'tmp'
$env:PYTHONUTF8 = '1'
$env:QD_OFFLINE_CACHE_FILE_PATH = Join-Path $cacheRoot 'all-five-q12\quadrants'
$env:GS_CACHE_FILE_PATH = Join-Path $cacheRoot 'genesis'
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
$python = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
$validator = Join-Path $PSScriptRoot 'validate_multiphysics_checkpoint.py'
$scenarios = @('coffee_water', 'table_wiping', 'litter_scoop', 'garment_folding', 'butter_spreading')

foreach ($scenario in $scenarios) {
    $logPath = Join-Path $resultRoot "$scenario.checkpoint.log"
    $exitPath = Join-Path $resultRoot "$scenario.checkpoint.exitcode"
    Add-Content -LiteralPath $coordinatorLog -Value "Starting $scenario checkpoint validation at $((Get-Date).ToString('o'))"
    & $python $validator $scenario 2>&1 | Tee-Object -FilePath $logPath
    $validationExitCode = $LASTEXITCODE
    Set-Content -LiteralPath $exitPath -Value $validationExitCode -Encoding ascii
    if ($validationExitCode -ne 0) {
        Add-Content -LiteralPath $coordinatorLog -Value "$scenario checkpoint validation failed: $validationExitCode"
        exit $validationExitCode
    }
}

Add-Content -LiteralPath $coordinatorLog -Value "All checkpoint validations passed at $((Get-Date).ToString('o'))"
exit 0
