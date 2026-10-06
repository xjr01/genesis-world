param(
    [Parameter(Mandatory = $true)]
    [int]$CheckpointCoordinatorProcessId
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$resultRoot = Join-Path $repositoryRoot 'out\full_trajectory_validation'
$coordinatorLog = Join-Path $resultRoot 'final_table_coordinator.log'
$scenarios = @('coffee_water', 'table_wiping', 'litter_scoop', 'garment_folding', 'butter_spreading')

Set-Location -LiteralPath $repositoryRoot
Set-Content -LiteralPath $coordinatorLog -Value (
    "Waiting for checkpoint coordinator $CheckpointCoordinatorProcessId at $((Get-Date).ToString('o'))"
)
Wait-Process -Id $CheckpointCoordinatorProcessId -ErrorAction SilentlyContinue

foreach ($scenario in $scenarios) {
    $exitPath = Join-Path $resultRoot "$scenario.checkpoint.exitcode"
    if (-not (Test-Path -LiteralPath $exitPath)) {
        Add-Content -LiteralPath $coordinatorLog -Value "$scenario checkpoint exit code is missing."
        exit 1
    }
    $validationExitCode = [int](Get-Content -LiteralPath $exitPath -Raw)
    if ($validationExitCode -ne 0) {
        Add-Content -LiteralPath $coordinatorLog -Value "$scenario checkpoint validation failed: $validationExitCode"
        exit $validationExitCode
    }
}

Add-Content -LiteralPath $coordinatorLog -Value "Checkpoint queue passed; rerunning final table trajectory."
& (Join-Path $PSScriptRoot 'run_full_trajectory.ps1') `
    -Module 'examples.multiphysics.table_wiping.run' `
    -Name 'table_wiping_final'
$tableExitCode = $LASTEXITCODE
Add-Content -LiteralPath $coordinatorLog -Value (
    "Final table trajectory ended with exit code $tableExitCode at $((Get-Date).ToString('o'))"
)
exit $tableExitCode
