param(
    [Parameter(Mandatory = $true)]
    [int]$FinalTableCoordinatorProcessId
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$resultRoot = Join-Path $repositoryRoot 'out\full_trajectory_validation'
$coordinatorLog = Join-Path $resultRoot 'final_litter_coordinator.log'
$tableExitPath = Join-Path $resultRoot 'table_wiping_final.exitcode'

Set-Location -LiteralPath $repositoryRoot
Set-Content -LiteralPath $coordinatorLog -Value (
    "Waiting for final table coordinator $FinalTableCoordinatorProcessId at $((Get-Date).ToString('o'))"
)
Wait-Process -Id $FinalTableCoordinatorProcessId -ErrorAction SilentlyContinue

if (-not (Test-Path -LiteralPath $tableExitPath)) {
    Add-Content -LiteralPath $coordinatorLog -Value 'Final table trajectory exit code is missing.'
    exit 1
}
$tableExitCode = [int](Get-Content -LiteralPath $tableExitPath -Raw)
if ($tableExitCode -ne 0) {
    Add-Content -LiteralPath $coordinatorLog -Value "Final table trajectory failed: $tableExitCode"
    exit $tableExitCode
}

Add-Content -LiteralPath $coordinatorLog -Value "Final table passed; rerunning final litter trajectory."
& (Join-Path $PSScriptRoot 'run_full_trajectory.ps1') `
    -Module 'examples.multiphysics.litter_scoop.run' `
    -Name 'litter_scoop_final'
$litterExitCode = $LASTEXITCODE
Add-Content -LiteralPath $coordinatorLog -Value (
    "Final litter trajectory ended with exit code $litterExitCode at $((Get-Date).ToString('o'))"
)
exit $litterExitCode
