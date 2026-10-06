param(
    [Parameter(Mandatory = $true)]
    [int]$CoffeeProcessId
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$resultRoot = Join-Path $repositoryRoot 'out\full_trajectory_validation'
$coffeeExitPath = Join-Path $resultRoot 'coffee_water.exitcode'
$coordinatorLog = Join-Path $resultRoot 'remaining_trajectory_coordinator.log'

Set-Content -LiteralPath $coordinatorLog -Value "Waiting for coffee process $CoffeeProcessId at $((Get-Date).ToString('o'))"
Wait-Process -Id $CoffeeProcessId -ErrorAction SilentlyContinue

for ($attempt = 0; $attempt -lt 30 -and -not (Test-Path -LiteralPath $coffeeExitPath); $attempt++) {
    Start-Sleep -Seconds 1
}
if (-not (Test-Path -LiteralPath $coffeeExitPath)) {
    Add-Content -LiteralPath $coordinatorLog -Value 'Coffee process ended without writing an exit code.'
    exit 1
}

$coffeeExitCode = [int](Get-Content -LiteralPath $coffeeExitPath -Raw)
if ($coffeeExitCode -ne 0) {
    Add-Content -LiteralPath $coordinatorLog -Value "Coffee failed with exit code $coffeeExitCode; butter was not started."
    exit $coffeeExitCode
}

Add-Content -LiteralPath $coordinatorLog -Value "Coffee passed; starting butter at $((Get-Date).ToString('o'))"
& (Join-Path $PSScriptRoot 'run_full_trajectory.ps1') `
    -Module 'examples.multiphysics.butter_spreading.run' `
    -Name 'butter_spreading'
$butterExitCode = $LASTEXITCODE
Add-Content -LiteralPath $coordinatorLog -Value "Butter ended with exit code $butterExitCode at $((Get-Date).ToString('o'))"
exit $butterExitCode
