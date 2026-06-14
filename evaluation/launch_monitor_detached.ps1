param(
  [Parameter(Mandatory = $true)][string]$RunDir,
  [string]$OutputDir = "evaluation/outputs/full_judge_run/logs",
  [int]$IntervalSeconds = 30
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = (Get-Command python -ErrorAction Stop).Source
$OutputDirAbs = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputDir))
New-Item -ItemType Directory -Force -Path $OutputDirAbs | Out-Null

$StdoutLog = Join-Path $OutputDirAbs "monitor.stdout.log"
$StderrLog = Join-Path $OutputDirAbs "monitor.stderr.log"
$argList = @(
  "evaluation/monitor_full_judge_run.py",
  "--run-dir", $RunDir,
  "--output-dir", $OutputDir,
  "--interval-seconds", $IntervalSeconds
)

$process = Start-Process -FilePath $PythonExe -ArgumentList $argList -WorkingDirectory $ProjectRoot -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru -WindowStyle Hidden

$meta = @{
  pid = $process.Id
  run_dir = $RunDir
  output_dir = $OutputDirAbs
  stdout_log = $StdoutLog
  stderr_log = $StderrLog
  interval_seconds = $IntervalSeconds
  started_at = (Get-Date).ToString("s")
}
$metaPath = Join-Path $OutputDirAbs "monitor.launcher.json"
$meta | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $metaPath

Write-Output "pid=$($process.Id)"
Write-Output "output_dir=$OutputDirAbs"
Write-Output "stdout_log=$StdoutLog"
Write-Output "stderr_log=$StderrLog"
Write-Output "meta=$metaPath"
