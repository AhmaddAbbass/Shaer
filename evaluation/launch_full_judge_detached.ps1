param(
  [Parameter(Mandatory = $true)][string]$RunDir,
  [int]$Workers = 4,
  [string]$JudgeModel = "qwen/qwen3-235b-a22b-2507",
  [string[]]$Datasets = @(),
  [switch]$AllowRowErrors,
  [string]$PromptFile = "evaluation/judge_prompts.yaml",
  [string]$PreparedInputRoot = ""
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunDir = [System.IO.Path]::GetFullPath($RunDir)
$LogDir = Join-Path $RunDir "launcher"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$StdoutLog = Join-Path $LogDir "stdout.log"
$StderrLog = Join-Path $LogDir "stderr.log"
$PythonExe = (Get-Command python -ErrorAction Stop).Source
$argList = @(
  "evaluation/full_judge_orchestrator.py",
  "--run-dir", $RunDir,
  "--workers", $Workers,
  "--judge-model", $JudgeModel,
  "--prompt-file", $PromptFile
)
if ($Datasets.Count -gt 0) {
  $argList += "--datasets"
  $argList += $Datasets
}
if ($AllowRowErrors) {
  $argList += "--allow-row-errors"
}
if ($PreparedInputRoot -ne "") {
  $argList += "--prepared-input-root"
  $argList += $PreparedInputRoot
}

$process = Start-Process -FilePath $PythonExe -ArgumentList $argList -WorkingDirectory $ProjectRoot -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru -WindowStyle Hidden

$meta = @{
  pid = $process.Id
  run_dir = $RunDir
  workers = $Workers
  judge_model = $JudgeModel
  datasets = $Datasets
  allow_row_errors = [bool]$AllowRowErrors
  prompt_file = $PromptFile
  prepared_input_root = $PreparedInputRoot
  stdout_log = $StdoutLog
  stderr_log = $StderrLog
  started_at = (Get-Date).ToString("s")
}
$metaPath = Join-Path $LogDir "launcher.json"
$meta | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $metaPath

Write-Output "pid=$($process.Id)"
Write-Output "run_dir=$RunDir"
Write-Output "stdout_log=$StdoutLog"
Write-Output "stderr_log=$StderrLog"
Write-Output "meta=$metaPath"
