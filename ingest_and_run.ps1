param(
  # Either pass -SdDrive "E:" or let it auto-detect by label / heuristic.
  [string]$SdDrive = "",

  # Folder on the SD card that contains RUN#### files. "" means SD root.
  [string]$SdSubdir = "",

  # If your SD volume label is stable, set it to auto-detect (optional).
  [string]$SdLabel = "",

  # If true, archive the SD folder into SD_dump\_raw\<timestamp>\ (optional)
  [bool]$ArchiveRaw = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-SdDrive {
  param([string]$Drive, [string]$Label)

  if ($Drive -and (Test-Path "$Drive\")) { return $Drive.TrimEnd('\') }

  if ($Label) {
    $vol = Get-Volume | Where-Object { $_.FileSystemLabel -eq $Label -and $_.DriveLetter } | Select-Object -First 1
    if ($vol) { return ($vol.DriveLetter + ":") }
  }

  # Heuristic: pick first removable drive that contains RUN*.CSV (root or first-level folder)
  $removable = Get-Volume | Where-Object { $_.DriveType -eq "Removable" -and $_.DriveLetter } |
    ForEach-Object { $_.DriveLetter + ":" }

  foreach ($d in $removable) {
    $root = "$d\"
    if (Get-ChildItem -Path $root -Filter "RUN*.CSV" -File -ErrorAction SilentlyContinue | Select-Object -First 1) {
      return $d
    }
    $sub = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue
    foreach ($s in $sub) {
      if (Get-ChildItem -Path $s.FullName -Filter "RUN*.CSV" -File -ErrorAction SilentlyContinue | Select-Object -First 1) {
        return $d
      }
    }
  }

  throw "Could not auto-detect SD drive. Pass -SdDrive `"E:`" or set -SdLabel."
}

function Ensure-Venv {
  param([string]$RepoRoot)

  $venvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
  if (!(Test-Path $venvPy)) {
    Write-Host "No .venv found at repo root. Creating it..."
    python -m venv (Join-Path $RepoRoot ".venv")
  }

  & $venvPy -m pip install --upgrade pip | Out-Null
  & $venvPy -m pip install -r (Join-Path $RepoRoot "bldc_loader\apps\run_compare\requirements.txt") | Out-Null

  return $venvPy
}

function Get-RunIdFromName {
  param([string]$FileName)
  if ($FileName -match '^(?i)RUN(\d{4})\.CSV$') {
    return [int]$Matches[1]
  }
  return $null
}

function Format-RunName {
  param([int]$RunId, [string]$Ext)
  return ("RUN{0:D4}{1}" -f $RunId, $Ext)
}

function Get-MaxExistingRunId {
  param([string]$DestDir)

  $max = 0
  $files = Get-ChildItem -Path $DestDir -Filter "RUN*.CSV" -File -ErrorAction SilentlyContinue
  foreach ($f in $files) {
    $rid = Get-RunIdFromName -FileName $f.Name
    if ($rid -ne $null -and $rid -gt $max) { $max = $rid }
  }
  return $max
}

function Get-FileSha256 {
  param([string]$Path)
  return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLower()
}

# ---- main ----

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$sd = Resolve-SdDrive -Drive $SdDrive -Label $SdLabel
$sdRoot = "$sd\"
$src = if ($SdSubdir) { Join-Path $sdRoot $SdSubdir } else { $sdRoot }

if (!(Test-Path $src)) {
  throw "Source path not found on SD: $src"
}

$destDump = Join-Path $RepoRoot "SD_dump"
New-Item -ItemType Directory -Force -Path $destDump | Out-Null

Write-Host "RepoRoot:   $RepoRoot"
Write-Host "SD source:  $src"
Write-Host "Dest dump:  $destDump"

if ($ArchiveRaw) {
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $rawDir = Join-Path $destDump "_raw\$ts"
  New-Item -ItemType Directory -Force -Path $rawDir | Out-Null
  Write-Host "Archiving SD folder to: $rawDir"
  Copy-Item -Path (Join-Path $src "*") -Destination $rawDir -Recurse -Force
}

# Gather and sort source runs
$srcCsvs = Get-ChildItem -Path $src -Filter "RUN*.CSV" -File -ErrorAction SilentlyContinue
if (!$srcCsvs -or $srcCsvs.Count -eq 0) {
  throw "No RUN*.CSV found in $src"
}

$runPairs = @()
foreach ($c in $srcCsvs) {
  $rid = Get-RunIdFromName -FileName $c.Name
  if ($rid -ne $null) {
    $runPairs += [pscustomobject]@{ RunId = $rid; Csv = $c }
  }
}
$runPairs = $runPairs | Sort-Object RunId

Write-Host ("Found {0} run CSV(s) on SD." -f $runPairs.Count)

# Destination numbering
$maxExisting = Get-MaxExistingRunId -DestDir $destDump
$nextId = $maxExisting + 1

Write-Host ("Max existing RUN id in SD_dump: RUN{0:D4}" -f $maxExisting)
Write-Host ("New runs will be written starting at: RUN{0:D4}" -f $nextId)

# Manifest with UID + hashes
$ts2 = Get-Date -Format "yyyyMMdd_HHmmss"
$manifestPath = Join-Path $destDump ("_ingest_manifest_{0}.csv" -f $ts2)
"timestamp,src_run_id,src_csv,src_json,dst_run_id,dst_csv,dst_json,run_uid,csv_sha256,json_sha256" | Out-File -FilePath $manifestPath -Encoding utf8

foreach ($rp in $runPairs) {
  $srcCsv = $rp.Csv.FullName
  $srcJson = [System.IO.Path]::ChangeExtension($srcCsv, ".JSON")

  $dstRunId = $nextId
  $nextId += 1

  $dstCsvName = Format-RunName -RunId $dstRunId -Ext ".CSV"
  $dstJsonName = Format-RunName -RunId $dstRunId -Ext ".JSON"

  $dstCsv = Join-Path $destDump $dstCsvName
  $dstJson = Join-Path $destDump $dstJsonName

  Copy-Item -Path $srcCsv -Destination $dstCsv -Force

  $dstJsonShown = ""
  $jsonSha = ""
  if (Test-Path $srcJson) {
    Copy-Item -Path $srcJson -Destination $dstJson -Force
    $dstJsonShown = $dstJson
    $jsonSha = Get-FileSha256 -Path $dstJson
  }

  # UID is derived from destination CSV bytes (stable for the ingested artifact)
  $csvSha = Get-FileSha256 -Path $dstCsv
  $uid = $csvSha.Substring(0, 16)

  $srcJsonShown = ""
  if (Test-Path $srcJson) { $srcJsonShown = $srcJson }

  $line = "{0},{1:D4},{2},{3},{4:D4},{5},{6},{7},{8},{9}" -f `
    (Get-Date -Format "o"), `
    ([int]$rp.RunId), `
    ($srcCsv -replace ",",";"), `
    ($srcJsonShown -replace ",",";"), `
    ([int]$dstRunId), `
    ($dstCsv -replace ",",";"), `
    ($dstJsonShown -replace ",",";"), `
    $uid, `
    $csvSha, `
    $jsonSha
    
  Add-Content -Path $manifestPath -Value $line
  Write-Host ("Copied RUN{0:D4} -> RUN{1:D4}  uid={2}" -f $rp.RunId, $dstRunId, $uid)
}

Write-Host "Manifest written: $manifestPath"

# Rebuild index + launch Streamlit
$py = Ensure-Venv -RepoRoot $RepoRoot

Write-Host "Re-indexing..."
& $py (Join-Path $RepoRoot "bldc_loader\apps\run_compare\index_runs.py") `
  --runs-dir $destDump `
  --out (Join-Path $RepoRoot "bldc_loader\data\runs_index.csv")

Write-Host "Launching Streamlit..."
& $py -m streamlit run (Join-Path $RepoRoot "bldc_loader\apps\run_compare\app.py") --server.fileWatcherType none
