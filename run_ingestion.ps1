# run_ingestion.ps1
# Usage (from project root):
# powershell -ExecutionPolicy Bypass -File .\run_ingestion.ps1

$ErrorActionPreference = "Stop"

function Run-Step($Name, $WorkingDir, $Command) {
  Write-Host ""
  Write-Host "============================================================"
  Write-Host "▶ $Name"
  Write-Host "   dir: $WorkingDir"
  Write-Host "   cmd: $Command"
  Write-Host "============================================================"

  Push-Location $WorkingDir
  try {
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
      throw "$Name failed with exit code $LASTEXITCODE"
    }
  } finally {
    Pop-Location
  }
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WebScrapers = Join-Path $ProjectRoot "webScrapers"

# 1) NOS (run from project root, as you do)
Run-Step "NOS RSS" $ProjectRoot "python webScrapers\scrape_nos_feeds.py --max_feeds 10 --max_items_per_feed 5"

# 2) BD (run from webScrapers, as you do)
Run-Step "BD RSS" $WebScrapers "python scrape_bd.py"

# 3) Gelderlander (run from webScrapers, as you do)
Run-Step "Gelderlander RSS" $WebScrapers "python scrape_gelderlander.py"

# 4) Politie (menu-based): choose 2 (Update), then 3 (Exit)
Run-Step "Politie update" $ProjectRoot "python webScrapers\scrape_police.py --update"

# 5) L1 (run from webScrapers, as you do)
Run-Step "L1 RSS" $WebScrapers "python scrape_l1.py"

# 6) Omroep West (run from webScrapers, as you do)
Run-Step "Omroep West RSS" $WebScrapers "python scrape_omroep_west.py"

# 7) RTV Noord (run from webScrapers, as you do)
Run-Step "RTV Noord RSS" $WebScrapers "python scrape_rtv_noord.py"

# 8) Merge + Refresh (run from project root)
Run-Step "Merge + Refresh" $ProjectRoot "python merge_jsons.py"

Write-Host ""
Write-Host "✅ Ingestion pipeline finished successfully."
