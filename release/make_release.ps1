# Builds the public release folder (default D:\logmit-release) as a fresh git repo with one commit.
# Copies code, infra scripts, results and the release README/LICENSE; no raw logs, paper sources,
# kubeconfigs or credentials. Runs a secret scan and stops if anything matches. Does not push.
#   .\release\make_release.ps1
param([string]$Dest = 'D:\logmit-release')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
if (Test-Path $Dest) { throw "$Dest already exists; remove it first" }
New-Item -ItemType Directory $Dest | Out-Null

$include = @('logmit\*.py', 'infra\*.ps1', 'infra\*.yaml', 'requirements.txt',
             'results\*.json', 'results\figures\*', 'results\paper_inputs\*')
foreach ($pattern in $include) {
    foreach ($f in Get-ChildItem (Join-Path $root $pattern) -File) {
        if ($f.Length -gt 50MB) { Write-Warning "skipping $($f.Name): larger than 50 MB"; continue }
        $rel = $f.FullName.Substring($root.Length + 1)
        New-Item -ItemType Directory -Force (Split-Path (Join-Path $Dest $rel)) | Out-Null
        Copy-Item $f.FullName (Join-Path $Dest $rel)
    }
}
Copy-Item (Join-Path $PSScriptRoot 'README.md') $Dest
Copy-Item (Join-Path $PSScriptRoot 'LICENSE') $Dest
"data/`n__pycache__/`n*.pt`nbuild/" | Set-Content -Encoding ascii (Join-Path $Dest '.gitignore')

# secret scan: credentials, keys, kubeconfig material, personal e-mail addresses
$pattern = '(?i)(password\s*[:=]|passwd|api[_-]?key|secret[_-]?key|access[_-]?token|BEGIN [A-Z ]*PRIVATE KEY|client-key-data|client-certificate-data|certificate-authority-data|aws_access_key|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|xox[bp]-|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,})'
$hits = Get-ChildItem $Dest -Recurse -File | Select-String -Pattern $pattern
$allowed = 'noreply@anthropic\.com'
$real = $hits | Where-Object { $_.Line -notmatch $allowed }
"secret scan: $(@($hits).Count) matches, $(@($real).Count) not on the allow list"
$real | ForEach-Object { "  $($_.Path.Substring($Dest.Length + 1)):$($_.LineNumber): $($_.Line.Trim())" }
if ($real) { throw "secret scan found matches; review them before committing" }

Push-Location $Dest
$ErrorActionPreference = 'Continue'   # git writes warnings to stderr; PowerShell 5.1 would treat them as fatal
git init -q -b main
git add -A
git commit -q -m "Code, scripts and measured results for log-based proactive mitigation on Kubernetes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git log --oneline -1
Pop-Location
"files: $((Get-ChildItem $Dest -Recurse -File | Where-Object { $_.FullName -notmatch '\\.git\\' }).Count)"
