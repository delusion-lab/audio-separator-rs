# Release packaging: build release, bundle exe + runtime deps + models.json + README into a zip.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/package-release.ps1 [-Version x.y.z]
param(
    [string]$Version = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot   # parent of scripts/ = repo root

if (-not $Version) {
    $m = Select-String -Path "$root\Cargo.toml" -Pattern '^version\s*=\s*"([^"]+)"'
    if (-not $m) { throw "cannot read version from workspace Cargo.toml" }
    $Version = $m.Matches[0].Groups[1].Value
}

Push-Location $root
try {
    cargo build --release --workspace
    if ($LASTEXITCODE -ne 0) { throw "release build failed" }

    $dist = Join-Path $root "dist"
    $pkg  = Join-Path $dist "asep-$Version-windows-x64"
    if (Test-Path $pkg) { Remove-Item $pkg -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $pkg | Out-Null

    Copy-Item "$root\target\release\asep.exe" $pkg
    Copy-Item "$root\target\release\audio-separator-server.exe" $pkg
    # ONNX Runtime DLL (needed by the mdx architecture; roformer is pure Rust).
    # fetch-ort.ps1 normally copies it into target\release.
    if (Test-Path "$root\target\release\onnxruntime.dll") {
        Copy-Item "$root\target\release\onnxruntime.dll" $pkg
    } else {
        Write-Warning "onnxruntime.dll not found in target\release - MDX models will not run without it. Run scripts/fetch-ort.ps1 first."
    }
    # models.json: default local model list, bundled so the release works offline.
    if (Test-Path "$root\models.json") { Copy-Item "$root\models.json" $pkg }
    # rankings.json: community guide / MVSEP leaderboard ranking data (optional, bundled with release).
    if (Test-Path "$root\rankings.json") { Copy-Item "$root\rankings.json" $pkg }
    Copy-Item "$root\README.md" $pkg
    Copy-Item "$root\README.zh-CN.md" $pkg
    Copy-Item "$root\README.ja.md" $pkg
    Copy-Item "$root\README.ko.md" $pkg

    $zip = Join-Path $dist "asep-$Version-windows-x64.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path "$pkg\*" -DestinationPath $zip

    Write-Host "packaged: $zip"
    Get-ChildItem $pkg | Select-Object Name, Length
}
finally {
    Pop-Location
}
