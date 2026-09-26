param(
    [string]$Url,
    [string]$Out,
    [int]$Parts = 8,
    [string]$Proxy = ""
)
$ErrorActionPreference = "Stop"
$tmp = "$Out.part"
1..$Parts | ForEach-Object { Remove-Item "$tmp$($_ - 1)" -ErrorAction SilentlyContinue }

$resp = Invoke-WebRequest -Uri $Url -Method Head -MaximumRedirection 5 -TimeoutSec 60 -UseBasicParsing
$len = [long]$resp.Headers["Content-Length"]
$seg = [long][math]::Floor($len / $Parts)
Write-Host "total=$len parts=$Parts seg=$seg"

$proxyArgs = @()
if ($Proxy -ne "") {
    $proxyArgs = @("-x", $Proxy)
}

$procs = @()
for ($i = 0; $i -lt $Parts; $i++) {
    $start = $i * $seg
    $end = if ($i -eq $Parts - 1) { $len - 1 } else { ($i + 1) * $seg - 1 }
    $args = @("-L", "--fail", "--retry", "5", "--retry-delay", "2") + $proxyArgs + @("-r", "$start-$end", "-o", "$tmp$i", $Url)
    $p = Start-Process -FilePath "curl.exe" -ArgumentList $args -NoNewWindow -PassThru
    $procs += $p
    Write-Host "part $i : $start-$end started"
}
$procs | Wait-Process
$failed = $procs | Where-Object { $_.ExitCode -ne 0 }
if ($failed) {
    Write-Host "FAILED parts: $($failed.ExitCode -join ',')"
    exit 1
}

$fs = [System.IO.File]::Open($Out, [System.IO.FileMode]::Create)
try {
    for ($i = 0; $i -lt $Parts; $i++) {
        $bytes = [System.IO.File]::ReadAllBytes("$tmp$i")
        $fs.Write($bytes, 0, $bytes.Length)
        Remove-Item "$tmp$i" -ErrorAction SilentlyContinue
    }
} finally {
    $fs.Close()
}
$final = (Get-Item $Out).Length
Write-Host "merged: $final (expected $len)"
if ($final -ne $len) { exit 1 }
exit 0
