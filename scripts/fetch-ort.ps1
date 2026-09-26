# 获取官方 ONNX Runtime 动态库并配置 ORT_LIB_PATH（Windows）。
#
# 背景：crates.io 的 ort 2.x 仅发布 rc（当前 2.0.0-rc.13，包装 ONNX Runtime 1.28），
# 且 ort-sys 默认下载的是静态库（MSVC STL 版本与本机工具链冲突，链接失败）。
# 本脚本改用官方动态库（onnxruntime.dll + 导入库），规避静态库 STL 问题。
#
# 用法：powershell -ExecutionPolicy Bypass -File scripts/fetch-ort.ps1
# 产物：%LOCALAPPDATA%\asep-ort\onnxruntime-win-x64-<ver>\lib\onnxruntime.lib
# 之后 cargo 通过 .cargo/config.toml 的 [env] ORT_LIB_PATH 找到 lib 目录；
# onnxruntime.dll 需随可执行文件分发（构建脚本会复制到 target/debug）。

$ErrorActionPreference = "Stop"
$ver = "1.28.0"
$dest = Join-Path $env:LOCALAPPDATA "asep-ort"
$lib = Join-Path $dest "onnxruntime-win-x64-$ver\lib"
if (Test-Path (Join-Path $lib "onnxruntime.lib")) {
    Write-Output "已存在: $lib"
    exit 0
}
$zip = Join-Path $dest "onnxruntime-win-x64-$ver.zip"
if (-not (Test-Path $zip)) {
    $url = "https://github.com/microsoft/onnxruntime/releases/download/v$ver/onnxruntime-win-x64-$ver.zip"
    Write-Output "下载 $url ..."
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -TimeoutSec 600
}
New-Item -ItemType Directory -Force $dest | Out-Null
Expand-Archive -Path $zip -DestinationPath $dest -Force
Write-Output "完成。lib 目录: $lib"
Write-Output "提示: 将 ORT_LIB_PATH 写入 .cargo/config.toml 的 [env]（参见 .gitignore 说明），"
Write-Output "并将 onnxruntime.dll 复制到可执行文件目录。"
