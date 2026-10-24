# SPDX-License-Identifier: AGPL-3.0-only
param(
    [Parameter(Mandatory=$true)][string]$Archive,
    [Parameter(Mandatory=$true)][string]$Destination,
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedSha256,
    [Parameter(Mandatory=$true)][long]$ExpectedBytes
)
$ErrorActionPreference = 'Stop'
function FileSha256([string]$Path) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
$archivePath = (Resolve-Path -LiteralPath $Archive).Path
$stagePath = [IO.Path]::GetFullPath($Destination)
if ((Split-Path -Leaf $stagePath) -notmatch '^\.staging-[a-f0-9-]{36}$' -or (Test-Path -LiteralPath $stagePath)) {
    throw 'Expected a new, dedicated staging directory.'
}
if ((Get-Item -LiteralPath (Split-Path -Parent $stagePath)).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Runtime parent must not be a reparse point.'
}
if ((Get-Item -LiteralPath $archivePath).Length -ne $ExpectedBytes -or
    (FileSha256 $archivePath) -ne $ExpectedSha256) {
    throw 'Runtime archive checksum mismatch.'
}
function SafePath([string]$Name) {
    if ([string]::IsNullOrWhiteSpace($Name) -or $Name.Contains('\') -or $Name.Contains(':') -or
        [IO.Path]::IsPathRooted($Name)) { throw "Unsafe runtime path: $Name" }
    foreach ($part in $Name.TrimEnd('/').Split('/')) {
        if ($part -in @('', '.', '..') -or $part -match '[. ]$' -or
            $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)') { throw "Unsafe runtime path: $Name" }
    }
    $target = [IO.Path]::GetFullPath((Join-Path $stagePath $Name))
    if (-not $target.StartsWith($stagePath + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Runtime path escaped staging.' }
    return $target
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    [long]$expanded = 0
    foreach ($entry in $zip.Entries) {
        $null = SafePath $entry.FullName
        if (-not $seen.Add($entry.FullName.TrimEnd('/')) -or
            (($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000 -or
            ($entry.ExternalAttributes -band 0x400)) { throw 'Duplicate or linked runtime member.' }
        $expanded += $entry.Length
        if ($expanded -gt 3221225472) { throw 'Runtime exceeds the 3 GiB expansion limit.' }
    }
    $null = New-Item -ItemType Directory -Path $stagePath
    foreach ($entry in $zip.Entries) {
        $target = SafePath $entry.FullName
        if ($entry.FullName.EndsWith('/')) { $null = New-Item -ItemType Directory -Path $target -Force; continue }
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force
        $inputStream = $entry.Open()
        $outputStream = [IO.File]::Open($target, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
        try { $inputStream.CopyTo($outputStream) } finally { $outputStream.Dispose(); $inputStream.Dispose() }
    }
} finally { $zip.Dispose() }
$manifest = Get-Content -LiteralPath (Join-Path $stagePath 'CPU-FILES.json') -Raw | ConvertFrom-Json
$files = @($manifest.files)
if ($manifest.schema_version -ne 1 -or $files.Count -lt 1 -or
    @(Get-ChildItem -LiteralPath $stagePath -File -Recurse).Count -ne $files.Count + 1) { throw 'Invalid CPU runtime inventory.' }
$inventory = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($entry in $files) {
    $target = SafePath $entry.path
    if (-not $inventory.Add($entry.path) -or -not (Test-Path -LiteralPath $target -PathType Leaf) -or
        (Get-Item -LiteralPath $target).Length -ne $entry.bytes -or
        (FileSha256 $target) -ne $entry.sha256) {
        throw "CPU component checksum mismatch: $($entry.path)"
    }
}
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:YOLO_CONFIG_DIR = Join-Path $stagePath '.probe-config'
$env:MPLCONFIGDIR = $env:YOLO_CONFIG_DIR
$null = New-Item -ItemType Directory -Path $env:YOLO_CONFIG_DIR -Force
& (Join-Path $stagePath 'python.exe') -I -B -c 'import torch,torchvision,cv2,numpy,onnx,netron,ultralytics,PIL,yaml,matplotlib; assert not torch.version.cuda; print(1)'
if ($LASTEXITCODE -ne 0) { throw 'CPU dependency self-test failed.' }
$probe = [IO.Path]::GetFullPath((Join-Path $stagePath '.probe-config'))
if ($probe.StartsWith($stagePath + '\', [StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath $probe)) {
    Remove-Item -LiteralPath $probe -Recurse -Force
}
