# SPDX-License-Identifier: AGPL-3.0-only
param(
    [Parameter(Mandatory=$true)][string]$InstallRoot,
    [string]$IncomingManifest = '',
    [Parameter(Mandatory=$true)][string]$RuntimeDescriptor,
    [switch]$RuntimeOnly
)
$ErrorActionPreference = 'Stop'
function FileSha256([string]$Path) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
$root = [IO.Path]::GetFullPath($InstallRoot).TrimEnd('\')
if ($root.Length -lt 5 -or -not (Test-Path -LiteralPath (Join-Path $root 'windows-install.json')) -or
    (Get-Item -LiteralPath $root).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Invalid installation root.' }
$descriptor = Get-Content -LiteralPath $RuntimeDescriptor -Raw | ConvertFrom-Json
$inventory = Join-Path $root 'runtime\CPU-FILES.json'
if ($RuntimeOnly) {
    if (-not (Test-Path -LiteralPath $inventory -PathType Leaf)) { exit 0 }
    if (-not $descriptor.inventory_sha256 -or (FileSha256 $inventory) -ne $descriptor.inventory_sha256) {
        Write-Output 'Unknown or modified bundled runtime inventory preserved.'; exit 0
    }
    $cpu = Get-Content -LiteralPath $inventory -Raw | ConvertFrom-Json
    $old = @($cpu.files | ForEach-Object { @{path=('runtime/'+$_.path);bytes=$_.bytes;sha256=$_.sha256} })
    $old += @{path='runtime/CPU-FILES.json';bytes=(Get-Item -LiteralPath $inventory).Length;sha256=$descriptor.inventory_sha256}
    $incoming = @()
} else {
    $old = Get-Content -LiteralPath (Join-Path $root 'WINDOWS-FILES.json') -Raw | ConvertFrom-Json
    $incoming = Get-Content -LiteralPath $IncomingManifest -Raw | ConvertFrom-Json
}
$names = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($entry in $incoming) { $null = $names.Add($entry.path) }
$keepRuntime = $false
if (-not $RuntimeOnly -and (Test-Path -LiteralPath $inventory -PathType Leaf) -and $descriptor.inventory_sha256) {
    $keepRuntime = (FileSha256 $inventory) -eq $descriptor.inventory_sha256
}
$removed = @(); $preserved = @()
# Validate the entire old manifest before deleting anything.
$safe = @()
foreach ($entry in $old) {
    $name = [string]$entry.path
    if ([string]::IsNullOrWhiteSpace($name) -or $name.Contains('\') -or [IO.Path]::IsPathRooted($name) -or
        $name -match '[<>:"|?*\x00]' -or @($name.Split('/') | Where-Object { $_ -in @('', '.', '..') -or $_ -match '[. ]$' }).Count) {
        throw 'Unsafe path in previous installation manifest.'
    }
    $target = [IO.Path]::GetFullPath((Join-Path $root $name))
    if (-not $target.StartsWith($root+'\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Previous payload path escaped installation.' }
    $safe += @{entry=$entry;target=$target}
}
foreach ($item in $safe) {
    $entry = $item.entry; $target = $item.target
    if ($names.Contains($entry.path) -or ($keepRuntime -and $entry.path.StartsWith('runtime/'))) { continue }
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { continue }
    $linked = $false; $cursor = $target
    while ($cursor.Length -gt $root.Length) {
        if ((Get-Item -LiteralPath $cursor).Attributes -band [IO.FileAttributes]::ReparsePoint) { $linked = $true; break }
        $cursor = Split-Path -Parent $cursor
    }
    if ($linked -or (Get-Item -LiteralPath $target).Length -ne $entry.bytes -or
        (FileSha256 $target) -ne $entry.sha256) {
        $preserved += $entry.path; continue
    }
    Remove-Item -LiteralPath $target -Force
    $removed += $entry.path
    $folder = Split-Path -Parent $target
    while ($folder.Length -gt $root.Length -and -not @(Get-ChildItem -LiteralPath $folder -Force).Count) {
        Remove-Item -LiteralPath $folder -Force
        $folder = Split-Path -Parent $folder
    }
}
$report = @{removed=$removed;preserved_modified=$preserved;retained_compatible_runtime=$keepRuntime} | ConvertTo-Json -Depth 5
if ($RuntimeOnly) { Write-Output $report }
else { $report | Set-Content -LiteralPath (Join-Path (Split-Path -Parent $IncomingManifest) 'cleanup-report.json') -Encoding UTF8 }
