param(
    [Parameter(Mandatory=$true)][string]$Installer,
    [Parameter(Mandatory=$true)][string]$TestRoot,
    [string]$GpuRuntime = ''
)
$ErrorActionPreference = 'Stop'
$installerPath = (Resolve-Path -LiteralPath $Installer).Path
$qaRoot = [IO.Path]::GetFullPath($TestRoot)
if (Test-Path -LiteralPath $qaRoot) { throw 'Use a new dedicated test directory.' }
New-Item -ItemType Directory -Path $qaRoot | Out-Null
$installRoot = Join-Path $qaRoot '安装路径 Space\VisionStudio'
$dataRoot = Join-Path $qaRoot 'personal-data'
New-Item -ItemType Directory -Path $dataRoot | Out-Null
[IO.File]::WriteAllText((Join-Path $dataRoot 'preserve-user-model.pt'), 'preserve-model-and-settings')
$env:VISION_STUDIO_DATA_DIR = $dataRoot
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
foreach ($name in @('VISION_STUDIO_HOME','VISION_STUDIO_PYTHON','QT_PLUGIN_PATH','QT_QPA_PLATFORM','QT_QPA_PLATFORM_PLUGIN_PATH',
    'QTWEBENGINEPROCESS_PATH','QTWEBENGINE_RESOURCES_PATH','QTWEBENGINE_LOCALES_PATH','QML_IMPORT_PATH','QML2_IMPORT_PATH','PYTHONHOME','PYTHONPATH')) {
    Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
}
$checks = [Collections.Generic.List[object]]::new()
function RunChecked([string]$File, [string[]]$Arguments, [string]$Label) {
    $process = Start-Process -FilePath $File -ArgumentList $Arguments -PassThru
    while (-not $process.WaitForExit(30000)) { Write-Output "$Label is still running." }
    if ($process.ExitCode -ne 0) { throw "$Label failed: $($process.ExitCode)" }
    Write-Output "$Label passed."
    $checks.Add(@{name=$Label;passed=$true;exit_code=$process.ExitCode})
}
function InstallPackage([string]$Label) {
    RunChecked $installerPath @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART',
        '/LANG=chinesesimplified',('/DIR="'+$installRoot+'"'),('/LOG="'+(Join-Path $qaRoot ($Label+'.log'))+'"')) $Label
}
function Smoke([string]$Option, [string]$Name, [string[]]$Extra=@()) {
    $destination = Join-Path $qaRoot $Name
    RunChecked (Join-Path $installRoot 'bin\vision-studio.exe') (@($Option,('"'+$destination+'"'))+$Extra) $Name
    $reportName = if ($Option -eq '--smoke-conversion') {'conversion-smoke-report.json'} elseif ($Option -eq '--smoke-model') {'model-display-report.json'} else {'smoke-report.json'}
    $report = Get-Content -LiteralPath (Join-Path $destination $reportName) -Raw | ConvertFrom-Json
    if ($report.success -ne $true) { throw "$Name did not report success." }
}
InstallPackage 'install-custom-directory'
$manifest = Get-Content -LiteralPath (Join-Path $installRoot 'WINDOWS-FILES.json') -Raw | ConvertFrom-Json
$important = @('bin/vision-studio.exe','bin/Qt6Core.dll','bin/Qt6WebEngineCore.dll',
    'runtime/python.exe','runtime/python311.dll','runtime/Lib/site-packages/torch/lib/torch_cpu.dll',
    'models/yolov5n.onnx','models/yolov8n.pt','models/yolov5n.pt','scripts/pt_worker.py',
    'scripts/netron_server.py','scripts/model_convert.py','scripts/gpu_windows.py')
foreach ($entry in $manifest) {
    $target = Join-Path $installRoot $entry.path
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw "Missing payload: $($entry.path)" }
    if ((Get-Item -LiteralPath $target).Length -ne $entry.bytes) { throw "Wrong payload size: $($entry.path)" }
    if ($important -contains $entry.path) {
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Wrong payload SHA256: $($entry.path)" }
    }
}
$checks.Add(@{name='all-payload-paths-and-sizes';passed=$true;files=$manifest.Count;critical_sha256_files=$important.Count})
Smoke '--smoke' 'installed-onnx-cpu' @('--device','cpu')
Smoke '--smoke-pt' 'installed-pt-cpu' @('--device','cpu')
Smoke '--smoke-model' 'installed-model-views' @('--display-model',('"'+(Join-Path $installRoot 'models\yolov5n.onnx')+'"'))
Smoke '--smoke-conversion' 'installed-conversion-onnx' @('--conversion-source',('"'+(Join-Path $installRoot 'models\yolov8n.pt')+'"'),'--conversion-format','onnx')
Smoke '--smoke-conversion' 'installed-conversion-torchscript' @('--conversion-source',('"'+(Join-Path $installRoot 'models\yolov5n.pt')+'"'),'--conversion-format','torchscript')
if ($GpuRuntime) {
    $env:VISION_STUDIO_GPU_RUNTIME_DIR = [IO.Path]::GetFullPath($GpuRuntime)
    Smoke '--smoke' 'installed-onnx-cuda' @('--device','cuda')
    Smoke '--smoke-pt' 'installed-pt-cuda' @('--device','cuda')
}
[IO.File]::WriteAllText((Join-Path $installRoot 'preserve-imported-model.pt'), 'user-file-inside-install-tree')
$preservedHashes = @{}
foreach ($file in @((Join-Path $dataRoot 'preferences.ini'),(Join-Path $dataRoot 'history.json'),
    (Join-Path $dataRoot 'preserve-user-model.pt'),(Join-Path $installRoot 'preserve-imported-model.pt'))) {
    $preservedHashes[$file] = (Get-FileHash -LiteralPath $file).Hash
}
InstallPackage 'upgrade-reinstall'
foreach ($entry in $preservedHashes.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Key) -or (Get-FileHash -LiteralPath $entry.Key).Hash -ne $entry.Value) { throw 'Upgrade changed personal settings, history or a user model.' }
}
$checks.Add(@{name='upgrade-preserves-user-files';passed=$true})
RunChecked (Join-Path $installRoot 'unins000.exe') @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART',('/LOG="'+(Join-Path $qaRoot 'uninstall.log')+'"')) 'uninstall'
if (Test-Path -LiteralPath (Join-Path $installRoot 'bin\vision-studio.exe')) { throw 'Uninstall left the application executable.' }
if ((Get-Content -LiteralPath (Join-Path $dataRoot 'preserve-user-model.pt') -Raw) -ne 'preserve-model-and-settings') { throw 'Uninstall changed personal data.' }
if ((Get-Content -LiteralPath (Join-Path $installRoot 'preserve-imported-model.pt') -Raw) -ne 'user-file-inside-install-tree') { throw 'Uninstall changed an imported user file.' }
foreach ($entry in $preservedHashes.GetEnumerator()) {
    if ((Get-FileHash -LiteralPath $entry.Key).Hash -ne $entry.Value) { throw 'Uninstall changed personal settings, history or a user model.' }
}
$checks.Add(@{name='uninstall-preserves-user-files';passed=$true})
$report = @{success=$true;installer=$installerPath;sha256=(Get-FileHash -LiteralPath $installerPath).Hash.ToLowerInvariant();
    installed_path=$installRoot;data_path=$dataRoot;checks=$checks;tested_at=(Get-Date).ToString('o');
    scope='Current Windows host, custom Unicode/space path, isolated developer PATH; no clean VM claimed.'}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $qaRoot 'installer-qa.json') -Encoding utf8
Write-Output (Join-Path $qaRoot 'installer-qa.json')
