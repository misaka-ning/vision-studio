#ifndef AppVersion
  #error AppVersion is required
#endif
#ifndef StageDir
  #error StageDir is required
#endif
#ifndef OutputDir
  #error OutputDir is required
#endif
#ifndef InstallProfile
  #define InstallProfile "offline"
#endif

[Setup]
AppId={{02E34A30-7577-45F3-A6CD-420EAA1D7021}
AppName=Vision Studio
AppVersion={#AppVersion}
AppPublisher=misaka_ning
AppPublisherURL=https://github.com/misaka-ning/vision-studio
AppSupportURL=https://github.com/misaka-ning/vision-studio/issues
DefaultDirName={%USERPROFILE}\VisionStudio
DefaultGroupName=Vision Studio
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
DisableDirPage=no
DisableProgramGroupPage=yes
LicenseFile={#StageDir}\LICENSE
OutputDir={#OutputDir}
OutputBaseFilename=vision-studio-{#AppVersion}-windows-x64-{#InstallProfile}-setup
Compression=lzma2/normal
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
UninstallDisplayIcon={app}\bin\vision-studio.exe
; The hash-checked migration removes obsolete owned files. A new log leaves
; modified legacy models/documents outside the new installer's ownership.
UninstallLogMode=new
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
AppMutex=VisionStudioDesktop
ChangesEnvironment=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
#if FileExists(SourcePath + "ChineseSimplified.isl")
Name: "chinesesimplified"; MessagesFile: "ChineseSimplified.isl"
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#StageDir}\WINDOWS-FILES.json"; DestName: "incoming-files.json"; Flags: dontcopy
Source: "{#StageDir}\cpu-runtime.json"; DestName: "incoming-runtime.json"; Flags: dontcopy
Source: "cleanup_install.ps1"; Flags: dontcopy

[Icons]
Name: "{group}\Vision Studio"; Filename: "{app}\bin\vision-studio.exe"; WorkingDir: "{app}"
Name: "{userdesktop}\Vision Studio"; Filename: "{app}\bin\vision-studio.exe"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{group}\{cm:UninstallProgram,Vision Studio}"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\bin\vision-studio.exe"; Description: "{cm:LaunchProgram,Vision Studio}"; Flags: nowait postinstall skipifsilent

; Only installer-owned files are removed. User models, settings, recordings and
; the optional GPU environment are stored outside {app} and remain untouched.

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Target: String;
  Existing: TFindRec;
  ExitCode: Integer;
  Arguments: String;
begin
  Result := '';
  Target := RemoveBackslashUnlessRoot(ExpandFileName(WizardDirValue));
  if (Length(Target) <= 3) or
     (CompareText(Target, ExpandConstant('{%USERPROFILE}')) = 0) or
     (CompareText(Target, ExpandConstant('{sd}\Users')) = 0) then
  begin
    Result := 'Please select a dedicated VisionStudio folder.';
    exit;
  end;
  if DirExists(Target) and not FileExists(Target + '\windows-install.json') then
  begin
    if FindFirst(Target + '\*', Existing) then
    begin
      try
        repeat
          if (Existing.Name <> '.') and (Existing.Name <> '..') then
          begin
            Result := 'The selected folder contains other files. Please select a new VisionStudio folder.';
            exit;
          end;
        until not FindNext(Existing);
      finally
        FindClose(Existing);
      end;
    end;
  end;
  if FileExists(Target + '\windows-install.json') and FileExists(Target + '\WINDOWS-FILES.json') then
  begin
    ExtractTemporaryFile('cleanup_install.ps1');
    ExtractTemporaryFile('incoming-files.json');
    ExtractTemporaryFile('incoming-runtime.json');
    Arguments := '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' +
      ExpandConstant('{tmp}\cleanup_install.ps1') + '" -InstallRoot "' + Target +
      '" -IncomingManifest "' + ExpandConstant('{tmp}\incoming-files.json') +
      '" -RuntimeDescriptor "' + ExpandConstant('{tmp}\incoming-runtime.json') + '"';
    if not Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Arguments,
      '', SW_HIDE, ewWaitUntilTerminated, ExitCode) or (ExitCode <> 0) then
      Result := 'Unable to safely check the previous installation. No new files were installed. See setup log.';
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Target: String;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    Target := RemoveBackslashUnlessRoot(ExpandFileName(WizardDirValue));
    if (Length(Target) <= 3) or
       (CompareText(Target, ExpandConstant('{%USERPROFILE}')) = 0) or
       (CompareText(Target, ExpandConstant('{sd}\Users')) = 0) then
    begin
      MsgBox('Please select a dedicated VisionStudio folder.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;
