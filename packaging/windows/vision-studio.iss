#ifndef AppVersion
  #error AppVersion is required
#endif
#ifndef StageDir
  #error StageDir is required
#endif
#ifndef OutputDir
  #error OutputDir is required
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
OutputBaseFilename=vision-studio-{#AppVersion}-windows-x64-setup
Compression=lzma2/normal
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
UninstallDisplayIcon={app}\bin\vision-studio.exe
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
