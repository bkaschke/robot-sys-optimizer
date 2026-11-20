; Inno Setup script: per-user installer (no admin rights) for the PyInstaller folder.
;   iscc /DAppVersion=1.0 /DSourceDir=dist\RobotCellOptimizer packaging\RobotCellOptimizer.iss
; scripts/build_exe.py --installer passes the defines (AppVersion from git describe, NumericVersion for the file version). Nothing is downloaded at install time.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef NumericVersion
  #define NumericVersion "0.0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\RobotCellOptimizer"
#endif

[Setup]
AppId={{6F2C9B1E-4B7A-4E36-9D0A-5B3F1C7A2E10}
AppName=Robot Cell Optimizer
AppVersion={#AppVersion}
VersionInfoVersion={#NumericVersion}
AppPublisher=Robot Cell Optimizer
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\RobotCellOptimizer
DefaultGroupName=Robot Cell Optimizer
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\RobotCellOptimizer.exe
OutputBaseFilename=RobotCellOptimizer-{#AppVersion}-setup
Compression=lzma2/normal
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Icons]
Name: "{autoprograms}\Robot Cell Optimizer"; Filename: "{app}\RobotCellOptimizer.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Robot Cell Optimizer"; Filename: "{app}\RobotCellOptimizer.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\RobotCellOptimizer.exe"; Description: "Start Robot Cell Optimizer"; Flags: nowait postinstall skipifsilent

; Projects, caches and tools live in %LOCALAPPDATA%\RobotCellOptimizer. The uninstaller does NOT touch
; that folder (no uninstall-delete entry for it): the user's work survives an uninstall or update.
