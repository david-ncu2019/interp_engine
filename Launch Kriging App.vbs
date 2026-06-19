Set WshShell = CreateObject("WScript.Shell")
batPath = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\")) & "kriging.bat"
WshShell.Run "cmd /c """ & batPath & """", 0, False
