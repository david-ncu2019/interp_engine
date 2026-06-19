Set WshShell = CreateObject("WScript.Shell")
batPath = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\")) & "run.bat"
WshShell.Run "cmd /c """ & batPath & """", 1, True
