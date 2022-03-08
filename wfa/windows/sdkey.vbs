Set Arg = WScript.Arguments
set WshShell = WScript.CreateObject("WScript.Shell")
WshShell.SendKeys Arg(0)