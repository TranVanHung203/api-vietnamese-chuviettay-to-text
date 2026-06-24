Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

root = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = fso.BuildPath(root, "run_server.bat")

sh.CurrentDirectory = root
sh.Run Chr(34) & batPath & Chr(34), 0, False

Set fso = Nothing
Set sh = Nothing
