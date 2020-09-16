@echo off
set a=0
setlocal EnableDelayedExpansion
dir /b .\*.jpg | find /c /v "" >> .\tmp.txt
set /p c=<.\tmp.txt
del /a /f /q .\tmp.txt 
 
for %%i in (*.jpg) do (
set /a a+=1
if !a! gtr %c% (goto aa)
echo !a!
echo %%i 
ren "%%i" "!a!.jpg"
)
:aa
pause
