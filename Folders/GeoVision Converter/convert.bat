@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
set tmpfile=

for /f "usebackq delims=|" %%f in (`dir /b/s %1\*.avi`) do (

set tmpfile=%%~pfnew_%%~nxf 
echo Converting file %%f
REM set tmpfile=!tmpfile:~0,-4!_new.avi
echo Output file: !tmpfile!
 
mencoder "%%f" -o "!tmpfile!" -ovc lavc -oac lavc -lavcopts abitrate=160

)

ENDLOCAL
