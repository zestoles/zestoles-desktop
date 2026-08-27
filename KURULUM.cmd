@echo off
title ZESTOLES Kurulum
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launcher\install-zestoles.ps1"
if errorlevel 1 pause
