@echo off
rem Double-click entry point. The PowerShell launcher owns diagnostics and
rem interpreter selection; this file exists because Windows opens .ps1 files in
rem an editor by default.
title ZESTOLES
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launcher\zestoles.ps1"
