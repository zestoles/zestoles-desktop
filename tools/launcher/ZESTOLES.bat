@echo off
rem Double-click entry point for ZESTOLES.
title ZESTOLES
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0zestoles.ps1"
