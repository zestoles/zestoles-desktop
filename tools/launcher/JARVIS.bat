@echo off
rem Double-click entry point for JARVIS. Everything real lives in jarvis.ps1;
rem this exists because a .lnk to a .ps1 opens an editor on most machines.
rem
rem -NoProfile keeps a slow or broken user profile out of the startup path.
title JARVIS
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0jarvis.ps1"
