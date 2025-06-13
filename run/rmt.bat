@echo off
REM -------------------------------------------------------------
REM Batch-script to evaluate all compressed trees with every mode
REM -------------------------------------------------------------
setlocal EnableDelayedExpansion

REM --- where the trees live and where to store results ----------
set TREEDIR=results\compressed_tree
set OUTDIR=results\rmt

REM --- python executable (adjust if needed) ---------------------
set PY=python

REM --- which modes to run --------------------------------------
set MODELIST=naive priority
REM -------------------------------------------------------------

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

for %%F in ("%TREEDIR%\*.json") do (
    REM strip path → get file name without extension
    set BASE=%%~nF

    for %%M in (%MODELIST%) do (
        echo Processing %%~nxF with mode %%M

        "%PY%" tree_to_rmt.py ^
            --mode %%M ^
            --input "%%F" ^
            --output "%OUTDIR%\!BASE!_%%M.json"

    )
)

echo All runs complete.
pause
