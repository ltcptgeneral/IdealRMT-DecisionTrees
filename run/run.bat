@echo off
REM --- settings --------------------------------------------------------
set INPUT=..\data\combined\data.csv
set OUTDIR=results\tree
set DEPTH_LIST=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
set BITS_LIST=0 1 3
set PY=python
REM ---------------------------------------------------------------------

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

for %%D in (%DEPTH_LIST%) do (
    for %%B in (%BITS_LIST%) do (
        echo Running depth=%%D bits=%%B
        %PY% decision_tree.py ^
            --input "%INPUT%" ^
            --output "%OUTDIR%\tree_d%%D_b%%B.json" ^
            --depth %%D ^
            --nudge --bits %%B
    )
)

echo All runs complete
pause
