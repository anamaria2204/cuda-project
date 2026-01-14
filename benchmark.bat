@echo off
REM Script de benchmark pentru convolutie CUDA

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Utilizare: %~nx0 [n] [m] [k] [p] [b]
    echo Exemplu: %~nx0 1000 1000 3 256 4
    exit /b 1
)

set n=%~1
set m=%~2
set k=%~3
set p=%~4
set b=%~5

if "!k!"=="" set k=3
if "!p!"=="" set p=256
if "!b!"=="" set b=0

echo ====================================
echo BENCHMARK CONVOLUTIE CUDA p=!p!
echo ====================================
echo Configurare: n=!n! m=!m! k=!k! p=!p!
echo.

REM Genereaza input
echo Generare matrice de intrare...
initialize.exe !n! !m! !k!

if not exist "input.txt" (
    echo Eroare: Nu s-au generat matricele de intrare!
    exit /b 1
)

REM Benchmark versiune secventiala
echo.
echo ====================================
echo Benchmark SEQUENTIAL (CPU)
echo ====================================

set count=10
set "sum=0"

for /l %%i in (1,1,!count!) do (
    echo Iteratia %%i...
    for /f "usebackq delims=" %%a in (`sequential.exe !n! !m! !k!`) do (
        set lastLine=%%a
    )
    echo Timp: !lastLine! ms
    
    for /f %%s in ('powershell -NoProfile -Command "[double](!sum!) + [double](!lastLine!)"') do set sum=%%s
)

for /f %%m in ('powershell -NoProfile -Command "([double]!sum!) / !count!"') do set media_seq=%%m

REM Salveaza baseline-ul pentru verificare
copy output_sequential.txt output_baseline.txt > nul

echo.
echo Media SEQUENTIAL: !media_seq! ms

REM Benchmark versiune paralela GPU
echo.
echo ====================================
echo Benchmark PARALLEL (GPU CUDA)
echo ====================================

set sum=0

for /l %%i in (1,1,!count!) do (
    echo Iteratia %%i...
    for /f "usebackq delims=" %%a in (`parallel.exe !n! !m! !k! !p! !b!`) do (
        set lastLine=%%a
    )
    echo Timp: !lastLine! ms
    
    REM TESTARE: Compara output-ul cu baseline-ul (din sequential)
    echo   Testare corectitudine...
    fc output_baseline.txt output_parallel.txt > nul 2>&1
    if !errorlevel! neq 0 (
        echo EROARE: Rezultatul paralel nu corespunde celui secvential!
        type output_baseline.txt | head -5 > temp1.txt
        type output_parallel.txt | head -5 > temp2.txt
        echo Primele 5 linii baseline:
        type temp1.txt
        echo Primele 5 linii parallel:
        type temp2.txt
        exit /b 1
    ) else (
        echo   OK - Rezultate identice!
    )
    
    for /f %%s in ('powershell -NoProfile -Command "[double](!sum!) + [double](!lastLine!)"') do set sum=%%s
)

for /f %%m in ('powershell -NoProfile -Command "([double]!sum!) / !count!"') do set media_par=%%m

echo.
echo Media PARALLEL: !media_par! ms

REM Calculeaza speedup
echo.
echo ====================================
echo REZULTATE FINALE
echo ====================================
echo Sequential: !media_seq! ms
echo Parallel:   !media_par! ms

for /f %%s in ('powershell -NoProfile -Command "([double]!media_seq!) / [double]!media_par!"') do set speedup=%%s

echo Speedup:    !speedup!x

endlocal
