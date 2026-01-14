@echo off
REM Script de benchmark complet pentru tema

setlocal enabledelayedexpansion

echo ====================================
echo BENCHMARK TEMA - CONVOLUTIE CUDA
echo ====================================
echo.

REM Configuratiile din tema
REM Test 1: N=M=10, n=m=3, p=2 + secvential
REM Test 2: N=M=1000, n=m=3, p=2,4,8,16 + secvential
REM Test 3: N=M=10000, n=m=3, p=2,4,8,16 + secvential

set iterations=10

REM ====================================
REM TEST 1: N=M=10
REM ====================================
echo.
echo ====================================
echo TEST 1: N=M=10, k=3, p=2
echo ====================================

set n=10
set m=10
set k=3
set p=2

echo Generare input...
initialize.exe !n! !m! !k!

echo.
echo SEQUENTIAL:
set sum=0
for /l %%i in (1,1,!iterations!) do (
    for /f "usebackq delims=" %%a in (`sequential.exe !n! !m! !k!`) do set lastLine=%%a
    for /f %%s in ('powershell -NoProfile -Command "[double](!sum!) + [double](!lastLine!)"') do set sum=%%s
)
for /f %%m in ('powershell -NoProfile -Command "([double]!sum!) / !iterations!"') do set media_seq=%%m
echo Media: !media_seq! ms

echo.
echo PARALLEL (p=!p!):
set sum=0
for /l %%i in (1,1,!iterations!) do (
    for /f "usebackq delims=" %%a in (`parallel.exe !n! !m! !k! !p!`) do set lastLine=%%a
    fc output_sequential.txt output_parallel.txt > nul 2>&1
    if !errorlevel! neq 0 (
        echo EROARE: Rezultate nu corespund!
        exit /b 1
    )
    for /f %%s in ('powershell -NoProfile -Command "[double](!sum!) + [double](!lastLine!)"') do set sum=%%s
)
for /f %%m in ('powershell -NoProfile -Command "([double]!sum!) / !iterations!"') do set media_par=%%m
echo Media: !media_par! ms
for /f %%s in ('powershell -NoProfile -Command "([double]!media_seq!) / [double]!media_par!"') do set speedup=%%s
echo Speedup: !speedup!x

REM ====================================
REM TEST 2: N=M=1000
REM ====================================
echo.
echo ====================================
echo TEST 2: N=M=1000, k=3
echo ====================================

set n=1000
set m=1000
set k=3

echo Generare input...
initialize.exe !n! !m! !k!

echo.
echo SEQUENTIAL:
set sum=0
for /l %%i in (1,1,!iterations!) do (
    for /f "usebackq delims=" %%a in (`sequential.exe !n! !m! !k!`) do set lastLine=%%a
    for /f %%s in ('powershell -NoProfile -Command "[double](!sum!) + [double](!lastLine!)"') do set sum=%%s
)
for /f %%m in ('powershell -NoProfile -Command "([double]!sum!) / !iterations!"') do set media_seq=%%m
echo Media: !media_seq! ms

REM Test cu p=2,4,8,16
for %%p in (2 4 8 16) do (
    echo.
    echo PARALLEL (p=%%p):
    set sum=0
    for /l %%i in (1,1,!iterations!) do (
        for /f "usebackq delims=" %%%%a in (`parallel.exe !n! !m! !k! %%p`) do set lastLine=%%%%a
        fc output_sequential.txt output_parallel.txt > nul 2>&1
        if !errorlevel! neq 0 (
            echo EROARE: Rezultate nu corespund!
            exit /b 1
        )
        for /f %%%%s in ('powershell -NoProfile -Command "[double](!sum!) + [double]!lastLine!"') do set sum=%%%%s
    )
    for /f %%%%m in ('powershell -NoProfile -Command "([double]!sum!) / !iterations!"') do set media_par=%%%%m
    echo   Media: !media_par! ms
    for /f %%%%s in ('powershell -NoProfile -Command "([double]!media_seq!) / [double]!media_par!"') do set speedup=%%%%s
    echo   Speedup: !speedup!x
)

REM ====================================
REM TEST 3: N=M=10000
REM ====================================
echo.
echo ====================================
echo TEST 3: N=M=10000, k=3
echo ====================================

set n=10000
set m=10000
set k=3

echo Generare input...
initialize.exe !n! !m! !k!

echo.
echo SEQUENTIAL:
set sum=0
for /l %%i in (1,1,!iterations!) do (
    for /f "usebackq delims=" %%a in (`sequential.exe !n! !m! !k!`) do set lastLine=%%a
    for /f %%s in ('powershell -NoProfile -Command "[double](!sum!) + [double](!lastLine!)"') do set sum=%%s
)
for /f %%m in ('powershell -NoProfile -Command "([double]!sum!) / !iterations!"') do set media_seq=%%m
echo Media: !media_seq! ms

REM Test cu p=2,4,8,16
for %%p in (2 4 8 16) do (
    echo.
    echo PARALLEL (p=%%p):
    set sum=0
    for /l %%i in (1,1,!iterations!) do (
        for /f "usebackq delims=" %%%%a in (`parallel.exe !n! !m! !k! %%p`) do set lastLine=%%%%a
        fc output_sequential.txt output_parallel.txt > nul 2>&1
        if !errorlevel! neq 0 (
            echo EROARE: Rezultate nu corespund!
            exit /b 1
        )
        for /f %%%%s in ('powershell -NoProfile -Command "[double](!sum!) + [double]!lastLine!"') do set sum=%%%%s
    )
    for /f %%%%m in ('powershell -NoProfile -Command "([double]!sum!) / !iterations!"') do set media_par=%%%%m
    echo   Media: !media_par! ms
    for /f %%%%s in ('powershell -NoProfile -Command "([double]!media_seq!) / [double]!media_par!"') do set speedup=%%%%s
    echo   Speedup: !speedup!x
)

echo.
echo ====================================
echo BENCHMARK FINALIZAT
echo ====================================

endlocal
