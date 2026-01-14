@echo off
REM Script de compilare pentru CUDA pe Windows

REM Setează path pentru Visual Studio
set "VS_PATH=D:\vss\vs"

if not exist "%VS_PATH%" (
    echo Visual Studio nu gasit la %VS_PATH%
    exit /b 1
)

echo Folosind Visual Studio: %VS_PATH%

REM Seteaza variabilele de mediu
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo ====================================
echo Compilare programe CUDA
echo ====================================

REM Compileaza programul de initializare
echo.
echo 1. Compilare initialize.cu...
nvcc -std=c++17 -arch=sm_61 -O2 -allow-unsupported-compiler -o initialize.exe initialize.cu
if %errorlevel% equ 0 (
    echo   SUCCESS!
) else (
    echo   FAILED!
    exit /b 1
)

REM Compileaza versiunea secventiala
echo.
echo 2. Compilare sequential.cu...
nvcc -std=c++17 -arch=sm_61 -O2 -allow-unsupported-compiler -o sequential.exe sequential.cu
if %errorlevel% equ 0 (
    echo   SUCCESS!
) else (
    echo   FAILED!
    exit /b 1
)

REM Compileaza versiunea paralela
echo.
echo 3. Compilare parallel.cu...
nvcc -std=c++17 -arch=sm_61 -O2 -allow-unsupported-compiler -o parallel.exe parallel.cu
if %errorlevel% equ 0 (
    echo   SUCCESS!
) else (
    echo   FAILED!
    exit /b 1
)

echo.
echo ====================================
echo Compilare finalizata cu succes!
echo ====================================
echo.
echo Pentru a rula benchmark:
echo   benchmark.bat 1000 1000 3
echo.