@echo off
title Dong goi Computer Vision App
echo ========================================================
echo        TOOL DONG GOI PYTHON THANH FILE .EXE
echo ========================================================

:: 1. Kiem tra va cai dat PyInstaller
echo.
echo [Buoc 1/3] Dang kiem tra va cai dat PyInstaller...
pip install pyinstaller
if %errorlevel% neq 0 (
    echo Loi: Khong the cai dat PyInstaller. Hay kiem tra lai Python/Internet.
    pause
    exit /b
)

:: 2. Don dep file cu
echo.
echo [Buoc 2/3] Dang don dep thu muc cu...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del *.spec

:: 3. Chay lenh dong goi
echo.
echo [Buoc 3/3] Dang dong goi "main.py" thanh file .exe...
echo Qua trinh nay co the mat 1-2 phut. Vui long doi...
echo.

pyinstaller --noconfirm --onefile --windowed --name "DoAn_ThiGiacMayTinh" main.py

if %errorlevel% neq 0 (
    echo.
    echo LOI: Dong goi that bai! Hay xem thong bao loi o tren.
    pause
    exit /b
)

echo.
echo ========================================================
echo             XONG ROI! (SUCCESS)
echo ========================================================
echo.
echo File .exe cua ban nam trong thu muc: dist\DoAn_ThiGiacMayTinh.exe
echo.
pause