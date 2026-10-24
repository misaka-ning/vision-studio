@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
set "PATH=D:\Qt\6.8.3\msvc2022_64\bin;D:\Opencv-C++\opencv\build\x64\vc16\bin;D:\Qt\Tools\Ninja;D:\Qt\bin;%PATH%"
D:\Qt\bin\cmake.exe -S E:\vision-studio -B E:\vision-studio\build-windows-2.9.1 -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_MAKE_PROGRAM=D:/Qt/Tools/Ninja/ninja.exe -DCMAKE_PREFIX_PATH=D:/Qt/6.8.3/msvc2022_64 -DOpenCV_DIR=D:/Opencv-C++/opencv/build/x64/vc16/lib -DPython3_EXECUTABLE=E:/vision-studio/build-windows/python/python.exe -DVISION_STUDIO_TEST_PYTHON=E:/vision-studio/build-windows/python/python.exe -DVISION_RESOURCE_FALLBACK=.
if errorlevel 1 exit /b %errorlevel%
D:\Qt\bin\cmake.exe --build E:\vision-studio\build-windows-2.9.1 --parallel 6
