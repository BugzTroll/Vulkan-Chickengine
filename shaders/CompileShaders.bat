SET SLANG_EXE=C:\Users\Marie\Downloads\slang-2026.1-windows-x86_64\bin\slangc.exe

%SLANG_EXE% triangle.slang -target spirv -o triangle.spv
%SLANG_EXE% coloredTriangle.slang -target spirv -o coloredTriangle.spv
%SLANG_EXE% baseMesh.slang -target spirv -o baseMesh.spv
%SLANG_EXE% textureLit.slang -target spirv -o textureLit.spv