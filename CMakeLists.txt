# CMakeList.txt : CMake project for vulkan_guide, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("vulkan_guide")


find_package(Vulkan REQUIRED)

add_subdirectory(third_party)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_subdirectory(src)

#find_program(SLANG_COMPILER slangc PATH "C:/Users/Marie/Downloads/slang-2026.1-windows-x86_64/bin/" NO_DEFAULT_PATH) //not working
set(SLANG_COMPILER "C:/Users/Marie/Downloads/slang-2026.1-windows-x86_64/bin/slangc.exe")

file(GLOB_RECURSE SLANG_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.slang"
    )

foreach(SLANG ${SLANG_SOURCE_FILES})
  message(STATUS "BUILDING SHADER")
  get_filename_component(FILE_NAME ${SLANG} NAME_WE)
  get_filename_component(FILE_NAME_EXT ${SLANG} NAME)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")
  message(STATUS ${SLANG} " -> " ${SPIRV})
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${SLANG_COMPILER} ${SLANG} -target spirv -o ${SPIRV}
    DEPENDS ${SLANG})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(SLANG)

add_custom_target(
    Shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
    )
