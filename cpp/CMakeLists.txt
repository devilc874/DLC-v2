cmake_minimum_required(VERSION 3.16)
project(dlc VERSION 0.1.0 LANGUAGES CXX)

# ── C++17 Standard ──────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler Warnings ───────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# ── Find zlib ────────────────────────────────────────────────────────────────
find_package(ZLIB REQUIRED)

# ── Static Library (core engine) ─────────────────────────────────────────────
add_library(dlc_lib STATIC
    src/format.cpp
    src/windowing.cpp
    src/models.cpp
    src/encoders.cpp
    src/pipeline.cpp
    src/codec.cpp
)
target_include_directories(dlc_lib PUBLIC include)
target_link_libraries(dlc_lib PRIVATE ZLIB::ZLIB)

# ── Static linking (portable exe, no DLLs needed) ────────────────────────────
if(MINGW)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -static-libgcc -static-libstdc++")
    # Use the static zlib (.a) instead of the dynamic one (.dll.a)
    set(ZLIB_USE_STATIC_LIBS ON)
    find_library(ZLIB_STATIC_LIB libz.a PATHS "${CMAKE_PREFIX_PATH}/lib" NO_DEFAULT_PATH)
    if(NOT ZLIB_STATIC_LIB)
        find_library(ZLIB_STATIC_LIB libz.a)
    endif()
endif()

# ── CLI Executable ───────────────────────────────────────────────────────────
add_executable(dlc_exe src/main.cpp)
set_target_properties(dlc_exe PROPERTIES OUTPUT_NAME dlc)
if(MINGW AND ZLIB_STATIC_LIB)
    target_link_libraries(dlc_exe PRIVATE dlc_lib ${ZLIB_STATIC_LIB})
    target_link_options(dlc_exe PRIVATE -static -static-libgcc -static-libstdc++)
else()
    target_link_libraries(dlc_exe PRIVATE dlc_lib ZLIB::ZLIB)
endif()

# ── Testing (GTest via FetchContent) ─────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# Prevent overriding parent project compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(dlc_tests
    tests/test_format.cpp
    tests/test_windowing.cpp
    tests/test_models.cpp
    tests/test_encoders.cpp
    tests/test_pipeline.cpp
    tests/test_codec.cpp
)
target_link_libraries(dlc_tests PRIVATE dlc_lib ZLIB::ZLIB GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(dlc_tests)
