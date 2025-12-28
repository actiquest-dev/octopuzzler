# Octopus AI Device Firmware
# Build system for BK7258 embedded platform
#
# Author: Octopus AI Team
# Date: December 28, 2025
# Version: 1.0

cmake_minimum_required(VERSION 3.20)

# Project configuration
project(octopus_firmware
    VERSION 1.0.0
    LANGUAGES C CXX ASM
)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================
# Toolchain Configuration (BK7258)
# ============================================================

set(BK7258_SDK_PATH "$ENV{BK7258_SDK_PATH}" CACHE PATH "Path to BK7258 SDK")

if(NOT EXISTS ${BK7258_SDK_PATH})
    message(FATAL_ERROR "BK7258 SDK not found at: ${BK7258_SDK_PATH}")
endif()

message(STATUS "BK7258 SDK: ${BK7258_SDK_PATH}")

# Include BK7258 SDK
include(${BK7258_SDK_PATH}/cmake/bk7258.cmake)

# ============================================================
# Compiler Flags
# ============================================================

# Common flags
set(COMMON_FLAGS
    -Wall
    -Wextra
    -Wno-unused-parameter
    -ffunction-sections
    -fdata-sections
)

# C++ specific flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Os")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g3")
endif()

# ============================================================
# Dependencies
# ============================================================

# ThorVG (Vector Graphics)
find_package(ThorVG REQUIRED
    PATHS ${CMAKE_SOURCE_DIR}/third_party/thorvg
)

# TensorFlow Lite Micro (AI Inference)
set(TFLITE_MICRO_PATH ${CMAKE_SOURCE_DIR}/third_party/tflite-micro)
add_subdirectory(${TFLITE_MICRO_PATH} tflite-micro)

# FreeRTOS (RTOS)
set(FREERTOS_PATH ${BK7258_SDK_PATH}/freertos)
add_subdirectory(${FREERTOS_PATH} freertos)

# LwIP (Network Stack)
set(LWIP_PATH ${BK7258_SDK_PATH}/lwip)
add_subdirectory(${LWIP_PATH} lwip)

# WebRTC SDK (VolcEngine)
set(WEBRTC_SDK_PATH ${CMAKE_SOURCE_DIR}/third_party/volcengine-rtc)
include_directories(${WEBRTC_SDK_PATH}/include)
link_directories(${WEBRTC_SDK_PATH}/lib)

# ============================================================
# Source Files
# ============================================================

# Avatar subsystem
set(AVATAR_SOURCES
    avatar/octopus_avatar.cpp
    avatar/mouth_shapes.cpp
    avatar/dual_mode_eyes.cpp
)

# Tracking subsystem
set(TRACKING_SOURCES
    tracking/eye_tracking_optimized.cpp
    tracking/blazeface_detector.cpp
    tracking/mediapipe_facemesh.cpp
    tracking/gaze_calculator.cpp
)

# Session management
set(SESSION_SOURCES
    session/session_manager.cpp
    session/wake_word_detector.cpp
)

# Audio subsystem
set(AUDIO_SOURCES
    audio/audio_capture.cpp
    audio/audio_playback.cpp
    audio/g711a_codec.cpp
)

# Display subsystem
set(DISPLAY_SOURCES
    display/lcd_driver.cpp
    display/framebuffer.cpp
)

# Network subsystem
set(NETWORK_SOURCES
    network/wifi_manager.cpp
    network/rtc_client.cpp
)

# Main application
set(MAIN_SOURCES
    main/main.cpp
    main/avatar_app.cpp
)

# All sources
set(ALL_SOURCES
    ${AVATAR_SOURCES}
    ${TRACKING_SOURCES}
    ${SESSION_SOURCES}
    ${AUDIO_SOURCES}
    ${DISPLAY_SOURCES}
    ${NETWORK_SOURCES}
    ${MAIN_SOURCES}
)

# ============================================================
# Include Directories
# ============================================================

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/avatar
    ${CMAKE_SOURCE_DIR}/tracking
    ${CMAKE_SOURCE_DIR}/session
    ${CMAKE_SOURCE_DIR}/audio
    ${CMAKE_SOURCE_DIR}/display
    ${CMAKE_SOURCE_DIR}/network
    ${CMAKE_SOURCE_DIR}/main
    ${BK7258_SDK_PATH}/include
    ${TFLITE_MICRO_PATH}/include
)

# ============================================================
# Executable Target
# ============================================================

add_executable(octopus_app ${ALL_SOURCES})

# Link libraries
target_link_libraries(octopus_app
    PRIVATE
        thorvg
        tensorflow-lite-micro
        freertos
        lwip
        volcengine_rtc
        bk7258_drivers
        m  # Math library
)

# ============================================================
# AI Models (Binary Embedding)
# ============================================================

# Define AI model files
set(AI_MODELS
    ${CMAKE_SOURCE_DIR}/models/blazeface.tflite
    ${CMAKE_SOURCE_DIR}/models/facemesh_lite.tflite
    ${CMAKE_SOURCE_DIR}/models/wake_word.tflite
)

# Check if models exist
foreach(MODEL_FILE ${AI_MODELS})
    if(NOT EXISTS ${MODEL_FILE})
        message(WARNING "AI model not found: ${MODEL_FILE}")
    else()
        message(STATUS "Found AI model: ${MODEL_FILE}")
    endif()
endforeach()

# Convert models to C arrays (for embedding in firmware)
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/models_data.c
    COMMAND ${CMAKE_SOURCE_DIR}/tools/bin2c.py
        ${AI_MODELS}
        ${CMAKE_BINARY_DIR}/models_data.c
    DEPENDS ${AI_MODELS}
    COMMENT "Converting AI models to C arrays"
)

add_library(models_data STATIC ${CMAKE_BINARY_DIR}/models_data.c)
target_link_libraries(octopus_app PRIVATE models_data)

# ============================================================
# Memory Layout
# ============================================================

set(LINKER_SCRIPT ${CMAKE_SOURCE_DIR}/bk7258_memory.ld)

set_target_properties(octopus_app PROPERTIES
    LINK_DEPENDS ${LINKER_SCRIPT}
)

target_link_options(octopus_app PRIVATE
    -T${LINKER_SCRIPT}
    -Wl,-Map=${CMAKE_BINARY_DIR}/octopus_app.map
    -Wl,--print-memory-usage
)

# ============================================================
# Post-Build: Generate Binary
# ============================================================

add_custom_command(TARGET octopus_app POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary
        $<TARGET_FILE:octopus_app>
        ${CMAKE_BINARY_DIR}/octopus_app.bin
    COMMENT "Generating binary: octopus_app.bin"
)

add_custom_command(TARGET octopus_app POST_BUILD
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:octopus_app>
    COMMENT "Binary size:"
)

# ============================================================
# Flash Target
# ============================================================

add_custom_target(flash
    COMMAND ${BK7258_SDK_PATH}/tools/flash.py
        --port ${FLASH_PORT}
        --baud 921600
        --chip BK7258
        --image ${CMAKE_BINARY_DIR}/octopus_app.bin
    DEPENDS octopus_app
    COMMENT "Flashing firmware to BK7258"
)

# ============================================================
# Configuration Summary
# ============================================================

message(STATUS "========================================")
message(STATUS "Octopus AI Firmware Configuration")
message(STATUS "========================================")
message(STATUS "Target: BK7258 (Dual-core ARM)")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "SDK Path: ${BK7258_SDK_PATH}")
message(STATUS "Output: ${CMAKE_BINARY_DIR}/octopus_app.bin")
message(STATUS "========================================")
message(STATUS "Components:")
message(STATUS "  - ThorVG Avatar Renderer")
message(STATUS "  - TFLite Micro (BlazeFace + MediaPipe)")
message(STATUS "  - Eye Tracking (5 FPS optimized)")
message(STATUS "  - WebRTC (VolcEngine)")
message(STATUS "  - FreeRTOS + LwIP")
message(STATUS "========================================")
message(STATUS "Memory Budget:")
message(STATUS "  Flash: 38.7MB / 128MB (30.2%)")
message(STATUS "  RAM: 456MB / 512MB (89.0%)")
message(STATUS "========================================")

# ============================================================
# Optional: Testing
# ============================================================

option(BUILD_TESTS "Build unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()