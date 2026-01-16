#!/bin/bash
  set -euo pipefail

  RUY_DIR=/Users/miguelaprossine/octopuzzler/build/_deps/ruy-build/ruy
  FFT_DIR=/Users/miguelaprossine/octopuzzler/build/_deps/fft2d-build
  FARMHASH=/Users/miguelaprossine/octopuzzler/build/_deps/farmhash-build/libfarmhash.a
  TFLITE=/Users/miguelaprossine/octopuzzler/build/libtensorflow-lite.a
  CPUINFO=/Users/miguelaprossine/octopuzzler/build/_deps/cpuinfo-build
  ABSL_DIR=/Users/miguelaprossine/octopuzzler/build/_deps/abseil-cpp-build

  RUY_FLAGS=$(for f in "$RUY_DIR"/*.a; do printf -- "-Wl,-force_load,%s " "$f"; done)
  FFT_FLAGS="-Wl,-force_load,$FFT_DIR/libfft2d_fftsg.a -Wl,-force_load,$FFT_DIR/libfft2d_fftsg2d.a -Wl,-force_load,$FFT_DIR/libfft2d_alloc.a"
  ABSL_FLAGS=$(find "$ABSL_DIR" -name "libabsl_*.a" -print0 | xargs -0 -I{} printf -- "-Wl,-force_load,%s " "{}")

  clang++ -std=c++17 -fobjc-arc \
    /Users/miguelaprossine/octopuzzler/code/tools/gaze_capture_macos.mm \
    /Users/miguelaprossine/octopuzzler/code/device/tracking/eye_tracking_optimized.cpp \
    /Users/miguelaprossine/octopuzzler/code/device/tracking/blazeface_detector.cpp \
    /Users/miguelaprossine/octopuzzler/code/device/tracking/mediapipe_facemesh.cpp \
    /Users/miguelaprossine/octopuzzler/code/device/tracking/gaze_calculator.cpp \
    /Users/miguelaprossine/octopuzzler/code/device/tracking/gaze_filter.cpp \
    /Users/miguelaprossine/octopuzzler/code/device/tracking/mp_tflite_attr_parsers.cc \
    /Users/miguelaprossine/octopuzzler/third_party/mediapipe/util/tflite/operations/landmarks_to_transform_matrix.cc \
    /Users/miguelaprossine/octopuzzler/third_party/mediapipe/util/tflite/operations/transform_landmarks.cc \
    /Users/miguelaprossine/octopuzzler/third_party/mediapipe/util/tflite/operations/transform_tensor_bilinear.cc \
    -o /Users/miguelaprossine/octopuzzler/code/tools/gaze_capture_macos \
    -I/Users/miguelaprossine/octopuzzler/code \
    -I/Users/miguelaprossine/octopuzzler/third_party/abseil-cpp \
    -I/Users/miguelaprossine/octopuzzler/third_party \
    -I/Users/miguelaprossine/octopuzzler/third_party/tensorflow \
    -I/Users/miguelaprossine/octopuzzler/third_party/tensorflow/tensorflow/lite \
    -I/Users/miguelaprossine/octopuzzler/third_party/FP16/include \
    -I/opt/homebrew/include/eigen3 \
    -I/Users/miguelaprossine/octopuzzler/build/gemmlowp \
    -I/Users/miguelaprossine/octopuzzler/third_party/tflite_build/gemmlowp \
    -I/Users/miguelaprossine/octopuzzler/build/flatbuffers/include \
    -I/Users/miguelaprossine/octopuzzler/build/flatbuffers-flatc/include \
    -I/Users/miguelaprossine/octopuzzler/build/_deps/abseil-cpp-src \
    $RUY_FLAGS $FFT_FLAGS $ABSL_FLAGS \
    -Wl,-force_load,$FARMHASH \
    -Wl,-force_load,$CPUINFO/libcpuinfo.a \
    $TFLITE \
    -framework AVFoundation -framework Foundation -framework CoreMedia -framework CoreVideo -framework CoreGraphics -framework ImageIO
