#!/bin/bash
# Build script for Octopus AI firmware

set -e

# Configuration
BUILD_TYPE=${1:-Release}  # Debug or Release
BUILD_DIR="build"
JOBS=$(nproc)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "==== Building Octopus AI Firmware ===="
echo "Build type: $BUILD_TYPE"
echo "Jobs: $JOBS"
echo ""

# Check BK7258 SDK
if [ -z "$BK7258_SDK_PATH" ]; then
    echo -e "${RED}ERROR: BK7258_SDK_PATH not set${NC}"
    echo "Please set BK7258_SDK_PATH environment variable"
    exit 1
fi

echo -e "${GREEN}✓ BK7258 SDK found:${NC} $BK7258_SDK_PATH"

# Check for AI models
echo ""
echo "Checking AI models..."

MODELS_DIR="models"
REQUIRED_MODELS=(
    "blazeface.tflite"
    "facemesh_lite.tflite"
    "wake_word.tflite"
)

MODELS_OK=true

for model in "${REQUIRED_MODELS[@]}"; do
    if [ -f "$MODELS_DIR/$model" ]; then
        SIZE=$(du -h "$MODELS_DIR/$model" | cut -f1)
        echo -e "${GREEN}✓${NC} $model ($SIZE)"
    else
        echo -e "${RED}✗${NC} $model (missing)"
        MODELS_OK=false
    fi
done

if [ "$MODELS_OK" = false ]; then
    echo ""
    echo -e "${YELLOW}WARNING: Some AI models are missing${NC}"
    echo "Download models using: ./download_models.sh"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Clean previous build (optional)
if [ "$2" = "clean" ]; then
    echo ""
    echo "Cleaning previous build..."
    rm -rf $BUILD_DIR
fi

# Create build directory
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
echo ""
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_TOOLCHAIN_FILE=$BK7258_SDK_PATH/cmake/toolchain.cmake \
    -DBUILD_TESTS=OFF

# Build
echo ""
echo "Building firmware..."
cmake --build . --parallel $JOBS

# Check build result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    
    # Show binary info
    BIN_FILE="octopus_app.bin"
    if [ -f "$BIN_FILE" ]; then
        SIZE=$(du -h $BIN_FILE | cut -f1)
        echo "Firmware: $BIN_FILE ($SIZE)"
        
        # Show memory usage
        echo ""
        echo "Memory usage:"
        arm-none-eabi-size octopus_app
        
        echo ""
        echo "Flash usage details:"
        echo "  Total available: 128MB"
        echo "  Firmware size: $SIZE"
        
        # Calculate percentage (rough estimate)
        SIZE_BYTES=$(stat -c%s "$BIN_FILE")
        FLASH_TOTAL=$((128 * 1024 * 1024))
        PERCENT=$((SIZE_BYTES * 100 / FLASH_TOTAL))
        echo "  Usage: ${PERCENT}%"
    fi
    
    echo ""
    echo "Next steps:"
    echo "  Flash firmware: ./flash.sh /dev/ttyUSB0"
    echo "  Or use: cmake --build . --target flash"
else
    echo ""
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi