#!/bin/bash
# Flash firmware to BK7258 device

set -e

FIRMWARE="build/octopus_app.bin"
PORT=${1:-/dev/ttyUSB0}
BAUD=921600

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "==== Flashing Octopus AI Firmware ===="
echo "Firmware: $FIRMWARE"
echo "Port: $PORT"
echo "Baud rate: $BAUD"

# Check if firmware exists
if [ ! -f "$FIRMWARE" ]; then
    echo -e "${RED}ERROR: Firmware not found at $FIRMWARE${NC}"
    echo "Run ./build.sh first"
    exit 1
fi

# Check if device exists
if [ ! -e "$PORT" ]; then
    echo -e "${RED}ERROR: Device not found at $PORT${NC}"
    echo ""
    echo "Available USB serial devices:"
    ls -l /dev/ttyUSB* 2>/dev/null || echo "No USB serial devices found"
    exit 1
fi

# Show firmware info
SIZE=$(du -h $FIRMWARE | cut -f1)
echo ""
echo "Firmware size: $SIZE"

# Instructions
echo ""
echo -e "${YELLOW}Put device in flash mode:${NC}"
echo "1. Hold BOOT button"
echo "2. Press RESET button"
echo "3. Release BOOT button"
echo ""
read -p "Press Enter when ready..."

# Flash using BK7258 flash tool
echo ""
echo "Flashing..."

python3 $BK7258_SDK_PATH/tools/flash.py \
    --port $PORT \
    --baud $BAUD \
    --chip BK7258 \
    --image $FIRMWARE \
    --verify

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Flashing complete!${NC}"
    echo ""
    echo "Reset the device to start Octopus firmware"
    echo ""
    echo "To view logs:"
    echo "  screen $PORT 115200"
    echo "  (Press Ctrl+A, then K to exit)"
else
    echo ""
    echo -e "${RED}✗ Flashing failed${NC}"
    exit 1
fi