#!/bin/bash
# RoArm M3 Professional Control System
# Startup script with automatic diagnostics

echo "=========================================="
echo "🤖 RoArm M3 Professional Control System"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✅ Python installed: $python_version${NC}"
else
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.7 or higher.${NC}"
    exit 1
fi

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✅ Virtual environment active: $VIRTUAL_ENV${NC}"
else
    echo -e "${YELLOW}⚠️  No virtual environment detected${NC}"
    echo "   Consider creating one with: python3 -m venv venv"
    echo "   And activating with: source venv/bin/activate"
fi

# Install/update dependencies if needed
echo ""
echo "Checking dependencies..."
pip_output=$(pip3 list 2>/dev/null | grep -E "pyserial|pyyaml|numpy|colorama")

if [[ -z "$pip_output" ]]; then
    echo -e "${YELLOW}⚠️  Some dependencies may be missing${NC}"
    read -p "Install dependencies now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing dependencies..."
        pip3 install -r requirements.txt
    fi
else
    echo -e "${GREEN}✅ Core dependencies found${NC}"
fi

# Run quick diagnostics
echo ""
echo "Running system diagnostics..."
python3 -c 
import sys
sys.path.insert(0, '.')
try:
    from utils.debug_tool import SystemDebugger
    debugger = SystemDebugger(verbose=False)
    if debugger.quick_test():
        print('✅ System check PASSED')
        sys.exit(0)
    else:
        print('❌ System check FAILED')
        print('   Run full diagnostics with: python3 utils/debug_tool.py')
        sys.exit(1)
except Exception as e:
    print(f'⚠️  Could not run diagnostics: {e}')
    sys.exit(2)


diagnostic_result=$?

# Ask user if they want to continue if diagnostics failed
if [[ $diagnostic_result -ne 0 ]]; then
    echo ""
    echo -e "${YELLOW}System diagnostics reported issues.${NC}"
    read -p "Do you want to run full diagnostics? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 utils/debug_tool.py
        echo ""
        read -p "Continue to main program? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Parse command line arguments
PORT="/dev/tty.usbserial-110"
BAUDRATE="115200"
SPEED="1.0"
DEBUG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --baudrate)
            BAUDRATE="$2"
            shift 2
            ;;
        --speed)
            SPEED="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --calibrate)
            CALIBRATE="--calibrate"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --port PORT         Serial port (default: /dev/tty.usbserial-110)"
            echo "  --baudrate RATE     Baud rate (default: 115200)"
            echo "  --speed SPEED       Speed factor (default: 1.0)"
            echo "  --debug             Enable debug mode"
            echo "  --calibrate         Run auto calibration"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Start the main program
echo ""
echo "=========================================="
echo "Starting RoArm Control System..."
echo "Port: $PORT"
echo "Baudrate: $BAUDRATE"
echo "Speed: $SPEED"
echo "=========================================="
echo ""

# Run main program with error handling
python3 main.py --port "$PORT" --baudrate "$BAUDRATE" --speed "$SPEED" $DEBUG $CALIBRATE

# Check exit code
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}✅ Program exited normally${NC}"
else
    echo ""
    echo -e "${RED}❌ Program exited with error${NC}"
    echo "Check logs/roarm_system.log for details"
fi
