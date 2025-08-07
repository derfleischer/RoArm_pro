#!/bin/bash
# RoArm M3 Automatic Installation Script
# Installs and configures everything needed

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Header
echo -e "${CYAN}${BOLD}"
echo "=============================================="
echo "🤖 RoArm M3 Automatic Installation"
echo "=============================================="
echo -e "${NC}"

# Function to print colored messages
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    print_success "Python $PYTHON_VERSION found"
    
    # Check if version is sufficient
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3,7) else 1)'; then
        print_success "Python version is sufficient (>=3.7)"
    else
        print_warning "Python version is old, consider upgrading"
    fi
else
    print_error "Python 3 not found! Please install Python 3.7 or higher"
    exit 1
fi

# Check if we're in the right directory
print_status "Checking current directory..."
if [ -f "main.py" ] && [ -f "config.yaml" ]; then
    print_success "In correct directory (RoArmM3v1)"
else
    print_error "Not in RoArmM3v1 directory!"
    print_warning "Please cd to RoArmM3v1 directory and run again"
    exit 1
fi

# Create virtual environment (optional but recommended)
read -p "$(echo -e ${YELLOW}Create virtual environment? [Y/n]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    print_status "Creating virtual environment..."
    if python3 -m venv venv 2>/dev/null; then
        print_success "Virtual environment created"
        
        # Activate virtual environment
        print_status "Activating virtual environment..."
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_warning "Could not create virtual environment, continuing without it"
    fi
fi

# Install/upgrade pip
print_status "Upgrading pip..."
python3 -m pip install --upgrade pip --quiet
print_success "pip upgraded"

# Install dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    if python3 -m pip install -r requirements.txt --quiet; then
        print_success "Dependencies installed"
    else
        print_error "Failed to install some dependencies"
        print_warning "Try manually: pip install -r requirements.txt"
    fi
else
    print_warning "requirements.txt not found, installing basic packages..."
    python3 -m pip install pyserial pyyaml numpy colorama --quiet
    print_success "Basic packages installed"
fi

# Run QuickFix to create missing files/directories
print_status "Running QuickFix to setup system..."
if [ -f "quickfix.py" ]; then
    python3 quickfix.py
    print_success "QuickFix completed"
else
    print_warning "quickfix.py not found, creating directories manually..."
    
    # Create directories
    for dir in core motion patterns teaching calibration safety utils logs sequences; do
        mkdir -p $dir
        touch $dir/__init__.py
    done
    print_success "Directories created"
fi

# Set executable permissions
print_status "Setting file permissions..."
chmod +x *.py 2>/dev/null || true
chmod +x *.sh 2>/dev/null || true
print_success "Permissions set"

# Setup Python path
print_status "Setting up Python path..."
echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)\"" > setup_env.sh
chmod +x setup_env.sh
source setup_env.sh
print_success "Python path configured"

# Run validation
print_status "Running system validation..."
echo ""
if python3 validate_system.py 2>/dev/null; then
    VALIDATION_RESULT=$?
else
    VALIDATION_RESULT=1
fi

echo ""

# Find serial ports
print_status "Searching for serial ports..."
echo -e "${CYAN}Available serial ports:${NC}"
python3 -c "
import serial.tools.list_ports
ports = list(serial.tools.list_ports.comports())
if ports:
    for i, p in enumerate(ports, 1):
        print(f'  {i}. {p.device} - {p.description}')
else:
    print('  No serial ports found')
" 2>/dev/null || echo "  Could not list ports"

# Update config with serial port
echo ""
read -p "$(echo -e ${YELLOW}Update serial port in config.yaml? [Y/n]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    read -p "Enter your serial port (e.g., /dev/tty.usbserial-110): " PORT
    if [ ! -z "$PORT" ]; then
        # Update config.yaml
        if command -v sed &>/dev/null; then
            sed -i.bak "s|port:.*|port: \"$PORT\"|g" config.yaml
            print_success "Updated serial port to: $PORT"
        else
            print_warning "Please manually update port in config.yaml to: $PORT"
        fi
    fi
fi

# Test connection
echo ""
read -p "$(echo -e ${YELLOW}Test robot connection? [Y/n]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    print_status "Testing connection..."
    if python3 test_connection.py 2>/dev/null; then
        print_success "Connection test successful!"
    else
        print_warning "Connection test failed - check cable and power"
    fi
fi

# Create startup script
print_status "Creating startup script..."
cat > start_roarm.sh << 'EOF'
#!/bin/bash
# RoArm M3 Startup Script

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Starting RoArm M3 Control System...${NC}"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Run main program
python3 main.py "$@"

echo -e "${GREEN}RoArm M3 shut down successfully${NC}"
EOF

chmod +x start_roarm.sh
print_success "Startup script created: start_roarm.sh"

# Summary
echo ""
echo -e "${CYAN}${BOLD}=============================================="
echo "📊 Installation Summary"
echo "=============================================="
echo -e "${NC}"

if [ $VALIDATION_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ System is READY TO RUN!${NC}"
    echo ""
    echo "To start the RoArm control system:"
    echo -e "  ${CYAN}./start_roarm.sh${NC}"
    echo ""
    echo "Or manually:"
    echo -e "  ${CYAN}source setup_env.sh${NC}"
    echo -e "  ${CYAN}python3 main.py${NC}"
else
    echo -e "${YELLOW}⚠️ System has some issues${NC}"
    echo ""
    echo "Recommended actions:"
    echo "  1. Run: python3 validate_system.py"
    echo "  2. Fix any reported issues"
    echo "  3. Run: python3 test_connection.py"
fi

echo ""
echo "Important files:"
echo "  • config.yaml - Configuration (update serial port!)"
echo "  • main.py - Main program"
echo "  • test_connection.py - Connection test"
echo "  • validate_system.py - System validation"
echo "  • quickfix.py - Automatic fixes"
echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo "=============================================="

# Save installation log
echo "Installation completed at $(date)" > install.log
echo "Python version: $(python3 --version)" >> install.log
echo "Directory: $(pwd)" >> install.log

# Offer to start program
echo ""
read -p "$(echo -e ${YELLOW}Start RoArm control system now? [Y/n]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo -e "${CYAN}Starting RoArm M3...${NC}"
    echo ""
    python3 main.py
fi
