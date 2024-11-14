#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display messages with colors
function echo_info() {
    echo -e "\033[1;34m$1\033[0m"  # Blue color
}

function echo_success() {
    echo -e "\033[1;32m$1\033[0m"  # Green color
}

function echo_error() {
    echo -e "\033[1;31m$1\033[0m"  # Red color
}

# Detect Operating System
OS_TYPE="$(uname)"
echo_info "Detected OS: $OS_TYPE"

# Function to initialize Conda
initialize_conda() {
    local conda_init_script="$1"
    if [ -f "$conda_init_script" ]; then
        source "$conda_init_script"
        echo_success "Conda initialized using $(basename "$conda_init_script")."
        return 0
    else
        return 1
    fi
}

# Load Conda into the current shell session
echo_info "Loading Conda..."

# Determine the Conda base directory
CONDA_BASE=$(conda info --base 2>/dev/null || true)

# Initialize Conda based on OS with fallback for Linux
case "$OS_TYPE" in
    Linux*)
        # Attempt to initialize using conda.sh from the base installation
        if initialize_conda "$CONDA_BASE/etc/profile.d/conda.sh"; then
            :
        else
            # Fallback: Check common installation directories
            COMMON_CONDA_PATHS=(
                "/usr/share/conda/etc/profile.d/conda.sh"
                "/etc/conda/etc/profile.d/conda.sh"
                "/usr/local/etc/profile.d/conda.sh"
            )
            for path in "${COMMON_CONDA_PATHS[@]}"; do
                if initialize_conda "$path"; then
                    break
                fi
            done
            # If still not found, attempt to locate conda.sh using 'whereis'
            if ! conda_initialized; then
                CONDA_LOCATIONS=$(whereis conda | tr ' ' '\n' | grep -v "^conda:$")
                for conda_path in $CONDA_LOCATIONS; do
                    # Assuming conda.sh is relative to the conda executable
                    potential_script="$(dirname "$conda_path")/etc/profile.d/conda.sh"
                    if initialize_conda "$potential_script"; then
                        break
                    fi
                done
            fi
        fi
        ;;
    Darwin*)
        # For macOS systems
        if initialize_conda "$CONDA_BASE/etc/profile.d/conda.sh"; then
            :
        else
            echo_error "Conda initialization script conda.sh not found in $CONDA_BASE/etc/profile.d/."
            echo_error "Please ensure Conda is installed correctly."
            exit 1
        fi
        ;;
    MINGW*|MSYS*)
        # For Windows systems using Git Bash or similar
        if [ -f "$CONDA_BASE/Scripts/activate" ]; then
            source "$CONDA_BASE/Scripts/activate"
            echo_success "Conda initialized using Scripts/activate for Windows."
        elif [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            echo_success "Conda initialized using conda.sh for Windows (Git Bash)."
        else
            echo_error "Conda initialization scripts not found in $CONDA_BASE/Scripts/ or $CONDA_BASE/etc/profile.d/."
            echo_error "Please ensure Conda is installed correctly."
            exit 1
        fi
        ;;
    *)
        echo_error "Unsupported OS type: $OS_TYPE"
        echo_error "This script supports Linux, macOS, and Windows (Git Bash/WSL)."
        exit 1
        ;;
esac

# Verify if Conda was initialized
function conda_initialized() {
    command -v conda >/dev/null 2>&1
}

if ! conda_initialized; then
    echo_error "Conda could not be initialized. Please check your Conda installation."
    exit 1
fi

# Define the environment name
ENV_NAME="perception"

# Check if the environment already exists
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo_info "Environment '${ENV_NAME}' already exists. Updating..."
    conda env update -n "${ENV_NAME}" -f environment.yml --prune
    echo_success "Environment '${ENV_NAME}' updated successfully."
else
    echo_info "Creating environment '${ENV_NAME}'..."
    conda env create -n "${ENV_NAME}" -f environment.yml
    echo_success "Environment '${ENV_NAME}' created successfully."
fi

# Activate the environment
echo_info "Activating environment '${ENV_NAME}'..."
conda activate "${ENV_NAME}"
echo_success "Environment '${ENV_NAME}' activated."

# Install additional Python packages if requirements.txt exists
if [[ -f requirements.txt ]]; then
    echo_info "Installing additional packages from requirements.txt..."
    pip install --no-cache-dir -r requirements.txt
    echo_success "Additional packages installed successfully."
else
    echo_info "No requirements.txt found. Skipping additional package installation."
fi

echo_success "Environment setup complete."
