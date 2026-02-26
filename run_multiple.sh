#!/bin/bash
# Bash script to run a Python file multiple times

# ===== CONFIGURATION =====
PYTHON_FILE="logic.py"   # Path to your Python file
RUNS=15                    # Number of times to run the script
PYTHON_CMD="python3"      # Python command (python or python3)

# ===== VALIDATION =====
if [ ! -f "$PYTHON_FILE" ]; then
    echo "Error: Python file '$PYTHON_FILE' not found."
    exit 1
fi

if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "Error: '$PYTHON_CMD' is not installed or not in PATH."
    exit 1
fi

# ===== EXECUTION =====
for ((i=1; i<=RUNS; i++)); do
    echo "Run #$i..."
    if ! $PYTHON_CMD "$PYTHON_FILE"; then
        echo "Error: Python script failed on run #$i"
        exit 1
    fi
done

echo "✅ Completed $RUNS runs successfully."

