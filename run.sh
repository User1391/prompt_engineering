#!/bin/bash

# Load API key from secure file (should be chmod 600)
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found. Please create it with your OpenAI API key"
    echo "Example: echo 'export OPENAI_API_KEY=your-key-here' > .env"
    exit 1
fi

# Run the watermarking process
python3 runprompt.py > watermarked_output.txt

echo "=== Detector Output ==="
python3 detect_watermark.py < watermarked_output.txt

