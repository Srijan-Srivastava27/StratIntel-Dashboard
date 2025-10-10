#!/bin/bash

echo "Deploying Strategic Intelligence Platform..."

# Create necessary directories
mkdir -p data/raw data/processed data/models logs

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

if [ $? -eq 0 ]; then
    echo "Tests passed successfully"
else
    echo "Tests failed. Deployment aborted."
    exit 1
fi

# Setup logging
touch logs/strategic_intelligence.log

# Check environment variables
python -c "
import os
from dotenv import load_dotenv

load_dotenv()

required_vars = [
    'OPENAI_API_KEY',
    'TWITTER_BEARER_TOKEN',
    'SLACK_WEBHOOK_URL'
]

missing_vars = []
for var in required_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print(f'Missing environment variables: {missing_vars}')
    exit(1)
else:
    print('All environment variables are set')
"

if [ $? -eq 0 ]; then
    echo "Environment check passed"
else
    echo "Environment check failed. Please set required environment variables."
    exit 1
fi

echo "Deployment completed successfully!"
echo "Run the platform with: python src/main.py"
