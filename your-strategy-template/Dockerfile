FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy base bot framework
COPY ../base-bot-template ./base-bot-template

# Copy strategy folder
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command — executes startup.py
CMD ["python", "startup.py"]