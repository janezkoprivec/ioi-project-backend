FROM python:3.11-slim

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Expose the API port
EXPOSE 8000

# Use entrypoint script to download data then start server
ENTRYPOINT ["/app/entrypoint.sh"]

