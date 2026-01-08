#!/bin/bash
# Simple test script to verify Docker setup is working

echo "======================================"
echo "Testing Docker Setup"
echo "======================================"
echo ""

# Check if Docker is running
if ! docker ps &> /dev/null; then
    echo "✗ Docker is not running. Please start Docker first."
    exit 1
fi
echo "✓ Docker is running"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "✗ .env file not found. Please create it with your Copernicus Marine credentials."
    echo "  Example:"
    echo "    COPERNICUSMARINE_USERNAME=your_username"
    echo "    COPERNICUSMARINE_PASSWORD=your_password"
    exit 1
fi
echo "✓ .env file found"

# Check if API is responding
echo ""
echo "Checking if API is running..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ API is running at http://localhost:8000"
    echo ""
    echo "Health check response:"
    curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8000/health
else
    echo "✗ API is not responding at http://localhost:8000"
    echo "  Run: docker-compose up --build"
    exit 1
fi

echo ""
echo "======================================"
echo "✓ All checks passed!"
echo "======================================"
echo ""
echo "Try these commands:"
echo "  - Health check: curl http://localhost:8000/health"
echo "  - API docs: http://localhost:8000/docs"
echo "  - Example query:"
echo "    curl 'http://localhost:8000/subset?dataset=reanalysis&variable=thetao&min_lon=-10&max_lon=-5&min_lat=30&max_lat=35&time=2011-07-01&depth=0&stride=10&fmt=json'"
echo ""
echo "See EXAMPLES.md for more usage examples."

