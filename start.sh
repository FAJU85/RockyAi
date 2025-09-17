#!/bin/bash

# Rocky AI Startup Script
# This script starts the enhanced Rocky AI system with all services

set -e

echo "🚀 Starting Rocky AI Enhanced System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "⚠️  Please update .env file with your configuration before running again."
    echo "   Especially change the database password and secret key!"
    exit 1
fi

# Pull latest images
echo "📥 Pulling latest Docker images..."
docker-compose pull

# Build services
echo "🔨 Building services..."
docker-compose build

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check API health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is healthy"
else
    echo "❌ API is not responding"
fi

# Check UI health
if curl -f http://localhost:5173 > /dev/null 2>&1; then
    echo "✅ UI is healthy"
else
    echo "❌ UI is not responding"
fi

# Check DMR health
if curl -f http://localhost:11434/health > /dev/null 2>&1; then
    echo "✅ DMR is healthy"
else
    echo "❌ DMR is not responding"
fi

# Check database health
if docker-compose exec -T postgres pg_isready -U rocky -d rocky_ai > /dev/null 2>&1; then
    echo "✅ Database is healthy"
else
    echo "❌ Database is not responding"
fi

# Check Redis health
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis is not responding"
fi

echo ""
echo "🎉 Rocky AI Enhanced System is starting up!"
echo ""
echo "📊 Services:"
echo "   • API: http://localhost:8000"
echo "   • UI: http://localhost:5173"
echo "   • DMR: http://localhost:11434"
echo "   • Database: localhost:5432"
echo "   • Redis: localhost:6379"
echo ""
echo "📚 Documentation:"
echo "   • API Docs: http://localhost:8000/docs"
echo "   • Health Check: http://localhost:8000/health"
echo ""
echo "🔧 Management:"
echo "   • View logs: docker-compose logs -f"
echo "   • Stop services: docker-compose down"
echo "   • Restart services: docker-compose restart"
echo ""
echo "✨ Ready to analyze your data with AI!"
