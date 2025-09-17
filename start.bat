@echo off
REM Rocky AI Startup Script for Windows
REM This script starts the enhanced Rocky AI system with all services

echo 🚀 Starting Rocky AI Enhanced System...

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose and try again.
    pause
    exit /b 1
)

REM Create data directory if it doesn't exist
if not exist data mkdir data

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy env.example .env
    echo ⚠️  Please update .env file with your configuration before running again.
    echo    Especially change the database password and secret key!
    pause
    exit /b 1
)

REM Pull latest images
echo 📥 Pulling latest Docker images...
docker-compose pull

REM Build services
echo 🔨 Building services...
docker-compose build

REM Start services
echo 🚀 Starting services...
docker-compose up -d

REM Wait for services to be healthy
echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check service health
echo 🔍 Checking service health...

REM Check API health
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ API is healthy
) else (
    echo ❌ API is not responding
)

REM Check UI health
curl -f http://localhost:5173 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ UI is healthy
) else (
    echo ❌ UI is not responding
)

REM Check DMR health
curl -f http://localhost:11434/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ DMR is healthy
) else (
    echo ❌ DMR is not responding
)

REM Check database health
docker-compose exec -T postgres pg_isready -U rocky -d rocky_ai >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Database is healthy
) else (
    echo ❌ Database is not responding
)

REM Check Redis health
docker-compose exec -T redis redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Redis is healthy
) else (
    echo ❌ Redis is not responding
)

echo.
echo 🎉 Rocky AI Enhanced System is starting up!
echo.
echo 📊 Services:
echo    • API: http://localhost:8000
echo    • UI: http://localhost:5173
echo    • DMR: http://localhost:11434
echo    • Database: localhost:5432
echo    • Redis: localhost:6379
echo.
echo 📚 Documentation:
echo    • API Docs: http://localhost:8000/docs
echo    • Health Check: http://localhost:8000/health
echo.
echo 🔧 Management:
echo    • View logs: docker-compose logs -f
echo    • Stop services: docker-compose down
echo    • Restart services: docker-compose restart
echo.
echo ✨ Ready to analyze your data with AI!
pause
