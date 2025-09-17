#!/bin/bash

# Rocky AI Test Script
# This script runs the complete test suite

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
PARALLEL=false
BENCHMARK=false
TIMEOUT="300"
MAX_FAILURES="5"
REPORT_FORMAT="html"
OUTPUT_DIR="./test-results"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -t, --type TYPE         Test type (unit, integration, e2e, performance, all) [default: all]
    -c, --coverage          Generate coverage report
    -v, --verbose           Verbose output
    -p, --parallel          Run tests in parallel
    -b, --benchmark         Run benchmark tests
    --timeout SECONDS       Test timeout in seconds [default: 300]
    --max-failures NUM      Maximum number of failures [default: 5]
    --report-format FORMAT  Report format (html, xml, json) [default: html]
    -o, --output DIR        Output directory [default: ./test-results]
    -h, --help              Show this help message

Examples:
    $0 --type unit --coverage
    $0 --type integration --verbose --parallel
    $0 --type e2e --timeout 600
    $0 --type performance --benchmark
    $0 --type all --coverage --report-format html
EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed or not in PATH"
        exit 1
    fi
    
    # Check if pip is installed
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed or not in PATH"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to setup test environment
setup_test_environment() {
    print_status "Setting up test environment..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Install Python dependencies
    print_status "Installing Python dependencies..."
    pip3 install -r apps/api/requirements.txt
    pip3 install -r apps/api/requirements-test.txt
    
    # Install Node.js dependencies
    print_status "Installing Node.js dependencies..."
    cd apps/ui
    npm ci
    cd ../..
    
    print_success "Test environment setup completed"
}

# Function to run unit tests
run_unit_tests() {
    print_status "Running unit tests..."
    
    cd apps/api
    
    # Build pytest command
    PYTEST_CMD="pytest tests/unit/ -v"
    
    if [ "$COVERAGE" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --cov=apps --cov-report=$REPORT_FORMAT --cov-report=term-missing"
    fi
    
    if [ "$VERBOSE" = true ]; then
        PYTEST_CMD="$PYTEST_CMD -s"
    fi
    
    if [ "$PARALLEL" = true ]; then
        PYTEST_CMD="$PYTEST_CMD -n auto"
    fi
    
    if [ "$BENCHMARK" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --benchmark-only"
    else
        PYTEST_CMD="$PYTEST_CMD --benchmark-skip"
    fi
    
    PYTEST_CMD="$PYTEST_CMD --timeout=$TIMEOUT"
    PYTEST_CMD="$PYTEST_CMD --maxfail=$MAX_FAILURES"
    PYTEST_CMD="$PYTEST_CMD --junitxml=$OUTPUT_DIR/unit-test-results.xml"
    
    # Run tests
    if eval "$PYTEST_CMD"; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed"
        return 1
    fi
    
    cd ../..
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    
    cd apps/api
    
    # Build pytest command
    PYTEST_CMD="pytest tests/integration/ -v"
    
    if [ "$COVERAGE" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --cov=apps --cov-report=$REPORT_FORMAT --cov-report=term-missing"
    fi
    
    if [ "$VERBOSE" = true ]; then
        PYTEST_CMD="$PYTEST_CMD -s"
    fi
    
    if [ "$PARALLEL" = true ]; then
        PYTEST_CMD="$PYTEST_CMD -n auto"
    fi
    
    if [ "$BENCHMARK" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --benchmark-only"
    else
        PYTEST_CMD="$PYTEST_CMD --benchmark-skip"
    fi
    
    PYTEST_CMD="$PYTEST_CMD --timeout=$TIMEOUT"
    PYTEST_CMD="$PYTEST_CMD --maxfail=$MAX_FAILURES"
    PYTEST_CMD="$PYTEST_CMD --junitxml=$OUTPUT_DIR/integration-test-results.xml"
    
    # Run tests
    if eval "$PYTEST_CMD"; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
        return 1
    fi
    
    cd ../..
}

# Function to run end-to-end tests
run_e2e_tests() {
    print_status "Running end-to-end tests..."
    
    cd apps/api
    
    # Build pytest command
    PYTEST_CMD="pytest tests/e2e/ -v"
    
    if [ "$COVERAGE" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --cov=apps --cov-report=$REPORT_FORMAT --cov-report=term-missing"
    fi
    
    if [ "$VERBOSE" = true ]; then
        PYTEST_CMD="$PYTEST_CMD -s"
    fi
    
    if [ "$PARALLEL" = true ]; then
        PYTEST_CMD="$PYTEST_CMD -n auto"
    fi
    
    if [ "$BENCHMARK" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --benchmark-only"
    else
        PYTEST_CMD="$PYTEST_CMD --benchmark-skip"
    fi
    
    PYTEST_CMD="$PYTEST_CMD --timeout=$TIMEOUT"
    PYTEST_CMD="$PYTEST_CMD --maxfail=$MAX_FAILURES"
    PYTEST_CMD="$PYTEST_CMD --junitxml=$OUTPUT_DIR/e2e-test-results.xml"
    
    # Run tests
    if eval "$PYTEST_CMD"; then
        print_success "End-to-end tests passed"
    else
        print_error "End-to-end tests failed"
        return 1
    fi
    
    cd ../..
}

# Function to run performance tests
run_performance_tests() {
    print_status "Running performance tests..."
    
    cd apps/api
    
    # Build pytest command
    PYTEST_CMD="pytest tests/performance/ -v"
    
    if [ "$COVERAGE" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --cov=apps --cov-report=$REPORT_FORMAT --cov-report=term-missing"
    fi
    
    if [ "$VERBOSE" = true ]; then
        PYTEST_CMD="$PYTEST_CMD -s"
    fi
    
    if [ "$PARALLEL" = true ]; then
        PYTEST_CMD="$PYTEST_CMD -n auto"
    fi
    
    if [ "$BENCHMARK" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --benchmark-only"
    else
        PYTEST_CMD="$PYTEST_CMD --benchmark-skip"
    fi
    
    PYTEST_CMD="$PYTEST_CMD --timeout=$TIMEOUT"
    PYTEST_CMD="$PYTEST_CMD --maxfail=$MAX_FAILURES"
    PYTEST_CMD="$PYTEST_CMD --junitxml=$OUTPUT_DIR/performance-test-results.xml"
    
    # Run tests
    if eval "$PYTEST_CMD"; then
        print_success "Performance tests passed"
    else
        print_error "Performance tests failed"
        return 1
    fi
    
    cd ../..
}

# Function to run Node.js tests
run_node_tests() {
    print_status "Running Node.js tests..."
    
    cd apps/ui
    
    # Build npm test command
    NPM_CMD="npm test"
    
    if [ "$VERBOSE" = true ]; then
        NPM_CMD="$NPM_CMD -- --verbose"
    fi
    
    if [ "$COVERAGE" = true ]; then
        NPM_CMD="$NPM_CMD -- --coverage"
    fi
    
    NPM_CMD="$NPM_CMD -- --watchAll=false"
    
    # Run tests
    if eval "$NPM_CMD"; then
        print_success "Node.js tests passed"
    else
        print_error "Node.js tests failed"
        return 1
    fi
    
    cd ../..
}

# Function to generate test report
generate_test_report() {
    print_status "Generating test report..."
    
    # Create test report summary
    cat > "$OUTPUT_DIR/test-summary.md" << EOF
# Rocky AI Test Report

## Test Summary
- Test Type: $TEST_TYPE
- Coverage: $COVERAGE
- Parallel: $PARALLEL
- Benchmark: $BENCHMARK
- Timeout: $TIMEOUT seconds
- Max Failures: $MAX_FAILURES

## Test Results
- Unit Tests: $(if [ -f "$OUTPUT_DIR/unit-test-results.xml" ]; then echo "✅ Passed"; else echo "❌ Failed"; fi)
- Integration Tests: $(if [ -f "$OUTPUT_DIR/integration-test-results.xml" ]; then echo "✅ Passed"; else echo "❌ Failed"; fi)
- End-to-End Tests: $(if [ -f "$OUTPUT_DIR/e2e-test-results.xml" ]; then echo "✅ Passed"; else echo "❌ Failed"; fi)
- Performance Tests: $(if [ -f "$OUTPUT_DIR/performance-test-results.xml" ]; then echo "✅ Passed"; else echo "❌ Failed"; fi)
- Node.js Tests: $(if [ -f "$OUTPUT_DIR/node-test-results.xml" ]; then echo "✅ Passed"; else echo "❌ Failed"; fi)

## Coverage Report
$(if [ "$COVERAGE" = true ] && [ -f "$OUTPUT_DIR/coverage.xml" ]; then echo "Coverage report generated: $OUTPUT_DIR/coverage.xml"; else echo "No coverage report generated"; fi)

## Test Artifacts
- Test Results: $OUTPUT_DIR/
- Coverage Report: $OUTPUT_DIR/coverage.xml
- Test Summary: $OUTPUT_DIR/test-summary.md

Generated on: $(date)
EOF
    
    print_success "Test report generated: $OUTPUT_DIR/test-summary.md"
}

# Function to cleanup test environment
cleanup_test_environment() {
    print_status "Cleaning up test environment..."
    
    # Remove temporary files
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "htmlcov" -type d -exec rm -rf {} + 2>/dev/null || true
    
    print_success "Test environment cleanup completed"
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                TEST_TYPE="$2"
                shift 2
                ;;
            -c|--coverage)
                COVERAGE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -p|--parallel)
                PARALLEL=true
                shift
                ;;
            -b|--benchmark)
                BENCHMARK=true
                shift
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --max-failures)
                MAX_FAILURES="$2"
                shift 2
                ;;
            --report-format)
                REPORT_FORMAT="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    print_status "Starting Rocky AI test suite..."
    print_status "Test Type: $TEST_TYPE"
    print_status "Coverage: $COVERAGE"
    print_status "Verbose: $VERBOSE"
    print_status "Parallel: $PARALLEL"
    print_status "Benchmark: $BENCHMARK"
    print_status "Timeout: $TIMEOUT seconds"
    print_status "Max Failures: $MAX_FAILURES"
    print_status "Report Format: $REPORT_FORMAT"
    print_status "Output Directory: $OUTPUT_DIR"
    echo ""
    
    # Execute test steps
    check_prerequisites
    setup_test_environment
    
    # Run tests based on type
    case $TEST_TYPE in
        unit)
            run_unit_tests
            ;;
        integration)
            run_integration_tests
            ;;
        e2e)
            run_e2e_tests
            ;;
        performance)
            run_performance_tests
            ;;
        all)
            run_unit_tests
            run_integration_tests
            run_e2e_tests
            run_performance_tests
            run_node_tests
            ;;
        *)
            print_error "Invalid test type: $TEST_TYPE"
            show_usage
            exit 1
            ;;
    esac
    
    generate_test_report
    cleanup_test_environment
    
    print_success "Rocky AI test suite completed successfully!"
}

# Run main function
main "$@"
