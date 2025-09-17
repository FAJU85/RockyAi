#!/bin/bash

# Rocky AI Deployment Script
# This script handles deployment to different environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="staging"
NAMESPACE="rocky-ai"
RELEASE_NAME="rocky-ai"
CHART_PATH="./helm/rocky-ai"
VALUES_FILE=""
DRY_RUN=false
FORCE=false
WAIT=false
TIMEOUT="10m"

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
    -e, --environment ENV    Environment to deploy to (staging, production) [default: staging]
    -n, --namespace NS       Kubernetes namespace [default: rocky-ai]
    -r, --release RELEASE    Helm release name [default: rocky-ai]
    -c, --chart PATH         Path to Helm chart [default: ./helm/rocky-ai]
    -f, --values FILE        Values file to use
    -d, --dry-run           Perform a dry run
    -w, --wait              Wait for deployment to complete
    -t, --timeout DURATION  Timeout for deployment [default: 10m]
    --force                 Force deployment even if resources exist
    -h, --help              Show this help message

Examples:
    $0 --environment staging --dry-run
    $0 --environment production --values values-production.yaml --wait
    $0 --environment staging --force --timeout 15m
EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        print_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if helm is initialized
    if ! helm list &> /dev/null; then
        print_error "Helm is not initialized. Run 'helm init' first"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to validate environment
validate_environment() {
    print_status "Validating environment: $ENVIRONMENT"
    
    case $ENVIRONMENT in
        staging)
            VALUES_FILE=${VALUES_FILE:-"values-staging.yaml"}
            ;;
        production)
            VALUES_FILE=${VALUES_FILE:-"values-production.yaml"}
            ;;
        *)
            print_error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
            exit 1
            ;;
    esac
    
    # Check if values file exists
    if [ -n "$VALUES_FILE" ] && [ ! -f "$CHART_PATH/$VALUES_FILE" ]; then
        print_warning "Values file $CHART_PATH/$VALUES_FILE not found, using default values"
        VALUES_FILE=""
    fi
    
    print_success "Environment validation passed"
}

# Function to create namespace if it doesn't exist
create_namespace() {
    print_status "Checking namespace: $NAMESPACE"
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_status "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
        print_success "Namespace created: $NAMESPACE"
    else
        print_success "Namespace already exists: $NAMESPACE"
    fi
}

# Function to check if release exists
check_release() {
    print_status "Checking if release exists: $RELEASE_NAME"
    
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        print_warning "Release $RELEASE_NAME already exists in namespace $NAMESPACE"
        if [ "$FORCE" = false ]; then
            print_error "Use --force to upgrade existing release"
            exit 1
        fi
    else
        print_success "Release $RELEASE_NAME does not exist, will install"
    fi
}

# Function to deploy with Helm
deploy_with_helm() {
    print_status "Deploying with Helm..."
    
    # Build helm command
    HELM_CMD="helm"
    
    if [ "$DRY_RUN" = true ]; then
        HELM_CMD="$HELM_CMD --dry-run"
        print_warning "Running in dry-run mode"
    fi
    
    if [ -n "$VALUES_FILE" ]; then
        HELM_CMD="$HELM_CMD -f $CHART_PATH/$VALUES_FILE"
    fi
    
    if [ "$WAIT" = true ]; then
        HELM_CMD="$HELM_CMD --wait"
    fi
    
    HELM_CMD="$HELM_CMD --timeout $TIMEOUT"
    HELM_CMD="$HELM_CMD --namespace $NAMESPACE"
    HELM_CMD="$HELM_CMD $RELEASE_NAME $CHART_PATH"
    
    print_status "Executing: $HELM_CMD"
    
    # Execute helm command
    if eval "$HELM_CMD"; then
        print_success "Helm deployment completed successfully"
    else
        print_error "Helm deployment failed"
        exit 1
    fi
}

# Function to verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Wait for pods to be ready
    print_status "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=rocky-ai -n "$NAMESPACE" --timeout="$TIMEOUT" || {
        print_error "Pods did not become ready within timeout"
        exit 1
    }
    
    # Check pod status
    print_status "Checking pod status..."
    kubectl get pods -n "$NAMESPACE" -l app=rocky-ai
    
    # Check service status
    print_status "Checking service status..."
    kubectl get services -n "$NAMESPACE" -l app=rocky-ai
    
    # Check ingress status
    print_status "Checking ingress status..."
    kubectl get ingress -n "$NAMESPACE" -l app=rocky-ai
    
    print_success "Deployment verification completed"
}

# Function to show deployment status
show_status() {
    print_status "Deployment status:"
    echo ""
    
    print_status "Pods:"
    kubectl get pods -n "$NAMESPACE" -l app=rocky-ai
    echo ""
    
    print_status "Services:"
    kubectl get services -n "$NAMESPACE" -l app=rocky-ai
    echo ""
    
    print_status "Ingress:"
    kubectl get ingress -n "$NAMESPACE" -l app=rocky-ai
    echo ""
    
    print_status "Helm release:"
    helm list -n "$NAMESPACE" | grep "$RELEASE_NAME" || echo "No release found"
}

# Function to cleanup on failure
cleanup() {
    print_error "Deployment failed, cleaning up..."
    
    # Delete failed release
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        print_status "Deleting failed release..."
        helm uninstall "$RELEASE_NAME" -n "$NAMESPACE" || true
    fi
    
    print_warning "Cleanup completed"
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--release)
                RELEASE_NAME="$2"
                shift 2
                ;;
            -c|--chart)
                CHART_PATH="$2"
                shift 2
                ;;
            -f|--values)
                VALUES_FILE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -w|--wait)
                WAIT=true
                shift
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --force)
                FORCE=true
                shift
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
    
    # Set trap for cleanup on failure
    trap cleanup EXIT
    
    print_status "Starting Rocky AI deployment..."
    print_status "Environment: $ENVIRONMENT"
    print_status "Namespace: $NAMESPACE"
    print_status "Release: $RELEASE_NAME"
    print_status "Chart: $CHART_PATH"
    print_status "Values: ${VALUES_FILE:-"default"}"
    print_status "Dry run: $DRY_RUN"
    print_status "Wait: $WAIT"
    print_status "Timeout: $TIMEOUT"
    print_status "Force: $FORCE"
    echo ""
    
    # Execute deployment steps
    check_prerequisites
    validate_environment
    create_namespace
    check_release
    deploy_with_helm
    
    if [ "$DRY_RUN" = false ]; then
        verify_deployment
        show_status
        print_success "Rocky AI deployment completed successfully!"
    else
        print_success "Dry run completed successfully!"
    fi
    
    # Clear trap
    trap - EXIT
}

# Run main function
main "$@"
