#!/bin/bash

# Weather ML Project - Main Orchestration Script
# This script creates a Dataproc cluster, runs the ML training job, and cleans up

set -e

# Configuration - Set these environment variables or modify defaults
PROJECT_ID="dsa-project-weather"
REGION=${REGION:-"asia-east1"}
CLUSTER_NAME=${CLUSTER_NAME:-"weather-ml-cluster-$(date +%s)"}
BUCKET_NAME=${BUCKET_NAME:-"weather-ml-data-1760683141"}
OUTPUT_BUCKET=${OUTPUT_BUCKET:-$BUCKET_NAME}

# Cluster configuration
MASTER_MACHINE_TYPE=${MASTER_MACHINE_TYPE:-"n1-standard-2"}
WORKER_MACHINE_TYPE=${WORKER_MACHINE_TYPE:-"n1-standard-8"}
NUM_WORKERS=${NUM_WORKERS:-2}
NUM_SECONDARY_WORKERS=${NUM_SECONDARY_WORKERS:-1}
DISK_SIZE=${DISK_SIZE:-30}

PROPERTIES=spark:spark.executor.instances=4
PROPERTIES=$PROPERTIES,spark:spark.executor.cores=3
PROPERTIES=$PROPERTIES,spark:spark.executor.memory=11700m

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for cluster to be ready
wait_for_cluster() {
    local cluster_name=$1
    local region=$2
    
    print_status "Waiting for cluster to be ready..."
    while true; do
        local status=$(gcloud dataproc clusters describe $cluster_name --region=$region --format="value(status.state)" 2>/dev/null || echo "NOT_FOUND")
        
        case $status in
            "RUNNING")
                print_status "Cluster is ready!"
                return 0
                ;;
            "ERROR"|"CREATING"|"STARTING")
                print_status "Cluster status: $status. Waiting..."
                sleep 30
                ;;
            "NOT_FOUND")
                print_error "Cluster not found. Check if it was created successfully."
                return 1
                ;;
            *)
                print_warning "Unknown cluster status: $status. Waiting..."
                sleep 30
                ;;
        esac
    done
}

# Function to wait for job completion
wait_for_job() {
    local job_id=$1
    local region=$2
    
    print_status "Waiting for job $job_id to complete..."
    while true; do
        local status=$(gcloud dataproc jobs describe $job_id --region=$region --format="value(status.state)" 2>/dev/null || echo "NOT_FOUND")
        
        case $status in
            "DONE")
                print_status "Job completed successfully!"
                return 0
                ;;
            "ERROR"|"CANCELLED")
                print_error "Job failed with status: $status"
                return 1
                ;;
            "PENDING"|"SETUP_DONE"|"RUNNING")
                print_status "Job status: $status. Waiting..."
                sleep 30
                ;;
            "NOT_FOUND")
                print_error "Job not found. Check if it was submitted successfully."
                return 1
                ;;
            *)
                print_warning "Unknown job status: $status. Waiting..."
                sleep 30
                ;;
        esac
    done
}

# Function to check and fix service account permissions
check_and_fix_permissions() {
    print_status "Checking service account permissions..."
    
    PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
    SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    
    print_status "Granting storage permissions to service account: ${SERVICE_ACCOUNT}"
    
    # Grant storage admin role directly (this will work even if already granted)
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="roles/storage.admin" \
        --quiet
    
    print_status "Storage permissions granted successfully!"
}

# Main execution
main() {
    print_status "=== Weather ML Project Execution ==="
    print_status "Project ID: $PROJECT_ID"
    print_status "Region: $REGION"
    print_status "Cluster Name: $CLUSTER_NAME"
    print_status "Bucket: $BUCKET_NAME"
    print_status "Output Bucket: $OUTPUT_BUCKET"
    echo ""
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    if [ "$PROJECT_ID" = "your-project-id" ]; then
        print_error "Please set PROJECT_ID environment variable"
        echo "Usage: PROJECT_ID=your-gcp-project-id ./run_project.sh"
        exit 1
    fi
    
    if ! command_exists gcloud; then
        print_error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    
    if ! command_exists gsutil; then
        print_error "gsutil not found. Please install Google Cloud SDK."
        exit 1
    fi
    
    # Check if data exists in bucket
    print_status "Checking if data exists in bucket..."
    if ! gsutil ls gs://$BUCKET_NAME/data/ >/dev/null 2>&1; then
    #     print_warning "Data not found in gs://$BUCKET_NAME/data/"
    #     print_status "Attempting to upload data automatically..."
        
    #     # Try to run upload script (use simple version to avoid disk space issues)
    #     if [ -f "./upload_data_simple.sh" ]; then
    #         print_status "Running upload_data_simple.sh (avoids local disk space issues)..."
    #         if ./upload_data_simple.sh; then
    #             print_status "Data upload completed successfully"
    #         else
    #             print_error "Data upload failed. Please run ./upload_data_simple.sh manually first."
    #             exit 1
    #         fi
    #     elif [ -f "./upload_data.sh" ]; then
    #         print_status "Running upload_data.sh..."
    #         if ./upload_data.sh; then
    #             print_status "Data upload completed successfully"
    #         else
    #             print_error "Data upload failed. Please run ./upload_data.sh manually first."
    #             exit 1
    #         fi
    #     else
    #         print_error "No upload script found. Please upload data manually."
    #         exit 1
    #     fi
        print_status "Data not found in bucket"
        exit 1
    else
        print_status "Data found in bucket"
    fi
    
    # Set project
    print_status "Setting GCP project..."
    gcloud config set project $PROJECT_ID
    
    # Check and fix service account permissions
    # check_and_fix_permissions
    
    # Create Dataproc cluster
    print_status "Creating Dataproc cluster: $CLUSTER_NAME"
    
    gcloud dataproc clusters create $CLUSTER_NAME \
        --region $REGION \
        --image-version 2.2-debian12 \
        --zone "" \
        --public-ip-address \
        --enable-component-gateway \
        --delete-max-idle 30m \
        --master-machine-type $MASTER_MACHINE_TYPE \
        --master-boot-disk-type pd-balanced \
        --master-boot-disk-size $DISK_SIZE \
        --num-workers $NUM_WORKERS \
        --worker-machine-type $WORKER_MACHINE_TYPE \
        --worker-boot-disk-type pd-balanced \
        --worker-boot-disk-size $DISK_SIZE \
        --num-secondary-workers $NUM_SECONDARY_WORKERS \
        --secondary-worker-boot-disk-size $DISK_SIZE \
        --secondary-worker-type preemptible\
        --no-shielded-secure-boot \
        --optional-components JUPYTER \
        --metadata "enable-oslogin=TRUE" \
        --tags "weather-ml" \
        --properties $PROPERTIES \
        --scopes "https://www.googleapis.com/auth/cloud-platform" 
    
    # Wait for cluster to be ready
    if ! wait_for_cluster $CLUSTER_NAME $REGION; then
        print_error "Failed to create or start cluster"
        exit 1
    fi
    
    # Submit job asynchronously and wait for completion
    print_status "Submitting PySpark job..."
    echo "=========================================="
    
    # Run job asynchronously
    print_status "Submitting ML training job asynchronously..."
    JOB_OUTPUT=$(gcloud dataproc jobs submit pyspark \
        --cluster=$CLUSTER_NAME \
        --region=$REGION \
        --async \
        --py-files=data_loader.py,feature_pipeline.py,model_trainer.py \
        train_model.py \
        -- \
        --data-path=gs://$BUCKET_NAME/data/ \
        --output-path=gs://$OUTPUT_BUCKET/results/ \
        --train-ratio=0.7 2>&1)
    
    # Extract job ID from output
    JOB_ID=$(echo "$JOB_OUTPUT" | sed -n 's/.*Job \[\(.*\)\].*/\1/p')
    
    if [ -z "$JOB_ID" ]; then
        print_error "Failed to get job ID from submission"
        echo "$JOB_OUTPUT"
        exit 1
    fi
    
    print_status "Job submitted successfully with ID: $JOB_ID"
    # print_status "Waiting for job to complete..."
    
    # Wait for job to complete
    if gcloud dataproc jobs wait $JOB_ID --region=$REGION; then
        print_status "ML training job completed successfully!"
    else
        print_error "ML training job failed"
        print_error "Check cluster logs for details"
        gcloud dataproc clusters delete $CLUSTER_NAME --region=$REGION --quiet || true
        exit 1
    fi
    
    # echo "=========================================="
    
    # # Download results
    # print_status "Downloading results..."
    # mkdir -p results
    # gsutil -m cp -r gs://$OUTPUT_BUCKET/results/* results/ 2>/dev/null || print_warning "No results found to download"
    
    # # Display results summary
    # print_status "=== Results Summary ==="
    # if [ -f "results/metrics/part-00000" ]; then
    #     echo ""
    #     cat results/metrics/part-00000
    #     echo ""
    # else
    #     print_warning "Metrics file not found. Check job logs for details."
    # fi
    
    # # Clean up cluster
    # print_status "Cleaning up cluster..."
    # gcloud dataproc clusters delete $CLUSTER_NAME --region=$REGION --quiet
    
    # print_status "=== Project Execution Complete ==="
    # print_status "Results saved to: results/"
    # print_status "Models saved to: gs://$OUTPUT_BUCKET/results/models/"
    # print_status "Metrics saved to: gs://$OUTPUT_BUCKET/results/metrics/"
}

# # Trap to ensure cleanup on script exit
# cleanup() {
#     if [ ! -z "$CLUSTER_NAME" ]; then
#         print_warning "Cleaning up cluster due to script interruption..."
#         gcloud dataproc clusters delete $CLUSTER_NAME --region=$REGION --quiet 2>/dev/null || true
#     fi
# }

# trap cleanup EXIT INT TERM

# Run main function
main "$@"
