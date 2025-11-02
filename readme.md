# Weather ML Project

This project trains machine learning models on weather data using PySpark on Google Cloud Dataproc.

## Quick Start

1. **Prerequisites**

   - Google Cloud SDK (`gcloud`, `gsutil`) installed and authenticated
   - Weather data uploaded to `gs://BUCKET_NAME/data/`
   - Project ID configured in `run_project.sh` (line 10)

2. **Run the pipeline**

   ```bash
   ./run_project.sh
   ```

3. **What it does**
   - Creates a Dataproc cluster (2 workers + 1 preemptible secondary worker)
   - Submits the ML training job (`train_model.py`)
   - Waits for job completion
   - Outputs saved to `gs://BUCKET_NAME/results/`

## Configuration

Edit `run_project.sh` to customize:

- `PROJECT_ID`: Your GCP project ID
- `BUCKET_NAME`: GCS bucket with your data
- `WORKER_MACHINE_TYPE`: Worker VM type (default: `n1-highmem-8`)
- `NUM_WORKERS`: Number of worker nodes (default: 2)
- Spark executor settings (lines 23-25)

The cluster auto-deletes after 5 minutes of idle time.
