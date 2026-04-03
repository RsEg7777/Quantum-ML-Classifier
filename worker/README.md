# Quantum ML Worker

Python worker service for experiment, training, and inference jobs.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

   pip install -r worker/requirements.txt

3. Start the worker from repository root:

   uvicorn worker.app.main:app --reload --port 8000

4. Health check:

   GET http://localhost:8000/health

5. Submit a job:

   POST http://localhost:8000/jobs/run

## Current behavior

- Uses project dataset loaders from src/data/loaders/classification_datasets.py
- Runs quantum-enhanced SVM search for training jobs (multi-restart VQC features + tuned SVM)
- Includes automatic fallback to classical RBF baseline when quantum candidate underperforms
- Runs classical SVM baseline execution for inference jobs
- Runs feature-grid sweep for experiment jobs
- Returns structured accuracy, loss, notes, and details payload
