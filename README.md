# Quantum ML Classifier

A monorepo for quantum-classical ML research and production operations.

This repository now has two connected tracks:
- Research and experimentation: modular Python code for quantum encodings, variational circuits, hybrid models, noise analysis, and benchmark experiments.
- Production app stack: a Next.js control center that submits jobs to a FastAPI worker, with auth, CSV upload, queue integration, and storage fallbacks.

## Current Project State (April 2026)

- Research pipeline is operational end-to-end via [main.py](main.py).
- Latest tracked Iris benchmark is documented in [output/pipeline_output.txt](output/pipeline_output.txt):
  - Quantum-enhanced SVM test accuracy: 1.0000
  - Classical SVM (RBF) test accuracy: 0.9667
- Python unit suite results are documented in [output/test_results.txt](output/test_results.txt) (40 passing tests in the latest recorded run).
- Frontend app is production-oriented with protected APIs, signed-cookie auth, optional Google OAuth, CSV upload, and job lifecycle UI.
- Worker service supports training, inference, and experiment jobs, including quantum-enhanced training with classical fallback.
- CI quality gates are defined in [ci_cd/.github/workflows/quality-gates.yml](ci_cd/.github/workflows/quality-gates.yml) for frontend lint/tests/build/E2E and worker checks.

## Monorepo Components

| Component | Path | Purpose |
|---|---|---|
| Research core | [src/](src/) | Quantum circuits, encodings, kernels, hybrid training, analysis, visualization |
| Experiment framework | [experiments/](experiments/) + [configs/experiments/](configs/experiments/) | Reproducible YAML-driven experiment suites |
| Demo entrypoint | [main.py](main.py) | End-to-end quantum feature extraction + classifier benchmark |
| Frontend control center | [frontend/](frontend/) | Next.js 16 app for login, job creation, queue monitoring, health checks |
| Worker API | [worker/](worker/) | FastAPI worker executing training/inference/experiment jobs |
| Lightweight deploy worker | [worker_service_deploy/](worker_service_deploy/) | Minimal FastAPI variant for constrained deployment targets |
| Deployment docs | [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md), [VERCEL_ENV.md](VERCEL_ENV.md) | Vercel + worker deployment checklist and env templates |

## Repository Layout (Key Paths)

```text
Quantum ML Classifier/
|- main.py
|- src/
|- experiments/
|- configs/
|- tests/
|- output/
|- frontend/
|  |- src/app/
|  |- __tests__/
|  |- e2e/
|- worker/
|  |- app/
|- worker_service_deploy/
|- ci_cd/.github/workflows/quality-gates.yml
```

## Quick Start

### 1. Research Demo (Python)

1. Create and activate a virtual environment.
2. Install dependencies.
3. Run the demo pipeline.

PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The demo executes the full pipeline from dataset loading through quantum feature extraction and baseline comparison.

Notes:
- Project metadata targets Python >=3.9,<3.12 (see [pyproject.toml](pyproject.toml)).
- Full TensorFlow Quantum environments are typically easiest on Linux/WSL2.

### 2. Full Local App (Frontend + Worker)

Run worker and frontend in separate terminals.

Worker terminal (from repository root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r worker/requirements.txt
uvicorn worker.app.main:app --reload --port 8000
```

Frontend terminal:

```powershell
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

Open:
- Frontend: http://localhost:3000
- Frontend health endpoint: http://localhost:3000/api/health
- Worker health endpoint: http://localhost:8000/health

Minimum local env values from [frontend/.env.example](frontend/.env.example):
- AUTH_SECRET
- AUTH_USER_EMAIL
- AUTH_USER_PASSWORD
- WORKER_WEBHOOK_SECRET
- WORKER_BASE_URL (usually http://localhost:8000 for local dev)

## API Surface (Current)

Frontend API routes under [frontend/src/app/api/](frontend/src/app/api/):
- GET /api/health
- GET /api/datasets
- GET /api/jobs
- POST /api/jobs
- GET /api/jobs/:id
- POST /api/uploads
- GET /api/uploads/:id?token=...
- Auth routes under /api/auth/*
- Internal dispatch route: POST /api/internal/jobs/dispatch

Worker routes in [worker/app/main.py](worker/app/main.py):
- GET /health
- POST /jobs/run
- WS /ws/jobs

Supported job types:
- training
- inference
- experiment

Supported dataset IDs:
- iris
- breast_cancer
- wine
- mnist_binary
- csv_upload

## Experiments

Experiment configs live in [configs/experiments/](configs/experiments/).

Run all configured experiments:

```powershell
python -m experiments.run_all_experiments
```

## Testing and Quality Gates

Python test suite:

```powershell
python -m pytest tests -v
```

Worker pipeline tests:

```powershell
python -m pytest tests/unit/test_worker_pipeline.py -v
```

Frontend checks:

```powershell
cd frontend
npm run lint
npm test
npm run e2e
npm run build
```

CI automation is configured in [ci_cd/.github/workflows/quality-gates.yml](ci_cd/.github/workflows/quality-gates.yml).

## Deployment Notes

- Vercel root directory must be set to [frontend/](frontend/) for the Next.js app.
- Set auth, worker, and storage env vars before production traffic.
- Deploy worker separately (for example Railway/Render/Docker host) and set WORKER_BASE_URL accordingly.
- Full checklist: [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)
- Env reference: [VERCEL_ENV.md](VERCEL_ENV.md)

## Additional Docs

- Frontend details: [frontend/README.md](frontend/README.md)
- Worker details: [worker/README.md](worker/README.md)
- Implementation checkpoint: [checkpoint(current progress).md](checkpoint(current progress).md)
- System plan: [Quantum-Classical Hybrid ML System Implementation Plan.md](Quantum-Classical%20Hybrid%20ML%20System%20Implementation%20Plan.md)

## License

MIT License. See [LICENSE](LICENSE).
