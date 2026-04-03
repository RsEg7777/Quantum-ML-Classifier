# Vercel Environment Variables

Copy and paste these into Vercel Project Settings → Environment Variables.
Paste each line into a new env var entry.

## REQUIRED AUTH VARIABLES

AUTH_SECRET=your-secret-key-at-least-16-characters-long
AUTH_USER_EMAIL=admin@example.com
AUTH_USER_PASSWORD=your-secure-password-here

## REQUIRED GOOGLE OAUTH VARIABLES

GOOGLE_CLIENT_ID=31852239069-ljdhp5aec5tui66ibolrsturmfkj3553.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=https://quantum-ml-classifier.vercel.app/api/auth/google/callback
GOOGLE_AUTH_ALLOWED_EMAILS=*

## REQUIRED WORKER INTEGRATION VARIABLES

WORKER_WEBHOOK_SECRET=your-secure-webhook-secret
WORKER_BASE_URL=https://your-deployed-worker-url.com

## OPTIONAL - STORAGE (Pick one or both)

# For Vercel Blob (recommended for Vercel deployments)
BLOB_READ_WRITE_TOKEN=vercel_blob_rw_xxxxx

# For PostgreSQL (for durable job metadata)
POSTGRES_URL=postgresql://user:password@host:5432/dbname

## OPTIONAL - QUEUE INTEGRATION (QStash for async job dispatch)

QSTASH_URL=https://qstash.io
QSTASH_TOKEN=your-qstash-token
APP_BASE_URL=https://quantum-ml-classifier.vercel.app

## OPTIONAL - APP CONFIG

NEXT_PUBLIC_APP_NAME=Quantum ML Control Center

---

## HOW TO SET IN VERCEL

1. Go to: https://vercel.com/sohams-projects-1c740e47/quantum-ml-classifier/settings/environment-variables
2. Click "Add New"
3. Paste each variable name and value
4. Select "Production" (or Production, Preview, Development as needed)
5. Click "Save"
6. Redeploy production when done

## INSTRUCTIONS

### Step 1: Set REQUIRED vars (all must be set for login to work)
- AUTH_SECRET: Generate a random 16+ char string
- AUTH_USER_EMAIL: Your admin email
- AUTH_USER_PASSWORD: Your admin password
- GOOGLE_CLIENT_ID: Already filled (from Google Cloud Console)
- GOOGLE_CLIENT_SECRET: Already filled (mark as Sensitive in Vercel)
- GOOGLE_REDIRECT_URI: Already filled (production callback URL)
- GOOGLE_AUTH_ALLOWED_EMAILS: Set to * to allow all Google users
- WORKER_WEBHOOK_SECRET: Generate a random 16+ char string
- WORKER_BASE_URL: Your deployed worker URL (e.g., https://quantum-ml-worker.railway.app)

### Step 2: Set OPTIONAL storage vars (pick one or use both)
- If using Vercel Blob: Set BLOB_READ_WRITE_TOKEN (Vercel auto-generates this)
- If using Postgres: Set POSTGRES_URL

### Step 3: Set OPTIONAL queue vars (only if using QStash)
- QSTASH_URL, QSTASH_TOKEN, APP_BASE_URL

### Step 4: Redeploy
- Go to Deployments → Redeploy production
- Or push a commit to main

## VERIFY IT WORKS

After deployment, test at:
https://quantum-ml-classifier.vercel.app/api/health

You should see worker connectivity status. If `worker.reachable` is true, backend is connected.
