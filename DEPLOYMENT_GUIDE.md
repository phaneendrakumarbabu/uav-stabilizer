# UAV Stability Simulation — Deployment Guide

This project can be deployed as a **single unified container/service** (where FastAPI serves both the simulation REST endpoints and the Three.js 3D web frontend) or as a **decoupled frontend/backend**.

---

## Architecture Overview

```
       [ Client Browser ]
               │
               ▼
   https://your-app-domain.com
               │
 ┌─────────────┴─────────────┐
 │    FastAPI Web Server     │
 ├─────────────┬─────────────┤
 │   Static    │   Dynamic   │
 │  Frontend   │  REST APIs  │
 │  (Three.js) │ (/api/...)  │
 └─────────────┴─────────────┘
```

Because FastAPI is configured to automatically mount and serve `uav_project/frontend` at the root path (`/`), deploying the backend automatically serves the full 3D interactive interface without needing separate frontend hosting or CORS configuration.

---

## 1. Render.com (Recommended — Free & Easiest)

Render provides a generous free tier for web services with automated GitHub deployment.

### Steps:
1. Push your repository to GitHub.
2. Sign in to [Render.com](https://render.com) and click **New > Web Service**.
3. Connect your GitHub repository.
4. Configure the service settings:
   * **Name**: `uav-stability-simulation`
   * **Environment**: `Python 3`
   * **Region**: Choose closest to you (e.g., Oregon, Frankfurt, Singapore)
   * **Branch**: `main`
   * **Root Directory**: `.` (leave empty or project root)
   * **Build Command**:
     ```bash
     pip install -r uav_project/backend/requirements.txt
     ```
   * **Start Command**:
     ```bash
     cd uav_project/backend && uvicorn api:app --host 0.0.0.0 --port $PORT
     ```
5. Click **Deploy Web Service**.
6. Render will provide a free HTTPS URL (e.g. `https://uav-stability-simulation.onrender.com`).

---

## 2. Hugging Face Spaces (100% Free with 16GB RAM)

Hugging Face Spaces is an outstanding platform for engineering and machine learning simulations.

### Steps:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Set Space SDK to **Docker** (Blank).
3. Clone the Space repo locally:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/uav-simulation
   ```
4. Copy all project files (including the provided [`Dockerfile`](./Dockerfile)) into the cloned directory.
5. Push to Hugging Face:
   ```bash
   git add .
   git commit -m "Deploy UAV 6-DOF simulation"
   git push
   ```
6. Hugging Face will build the container and host your simulation on a dedicated public URL with free 2 vCPUs and 16GB RAM!

---

## 3. Railway.app (1-Click Docker Deployment)

Railway provides instant container deployment with zero setup:

### Steps:
1. Sign in to [Railway.app](https://railway.app).
2. Click **New Project > Deploy from GitHub repo**.
3. Select this repository.
4. Railway automatically detects the root [`Dockerfile`](./Dockerfile) and starts building.
5. In **Settings > Networking**, click **Generate Domain** to get your public HTTPS URL.

---

## 4. Google Cloud Run (Serverless Container)

For professional cloud deployment with automated scaling (including scale-to-zero when idle, costing virtually $0):

### Steps:
1. Install and authenticate the Google Cloud CLI (`gcloud auth login`).
2. Set your GCP project:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```
3. Deploy directly from the source directory:
   ```bash
   gcloud run deploy uav-simulation \
       --source . \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated \
       --port 8000 \
       --memory 1Gi \
       --cpu 1
   ```
4. Cloud Run will build the container via Cloud Build and output your production HTTPS URL.

---

## 5. Local Docker Deployment (Self-Hosted / VPS)

You can run the entire system inside Docker on any Linux, Mac, or Windows server:

### Build Image:
```bash
docker build -t uav-simulation .
```

### Run Container:
```bash
docker run -d -p 8000:8000 --name uav-app uav-simulation
```

Open `http://localhost:8000` in your browser.

---

## 6. Decoupled Static Hosting (Vercel / GitHub Pages + Cloud Backend)

If you prefer hosting the frontend on a static CDN (like Vercel or GitHub Pages) and the Python backend on Render or AWS:

1. **Deploy Backend**: Deploy `uav_project/backend` on Render or Railway (e.g. `https://uav-api.onrender.com`).
2. **Deploy Frontend**: Deploy `uav_project/frontend` to Vercel, Netlify, or GitHub Pages.
3. **Configure API Base**: In `uav_project/frontend/index.html`, add this script tag right before `<script src="js/geometry.js">`:
   ```html
   <script>
     window.UAV_API_BASE = "https://uav-api.onrender.com";
   </script>
   ```

