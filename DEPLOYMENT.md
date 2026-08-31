# Deployment Guide

## Option A: Hugging Face Spaces (recommended for portfolio demo)

Free hosting, works well as a shareable link on a CV or scholarship application.

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Docker**
   - Hardware: CPU basic (free tier) is fine for demo purposes
2. In your local repo, make sure `app/gradio_app.py`, `src/`, `models/best.pt`,
   and `requirements.txt` are present.
3. Add a `Dockerfile` at the Space's repo root — reuse `Dockerfile.gradio`
   from this repo (rename it to `Dockerfile` in the Space, or point HF's
   Space settings at it).
4. Push:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/wastevision
   git push space main
   ```
5. HF Spaces builds the Docker image and gives you a public URL
   (`https://huggingface.co/spaces/<your-username>/wastevision`) — put this
   in your README and application materials.

Note: `models/best.pt` needs to be committed to the Space (or pulled at
build time from a release asset / HF Hub model repo if you'd rather not
commit binary weights to git — HF Hub's `huggingface_hub` Python client can
download it in the Dockerfile's build step instead).

## Option B: Render / Railway (for the FastAPI service)

Both support "deploy from Dockerfile" directly from a GitHub repo:

1. Connect your GitHub repo.
2. Point the service at the root `Dockerfile` (the API, not `Dockerfile.gradio`).
3. Set the `WASTEVISION_WEIGHTS` env var if your weights live somewhere
   other than `/app/models/best.pt`.
4. Expose port 8000.

## Option C: Local Docker Compose (for local testing / video demo recording)

```bash
docker compose up --build
```
- API: http://localhost:8000/docs (FastAPI's auto-generated Swagger UI —
  useful to screen-record for your application if you don't want to pay
  for hosting)
- Demo UI: http://localhost:7860

## Verifying a deployment

```bash
curl https://<your-deployed-url>/health
curl -X POST -F "file=@sample.jpg" https://<your-deployed-url>/predict
```
