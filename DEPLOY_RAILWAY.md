# Railway Deployment Guide — WeldVision v2

## Step 1 — Bitbucket Push

```bash
cd weld-vision-v2
git init                          # if not already a git repo
git add .
git commit -m "feat: object-level S3 folders, Gemini commented out"
git remote add origin https://bitbucket.org/YOUR_WORKSPACE/weld-vision-v2.git
git push -u origin main
```

## Step 2 — Railway Setup

1. Go to https://railway.app → New Project
2. **"Deploy from GitHub/Bitbucket repo"** → connect your Bitbucket account
3. Select `weld-vision-v2` repo
4. Railway will auto-detect the `Dockerfile` and `railway.toml`

## Step 3 — Add PostgreSQL on Railway

1. In your Railway project → **"+ New"** → **"Database"** → **"Add PostgreSQL"**
2. Railway creates a Postgres instance automatically
3. Click the Postgres service → **"Variables"** tab → copy `DATABASE_URL`
   - It will look like: `postgresql://postgres:password@monorail.proxy.rlwy.net:PORT/railway`
   - **Change the prefix** from `postgresql://` to `postgresql+asyncpg://` for asyncpg driver

## Step 4 — Set Environment Variables

In your Railway app service → **"Variables"** tab → add these:

```
APP_ENV=production
GEMINI_API_KEY=AIzaSy_your_key_here
AWS_ACCESS_KEY_ID=AKIA_your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=ap-south-1
AWS_S3_BUCKET=your-bucket-name
DATABASE_URL=postgresql+asyncpg://postgres:password@host:PORT/railway
ALLOWED_ORIGINS=*
```

## Step 5 — Deploy

1. Railway auto-deploys on every push to `main`
2. Check **"Deployments"** tab → watch logs
3. Once green, click **"Settings"** → **"Domains"** → generate a public URL
4. Test: `GET https://your-app.railway.app/health`

## Step 6 — Verify

```bash
# Health check
curl https://your-app.railway.app/health

# Test POST with sample images
curl -X POST https://your-app.railway.app/api/v1/inspections/video \
  -F "object_id=A" \
  -F "object_name=Test Joint" \
  -F "scan_number=1" \
  -F "side=Top" \
  -F "images=@frame1.jpg" \
  -F "images=@frame2.jpg"
```

## Swagger Docs
`https://your-app.railway.app/docs`
