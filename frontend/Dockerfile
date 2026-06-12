# ── VLM Disaster Analyzer — Frontend ─────────────────────────────────────────
# Multi-stage build: Node builds the Vite app, nginx serves the static files.

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (cached layer)
COPY package.json package-lock.json* ./
RUN npm ci --prefer-offline

# Copy source and build
COPY . .

# VITE_API_URL is baked into the static bundle at build time.
# Pass it as a build arg: docker build --build-arg VITE_API_URL=http://...
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build

# ── Stage 2: Serve ────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine

# SPA routing: all unknown paths fall through to index.html
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
