#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-ocr-paddle-linux-amd64}"
INPUT_DIR="${INPUT_DIR:-./Document}"
OUT_DIR="${OUT_DIR:-./outputs}"
DB_DIR="${DB_DIR:-./chroma_db}"
CACHE_DIR="${ROOT_DIR}/.docker-cache"
INPUT_DIR_REL="${INPUT_DIR#./}"
OUT_DIR_REL="${OUT_DIR#./}"
DB_DIR_REL="${DB_DIR#./}"

mkdir -p "${CACHE_DIR}/uv" "${CACHE_DIR}/hf" "${CACHE_DIR}/paddlex" "${CACHE_DIR}/pip"

if [[ "${INPUT_DIR}" = /* || "${OUT_DIR}" = /* || "${DB_DIR}" = /* ]]; then
  echo "Please use relative paths for INPUT_DIR/OUT_DIR/DB_DIR (for Docker copy-back)."
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" && -f "${ROOT_DIR}/.env" ]]; then
  OPENAI_API_KEY="$(
    sed -n 's/^OPENAI_API_KEY=//p' "${ROOT_DIR}/.env" \
      | head -n 1 \
      | tr -d '\r'
  )"
  export OPENAI_API_KEY
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required. Export it or set it in .env."
  exit 1
fi

echo "[1/3] Building linux/amd64 Paddle ingest image: ${IMAGE_NAME}"
docker build \
  --platform linux/amd64 \
  -f "${ROOT_DIR}/docker/paddle-ingest.Dockerfile" \
  -t "${IMAGE_NAME}" \
  "${ROOT_DIR}"

echo "[2/3] Installing dependencies in container venv"
echo "[3/3] Running ingest with --ocr-engine paddle"
docker run --rm \
  --platform linux/amd64 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e OCR_CHAT_MODEL="${OCR_CHAT_MODEL:-gpt-4o-mini}" \
  -e OCR_VISION_MODEL="${OCR_VISION_MODEL:-gpt-4o-mini}" \
  -e OCR_EMBED_MODEL="${OCR_EMBED_MODEL:-text-embedding-3-small}" \
  -e UV_CACHE_DIR=/cache/uv \
  -e HF_HOME=/cache/hf \
  -e PIP_CACHE_DIR=/cache/pip \
  -e FLAGS_use_mkldnn=0 \
  -e FLAGS_enable_pir_api=0 \
  -e FLAGS_enable_pir_in_executor=0 \
  -e HOME=/tmp \
  -v "${ROOT_DIR}:/src" \
  -v "${CACHE_DIR}:/cache" \
  -v "${CACHE_DIR}/paddlex:/tmp/.paddlex" \
  -w /tmp \
  "${IMAGE_NAME}" \
  bash -lc "set -euo pipefail && rm -rf /tmp/workdir && mkdir -p /tmp/workdir && cd /src && tar cf - --exclude='.venv' --exclude='.venv_linux_amd64' --exclude='outputs' --exclude='chroma_db' --exclude='.docker-cache' . | tar xf - -C /tmp/workdir && cd /tmp/workdir && python -m venv .venv_linux_amd64 && .venv_linux_amd64/bin/pip install -U pip && .venv_linux_amd64/bin/pip install -e '.[ocr]' && .venv_linux_amd64/bin/pip install paddlepaddle && .venv_linux_amd64/bin/ocr ingest --input-dir \"${INPUT_DIR_REL}\" --out-dir \"${OUT_DIR_REL}\" --db \"${DB_DIR_REL}\" --ocr-engine paddle && rm -rf \"/src/${OUT_DIR_REL}\" \"/src/${DB_DIR_REL}\" && cp -a \"/tmp/workdir/${OUT_DIR_REL}\" \"/src/${OUT_DIR_REL}\" && cp -a \"/tmp/workdir/${DB_DIR_REL}\" \"/src/${DB_DIR_REL}\""

echo "Done. Outputs are in ${OUT_DIR} and ${DB_DIR}."
