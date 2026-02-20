#!/bin/bash
#SBATCH --job-name=fever-lotus
#SBATCH --output=logs/fever-lotus-%j.out
#SBATCH --error=logs/fever-lotus-%j.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mail-user=hcao@umass.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1

# --- CUDA Setup ---
module load cuda/11.8

nvidia-smi

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Use /tmp for model downloads (home dir has limited space)
export HF_HOME=/tmp/hf_cache_$SLURM_JOB_ID

# --- HuggingFace Auth (needed for gated models like Llama) ---
# Option 1: Set your token here directly
# export HF_TOKEN="hf_your_token_here"
#
# Option 2: Load from .env file (recommended)
if [ -f "$SLURM_SUBMIT_DIR/.env" ]; then
    set -a
    source "$SLURM_SUBMIT_DIR/.env"
    set +a
fi

if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN not set. Gated models (Llama) won't download."
    echo "   Set it in .env or run: huggingface-cli login"
fi

# --- Environment Setup ---
module --ignore_cache load "python/3.12"

cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Create venv and install deps if not present
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment and installing dependencies..."
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install vllm
    pip install lotus-ai "datasets<3.0" faiss-cpu
else
    source .venv/bin/activate
fi

echo "Running on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"

# --- Start vLLM Server ---
echo "🚀 Starting vLLM server (Qwen2.5-1.5B-Instruct)..."

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000 \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 &

VLLM_PID=$!

# Wait for vLLM to be ready (can take 1-2 min to load model)
echo "⏳ Waiting for vLLM server to be ready..."
MAX_WAIT=300
WAITED=0
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "❌ vLLM server failed to start within ${MAX_WAIT}s"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    echo "   Waited ${WAITED}s..."
done
echo "✅ vLLM server ready!"

# --- Run FEVER Experiment ---
echo "🔬 Running FEVER experiment..."
python run_lotus_fever.py || { echo "❌ Experiment failed"; kill $VLLM_PID; exit 1; }

# --- Cleanup ---
echo "🧹 Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "✅ Job finished successfully at $(date)"
