# vLLM Filesystem Queue Setup

End-to-end setup for running HuggingFace models on GPU via filesystem queue.

## Architecture

```
┌─────────────┐         ┌──────────────┐         ┌──────────────┐
│   lm.py     │  HTTP   │ api_server.py│  Files  │vllm_worker.py│
│  (DSPy)     │────────>│ (login node) │────────>│  (GPU node)  │
│             │         │              │         │              │
│ "local/..." │         │ VllmLlm      │         │ vLLM engine  │
└─────────────┘         └──────────────┘         └──────────────┘
                              │                         │
                              └─────────────────────────┘
                                Filesystem Queue
                          /nas/ucb/biddulph/shared/vllm_queue/
                            ├── requests/
                            ├── processing/
                            ├── responses/
                            └── heartbeat/
```

## Quick Start

### One-Command Startup (Recommended)

```bash
cd /nas/ucb/biddulph/gepa-legibility/serving

# Start both worker and API server with one command
./start_model_service.sh allenai/OLMo-2-1124-13B-SFT

# For larger models with more GPUs
./start_model_service.sh meta-llama/Meta-Llama-3-70B 4

# Custom port
./start_model_service.sh allenai/OLMo-2-1124-13B-SFT 1 8043
```

This script will:
1. Submit the Slurm worker job
2. Wait for the worker to be healthy (heartbeat detected)
3. Start the API server on the login node

The API server runs in the foreground. Press Ctrl+C to stop it.

### Manual Startup (Alternative)

If you prefer to start components separately:

#### 1. Start vLLM Worker on GPU Node

```bash
cd /nas/ucb/biddulph/gepa-legibility/serving

# Start worker with 1 GPU for a 7B model
./start_vllm_worker.sh 1 allenai/OLMo-2-1124-13B-SFT

# Or for a larger model with 4 GPUs
./start_vllm_worker.sh 4 meta-llama/Meta-Llama-3-70B

# Check Slurm job
squeue -u $USER

# Check logs
tail -f slurm-logs/start_vllm_worker-*.out
```

#### 2. Start API Server on Login Node

```bash
# In a separate terminal on login node
cd /nas/ucb/biddulph/gepa-legibility/serving

python api_server.py \
    --model allenai/OLMo-2-1124-13B-SFT \
    --port 8042

# Or run in background
nohup python api_server.py --model allenai/OLMo-2-1124-13B-SFT --port 8042 > api_server.log 2>&1 &
```

### Use in Your Code

```python
from lm import get_dspy_lm

# Use local model via filesystem queue
lm = get_dspy_lm("local/allenai/OLMo-2-1124-13B-SFT", temperature=0.7)

# Use like any other DSPy LM
response = lm("What is 2+2?")
print(response)

# Or with full messages
result = lm(messages=[
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Explain GEPA"}
])
```

## Configuration

### Queue Directory
Default: `/nas/ucb/biddulph/shared/vllm_queue`

To use a different directory:
```bash
# Worker
./start_vllm_worker.sh 1 allenai/OLMo-2-1124-13B-SFT /path/to/custom/queue

# API server
python api_server.py --model allenai/OLMo-2-1124-13B-SFT --queue-dir /path/to/custom/queue
```

### Model Selection
Use any HuggingFace model ID:
- `allenai/OLMo-2-1124-13B-SFT`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

### GPU Allocation
Number of GPUs depends on model size:
- 7B models: 1 GPU
- 13B models: 1-2 GPUs
- 70B models: 4-8 GPUs

## Troubleshooting

### Worker not found
```python
RuntimeError: No active vLLM workers found
```
**Solution**: Start the worker with `./start_vllm_worker.sh`

### Model mismatch
```python
RuntimeError: No vLLM worker found for model 'model-a'. Found active workers for: 'model-b'
```
**Solution**: Either:
1. Start worker for correct model
2. Change your code to use the running model

### Timeout
```python
TimeoutError: Job xyz timed out after 300 seconds
```
**Solution**:
- Check worker logs: `tail -f slurm-logs/start_vllm_worker-*.out`
- Ensure worker is running: `squeue -u $USER`
- Model might be loading (first request takes longer)

### Check Worker Health
```bash
curl http://localhost:8000/health
```

## Files

- `start_model_service.sh` - **One-command startup** (submits worker + starts API server)
- `vllm_worker.py` - GPU worker process (runs on Slurm node)
- `vllm_llm.py` - Client library for filesystem queue
- `api_server.py` - FastAPI wrapper providing OpenAI-compatible API
- `start_vllm_worker.sh` - Script to submit worker via Slurm
- `vllm_worker_template.sh` - Slurm job template
- `test_local_model.py` - End-to-end test script

## Advanced Usage

### Direct VllmLlm Usage (without API server)

```python
from serving.vllm_llm import VllmLlm

llm = VllmLlm(
    model_id="allenai/OLMo-2-1124-13B-SFT",
    temperature=0.7,
    max_tokens=2048,
    queue_dir="/nas/ucb/biddulph/shared/vllm_queue"
)

response = llm.generate_response([
    {"role": "user", "content": "Hello!"}
])
print(response)  # Just a string!
```

### Multiple Models
Run multiple workers with different models:

```bash
# Terminal 1: Start worker for model A
./start_vllm_worker.sh 1 allenai/OLMo-2-1124-13B-SFT /nas/ucb/biddulph/shared/queue_a

# Terminal 2: Start worker for model B
./start_vllm_worker.sh 1 Qwen/Qwen2.5-7B-Instruct /nas/ucb/biddulph/shared/queue_b

# Terminal 3: API server for model A
python api_server.py --model allenai/OLMo-2-1124-13B-SFT --queue-dir /nas/ucb/biddulph/shared/queue_a --port 8000

# Terminal 4: API server for model B
python api_server.py --model Qwen/Qwen2.5-7B-Instruct --queue-dir /nas/ucb/biddulph/shared/queue_b --port 8001
```

Then in code:
```python
# Model A on port 8000
lm_a = get_dspy_lm("local/allenai/OLMo-2-1124-13B-SFT")

# Model B on port 8001 (need to modify lm.py or use direct HTTP client)
```
