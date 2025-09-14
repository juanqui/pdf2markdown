## vLLM - NanoNets

```
vllm serve --tensor-parallel-size 2 --max-model-len 32768 --host 0.0.0.0 --port 7080 nanonets/Nanonets-OCR-s
```

## vLLM - MiniCPM-o-2_6

```
vllm serve --trust-remote-code --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 --enforce-eager --host 0.0.0.0 --port 7080 openbmb/MiniCPM-o-2_6
```

## vLLM - MiniCPM-V-4

(Honestly pretty bad performance.The omni v2.6 model is MUCH better)

DOES NOT WORK:

```
vllm serve --trust-remote-code --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 --enforce-eager --host 0.0.0.0 --port 7080 openbmb/MiniCPM-V-4
```

NOTE - As of Aug 16, 2025, vLLM does NOT support MiniCPM-V-4.

Using llama.cpp:

```
./llama-server -hf second-state/MiniCPM-V-4-GGUF:F16 \
  --host 0.0.0.0 \
  --port 7080 \
  -c 32768 \
  -ngl 999 \
  --jinja \
  --parallel 8 \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 100 \
  --repeat-penalty 1.05
```

Ref: https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-v4_llamacpp.md

## vLLM - GLM-4.1V-9B-Thinking

```
vllm serve --trust-remote-code --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 --enforce-eager --host 0.0.0.0 --port 7080 zai-org/GLM-4.1V-9B-Thinking
```

## vLLM - Qwen2.5-VL-7B-Instruct

This is the best!

```
vllm serve --trust-remote-code --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 --enforce-eager --host 0.0.0.0 --port 7080 Qwen/Qwen2.5-VL-7B-Instruct
```

## vLLM - olmOCR-7B-0825

Pretty decent. Better than `Qwen2.5-VL-7B-Instruct`

```
vllm serve --trust-remote-code --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 --enforce-eager --host 0.0.0.0 --port 7080 allenai/olmOCR-7B-0825
```

```
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/AI:/root/AI \
    -p 3001:3001 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model allenai/olmOCR-7B-0825-FP8 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 3001
```

## vLLM - dots.ocr

```
vllm serve --trust-remote-code --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 --enforce-eager --chat-template-content-format string --host 0.0.0.0 --port 7080 rednote-hilab/dots.ocr
```