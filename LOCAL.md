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