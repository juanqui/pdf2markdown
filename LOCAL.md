## vLLM - NanoNets

```
vllm serve --tensor-parallel-size 2 --max-model-len 32768 --host 0.0.0.0 --port 7080 nanonets/Nanonets-OCR-s
```

## vLLM - MiniCPM-o-2_6

```
vllm serve --trust-remote-code --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 --enforce-eager --host 0.0.0.0 --port 7080 openbmb/MiniCPM-o-2_6
```