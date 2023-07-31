#!/bin/bash

python3 -m llama_cpp.server \
  --model  /media/amine/models/llama/llama-2-7b-chat.ggmlv3.q4_0.bin \
  --f16_kv True \
  --n_gpu_layers 20


