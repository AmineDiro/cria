import time

from llama_cpp import Llama

llm = Llama(
    model_path="/media/amine/models/llama/llama-2-13b.ggmlv3.q4_K_M.bin",
    n_gpu_layers=32,
    n_ctx=200,
    verbose=False,
)

s = time.perf_counter()
output = llm(
    "Q: Name the planets in the solar system? A: ",
    max_tokens=30,
    # stop=["Q:", "\n"],
    echo=True,
)
e = time.perf_counter()
print(f"Generation took {e-s:.2f}s")
