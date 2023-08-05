# Cria is a herd of Llamas

The objective is to serve a local `llama-2` model by mimicking an OpenAI API service.
The llama2 model **runs on GPU** using `ggml-sys` crate with specific compilation flags.

## Quickstart:

1. Git clone project

```bash
git clone git@github.com:AmineDiro/cria.git
```

2. Build project ( I ❤️ cargo !)

```bash
cargo b --release
```

> NOTE: If you have issues building for GPU.

3. Download GGML `.bin` LLama-2 quantized model (for example [llama-2-7b](https://huggingface.co/TheBloke/Llama-2-7B-GGML/tree/main))
4. Run API

```bash
./target/cria llama-2 {MODEL_BIN_PATH} --use-gpu --gpu-layers 32
```

# Completion Example

You can use `openai` python client or directly use the `sseclient` python library and stream messages. Here is an example :

```python
import json
import sys
import time

import sseclient
import urllib3

url = "http://localhost:3000/v1/completions"


http = urllib3.PoolManager()
response = http.request(
    "POST",
    url,
    preload_content=False,
    headers={
        "Content-Type": "application/json",
    },
    body=json.dumps(
        {
            "prompt": "Morocco is a beautiful country situated in north africa.",
            "temperature": 0.1,
        }
    ),
)

client = sseclient.SSEClient(response)

s = time.perf_counter()
for event in client.events():
    chunk = json.loads(event.data)
    sys.stdout.write(chunk["choices"][0]["text"])
    sys.stdout.flush()
e = time.perf_counter()

print(f"\nGeneration from completion took {e-s:.2f} !")

```

Here is the llama-2 response:

```ipython
In [8]: %run test_sse.py
nobody knows how many people live there, but it's estimated that the population is around 3
0 million.
The Moroccans are very friendly and welcoming people. They love to meet foreigners and they will be happy if you speak their language (Arabic).
Morocco is a Muslim country so don't expect to see any women wearing bikinis on the beach or at the pool. You can find some of them in Marrakech though!
If you want to visit Morocco, I recommend you to go during spring or autumn because summer is too hot and winter is cold.
I hope you enjoy your stay in this beautiful country!

Generation from completion took 2.25 !
```

## Building with GPU issues

I had some issues compiling `llm` crate with `cuda` support for my RTX2070 Super (Turing architecture). After some debugging, I needed to provide nvcc with the correct gpu-architecture version, for now `ggml-sys` crates only supports. Here are the set of changes to the `build.rs` :

```diff
diff --git a/crates/ggml/sys/build.rs b/crates/ggml/sys/build.rs
index 3a6e841..ef1e1b0 100644
--- a/crates/ggml/sys/build.rs
+++ b/crates/ggml/sys/build.rs
@@ -330,8 +330,9 @@ fn enable_cublas(build: &mut cc::Build, out_dir: &Path) {
             .arg("--compile")
             .arg("-cudart")
             .arg("static")
-            .arg("--generate-code=arch=compute_52,code=[compute_52,sm_52]")
-            .arg("--generate-code=arch=compute_61,code=[compute_61,sm_61]")
+            //.arg("--generate-code=arch=compute_52,code=[compute_52,sm_52]")
+            //.arg("--generate-code=arch=compute_61,code=[compute_61,sm_61]")
+            .arg("--generate-code=arch=compute_75,code=[compute_75,sm_75]")
             .arg("-D_WINDOWS")
             .arg("-DNDEBUG")
             .arg("-DGGML_USE_CUBLAS")
@@ -361,8 +362,7 @@ fn enable_cublas(build: &mut cc::Build, out_dir: &Path) {
             .arg("-Illama-cpp/include/ggml")
             .arg("-mtune=native")
             .arg("-pthread")
-            .arg("--generate-code=arch=compute_52,code=[compute_52,sm_52]")
-            .arg("--generate-code=arch=compute_61,code=[compute_61,sm_61]")
+            .arg("--generate-code=arch=compute_75,code=[compute_75,sm_75]")
             .arg("-DGGML_USE_CUBLAS")
             .arg("-I/usr/local/cuda/include")
             .arg("-I/opt/cuda/include")
```

The only thing left to do is to change `Cargo.toml` file to

## TODO/ Roadmap:

- [x] Run Llama.cpp on CPU using llm-chain
- [x] Run Llama.cpp on GPU using llm-chain
- [x] Implement `/models` route
- [x] Implement basic `/completions` route
- [x] Implement streaming completions SSE
- [ ] Cleanup cargo features with llm
- [ ] Merge completions / completion_streaming routes in same endpoint
- [ ] Implement `/embeddings` route
- [ ] Implement route `chat/completions`
- [ ] Implement streaming chat completions SSE
- [ ] GPU use should be optional ?
- [ ] Batching requests(ala iouring):
  - For each response put an entry in a ringbuffer queue with : Entry(Flume mpsc (resp_rx,resp_tx))
  - Spawn a model in separate task reading from ringbuffer, get entry and put each token in response
  - Construct stream from flue resp_rx chan and return SSE(stream) to user.

## Routes

- Checkout : https://platform.openai.com/docs/api-reference/
