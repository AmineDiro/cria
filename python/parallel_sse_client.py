import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import sseclient
import urllib3

url = "http://localhost:3000/v1/completions_stream"


def sse_request(prompt: str):
    http = urllib3.PoolManager()

    response = http.request(
        "POST",
        url,
        preload_content=False,
        # headers={"Accept": "text/event-stream"},
        headers={
            "Content-Type": "application/json",
        },
        body=json.dumps(
            {
                "prompt": prompt,
                "temperature": 0.8,
                "max_tokens": 256,
                "stream": True,
            }
        ),
    )

    client = sseclient.SSEClient(response)

    s = time.perf_counter()
    txt = ""
    for event in client.events():
        chunk = json.loads(event.data)
        txt += chunk["choices"][0]["text"]
        sys.stdout.write(chunk["choices"][0]["text"])
        sys.stdout.flush()
    e = time.perf_counter()

    print(
        f"\n [{os.getpid()}] Generation from completion took {e-s:.2f} !\
            Result : {txt}"
    )


if __name__ == "__main__":
    prompts = [
        "Morocco is a beautiful country",
        "Engineering is ",
        "Soccer is a best ",
    ]
    # for p in prompts:
    #     sse_request(p)

    with ProcessPoolExecutor(4) as e:
        e.map(sse_request, prompts)
