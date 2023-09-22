import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import sseclient
import urllib3

url = "http://localhost:3000/v1/completions"


def sse_request(prompt: str):
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
                "prompt": prompt,
                "temperature": 0.8,
                "max_tokens": 10,
                "stream": True,
            }
        ),
    )

    client = sseclient.SSEClient(response)

    s = time.perf_counter()
    resp = ""
    for event in client.events():
        chunk = json.loads(event.data)
        sys.stdout.write(chunk["choices"][0]["text"])
        sys.stdout.flush()
        resp += chunk["choices"][0]["text"]
    e = time.perf_counter()
    print(f"{[os.getpid()]}\nGeneration from completion took {e-s:.2f}!")


if __name__ == "__main__":
    prompts = [
        "Morocco is a beautiful country",
        "Soccer is a beautiful sport because",
        "Engineering is an amazing job ",
    ]

    with ProcessPoolExecutor(5) as e:
        e.map(sse_request, prompts)

    # for p in prompts:
    #     sse_request(p)
