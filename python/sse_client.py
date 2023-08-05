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
    # headers={"Accept": "text/event-stream"},
    headers={
        "Content-Type": "application/json",
    },
    body=json.dumps(
        {
            "prompt": "Morocco is a beautiful country",
            "temperature": 0.8,
            "max_tokens": 100,
            "stream": True,
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
