from sys import stdout

import openai

openai.organization = "YOUR_ORG_ID"
openai.api_key = "test"
openai.api_base = "http://localhost:3000/v1"


completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=512,
    stream=True,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Hello! Can you give informations about morocco ? ",
        },
    ],
)

# Streaming
for chunk in completion:
    delta = chunk.choices[0].delta["content"]
    stdout.write(delta)
    stdout.flush()
