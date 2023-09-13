from sys import stdout

import openai

openai.organization = "YOUR_ORG_ID"
openai.api_key = "test"
openai.api_base = "http://localhost:3000/v1"

response = openai.Completion.create(
    model="llama-2",
    prompt="This is a story of a hero who went",
    stream=True,
)
for event in response:
    event_text = event["choices"][0]["text"]  # extract the text
    stdout.write(event_text)
    stdout.flush()
