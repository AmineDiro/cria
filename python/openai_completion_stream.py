import pprint

import openai

openai.organization = "YOUR_ORG_ID"
openai.api_key = "test"
openai.api_base = "http://localhost:3000/v1"
pprint.pprint(openai.Model.list())

result = openai.Completion.create(
    model="llama-2",
    prompt="This is a story of a hero.",
    stream=True,
)
for chunk in result:
    print(chunk)
