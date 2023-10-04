import pprint

import openai

openai.organization = "YOUR_ORG_ID"
openai.api_key = "test"
openai.api_base = "http://localhost:3000/v1"

pprint.pprint(openai.Model.list())

result = openai.Completion.create(
    model="llama-2",
    prompt="Say this is a test",
)
pprint.pprint(result)
