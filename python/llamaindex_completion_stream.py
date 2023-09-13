import os

import openai
from llama_index.llms import OpenAI

openai.organization = "YOUR_ORG_ID"
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:3000/v1"


# non-streaming
llm = OpenAI()
resp = llm.complete("Paul Graham is ")
print(resp)

# resp = llm.stream_complete("Paul Graham is ")
# for delta in resp:
#     print(delta, end="")
