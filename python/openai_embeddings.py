import openai

openai.organization = "YOUR_ORG_ID"
openai.api_key = "test"
openai.api_base = "http://localhost:3000/v1"


def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0][
        "embedding"
    ]


embedding = get_embedding("Your text goes here", model="llama-2")
print(embedding)
print(len(embedding))
