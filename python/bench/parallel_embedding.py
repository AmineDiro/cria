from concurrent.futures import ProcessPoolExecutor

import openai

openai.organization = "YOUR_ORG_ID"
openai.api_key = "test"
openai.api_base = "http://localhost:3000/v1"


def embed_req(text):
    response = openai.Embedding.create(input=text, model="llama-2")
    return response["data"][0]["embedding"]


if __name__ == "__main__":
    text = [
        "You are a helpful assistant.",
        # "Who won the world series in 2020?",
        # "The Los Angeles Dodgers won the World Series in 2020.",
        # "Where was it played?",
    ]

    with ProcessPoolExecutor(5) as e:
        result = e.map(embed_req, text)

    print(list(result))

    # for p in prompts:
    #     sse_request(p)
