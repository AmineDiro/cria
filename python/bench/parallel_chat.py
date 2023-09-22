from concurrent.futures import ProcessPoolExecutor

import openai

openai.organization = "YOUR_ORG_ID"
openai.api_key = "test"
openai.api_base = "http://localhost:3000/v1"


def chat_request(idx):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020.",
            },
            {"role": "user", "content": "Where was it played?"},
        ],
    )
    return idx, completion.choices[0].message


if __name__ == "__main__":
    with ProcessPoolExecutor(5) as e:
        result = e.map(chat_request, range(3))

    print(list(result))

    # for p in prompts:
    #     sse_request(p)
