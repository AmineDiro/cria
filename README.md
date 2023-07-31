# Cria is a herd of Llamas

The objective is to serve a local llama 2 model by mimicking an OpenAI API service.

## TODO :

- [ ] Run Llama.cpp on CPU using llm-chain
- [ ] Run Llama.cpp on GPU using llm-chain
- [ ] Implement `/models` route
- [ ] Implement `/embeddings` route
- [ ] Implement route

## Routes

```
GET - /v1/models
```

---

```
POST- /v1/chat/completions
```

- Params:

  **model**
  _string_
  Optional
  ID of the model to use. You can use the List models API to see all of your available models, or see our Model overview for descriptions of them.

  **prompt**
  _string or array_
  Required
  The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.

  Note that <|endoftext|> is the document separator that the model sees during training, so if a prompt is not specified the model will generate as if from the beginning of a new document.

  **stream**
  _boolean_
  Optional
  Defaults to false
  Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message. Example Python code.

  **stop**
  string or array
  Optional
  Defaults to null
  Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.

- Example :

  **Request**

  ```bash
  curl https:/localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
      "model": "gpt-3.5-turbo",
      "messages": [
      {
          "role": "system",
          "content": "You are a helpful assistant."
      },
      {
          "role": "user",
          "content": "Hello!"
      }
      ]
  }'

  ```

  **Response**

  ```json
  {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "\n\nHello there, how may I assist you today?"
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 9,
      "completion_tokens": 12,
      "total_tokens": 21
    }
  }
  ```
