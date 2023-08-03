# Cria is a herd of Llamas

The objective is to serve a local llama 2 model by mimicking an OpenAI API service.

## TODO :

- [x] Run Llama.cpp on CPU using llm-chain
- [x] Run Llama.cpp on GPU using llm-chain
- [x] Implement `/models` route
- [x] Implement basic `/completions` route
- [ ] Implement streaming completions (ala iouring):

  - For each response put an entry in a ringbuffer queue with : Entry(Flume mpsc (resp_rx,resp_tx))
  - Spawn a model in separate task reading from ringbuffer, get entry and put each token in response
  - Construct stream from flue resp_rx chan and return SSE(stream) to user.

- [ ] Implement `/embeddings` route
- [ ] Implement route `chat/completions`

## Routes

- Checkout : https://platform.openai.com/docs/api-reference/
