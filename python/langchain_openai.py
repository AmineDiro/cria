import os
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
os.environ["OPENAI_API_KEY"] = "api_key"


template = """
USER: You are a helpful, medical specialist. Always answer as helpfully as possible.  If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. You always answer medical questions based on facts.
ASSISTANT: Ok great ! I am a medical expert!
USER: {question} [/INST]
"""

template = """
USER: You are a helpful, medical specialist. Always answer as helpfully as possible.  If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. You always answer medical questions based on facts.
ASSISTANT: Ok great ! I am a medical expert!
USER: {question}
ASSISTANT:
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = OpenAI(openai_api_base="http://localhost:3000/v1")
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What can I do about glenoid cavity injury ?"

result = llm_chain.run(question)
print(result)
