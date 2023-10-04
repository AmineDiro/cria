import openai

openai.organization = "YOUR_ORG_ID"
openai.api_key = "test"
openai.api_base = "http://localhost:3000/v1"

print(openai.Model.list())

result = openai.Completion.create(
    model="llama-2",
    prompt="Say this is a test",
)
print(result)
