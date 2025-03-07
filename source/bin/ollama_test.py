

import ollama


response = ollama.chat(
model="llama3.2",
messages=[{"role": "user", "content": "this is a new test?"}])
print(response)