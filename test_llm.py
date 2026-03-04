from app.llm import generate_reply

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello in one sentence."}
]

reply = generate_reply(messages)

print(reply)