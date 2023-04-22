from langchain.llms import GPT4All

# Instantiate the model. Callbacks support token-wise streaming
model = GPT4All(model="../models/ggjt-model.bin", n_ctx=512, n_threads=8)

# Query text
text = "What is the fastest way to become a billionaire?"

# Feed Query text to model
print(model(text)) 