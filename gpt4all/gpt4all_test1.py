from langchain.llms import GPT4All

# Instantiate the model. Callbacks support token-wise streaming
model = GPT4All(model="../models/gpt4all-lora-quantized-ggml.bin", n_ctx=512, n_threads=8)

# Generate text
response = model("Once upon a time, ")
