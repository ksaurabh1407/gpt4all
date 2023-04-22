from nomic.gpt4all import GPT4AllGPU
m = GPT4AllGPU("../models/ggjt-model.bin")
config = {'num_beams': 2,
          'min_new_tokens': 10,
          'max_length': 100,
          'repetition_penalty': 2.0}
out = m.generate('write me a story about a lonely computer', config)
print(out)