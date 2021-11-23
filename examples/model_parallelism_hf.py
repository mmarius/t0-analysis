# Copied from https://gist.github.com/VictorSanh/049387b8dc0e6ac8f83ba55feb04917f
# Model parallelism
# Tested: 4 16GB V100 or 2 32GB V100 (basically need somewhere to fit ~42GB of fp32 params)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "bigscience/T0"

print("Loading model and tokenizer...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, cache_dir="/pre-trained-transformers"
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, cache_dir="/pre-trained-transformers"
)
print("Model and tokenizer loaded")

model.parallelize()
print("Moved model to GPUs")

inputs = tokenizer.encode(
    "Review: this is the best cast iron skillet you will ever buy. Is this review positive or negative?",
    return_tensors="pt",
)
inputs = inputs.to("cuda:0")
with torch.no_grad():
    outputs = model.generate(inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("FINISHED")
