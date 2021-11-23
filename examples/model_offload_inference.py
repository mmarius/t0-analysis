# Copied from https://gist.github.com/VictorSanh/26e93461acccff0dd5d5219b97266ce3
# Adapted from https://huggingface.co/transformers/main_classes/deepspeed.html#non-trainer-deepspeed-integration

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import os
import torch

# To avoid warnings about parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "bigscience/T0"

ds_config = {
    "fp16": {
        "enabled": False,
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "cpu", "pin_memory": True},
        # To tune. Current value was set to max out a 40GB A100 GPU.
        "stage3_param_persistence_threshold": 4e7,
    },
    "train_batch_size": 1,
}

_ = HfDeepSpeedConfig(ds_config)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, cache_dir="/pre-trained-transformers"
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, cache_dir="/pre-trained-transformers"
)
print("Model and tokenizer loaded")

inputs = tokenizer.encode(
    "Review: this is the best cast iron skillet you will ever buy. Is this review positive or negative?",
    return_tensors="pt",
)
inputs = inputs.to("cuda:0")

deepspeed_engine, _, _, _ = deepspeed.initialize(
    model=model,
    config_params=ds_config,
    model_parameters=None,
    optimizer=None,
    lr_scheduler=None,
)

deepspeed_engine.module.eval()
with torch.no_grad():
    outputs = deepspeed_engine.module.generate(inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("FINISHED")
