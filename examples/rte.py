from datasets import load_dataset
from transformers import AutoTokenizer

import torch


def rte_prompt(sample, template):
    # create a prompted RTE input
    return template.format(premise=sample["premise"], hypothesis=sample["hypothesis"])


print("Load dataset...")
# load dataset
dataset = load_dataset(
    # path="bigscience/P3",
    # name="super_glue_rte_based_on_the_previous_passage",
    path="super_glue",
    name="rte",
    cache_dir="/datasets/huggingface-datasets",
    split="validation",
    streaming=False,
)


print("Load tokenizer...")
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bigscience/T0", cache_dir="/pre-trained-transformers"
)

print("Tokenize dataset...")
# tokenize and format dataset
dataset = dataset.map(
    lambda sample: tokenizer(
        rte_prompt(
            sample, template='{premise} Are we justified in saying that "{hypothesis}"?'
        ),  # apply prompt to sample
        truncation=True,
        padding="max_length",
    ),
    batched=False,
)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# creata dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

# iterate over dataset in batches
for idx, batch in enumerate(dataloader):
    print(batch)
    break
