import random

from datasets import load_dataset
from transformers import AutoTokenizer

import torch


def shuffle_input(input, seed):
    shuffled_input = input.split()  # split based on whitespace
    random.seed(seed)
    random.shuffle(shuffled_input)
    shuffled_input = " ".join(shuffled_input)

    return shuffled_input


def prompted_rte(sample, template):
    # create a prompted RTE input
    prompted_sample = template.format(
        premise=sample["premise"], hypothesis=sample["hypothesis"]
    )
    return prompted_sample


PROMPTED_TASKS = {
    "rte": prompted_rte,
}
