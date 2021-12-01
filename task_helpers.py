import random

from datasets import load_dataset
from transformers import AutoTokenizer

import torch

TASK_LABELS = {
    "rte": {0: "entailment", 1: "not_entailment"},
    "cb": {
        0: "entailment",
        1: "contradiction",
        2: "neutral",
    },
    "wic": {
        0: "False",
        1: "True",
    },
}


def shuffle_input(input, seed):
    shuffled_input = input.split()  # split based on whitespace
    random.seed(seed)
    random.shuffle(shuffled_input)
    shuffled_input = " ".join(shuffled_input)

    return shuffled_input


def prompted_rte(sample, template):
    label = TASK_LABELS["rte"]
    # create a prompted RTE input
    prompted_sample = template.format(
        premise=sample["premise"], hypothesis=sample["hypothesis"]
    )
    sample_label = label[sample["label"]]
    return prompted_sample, sample_label


def prompted_cb(sample, template):
    label = TASK_LABELS["cb"]
    # create a prompted CB input
    prompted_sample = template.format(
        premise=sample["premise"], hypothesis=sample["hypothesis"]
    )
    sample_label = label[sample["label"]]
    return prompted_sample, sample_label


def prompted_wic(sample, template):
    label = TASK_LABELS["wic"]
    # create a prompted WIC input
    prompted_sample = template.format(
        sentence1=sample["sentence1"],
        sentence2=sample["sentence2"],
        word=sample["word"],
    )
    sample_label = label[sample["label"]]
    return prompted_sample, sample_label


PROMPTED_TASKS = {
    "rte": prompted_rte,
    "cb": prompted_cb,
    "wic": prompted_wic,
}
