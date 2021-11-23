from datasets import load_dataset
from transformers import AutoTokenizer

import torch


def prompted_rte(sample, template):
    # create a prompted RTE input
    return template.format(premise=sample["premise"], hypothesis=sample["hypothesis"])
