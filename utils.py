import h5py
import torch
import random

from tqdm import tqdm
import numpy as np
import pandas as pd

pd.set_option("display.max_colwidth", -1)


def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def avg_pooler(hidden_representation):
    # hidden_representation.shape = (batch_size, sequence_length, hidden_size)
    return torch.mean(hidden_representation, dim=1, keepdim=False)


def avg_pooler_no_padding(input_ids, hidden_representation, pad_token_id):
    # ignore padding tokens when taking the average
    # hidden_representation.shape = (batch_size, sequence_length, hidden_size)
    hidden_size = hidden_representation.shape[-1]
    all_hidden_states = []
    for idx, hidden_state in enumerate(hidden_representation):
        mask = torch.where(
            input_ids[idx] != pad_token_id,
            torch.tensor(True).to("cuda:0"),
            torch.tensor(False).to("cuda:0"),
        )

        candidates = torch.masked_select(hidden_state, mask[:, None]).reshape(
            -1, hidden_size
        )
        pooled_hidden_states = torch.mean(candidates, dim=0)
        all_hidden_states.append(pooled_hidden_states)

    hidden_states = torch.stack(all_hidden_states, dim=0)
    return hidden_states


def random_token_pooler(hidden_representation, seed=42):
    # hidden_representation.shape = (batch_size, sequence_length, hidden_size)
    random.seed(seed)
    idx = random.randint(0, hidden_representation.shape[1] - 1)
    return hidden_representation[:, idx, :]


def last_token_pooler(hidden_representation):
    # hidden_representation.shape = (batch_size, sequence_length, hidden_size)
    # TODO(mm): This doesn't make sense. It will return a padding token.
    return hidden_representation[:, -1, :]


def first_token_pooler(hidden_representation):
    # hidden_representation.shape = (batch_size, sequence_length, hidden_size)
    return hidden_representation[:, 0, :]


def load_hidden_representations_from_hdf5(input_file, silent=False):
    # the hdf5 files are dictionaries of length n_inputs containing numpy arrays of shape (embedding_dim, )
    f = h5py.File(input_file, "r")

    # NOTE: hdf5 file keys are strings which are sorted *alphabetically*: ['0', '1', '10', '12', ... ]. This will cause issues when assigning labels to samples.
    rows = list(f.keys())
    rows.sort(key=int)  # sort them numerically

    hidden_representations = []
    if silent:
        gen = rows
    else:
        gen = tqdm(rows, desc="Reading embeddings")
    for row in gen:
        hidden_representation = np.asarray(f[row])
        hidden_representations.append(hidden_representation)

    hidden_representations = np.asarray(hidden_representations).squeeze()

    return hidden_representations


def read_templates_from_file(file_name, sep=";"):
    df = pd.read_csv(file_name, sep=sep)
    return df


def read_predictions_from_file(file_name, sep=";"):
    df = pd.read_csv(file_name, sep=sep, names=["predicted_sequence"])
    return df


POOLER = {
    "avg": avg_pooler,
    "avg-nopad": avg_pooler_no_padding,
    "random": random_token_pooler,
    "last": last_token_pooler,
    "first": first_token_pooler,
}
