import os
import argparse
import torch
from pathlib import Path
import h5py

import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from task_helpers import PROMPTED_TASKS
from utils import set_seeds, POOLER


def check_args(args):
    assert args.batch_size <= args.max_inputs


def load_model(model_name_or_path, put_on_gpu=True):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path, cache_dir="/pre-trained-transformers"
    )
    # move model to GPUs
    if put_on_gpu:
        print("Putting model on GPU...")
        model.parallelize()  # this will use all visible GPUs
    return model


def load_tokenizer(tokenizer_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, cache_dir="/pre-trained-transformers"
    )
    return tokenizer


def tokenize_dataset(task, dataset, tokenizer, template):
    f = PROMPTED_TASKS[task]

    dataset = dataset.map(
        lambda s: tokenizer(
            f(s, template),
            truncation=True,
            padding="max_length",
        ),
        batched=False,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


def postprocess_hidden_representation(hidden_representation, pooler_type):
    # hidden_representation.shape =  (batch_size, sequence_length, hidden_size)
    # apply pooler
    pooled_hidden_representation = POOLER[pooler_type](hidden_representation)

    # convert to numpy array
    pooled_hidden_representation = pooled_hidden_representation.cpu().detach().numpy()

    return pooled_hidden_representation


def save_hidden_representations(args, hidden_representations_collection):
    # make sure output_dir exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for l, hidden_representations in hidden_representations_collection.items():
        if args.layer in [-1, l]:
            file_name = f"hidden_represenations_{args.model_name_or_path.replace('/', '-')}_layer{l}_{args.pooler_type}.hdf5"
            output_file = os.path.join(args.output_dir, file_name)
            create_hdf5_file(hidden_representations, output_file)


def create_hdf5_file(hidden_representations, output_file):
    # hidden_representations.shape = (batch_size, hidden_size)
    print(
        f"Saving hidden representations with shape {hidden_representations.shape} to: ",
        output_file,
    )

    with h5py.File(output_file, "w") as f:
        for idx, h in enumerate(
            tqdm(hidden_representations, desc="Creating hdf5 file")
        ):
            # Hidden represenations are indexed by their corresponding sample index. Which will be a str.
            # Be careful when loading the data. One has to use the same str index to get the sample back.
            f.create_dataset(
                str(idx),  # int doesn't work here
                h.shape,
                dtype="float32",
                data=h,
            )


def main(args):
    # fix seeds
    set_seeds(args.seed)

    # load tokenizer and model
    print(f"Loading {args.model_name_or_path} tokenizer...")
    tokenizer = load_tokenizer(args.model_name_or_path)

    print(f"Loading {args.model_name_or_path} model (this may take some time)...")
    model = load_model(args.model_name_or_path)

    # load dataset
    dataset = load_dataset(
        path="super_glue",
        name=args.task,
        cache_dir="/datasets/huggingface-datasets",
        split=args.split,
        streaming=False,
    )

    # tokenize dataset and create dataloader
    tokenized_dataset = tokenize_dataset(
        args.task,
        dataset,
        tokenizer,
        template=args.template,
    )
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset, batch_size=args.batch_size, shuffle=False
    )

    hidden_representations_collection = {}

    # iterate over dataset
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            # stop after max_inputs samples
            if hidden_representations_collection:
                if hidden_representations_collection[0].shape[0] >= args.max_inputs:
                    break

            # format batch and put data on GPU
            formatted_batch = {
                "input_ids": batch["input_ids"].to("cuda:0"),
                "attention_mask": batch["attention_mask"].to("cuda:0"),
                # "labels": batch["label"].to("cuda:0"),
            }
            outputs = model.generate(
                **formatted_batch,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

            # get encoder hidden representations
            encoder_hidden_representations = (
                outputs.encoder_hidden_states
            )  # Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            # of shape :obj:`(batch_size, sequence_length, hidden_size)`

            # get decoder hidden representations
            decoder_hidden_representations = (
                outputs.decoder_hidden_states
            )  # Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            # :obj:`torch.FloatTensor` of shape :obj:`(batch_size, generated_length, hidden_size)`

            # post-process hidden states
            for l, h in enumerate(encoder_hidden_representations):
                # post-process hidden state representation
                h = postprocess_hidden_representation(h, pooler_type=args.pooler_type)

                # collect post-processed hidden states for every layer
                if l in hidden_representations_collection:
                    hidden_representations_collection[l] = np.concatenate(
                        (hidden_representations_collection[l], h), axis=0
                    )
                else:
                    hidden_representations_collection[l] = h

            # decode prediction
            # generated_sequences = tokenizer.batch_decode(
            #     outputs.sequences, skip_special_tokens=True
            # )
            # print(generated_sequences)

    # store hidden states on disc
    save_hidden_representations(args, hidden_representations_collection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and store T0 hidden states.")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bigscience/T0_3B",
        choices=["bigscience/T0", "bigscience/T0_3B"],
        help="which model to use. defaults to bigscience/T0_3B",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="rte",
        help="super_glue task for which to compute hidden states. defaults to rte",  # TODO(mm): support other datasets as well
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="validation",
        help="data split. defaults to validation",
    )

    parser.add_argument(
        "--template",
        type=str,
        default='{premise} Are we justified in saying that "{hypothesis}"?',
        help="template used to construct prompted inputs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="batch size to use",
    )

    parser.add_argument(
        "--max_inputs",
        type=int,
        default=100,
        help="collect hidden representation for `--max_input` inputs",
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="save hidden representation only for layer `--layer`. if -1 save hidden representations of all layers.",
    )

    parser.add_argument(
        "--pooler_type",
        type=str,
        default="avg",
        help="pooler type to use",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/logfiles",
        help="where to store the hidden representations",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    args = parser.parse_args()

    # check args
    check_args(args)

    main(args)
