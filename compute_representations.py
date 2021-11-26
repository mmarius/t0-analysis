import os
import sys
import argparse
import random
from pathlib import Path

import torch
import h5py
import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from task_helpers import PROMPTED_TASKS, shuffle_input
from utils import set_seeds, read_templates_from_file, POOLER


def check_args(args):
    assert args.batch_size <= args.max_inputs


def create_template_output_dir(args, template):
    # modify output_dir
    args.template_output_dir = os.path.join(
        args.output_dir,
        args.task,
        args.model_name_or_path.replace("/", "-"),
        "decoder" if args.decoder else "encoder",
        template["name"],
    )

    # make sure output_dir exists
    Path(args.template_output_dir).mkdir(parents=True, exist_ok=True)


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


def tokenize_dataset(args, task, dataset, tokenizer, template):
    prompt = PROMPTED_TASKS[task]

    # save some prompted samples for inspection
    with open(
        os.path.join(args.template_output_dir, "prompted_samples.csv"),
        "w",
        encoding="utf-8",
    ) as f:
        for _, s in enumerate(dataset):
            prompted_sample = (
                shuffle_input(prompt(s, template["template"]), seed=args.seed)
                if template["shuffle"]
                else prompt(s, template["template"])
            )
            f.write(f"{prompted_sample}\n")

    # # save template as well
    with open(
        os.path.join(args.template_output_dir, "template.csv"), "w", encoding="utf-8"
    ) as f:
        f.write(f"name,template,category,shuffle\n")
        f.write(
            f"{template['name']},{template['template']},{template['category']},{template['shuffle']}\n"
        )

    # apply prompt and tokenize
    dataset = dataset.map(
        lambda s: tokenizer(
            shuffle_input(prompt(s, template["template"]), seed=args.seed)
            if template["shuffle"]
            else prompt(s, template["template"]),
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
    for l, hidden_representations in hidden_representations_collection.items():
        if args.layer in [-1, l]:
            file_name = f"hidden_represenations_layer{l}_{args.pooler_type}.hdf5"
            output_file = os.path.join(args.template_output_dir, file_name)
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

    # load tokenizer
    print(f"Loading {args.model_name_or_path} tokenizer...")
    tokenizer = load_tokenizer(args.model_name_or_path)

    # load model
    print(f"Loading {args.model_name_or_path} model (this may take some time)...")
    model = load_model(args.model_name_or_path)

    templates = []
    if args.template_file is not None:
        df = read_templates_from_file(args.template_file)
        for _, row in df.iterrows():
            if args.template_name == "all":
                templates.append(row)
            else:
                if row["name"] == args.template_name:
                    templates.append(row)

    # iterate over templates
    for template in tqdm(templates, desc="Iterating over templates"):
        # create output dir
        create_template_output_dir(args, template)

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
            args,
            args.task,
            dataset,
            tokenizer,
            template=template,
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
                # outputs = model.generate(
                outputs = model.forward(
                    input_ids=formatted_batch["input_ids"],
                    attention_mask=formatted_batch["attention_mask"],
                    decoder_input_ids=formatted_batch["input_ids"],
                    output_hidden_states=True,
                    # return_dict_in_generate=True,
                    return_dict=True,
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

                hidden_represenations = (
                    decoder_hidden_representations
                    if args.decoder
                    else encoder_hidden_representations
                )

                # post-process hidden states
                for l, h in enumerate(hidden_represenations):
                    # post-process hidden state representation
                    h = postprocess_hidden_representation(
                        h, pooler_type=args.pooler_type
                    )

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
        choices=["rte"],
        required=True,
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
        "--template_file",
        type=str,
        default=None,
        required=True,
        help=".csv file containing templates",
    )

    parser.add_argument(
        "--template_name",
        type=str,
        default=None,
        required=True,
        help="name of template to use. make sure to specify a `template_file`.",
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
        "--decoder",
        action="store_true",
        help="if set, extract decoder representations",
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