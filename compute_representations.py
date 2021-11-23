import argparse
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from task_helpers import prompted_rte

PROMPTED_TASK = {
    "rte": prompted_rte,
}


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
    f = PROMPTED_TASK[task]

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


def postprocess_hidden_states(outputs):
    # get encoder hidden representations
    encoder_hidden_states = (
        outputs.encoder_hidden_states
    )  # Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
    # of shape :obj:`(batch_size, sequence_length, hidden_size)`

    # get decoder hidden representations
    decoder_hidden_states = (
        outputs.decoder_hidden_states
    )  # Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
    # :obj:`torch.FloatTensor` of shape :obj:`(batch_size, generated_length, hidden_size)`

    for l, h in enumerate(encoder_hidden_states):
        print(l, h.shape)  # (batch_size, sequence_length, hidden_size)


def main(args):
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
        tokenized_dataset, batch_size=args.batch_size
    )

    # iterate over dataset
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
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

            # post-process hidden states
            postprocess_hidden_states(outputs)

            # decode prediction
            # generated_sequences = tokenizer.batch_decode(
            #     outputs.sequences, skip_special_tokens=True
            # )
            # print(generated_sequences)
            break


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
        help="batch_size to use",
    )

    args = parser.parse_args()

    main(args)
