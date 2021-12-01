# Copied from https://gist.github.com/VictorSanh/049387b8dc0e6ac8f83ba55feb04917f
# Model parallelism
# Tested: 4 16GB V100 or 2 32GB V100 (basically need somewhere to fit ~42GB of fp32 params)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np

model_name = "bigscience/T0_3B"

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

# embedding_matrix = model.shared.weight
# print(embedding_matrix.shape)

# # save embedding matrix
# with open("/logfiles/embeddings/embedding_matrix.npy", "wb") as f:
#     np.save(f, embedding_matrix.cpu().detach().numpy())

inputs = tokenizer.encode(
    "Review: the movie was not great. Is this review positive or negative?",
    return_tensors="pt",
)

# ids = tokenizer._convert_token_to_id_with_added_voc("<pad>")
# print(ids)

inputs = inputs.to("cuda:0")
with torch.no_grad():
    outputs = model.generate(
        inputs,
        output_hidden_states=True,
        output_scores=True,
        return_dict_in_generate=True,
    )

    sequences = outputs.sequences

    # Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
    # of shape :obj:`(batch_size, sequence_length, hidden_size)`.
    encoder_hidden_states = outputs.encoder_hidden_states

    # Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
    # :obj:`torch.FloatTensor` of shape :obj:`(batch_size, generated_length, hidden_size)`.
    decoder_hidden_states = outputs.decoder_hidden_states

print(tokenizer.batch_decode(sequences, skip_special_tokens=False))

print("sequences", sequences.shape)

for tid in torch.flatten(sequences):
    print(tid, tokenizer._convert_id_to_token(tid))

print("encoder_hidden_states")
for i, h in enumerate(encoder_hidden_states):  # iterate over encoder layers
    print(i, h.shape)

print("decoder_hidden_states:", len(decoder_hidden_states))
for i, t in enumerate(decoder_hidden_states):  # iterate over generated tokens
    for j, h in enumerate(t):  # iterate over decoder layers
        print(i, j, h.shape)

print("FINISHED")
