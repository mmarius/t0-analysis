export CUDA_VISIBLE_DEVICES=1,2,3

MODEL="bigscience/T0_3B" # 3B params
# MODEL="bigscience/T0" # 11B params

##  rte (has 277 dev samples and 2490 train examples)

### encoder

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "${MODEL}" \
# --task "rte" \
# --max_inputs 277 \
# --template_file "/t0-analysis/prompts/rte.csv" \
# --template_name "all" \
# --output_dir "/logfiles" \
# --layer 0

python /t0-analysis/compute_representations.py \
--model_name_or_path "${MODEL}" \
--task "rte" \
--max_inputs 277 \
--template_file "/t0-analysis/prompts/rte.csv" \
--template_name "all" \
--output_dir "/logfiles"


### decoder

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "${MODEL}" \
# --task "rte" \
# --max_inputs 277 \
# --template_file "/t0-analysis/prompts/rte.csv" \
# --template_name "all" \
# --output_dir "/logfiles" \
# --decoder \
# --layer 0

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "${MODEL}" \
# --task "rte" \
# --max_inputs 277 \
# --template_file "/t0-analysis/prompts/rte.csv" \
# --template_name "all" \
# --output_dir "/logfiles" \
# --decoder



##  cb (has 56 dev samples and 250 train examples)

### encoder

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "${MODEL}" \
# --task "cb" \
# --max_inputs 56 \
# --template_file "/t0-analysis/prompts/cb.csv" \
# --template_name "all" \
# --output_dir "/logfiles" \
# --layer 0

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "${MODEL}" \
# --task "cb" \
# --max_inputs 56 \
# --template_file "/t0-analysis/prompts/cb.csv" \
# --template_name "all" \
# --output_dir "/logfiles"


### decoder

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "${MODEL}" \
# --task "cb" \
# --max_inputs 56 \
# --template_file "/t0-analysis/prompts/cb.csv" \
# --template_name "all" \
# --output_dir "/logfiles" \
# --decoder \
# --layer 0

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "${MODEL}" \
# --task "cb" \
# --max_inputs 56 \
# --template_file "/t0-analysis/prompts/cb.csv" \
# --template_name "all" \
# --output_dir "/logfiles" \
# --decoder



##  wic (has 638 dev samples and 5428 train examples)

### encoder

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "${MODEL}" \
# --task "wic" \
# --max_inputs 638 \
# --template_file "/t0-analysis/prompts/wic.csv" \
# --template_name "all" \
# --output_dir "/logfiles" \
# --layer 0

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "${MODEL}" \
# --task "wic" \
# --max_inputs 638 \
# --template_file "/t0-analysis/prompts/wic.csv" \
# --template_name "all" \
# --output_dir "/logfiles"


### decoder

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "${MODEL}" \
# --task "wic" \
# --max_inputs 638 \
# --template_file "/t0-analysis/prompts/wic.csv" \
# --template_name "all" \
# --output_dir "/logfiles" \
# --decoder \
# --layer 0

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "${MODEL}" \
# --task "wic" \
# --max_inputs 638 \
# --template_file "/t0-analysis/prompts/wic.csv" \
# --template_name "all" \
# --output_dir "/logfiles" \
# --decoder
