export CUDA_VISIBLE_DEVICES=1,2,3

# "bigscience/T0_3B"

##  rte (has 277 dev samples)

### encoder

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "bigscience/T0_3B" \
# --task "rte" \
# --max_inputs 277 \
# --template_file "/t0-analysis/prompts/rte.csv" \
# --template_name "all" \
# --output_dir "/logfiles" \
# --layer 0

python /t0-analysis/compute_representations.py \
--model_name_or_path "bigscience/T0_3B" \
--task "rte" \
--max_inputs 277 \
--template_file "/t0-analysis/prompts/rte.csv" \
--template_name "all" \
--output_dir "/logfiles"


### decoder

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "bigscience/T0_3B" \
# --task "rte" \
# --max_inputs 277 \
# --template_file "/t0-analysis/prompts/rte.csv" \
# --template_name "all" \
# --output_dir "/logfiles" \
# --decoder \
# --layer 0

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "bigscience/T0_3B" \
# --task "rte" \
# --max_inputs 277 \
# --template_file "/t0-analysis/prompts/rte.csv" \
# --template_name "all" \
# --output_dir "/logfiles" \
# --decoder

