export CUDA_VISIBLE_DEVICES=1,2,3

# "bigscience/T0_3B"

##  rte (has 277 dev samples)

### encoder

#### null pattern

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "bigscience/T0_3B" \
# --task "rte" \
# --max_inputs 277 \
# --template "{premise} {hypothesis}" \
# --output_dir "/logfiles/null_pattern"

#### template 1

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "bigscience/T0_3B" \
# --task "rte" \
# --max_inputs 277 \
# --template '{premise} Are we justified in saying that "{hypothesis}"?' \
# --output_dir "/logfiles/template_1"

#### template 2

python /t0-analysis/compute_representations.py \
--model_name_or_path "bigscience/T0_3B" \
--task "rte" \
--max_inputs 277 \
--template "{premise} Question: {hypothesis} Yes or no?" \
--output_dir "/logfiles/template_2"

### decoder

#### null pattern

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "bigscience/T0_3B" \
# --task "rte" \
# --max_inputs 277 \
# --template "{premise} {hypothesis}" \
# --output_dir "/logfiles/null_pattern" \
# --decoder

#### template 1

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "bigscience/T0_3B" \
# --task "rte" \
# --max_inputs 277 \
# --template '{premise} Are we justified in saying that "{hypothesis}"?' \
# --output_dir "/logfiles/template_1" \
# --decoder

#### template 2

# python /t0-analysis/compute_representations.py \
# --model_name_or_path "bigscience/T0_3B" \
# --task "rte" \
# --max_inputs 277 \
# --template "{premise} Question: {hypothesis} Yes or no?" \
# --output_dir "/logfiles/template_2" \
# --decoder

