export CUDA_VISIBLE_DEVICES=1,2,3,4

# "bigscience/T0_3B"

##  rte

### null pattern
# python /t0-analysis/compute_representations.py \
# --model_name_or_path "bigscience/T0_3B" \
# --task "rte" \
# --template "{premise} {hypothesis}" \
# --output_dir "/logfiles/null_pattern"

### template 1
python /t0-analysis/compute_representations.py \
--model_name_or_path "bigscience/T0_3B" \
--task "rte" \
--template '{premise} Are we justified in saying that "{hypothesis}"?' \
--output_dir "/logfiles/template_1"
