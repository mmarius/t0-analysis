export CUDA_VISIBLE_DEVICES=1,2,3,4

# rte
python /t0-analysis/compute_representations.py --task "rte" --template '{premise} Are we justified in saying that "{hypothesis}"?'