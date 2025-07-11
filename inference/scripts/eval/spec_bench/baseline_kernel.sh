export CUDA_VISIBLE_DEVICES=0
Model_id="blockffn_3b_sft"
Bench_name="spec_bench"

python3 evaluation/inference_baseline.py \
    $@ \
    --cuda-graph \
    --model-id $Model_id/baseline_kernel \
    --memory-limit 0.5 \
    --bench-name $Bench_name \
    --dtype "bfloat16" \
    --use-kernel \
