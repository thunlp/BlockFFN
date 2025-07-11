eagle_file="data/spec_bench/model_answer/blockffn_3b_sft/eagle.jsonl"
base_file="data/spec_bench/model_answer/blockffn_3b_sft/baseline.jsonl"
eagle_kernel_file="data/spec_bench/model_answer/blockffn_3b_sft/eagle_kernel.jsonl"
base_kernel_file="data/spec_bench/model_answer/blockffn_3b_sft/baseline_kernel.jsonl"

echo "=== eagle (without ffn kernel) vs. baseline (without ffn kernel) ==="
python evaluation/spec_bench/speed.py \
    --file-path $eagle_file \
    --base-path $base_file \
    $@

echo "=== eagle (with ffn kernel) vs. baseline (with ffn kernel) ==="
python evaluation/spec_bench/speed.py \
    --file-path $eagle_kernel_file \
    --base-path $base_kernel_file \
    $@