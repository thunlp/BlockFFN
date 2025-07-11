# Inference

This code is modified based on [FR-Spec](https://github.com/thunlp/FR-Spec).

## Install

Change line 30 of setup.py from `arch = "87"` to your GPU architecture.
Default 87 for Jetson Orin NX.

- A100: `arch = "80"`
- RTX 3090: `arch = "86"`
- RTX 4090: `arch = "89"`
- Jetson Orin NX: `arch = "87"`
- H100: `arch = "90"`

Then run the following command to install the package.

```bash
python3 setup.py install
```

## Simple generate example

```bash
cd tests/
# echo "=== baseline ==="
python3 test_generate.py --model-type base  --model-path SparseLLM/blockffn_3b_sft
# echo "=== baseline + ffn kernel ==="
python3 test_generate.py --model-type base --use-kernel  --model-path SparseLLM/blockffn_3b_sft
# echo "=== eagle ==="
python3 test_generate.py --model-type eagle  --model-path SparseLLM/blockffn_3b_sft --eagle-path SparseLLM/blockffn_3b_sft-eagle
# echo "=== eagle + ffn kernel ==="
python3 test_generate.py --model-type eagle --use-kernel  --model-path SparseLLM/blockffn_3b_sft --eagle-path SparseLLM/blockffn_3b_sft-eagle
```

## Run the benchmark

```bash
bash scripts/eval/spec_bench/run_all.sh --model-path SparseLLM/blockffn_3b_sft --eagle-path SparseLLM/blockffn_3b_sft-eagle
```