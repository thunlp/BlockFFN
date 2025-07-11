echo "=== baseline ==="
python3 test_generate.py --model-type base $@ 2>&1 | tail -n 1
echo "=== baseline + kernel ==="
python3 test_generate.py --model-type base --use-kernel $@ 2>&1 | tail -n 1
echo "=== eagle ==="
python3 test_generate.py --model-type eagle $@ 2>&1 | tail -n 1
echo "=== eagle + kernel ==="
python3 test_generate.py --model-type eagle --use-kernel $@ 2>&1 | tail -n 1

