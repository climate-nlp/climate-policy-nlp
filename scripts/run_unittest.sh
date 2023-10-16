set -eu

python -m unittest src.test.test_evaluate_f1
python -m unittest src.test.validate_dataset
