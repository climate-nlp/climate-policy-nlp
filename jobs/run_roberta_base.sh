# Language model
export PLM="roberta-base"
# Train batch-size
export BS=8
# Gradient accumulation steps
export GS=1
# Max sequence length
export MAX_TRAIN_LEN=512
export MAX_INFER_LEN=512

./scripts/run_main_task_experiment.sh
