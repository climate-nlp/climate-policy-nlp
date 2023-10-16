# Language model
export PLM="allenai/longformer-large-4096"
# Train batch-size
export BS=4
# Gradient accumulation steps
export GS=2
# Max sequence length
export MAX_TRAIN_LEN=1532
export MAX_INFER_LEN=2048

./scripts/run_main_task_experiment.sh
