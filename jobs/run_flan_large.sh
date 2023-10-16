# Language model
export PLM="google/flan-t5-large"
export FSDP=0
# Train batch-size
export BS=1
# Gradient accumulation steps
export GS=32

./scripts/run_supplementary_task_experiment.sh
