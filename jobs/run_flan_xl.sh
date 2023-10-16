# Language model
export PLM="google/flan-t5-xl"
export FSDP=1
# Train batch-size
export BS=2
# Gradient accumulation steps
export GS=16

./scripts/run_supplementary_task_experiment.sh
