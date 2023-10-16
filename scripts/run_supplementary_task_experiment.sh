set -eu

# Hyper-parameters ---------
DATASET_ROOT=data/lobbymap_dataset
TRAIN_FILE=data/lobbymap_dataset/train.comment.json


if [[ -z "${PLM}" ]]; then
  echo "Environment variable PLM is not defined"
  exit
fi
if [[ -z "${BS}" ]]; then
  echo "Environment variable BS is not defined"
  exit
fi
if [[ -z "${GS}" ]]; then
  echo "Environment variable GS is not defined"
  exit
fi
if [[ -z "${FSDP}" ]]; then
  echo "Environment variable FSDP is not defined"
  exit
fi

PLM_ABBR=$(python -c "print('${PLM}'.split('/')[-1])")
LOG_ROOT=./log/${PLM_ABBR}

MAX_TRAIN_SRC_LEN=1532
MAX_INFER_SRC_LEN=2048
MIN_TGT_LEN=10
MAX_TGT_LEN=150
NUM_BEAM=3
SEED=42
TRAIN_EPOCH=10
EBS=4
# --------------------------


mkdir -p ${LOG_ROOT}


# -----------------------------------------
# 1. Fine-tuning
# -----------------------------------------

LOG1=${LOG_ROOT}/1_finetuning

if [ ! -e ${LOG1}/train.json ]; then
python src/flan-alpaca/data_loading.py preprocess_alpaca \
  --path_in ${TRAIN_FILE} \
  --path_out ${LOG1}/train.json
fi

if [ ! -e ${LOG1}/*.ckpt ]; then
  echo "Fine-tuning Flan-T5"
  if [ "${FSDP}" -eq "0" ]; then
    python src/flan-alpaca/training.py \
      --seed ${SEED} \
      --output_dir ${LOG1} \
      --use_compile \
      --max_source_length ${MAX_TRAIN_SRC_LEN} \
      --max_target_length ${MAX_TGT_LEN} \
      --train_epochs ${TRAIN_EPOCH} \
      --data_path ${LOG1}/train.json \
      --model_name_or_path ${PLM} \
      --train_batch_size ${BS} \
      --gradient_accumulation_steps ${GS} \
      --bf16
  else
    python src/flan-alpaca/training.py \
      --seed ${SEED} \
      --output_dir ${LOG1} \
      --use_fsdp \
      --max_source_length ${MAX_TRAIN_SRC_LEN} \
      --max_target_length ${MAX_TGT_LEN} \
      --train_epochs ${TRAIN_EPOCH} \
      --data_path ${LOG1}/train.json \
      --model_name_or_path ${PLM} \
      --train_batch_size ${BS} \
      --gradient_accumulation_steps ${GS} \
      --bf16
  fi
fi

# Prediction
for DATA_SPLIT in "test" "valid"
do
  if [ ! -e ${LOG1}/${DATA_SPLIT}/final_result.comment.json ]; then
    python src/flan-alpaca/inference.py predict_json \
      --model_path ${LOG1}/*.ckpt \
      --data_path ${DATASET_ROOT}/${DATA_SPLIT}.comment.json \
      --output_path ${LOG1}/${DATA_SPLIT}/final_result.comment.json \
      --max_source_length ${MAX_INFER_SRC_LEN} \
      --max_target_length ${MAX_TGT_LEN} \
      --min_target_length ${MIN_TGT_LEN} \
      --num_beams ${NUM_BEAM} \
      --bf16 \
      --device "cuda" \
      --batch_size ${EBS}
  fi
done


# -----------------------------------------
# 2. Evaluation
# -----------------------------------------

echo "Evaluating"

LOG2=${LOG_ROOT}/2_evaluation
mkdir -p ${LOG2}

for DATA_SPLIT in "test" "valid"
do
  if [ ! -e ${LOG2}/${DATA_SPLIT}.json ]; then
    python src/evaluate_rouge.py \
      -s ${LOG1}/${DATA_SPLIT}/final_result.comment.json \
      -g ${DATASET_ROOT}/${DATA_SPLIT}.comment.json  > ${LOG2}/${DATA_SPLIT}.json
  fi
done
