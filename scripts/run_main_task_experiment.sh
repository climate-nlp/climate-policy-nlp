set -eu

# Hyper-parameters ---------
DATASET_ROOT=data/lobbymap_dataset
TRAIN_FILE=data/lobbymap_dataset/train.jsonl

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
if [[ -z "${MAX_TRAIN_LEN}" ]]; then
  echo "Environment variable MAX_TRAIN_LEN is not defined"
  exit
fi
if [[ -z "${MAX_INFER_LEN}" ]]; then
  echo "Environment variable MAX_INFER_LEN is not defined"
  exit
fi

PLM_ABBR=$(python -c "print('${PLM}'.split('/')[-1])")
LOG_ROOT=./log/${PLM_ABBR}

LR=1e-5
WARMUP_RATIO=0.1
EBS=8
SEED=42
# --------------------------


mkdir -p ${LOG_ROOT}


# -----------------------------------------
# 1. Training of evidence page detection
# -----------------------------------------

LOG1=${LOG_ROOT}/1_detect_evidence

if [ ! -e ${LOG1}/pytorch_model.bin ]; then
  echo "[Fine-tuning] evidence detection"
  python src/model_1_detect_evidence.py \
    --model_name_or_path ${PLM} \
    --output_dir ${LOG1} \
    --train_file ${TRAIN_FILE} \
    --do_train \
    --seed ${SEED} \
    --learning_rate ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_steps 20000 \
    --max_seq_length ${MAX_TRAIN_LEN} \
    --per_device_train_batch_size ${BS} \
    --gradient_accumulation_steps ${GS} \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --save_steps 100 --logging_steps 5 \
    --fp16
fi

for DATA_SPLIT in "test" "valid"
do
  if [ ! -e ${LOG1}/${DATA_SPLIT}/predict_results.jsonl ]; then
    echo "[Prediction] evidence detection"
    python src/model_1_detect_evidence.py \
      --model_name_or_path ${LOG1} \
      --output_dir ${LOG1}/${DATA_SPLIT} \
      --test_file ${DATASET_ROOT}/${DATA_SPLIT}.jsonl \
      --do_predict \
      --seed ${SEED} \
      --max_seq_length ${MAX_INFER_LEN} \
      --per_device_eval_batch_size ${EBS} \
      --min_prediction_per_doc 1 \
      --fp16
  fi
done


# -----------------------------------------
# 2. Training of query classification
# -----------------------------------------

LOG2=${LOG_ROOT}/2_classify_query

if [ ! -e ${LOG2}/pytorch_model.bin ]; then
  echo "[Fine-tuning] query classification"
  python src/model_2_classify_query.py \
    --model_name_or_path ${PLM} \
    --output_dir ${LOG2} \
    --train_file ${TRAIN_FILE} \
    --do_train \
    --seed ${SEED} \
    --learning_rate ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_steps 20000 \
    --max_seq_length ${MAX_TRAIN_LEN} \
    --per_device_train_batch_size ${BS} \
    --gradient_accumulation_steps ${GS} \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --save_steps 100 --logging_steps 5 \
    --fp16
fi

# Prediction
for DATA_SPLIT in "test" "valid"
do
  if [ ! -e ${LOG2}/${DATA_SPLIT}/predict_results.jsonl ]; then
    echo "[Prediction] query classification"
    python src/model_2_classify_query.py \
      --model_name_or_path ${LOG2} \
      --output_dir ${LOG2}/${DATA_SPLIT} \
      --test_file ${LOG1}/${DATA_SPLIT}/predict_results.jsonl \
      --do_predict \
      --seed ${SEED} \
      --max_seq_length ${MAX_INFER_LEN} \
      --per_device_eval_batch_size ${EBS} \
      --fp16
  fi
done


# -----------------------------------------
# 3. Training of stance classification
# -----------------------------------------

LOG3=${LOG_ROOT}/3_classify_stance

if [ ! -e ${LOG_ROOT}/3_classify_stance/pytorch_model.bin ]; then
  echo "[Fine-tuning] stance classification"
  python src/model_3_classify_stance.py \
    --model_name_or_path ${PLM} \
    --output_dir ${LOG3} \
    --train_file ${TRAIN_FILE} \
    --do_train \
    --seed ${SEED} \
    --learning_rate ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_steps 20000 \
    --max_seq_length ${MAX_TRAIN_LEN} \
    --per_device_train_batch_size ${BS} \
    --gradient_accumulation_steps ${GS} \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --save_steps 100 --logging_steps 5 \
    --fp16
fi

# Prediction
for DATA_SPLIT in "test" "valid"
do
  if [ ! -e ${LOG3}/${DATA_SPLIT}/predict_results.jsonl ]; then
    echo "[Prediction] stance classification"
    python src/model_3_classify_stance.py \
      --model_name_or_path ${LOG3} \
      --output_dir ${LOG3}/${DATA_SPLIT} \
      --test_file ${LOG2}/${DATA_SPLIT}/predict_results.jsonl \
      --do_predict \
      --seed ${SEED} \
      --max_seq_length ${MAX_INFER_LEN} \
      --per_device_eval_batch_size ${EBS} \
      --fp16
  fi
done

# Post-process
for DATA_SPLIT in "test" "valid"
do
  if [ ! -e ${LOG3}/${DATA_SPLIT}/final_result.jsonl ]; then
    echo "Postprocessing the prediction file (${LOG3}/${DATA_SPLIT}/predict_results.jsonl)"
    python src/postprocess.py \
      --input_file ${LOG3}/${DATA_SPLIT}/predict_results.jsonl \
      --output_file ${LOG3}/${DATA_SPLIT}/final_result.jsonl
  fi
done



# -----------------------------------------
# 4. Evaluation
# -----------------------------------------

echo "Evaluating"

LOG4=${LOG_ROOT}/4_evaluation
mkdir -p ${LOG4}

# Scoring
for DATA_SPLIT in "test" "valid"
do
  if [ ! -e ${LOG4}/${DATA_SPLIT}.json ]; then
    python src/evaluate_f1.py \
      -s ${LOG3}/${DATA_SPLIT}/final_result.jsonl \
      -g ${DATASET_ROOT}/${DATA_SPLIT}.jsonl > ${LOG4}/${DATA_SPLIT}.json
  fi
done
