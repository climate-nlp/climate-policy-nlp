# An NLP Benchmark Dataset for Assessing Corporate Climate Policy Engagement

This is a research codebase to reproduce our work presented in the NeurIPS 2023 datasets and benchmark track.
The codebase here is for academic research purposes and not intended for use by the general public, investors, or business-related users.
Note that the accuracy of the current models is far from perfect and frequently produces incorrect results. Please refer to the paper for more detail of limitations.


### Setup

#### Recommended environment and device

- python 3.9.5
- gcc/8.3.1
- cuda/11.7/11.7.1 
- cudnn/8.8/8.8.1 
- nccl/2.13/2.13.4-1

- A100 GPUs

#### Install dependencies
```bash
pip install torch==2.0.0 torchvision
pip install -r requirements.txt
```

### Usage

- For the pipeline model with pre-trained language models, refer to the following scripts:
  - ```src/model_1_detect_evidence.py``` is the training and prediction code for the evidence page detection. See example usage in [run_main_task_experiment.sh](scripts%2Frun_main_task_experiment.sh)
  - ```src/model_2_classify_query.py``` is the training and prediction code for the query classification. See example usage in [run_main_task_experiment.sh](scripts%2Frun_main_task_experiment.sh)
  - ```src/model_3_classofy_stance.py``` is the training and prediction code for the stance classification. See example usage in [run_main_task_experiment.sh](scripts%2Frun_main_task_experiment.sh)

- For the comment generation, refer to the following scripts:
  - ```src/flan-alpaca/training.py``` is the training code. See example usage in [run_supplementary_task_experiment.sh](scripts%2Frun_supplementary_task_experiment.sh)
  - ```src/flan-alpaca/inference.py``` is the prediction code. See example usage in [run_supplementary_task_experiment.sh](scripts%2Frun_supplementary_task_experiment.sh)

- ```src/evaluate_f1.py``` is the official scorer for the main task (i.e., evidence page detection, query classification, and stance classification). 
  - Example usage: ```python src/evaluate_f1.py -s {{prediction_file}} -g data/lobbymap_dataset/test.jsonl```

- ```src/evaluate_rouge.py``` is the official scorer for the supplementary task (i.e., comment generation). 
  - Example usage: ```python src/evaluate_rouge.py -s {{prediction_file}} -g data/lobbymap_dataset/test.comment.json```


### Reproduce experiments


#### Prepare the dataset

1. Accept terms/conditions and download the dataset from [here](https://climate-nlp.github.io/)
2. Place the downloaded dataset under the project directory as follows:

```text
data/lobbymap_dataset
├── test.comment.json
├── test.jsonl
├── train.comment.json
├── train.jsonl
├── valid.comment.json
└── valid.jsonl
```


#### Fine-tune, predict and evaluate for the main task

```bash
# BERT (base)
./jobs/run_bert_base.sh
# ClimateBERT
./jobs/run_climatebert_base.sh
# RoBERTa (base)
./jobs/run_roberta_base.sh
# Longformer (base)
./jobs/run_longformer_base.sh
# Longformer (large)
./jobs/run_longformer_large.sh
```

The trained models, predictions and evaluation results will be located under the ```log/``` directory.
After the process finished, you can see evaluation results of the test set like:
```cat log/longformer-large-4096/4_evaluation/test.json```

Note that you may get slightly different scores than in our paper. Different computational resources and environments can cause this to happen.
Also, the Longformer model behaves in a non-deterministic manner even with the same random seed.


#### Fine-tune, predict and evaluate for the supplementary task (i.e., comment generation task)

```bash
# FlanT5 (large)
./jobs/run_flan_large.sh
# FlanT5 (xl)
./jobs/run_flan_xl.sh
```

The trained models, predictions and evaluation results will be located under the ```log/``` directory.
After the process finished, you can see evaluation results of the test set like:
```cat log/flan-t5-large/2_evaluation/test.json```

Note that you may get slightly different scores than in our paper. Different computational resources and environments can cause this to happen.
Also, the model may behave in a non-deterministic manner even with the same random seed.


### Using trained models

Make sure you have installed following packages:

```bash
pip install git+https://github.com/facebookresearch/detectron2.git@v0.5 datasets==2.10.0 nltk==3.8.1 python-doctr==0.6.0 pymupdf==1.21.1 pytesseract==0.3.10 Pillow==9.4.0 imutils==0.5.4 rapidfuzz==2.13.7
```

The trained models are available [here](https://huggingface.co/climate-nlp).
You can directly download and use the models as follows:

```bash
python src/demo.py \
  --pdf <your pdf file path> \
  --max_pages 100 \
  --batch_size 4 \
  --model_name_detect_evidence "climate-nlp/longformer-large-4096-1-detect-evidence" \
  --model_name_classify_query "climate-nlp/longformer-large-4096-2-classify-query" \
  --model_name_classify_stance "climate-nlp/longformer-large-4096-3-classify-stance"
```

The output result of the main task will be saved at the directory of the input PDF file.



### License

Some contents used in this project are drawn from external projects:
- Main task models: https://github.com/huggingface/transformers
- Comment generation models: https://github.com/declare-lab/flan-alpaca
- Scoring: https://github.com/jerbarnes/semeval22_structured_sentiment/

The license of these third-party codes depends on the license of the project. Please refer to each script file for more detail.
For the rest of our original work, please refer to the LICENSE file in this project. 
Please note that the licenses do not apply to the dataset and trained model of this project, and the terms and conditions of any rights holders apply. 


### Citation

```text
@inproceedings{morio-and-manning-2023-nlp,
 author = {Morio, Gaku and Manning, Christopher D},
 booktitle = {Advances in Neural Information Processing Systems: Datasets and Benchmarks Track},
 title = {An NLP Benchmark Dataset for Assessing Corporate Climate Policy Engagement},
 year = {2023}
}
```
