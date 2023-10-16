# This code is adapted from https://github.com/declare-lab/flan-alpaca
# See origin project and LICENSE for more detail

import os.path
import shutil
import json
import copy
from pathlib import Path

import torch
from fire import Fire
from huggingface_hub import HfApi
from lightning_fabric import seed_everything

from training import LightningModel


def test_model(
    path: str,
    prompt: str = "",
    max_length: int = 160,
    device: str = "cuda",
):
    if not prompt:
        prompt = "Write a short email to show that 42 is the optimal seed for training neural networks"

    model: LightningModel = LightningModel.load_from_checkpoint(path)
    tokenizer = model.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    seed_everything(model.hparams.seed)
    with torch.inference_mode():
        model.model.eval()
        model = model.to(device)
        input_ids = input_ids.to(device)
        outputs = model.model.generate(input_ids, max_length=max_length, do_sample=True)

    print(tokenizer.decode(outputs[0]))

    """
    Example output (outputs/model/base/epoch=2-step=2436.ckpt):
    <pad> Dear [Company Name], I am writing to demonstrate the feasibility of using 42 as an optimal seed
    for training neural networks. I am sure that this seed will be an invaluable asset for the training of 
    these neural networks, so let me know what you think.</s>
    """


def predict_json(
    model_path: str,
    data_path: str,
    output_path: str,
    max_source_length: int = 1024,
    max_target_length: int = 100,
    min_target_length: int = 1,
    device: str = "cuda",
    max_samples: int = -1,
    batch_size: int = 4,
    do_sample: bool = False,
    num_beams: int = 3,
    bf16: bool = False,
):
    with open(data_path, 'r') as f:
        jds = json.loads(f.read())

    if max_samples > 0:
        jds = jds[:max_samples]

    model: LightningModel = LightningModel.load_from_checkpoint(model_path)
    model.model.eval()
    model = model.to(dtype=torch.bfloat16 if bf16 else torch.float32, device=device)

    tokenizer = model.tokenizer

    seed_everything(model.hparams.seed)

    out_dir = os.path.dirname(output_path)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except:
        pass
    with open(output_path, 'w') as f:
        pass

    write_jds = []
    for i_batch in range(0, len(jds), batch_size):
        batch_jds = jds[i_batch: i_batch + batch_size]
        batch_prompt = [
            f"{jd['instruction']}\n{jd['input']}" for jd in batch_jds
        ]
        input_ids = tokenizer.batch_encode_plus(
            batch_prompt,
            return_tensors="pt",
            padding='max_length',
            max_length=max_source_length,
            truncation=True,
        ).input_ids

        with torch.inference_mode():
            input_ids = input_ids.to(device)
            gen_result = model.model.generate(
                input_ids,
                max_new_tokens=max_target_length,
                min_new_tokens=min_target_length,
                do_sample=do_sample,
                num_beams=num_beams,
            )
            outputs = tokenizer.batch_decode(
                gen_result,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            assert len(outputs) == len(batch_jds)

        for out, jd in zip(outputs, batch_jds):
            write_jd = copy.deepcopy(jd)
            write_jd['output'] = out
            write_jds.append(write_jd)

        with open(output_path, 'w') as f:
            f.write(json.dumps(write_jds, ensure_ascii=False, indent=4))


def export_to_hub(path: str, repo: str, temp: str = "temp"):
    if Path(temp).exists():
        shutil.rmtree(temp)

    model = LightningModel.load_from_checkpoint(path)
    model.model.save_pretrained(temp)
    model.tokenizer.save_pretrained(temp)
    del model  # Save memory?

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=temp)


if __name__ == "__main__":
    Fire()
