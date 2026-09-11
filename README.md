# H3-World: Turning Language Understanding into World Control

H3-World is the **first interactive world model** built on [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). It generates action-controlled video from an initial frame by converting keyboard states into per-latent language instructions and binding each instruction to its corresponding future video latent through directed attention routing. Using 8,000 gameplay clips from [ABot-World-Explorer-500h](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h), H3-World learns 65.6M LoRA parameters, only 0.199% of the 33B backbone.

<a href="https://arxiv.org/abs/2609.01560"><img src="https://img.shields.io/badge/arXiv-H3--World-A42C25.svg" alt="arXiv"></a>
<a href="https://huggingface.co/DANNY621/H3-World"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffbd45.svg" alt="Hugging Face model"></a>
<a href="https://danzer1xxxxchan.github.io/H3-World"><img src="https://img.shields.io/badge/Web-Project%20Page-1d72b8.svg" alt="Project page"></a>
<a href="https://modelscope.cn/models/DANNY621/H3-World"><img src="https://img.shields.io/badge/ModelScope-H3--World-purple?logo=modelscope" alt="ModelScope"></a>

https://github.com/user-attachments/assets/1c862995-8809-447e-bade-2c47bfdb2738

> **H3-World: Turning Language Understanding into World Control**<br>
> Danze Chen<sup>1,2</sup>, Zeqing Wang<sup>1,2</sup>, Ziyue Lin<sup>3</sup>, [Xingyi Yang](https://adamdad.github.io/)<sup>3</sup>, [Yeying Jin](https://jinyeying.github.io/)<sup>1,2</sup><br>
> <sup>1</sup>Tencent &nbsp; <sup>2</sup>National University of Singapore &nbsp; <sup>3</sup>The Hong Kong Polytechnic University

## ⚙️ Setup

Tested with Python 3.10 and CUDA 12.8.

```bash
conda create -n minimax_h3 python=3.10 -y
conda activate minimax_h3
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Use the exact DiffSynth revision and the H3-World attention patch.
git clone https://github.com/modelscope/DiffSynth-Studio.git DiffSynth-Studio-h3-v2
git -C DiffSynth-Studio-h3-v2 checkout "$(cat code/diffsynth_base_commit.txt)"
git -C DiffSynth-Studio-h3-v2 apply ../code/diffsynth_h3_action.patch

# Keep Hugging Face, Torch, and Triton caches inside this repository.
source env.sh
```

If `DiffSynth-Studio-h3-v2` already contains downloaded model weights, `git clone` will refuse to use the non-empty directory. Initialize the checkout in place instead:

```bash
git -C DiffSynth-Studio-h3-v2 init
git -C DiffSynth-Studio-h3-v2 remote add origin https://github.com/modelscope/DiffSynth-Studio.git
git -C DiffSynth-Studio-h3-v2 fetch --depth 1 origin "$(cat code/diffsynth_base_commit.txt)"
git -C DiffSynth-Studio-h3-v2 checkout --detach FETCH_HEAD
git -C DiffSynth-Studio-h3-v2 apply ../code/diffsynth_h3_action.patch
```

Run `git -C DiffSynth-Studio-h3-v2 remote add origin ...` only when the directory does not already have an `origin` remote.

Download the required weights into the following locations:

| Asset | Required for | Target location |
| --- | --- | --- |
| [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) FL2VA base weights | inference and training | `DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/` |
| [H3-World LoRA](https://huggingface.co/DANNY621/H3-World) | inference | `checkpoints/H3-World/step-10000.safetensors` |
| [ABot-World-Explorer-500h](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h) | training only | any local path, passed through `ABOT_SRC_ROOT` |

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'MiniMaxAI/MiniMax-H3',
    allow_patterns=['model_index.json', 'FL2VA/*'],
    local_dir='DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3'
)"

python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    'DANNY621/H3-World',
    'step-10000.safetensors',
    local_dir='checkpoints/H3-World'
)"
```

H3-World uses the original `FL2VA` checkpoint layout released by MiniMax, rather than the root-level Diffusers checkpoint. No weight conversion or key remapping is required.

The patch is required for the released checkpoint. Do not install `DiffSynth-Studio-h3-v2` in editable mode; the included training and inference scripts verify that they load the patched checkout.

## 🎬 Inference

The repository includes a held-out ABot test frame at `examples/first_frame.png`. The following fixed configuration generates a 5.2-second forward-motion video:

```bash
python3 code/abot/infer.py \
  --checkpoint checkpoints/H3-World/step-10000.safetensors \
  --first-frame examples/first_frame.png \
  --scene-prompt "A man in a yellow floral shirt stands in a dim, multi-level concrete parking garage." \
  --action-preset forward \
  --seed 2 \
  --steps 50 \
  --num-frames 124 \
  --cfg-scale 1.0 \
  --out outputs/example_forward.mp4
```

The included frame is sample `d0b768c6` from the held-out test split. To use a custom image, replace `examples/first_frame.png` and describe its static scene and subject with `--scene-prompt`. Inputs are center-cropped to 832x480 when needed. The built-in presets are `still`, `forward`, `back`, `strafe-left`, `strafe-right`, `tilt-up`, `tilt-down`, `pan-left`, `pan-right`, `pan-left-fast`, and `pan-right-fast`. The full key-to-language mapping is defined in [`code/abot/action_script.py`](code/abot/action_script.py).

## 🏋️ Training

Prepare the 7,872-clip training split, cache its latents, inject the per-latent action instructions, then train LoRA:

```bash
# 1. Build clips and the fixed train/test split from ABot.
ABOT_SRC_ROOT=/path/to/ABot-World-Explorer-500h \
  python3 code/abot/build_abot_clips.py --num-clips 8000 --workers 48
ABOT_SRC_ROOT=/path/to/ABot-World-Explorer-500h \
  python3 code/abot/build_abot_clips.py --verify 8
python3 code/abot/split_abot_metadata.py \
  --input data/abot_meta_8000.jsonl \
  --train-output data/abot_meta_train_7872.jsonl \
  --test-output data/abot_meta_test_128.jsonl \
  --clips-dir data/clips

# 2. Cache the video, audio, and text latents, then add action text.
bash code/cache.sh
python3 code/abot/inject_abot_text.py \
  --meta data/abot_meta_train_7872.jsonl \
  --cache output/minimax_h3_abot/7872-cache \
  --device cuda:0

# 3. Train on four GPUs by default.
bash code/train.sh
```

`code/train.sh` uses rank-32 LoRA on `qkv_proj` and `out_proj` for 20 epochs, saving checkpoints every 2,000 steps. Override the visible devices with `CUDA_VISIBLE_DEVICES=4,5,6,7 bash code/train.sh`.

## 📄 License

The source code in this repository is licensed under the [Apache License 2.0](LICENSE), except where otherwise noted.

`code/diffsynth_h3_action.patch` contains modifications to [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), which is distributed under the Apache License 2.0.

`examples/first_frame.png` is derived from the [ABot-World-Explorer-500h](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h) dataset and remains subject to its Apache License 2.0.

The H3-World LoRA checkpoint is not covered by this source-code license. It is a Model Derivative of MiniMax-H3 and is distributed subject to the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Use, modification, redistribution, and deployment of the checkpoint and resulting model must comply with that agreement.

## 🙏 Acknowledgements

- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [ABot-World-Explorer-500h](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h)

## 📚 Citation

```bibtex
@misc{chen2026h3worldturninglanguageunderstanding,
      title={H3-World: Turning Language Understanding into World Control},
      author={Danze Chen and Zeqing Wang and Ziyue Lin and Xingyi Yang and Yeying Jin},
      year={2026},
      eprint={2609.01560},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.01560},
}
```
