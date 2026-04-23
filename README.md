# SLAM-ASR-Encoder-Pruning-LoRA

Official code for **"On the Role of Encoder Depth: Pruning Whisper and LoRA Fine-Tuning in SLAM-ASR"** (SPEAKABLE Workshop @ LREC 2026).

**Paper:** [arXiv:2603.27981](https://arxiv.org/abs/2603.27981)

We investigate the role of encoder depth in SLAM-ASR by pruning layers from the Whisper encoder and examining how well LoRA fine-tuning on the downstream LLM compensates for the resulting WER degradation.

- **Models:** Whisper-Small / Medium / Large-v2 + Qwen2.5-3B
- **Languages:** Danish, Dutch, English (low-, mid-, high-resource)
- **Key finding:** removing two encoder layers incurs only 2–4% WER degradation, and combining this pruning with LoRA adaptation surpasses the unpruned baseline while reducing total parameters by 7–14%.

---

## Installation

Python ≥ 3.10 and a CUDA-capable GPU are required.

### Using `uv` (recommended)

```bash
git clone https://github.com/KarthikKolluriKB/SLAM-ASR-Encoder-Pruning-LoRA.git
cd SLAM-ASR-Encoder-Pruning-LoRA
uv sync
```

### Using `pip`

```bash
git clone https://github.com/KarthikKolluriKB/SLAM-ASR-Encoder-Pruning-LoRA.git
cd SLAM-ASR-Encoder-Pruning-LoRA
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -e .
```

### Optional: Weights & Biases

Logging to W&B is enabled by default in the example configs. Provide a key via environment variable (never hardcode it):

```bash
export WANDB_API_KEY=<your_key>          # or put it in a .env file
```

Set `log.use_wandb: false` in any config to disable.

---

## Dataset

We use [Common Voice 22.0](https://huggingface.co/datasets/fsicoli/common_voice_22_0) from the Hugging Face Hub. The helper script downloads audio + transcripts, resamples to 16 kHz, filters by duration, and saves a HuggingFace `DatasetDict` to disk.

```bash
# Download Danish split (default)
python scripts/download_dataset.py --language da

# Limit to 100 hours of training data
python scripts/download_dataset.py --language da --max-hours 100
```

Output goes to `data/cv22_hf/<language>/`, which is the path the example configs expect via `data.hf_dataset_path`.

---

## Training

Each config selects one encoder size (`whisper_small` / `whisper_medium` / `whisper_largev2`) and one of two training modes:

- **`baseline.yaml`** — projector-only (encoder and LLM frozen)
- **`baseline_lora.yaml`** — projector + LoRA adapters on the LLM

```bash
# Projector-only baseline (Whisper-small encoder, Danish)
python train.py --config configs/whisper_small/danish/train/baseline.yaml

# Projector + LoRA on LLM
python train.py --config configs/whisper_small/danish/train/baseline_lora.yaml
```

### Encoder pruning

To reproduce the encoder-depth ablation, set `model.encoder_num_layers: N` in any training config. `null` uses all layers; any integer keeps only the first `N` transformer blocks.

### Sequential runs

To queue multiple experiments with GPU cleanup between runs:

```bash
python scripts/run_sequential_training.py \
    configs/whisper_small/danish/train/baseline.yaml \
    configs/whisper_small/danish/train/baseline_lora.yaml \
    --gap 5 --gpu 0
```

---

## Evaluation

`eval.py` runs real generation (not teacher forcing) on a dataset split and reports WER, CER, and word/char accuracy. It also writes per-sample predictions to a JSONL file.

```bash
python eval.py --config configs/whisper_small/danish/eval/baseline.yaml

# With beam search
python eval.py --config configs/whisper_small/danish/eval/baseline.yaml --num_beams 4

# Override the checkpoint
python eval.py --config configs/whisper_small/danish/eval/baseline.yaml \
    --ckpt_path outputs/whisper_small/danish/baseline/checkpoint_best_wer.pt
```

Queue several evals at once:

```bash
python scripts/run_sequential_evals.py \
    configs/whisper_small/danish/eval/*.yaml \
    --save-results results/summary.json
```

---

## Inference

There is no standalone `inference.py`; single-file transcription is straightforward using `eval.py` with a one-sample dataset, or via the `ASRLLM.generate()` method directly:

```python
from omegaconf import OmegaConf
from models.model import model_builder

cfg = OmegaConf.load("configs/whisper_small/danish/eval/baseline.yaml")
model, tokenizer = model_builder(cfg.train, cfg.model, ckpt_path="path/to/checkpoint_best_wer.pt")
model.eval().to("cuda")

# Build input tensors (audio_mel, input_ids, attention_mask, modality_mask)
# following the same format SpeechDatasetHF.collator produces.
# Then:
ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    audio_mel=audio_mel,
    modality_mask=modality_mask,
    max_new_tokens=128,
    num_beams=2,
)
print(tokenizer.decode(ids[0], skip_special_tokens=True))
```

See `eval.py` for a full end-to-end generation loop.

---

## Project structure

```
.
├── models/                 # Encoder wrapper, projector variants, ASRLLM module
│   ├── encoder.py          # Whisper encoder wrapper with optional top-layer pruning
│   ├── projector.py        # Concat-linear, conv1d, and Q-Former projectors
│   └── model.py            # model_builder() and the ASRLLM nn.Module
├── datamodule/             # Data loading and preparation
│   ├── dataset.py          # SpeechDatasetHF + collator (HF Dataset backend)
│   ├── hf_data.py          # Orchestrator: download + preprocess + save
│   ├── download_data.py    # Common Voice transcript/audio download
│   └── preprocess_data.py  # Text normalization, audio resampling, filtering
├── utils/                  # Metrics, logging, W&B, checkpoint helpers
│   ├── metrics.py          # WER/CER, parameter counting, latency/memory
│   ├── preprocessing.py    # Language-specific text normalizers
│   ├── train_utils.py      # Model-size logging, example dump, W&B tables
│   ├── log_config.py       # Rotating-file + stream logger
│   ├── wand_config.py      # Safe W&B init (reads key from env only)
│   ├── utils.py            # Seeding, device, pad-token, checkpoint saving
│   └── compute_utils.py    # conv output-length helper
├── configs/                # Example YAML configs (Danish)
│   ├── whisper_small/
│   ├── whisper_medium/
│   └── whisper_largev2/
├── scripts/                # CLI helpers
│   ├── download_dataset.py      # Fetch + preprocess Common Voice
│   ├── run_sequential_training.py
│   └── run_sequential_evals.py
├── train.py                # Training entry point
├── eval.py                 # Evaluation entry point (WER/CER + beam search)
├── pyproject.toml
└── LICENSE                 # Apache 2.0
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kolluri2026encoderdepth,
  title         = {On the Role of Encoder Depth: Pruning Whisper and LoRA Fine-Tuning in SLAM-ASR},
  author        = {Kolluri, Ganesh Pavan Kartikeya Bharadwaj and Kampouridis, Michael and Shekhar, Ravi},
  booktitle     = {Proceedings of the SPEAKABLE Workshop at LREC 2026},
  year          = {2026},
  eprint        = {2603.27981},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

---

## License

This project is released under the [Apache License 2.0](LICENSE).
