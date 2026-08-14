# Setup and usage

Everything needed to go from a clean machine to the full set of tables and
figures reported in the paper.

---

## 1. Requirements

| | |
|---|---|
| Python | 3.10 – 3.12 (3.12 used for the reported run) |
| GPU | ≥ 12 GB VRAM for `--preset paper` at 384 px. CPU works but is impractically slow. |
| Disk | ~4 GB (dataset ~2.5 GB, preprocessing cache ~1 GB, checkpoints ~500 MB) |
| RAM | 16 GB |

The reported run used an NVIDIA GB10 (aarch64, compute 12.1) with
`torch 2.13.0+cu130`, but nothing in the code is specific to that hardware.

---

## 2. Get the code

```bash
git clone https://github.com/kowshir-bitto/hemofusion-net.git
cd hemofusion-net
```

---

## 3. Create the environment

**With `venv` (standard):**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**With `uv` (faster, what the reported run used):**

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -r requirements.txt
```

**To reproduce the exact package versions of the reported run:**

```bash
pip install -r requirements-lock.txt
```

### PyTorch and CUDA

`requirements.txt` pins only `torch>=2.6`, because the correct wheel depends on
your CUDA version. If the default PyPI wheel does not match your driver, install
torch first from the right index and then the rest:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.txt
```

Verify the GPU is visible before starting a long run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 4. Get the dataset

The PhysioNet **CT-ICH** cohort is not redistributed with this repository.
Download version 1.3.1 from <https://physionet.org/content/ct-ich/> and place it
**as a sibling of the repository folder**:

```
parent-directory/
├── hemofusion-net/                 <- this repository
│   ├── run_pipeline.py
│   └── src/
└── computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1/
    ├── ct_scans/                   <- 75 NIfTI volumes
    ├── masks/                      <- 75 NIfTI lesion masks
    ├── hemorrhage_diagnosis_raw_ct.csv
    └── Patient_demographics.csv
```

Download from the command line:

```bash
wget -r -N -c -np https://physionet.org/files/ct-ich/1.3.1/
```

To keep the data somewhere else, edit `DATA_ROOT` at the top of
[`src/config.py`](src/config.py) — it is the only place the path is defined.

Check the layout is correct before training:

```bash
python -c "from src.config import CT_DIR, MASK_DIR; import os; print(len(os.listdir(CT_DIR)), 'scans;', len(os.listdir(MASK_DIR)), 'masks')"
```

Expected: `75 scans; 75 masks`.

---

## 5. Run it

### Smoke test first (~10 minutes)

Touches every code path with one epoch, so a broken install fails fast instead of
two hours in:

```bash
python -u run_pipeline.py --preset paper --epochs 1 --no-ssl --workers 2
```

### The reported study (~2.5 hours)

384 px, single patient-wise split, 30 epochs, four baselines plus the proposed
model:

```bash
python -u run_pipeline.py --preset paper --workers 6
```

The first invocation builds the preprocessing cache (NIfTI → npz), which takes
several minutes; later runs at the same resolution reuse it.

### Other runs

Validation-only tuning of the proposed model (never touches test):

```bash
python -u ab_tune.py --arms capacity,slim_reg,slim_reg_pos70 --epochs 30
```

The 15-variant ablation grid, kept separate so it can be scheduled independently:

```bash
python -u run_pipeline.py --preset paper --stages ablation --ablation-epochs 15
```

Full 5-fold patient-wise cross-validation (overnight):

```bash
python -u run_pipeline.py --preset full --workers 6
```

Long runs are best detached:

```bash
nohup python -u run_pipeline.py --preset paper --workers 6 > outputs/logs/run.log 2>&1 &
```

---

## 6. Command-line reference

| Flag | Default | Meaning |
|---|---|---|
| `--preset` | `paper` | `smoke`, `quick`, `paper`, or `full` |
| `--stages` | `data,train,results,stats,qualitative,xai` | comma-separated subset to run; add `ablation` to include the grid |
| `--folds` | preset | override folds, e.g. `0,1,2` |
| `--epochs` | preset | override the training budget |
| `--ablation-epochs` | preset | shorter budget for ablation variants |
| `--img-size` | preset | 256 or 384 |
| `--batch-size` | preset | lower this first if you hit OOM |
| `--workers` | preset | dataloader workers |
| `--models` | all | restrict to a comma-separated subset, e.g. `unet,hemofusion` |
| `--no-ssl` | off | skip SimCLR pretraining |
| `--no-ablation` | off | force the ablation grid off |
| `--reuse-trained` | off | skip training where a checkpoint already exists; re-evaluate only |

### Stages

| Stage | Produces |
|---|---|
| `data` | dataset description, class distribution, preprocessing and augmentation figures, skull-strip audit |
| `train` | SimCLR pretraining, every model, checkpoints, per-slice prediction CSVs |
| `results` | performance tables, model-comparison and training-curve figures |
| `stats` | Wilcoxon, bootstrap CIs, Friedman/Nemenyi, DeLong, McNemar, Bland–Altman |
| `ablation` | the 15-variant grid and its significance table |
| `qualitative` | qualitative panels and threshold sensitivity |
| `xai` | Grad-CAM / Grad-CAM++ / LayerCAM and the faithfulness metrics |

Every stage persists its artefacts, so later stages re-run without retraining:

```bash
python -u run_pipeline.py --preset paper --stages results,stats
```

This regenerates every table and figure from the existing
`outputs/predictions/*.csv` in under a minute — which is how the numbers in the
README can be verified without a GPU.

---

## 7. Where the results land

```
outputs/
├── ICH_Hybrid_Results_paper.xlsx   every table, one sheet each
├── RESULTS_paper.md                paste-ready results section
├── config_paper.json               the exact configuration of the run
├── PAPER_Methods_Results_Discussion.md / .docx
├── tables/       T00–T25 as flat CSV
├── figures/      F01–F25 as 300 dpi PNG and vector PDF
├── predictions/  per-slice segmentation and classification scores
├── logs/         per-epoch history CSVs
├── models/       checkpoints (not versioned)
└── cache/        preprocessed npz slices (not versioned)
```

The workbook gathers **every** CSV in `outputs/tables`, so splitting the study
across several invocations still yields one complete result file.

Rebuild the Word manuscript draft after editing the markdown:

```bash
pip install python-docx
python make_docx.py
```

---

## 8. Tests

```bash
pip install pytest
pytest tests/ -v
```

`tests/test_stats_metrics.py` covers the metric implementations and the
statistical routines against hand-computed values — worth running after any edit
to `src/metrics.py` or `src/stats.py`.

---

## 9. Notebook

`notebooks/ICH_Hybrid_Report.ipynb` walks through the results narratively,
reading the CSVs in `outputs/`. It needs no GPU. Regenerate it after changing the
narrative with:

```bash
python notebooks/build_notebook.py
```

---

## 10. Troubleshooting

**`CUDA out of memory`** — lower the batch size, then the resolution:

```bash
python -u run_pipeline.py --preset paper --batch-size 4
python -u run_pipeline.py --preset paper --img-size 256 --batch-size 8
```

**`FileNotFoundError` on `ct_scans`** — the dataset is not where `src/config.py`
expects it. Re-check the layout in §4; the folder must be a *sibling* of the
repository, not inside it.

**First epoch is very slow** — the preprocessing cache is being built. It is
written once per `(img_size, skull_strip, clahe)` combination into
`outputs/cache/` and reused afterwards.

**`timm` cannot download pretrained weights** — the encoders are fetched from
Hugging Face on first use, so the machine needs network access once. Pre-warm it
with:

```bash
python -c "import timm; timm.create_model('resnet34', pretrained=True); timm.create_model('pvt_v2_b1', pretrained=True)"
```

**Dataloader workers hang on Windows** — use `--workers 0`, or run under WSL.

**Results differ slightly between runs** — the seed is fixed at 42, but cuDNN
autotuning and non-deterministic GPU reductions leave small variation. The
statistical conclusions are unaffected; the exact configuration of the reported
run is frozen in `outputs/config_paper.json`.
