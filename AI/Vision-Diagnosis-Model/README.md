# Vision Diagnosis Model (AgriGuard Voice)

Leaf disease and pest classifier for **rice and corn** (10 classes), built by fine-tuning
**BioCLIP 2** (ViT-L/14) on public datasets. On top of the classifier the pipeline fits the
calibration and uncertainty layers that produce the `s_vis` signal for the AgriGuard Voice safety
gate:

- Linear probe on frozen embeddings, then LoRA fine-tuning warm-started from that probe (LP-FT)
- Temperature scaling with ECE, adaptive ECE, classwise ECE, NLL, Brier and reliability diagrams
- Selective prediction (risk-coverage curves, AURC)
- Out-of-distribution detection (MSP, energy, KNN) against near-OOD and far-OOD pools
- Conformal prediction (THR, APS, RAPS) at a 90% target coverage
- Exported artifacts for serving: LoRA + head weights, calibration file, KNN bank

## Results

Test-set numbers from a full run with the default configuration. The final model is selected on
validation macro-F1, and the test split is only used for these final numbers.

| Metric (test split) | Linear probe | LoRA (selected) |
|---|---|---|
| Accuracy | 0.8383 | **0.9643** |
| Macro-F1 | 0.8309 | **0.9672** |
| Balanced accuracy | 0.8298 | 0.9664 |
| Temperature | 0.7176 | 1.5257 |
| ECE before → after scaling | 0.1039 → 0.0398 | 0.0227 → **0.0131** |
| NLL before → after scaling | 0.5282 → 0.4845 | 0.1211 → 0.0983 |
| Brier score | 0.2388 | 0.0534 |
| AURC | 0.0376 | 0.0018 |
| near-OOD AUROC / far-OOD AUROC | 0.7789 / 0.9332 | 0.8605 / **0.9940** |
| RAPS coverage (target 0.90) | 0.9765 | 0.9793 |
| RAPS average set size | 2.556 | 1.093 |

Per-class test F1 for the selected model ranges from 0.922 (rice blast) to 1.000 (corn healthy and
common rust). All 38 test errors are rice-versus-rice confusions, mostly hispa against blast.

Run profile: Tesla T4 (16 GB), 2.4 hours end to end, LoRA batch size auto-selected at 64 (128 hit
OOM, 96 needed 87% of memory), ~42 images/s, 8.97 GB peak. Splits contained 7,647 train / 1,077
validation / 1,096 calibration / 1,064 test images plus 600 + 600 near-OOD and 680 + 680 far-OOD
images; grouping merged 1,727 near-duplicate pairs before splitting.

KNN is a close alternative to the selected energy score (near-OOD AUROC 0.8608, far-OOD 0.9992);
the pipeline picks whichever method has the better mean AUROC on the OOD validation halves.

## Repository layout

```
Vision-Diagnosis-Model/
├── dataset/
│   ├── paddy_doctor/                 images/<label>/*.jpg, metadata.csv, dataset_info.json
│   └── plantvillage/                 same format
├── scripts/
│   ├── download_datasets.py          fetch the raw releases from Hugging Face
│   ├── prepare_datasets.py           convert them into the shared format
│   ├── config.py                     paths, classes and all hyperparameters
│   ├── utils.py                       logging, seeding, metrics, calibration metrics
│   ├── data.py                        loading, grouping, splitting, transforms, features
│   ├── model.py                       backbone, LoRA layers, head, batch-size probe
│   ├── calibration.py                temperature scaling, OOD scores, conformal, gate mapping
│   ├── plots.py                       figures
│   ├── train.py                       linear probe and LoRA training
│   └── test.py                        evaluation, artifact export, summary
├── outputs/                          run directories
├── requirements.txt
└── readme.md
```

## Datasets

| Dataset | Source | Images | Classes | Used for | License |
|---|---|---|---|---|---|
| Paddy Doctor | [Project-AgML/paddy_disease_classification](https://huggingface.co/datasets/Project-AgML/paddy_disease_classification) (labelled train split of the Kaggle *Paddy Disease Classification* competition) | 10,407 field photos at 480×640 | 10 | 6 in-distribution rice classes (normal, blast, bacterial leaf blight, brown spot, tungro, hispa); the other 4 become the near-OOD pool | See the original Paddy Doctor release |
| PlantVillage (color) | [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) | 54,305 lab photos at 256×256 | 38 | 4 in-distribution corn classes (healthy, gray leaf spot, common rust, northern leaf blight); the other crops become the far-OOD pool | CC BY-SA 3.0 |

Both datasets are converted to one shared layout so the training code treats them identically:

```
<dataset>/
├── images/<label>/<image_id>.<ext>
├── metadata.csv        image_id, relpath, label, group_id, source, original_name, width, height
└── dataset_info.json   source, license, citation, per-class counts
```

`group_id` is the physical leaf id from PlantVillage's `leaf-map.json`. Paddy Doctor ships no
grouping information, so there `group_id` equals `image_id` and near-duplicate grouping is left to
the perceptual hash step.

## Usage

```bash
pip install -r requirements.txt
python scripts/download_datasets.py     # raw releases -> dataset/_raw
python scripts/prepare_datasets.py      # shared format -> dataset/paddy_doctor, dataset/plantvillage
python scripts/train.py                 # linear probe + LoRA
python scripts/test.py                  # calibration, OOD, conformal, artifacts
```

`train.py --stage {all,linear_probe,lora}` runs a single stage, and `test.py --models lora`
evaluates a subset. Because `test.py` works from the embeddings saved during training, it needs
neither a GPU nor the backbone weights, so calibration can be re-fit in seconds.

Paths live at the top of `config.py` (`DATA_ROOT`, `OUTPUT_ROOT`, `PADDY_DIR`,
`PLANTVILLAGE_DIR`) and can be overridden with the `AGV_DATA_ROOT`, `AGV_OUTPUT_ROOT`,
`AGV_PADDY_DIR` and `AGV_PLANTVILLAGE_DIR` environment variables. Setting `AGV_SMOKE_TEST=1`
switches to a tiny CPU configuration (ViT-B/32 with random weights, a handful of images per class)
that only checks that the pipeline runs:

```bash
AGV_SMOKE_TEST=1 AGV_OUTPUT_ROOT=outputs/smoke_test python scripts/train.py
AGV_SMOKE_TEST=1 AGV_OUTPUT_ROOT=outputs/smoke_test python scripts/test.py
```

Hardware: a 16 GB GPU is enough for the default settings; the LoRA stage probes the candidate batch
sizes and keeps the largest one whose peak memory stays under 85% of VRAM, so it adapts to smaller
cards. Downloading the BioCLIP 2 weights (~1.7 GB) requires internet access on the first run.

## Method

| Step | Setting |
|---|---|
| Classes | 10 (6 rice, 4 corn), capped at 1,500 images per class |
| Leakage control | Split groups are the union of `group_id` and perceptual-hash clusters (Hamming ≤ 4); `StratifiedGroupKFold` over 10 folds gives 7 train / 1 validation / 1 calibration / 1 test, and the code asserts that no group spans two splits |
| Validation split | Early stopping, final model selection, OOD thresholds at 95% TPR |
| Calibration split | Temperature, conformal quantiles and the RAPS regularization rank |
| Test split | Final metrics only |
| OOD pools | near-OOD: the 4 unused Paddy Doctor classes (≤ 300 per class); far-OOD: 34 non-corn PlantVillage classes (≤ 40 per class). Each pool is halved by group: one half selects the scoring method, the other reports |
| Stage 1 | Linear probe on L2-normalized frozen embeddings; AdamW, lr 3e-3, batch 1024, ≤ 500 epochs, patience 40 |
| Stage 2 | LoRA (rank 16, alpha 32, dropout 0.05) on `mlp.c_fc` and `mlp.c_proj` of the last 12 blocks, head warm-started from stage 1; AdamW with lr 2e-4 (adapters) and 5e-4 (head) at the reference batch of 64, scaled by the square root of the actual batch; 1 warmup epoch then cosine decay, ≤ 40 epochs, patience 8, float16 autocast, gradient clipping at 1.0 |
| Augmentation | RandomResizedCrop (0.35–1.0), horizontal and vertical flips, colour jitter, Gaussian blur, JPEG compression, random erasing |
| Loss | Cross-entropy weighted by square-root inverse class frequency |
| Model selection | Validation macro-F1, with validation NLL as tie-break |

## Outputs

| Directory | Contents |
|---|---|
| `logs/` | `train.log`, `test.log`, resolved `config.json`, `environment.json`, `output_manifest.csv` |
| `splits/` | Per-image split assignment, per-class counts, grouping statistics |
| `checkpoints/` | `linear_probe_best.pt`, `lora_best.pt`, `lora_last.pt` (with optimizer, scheduler and scaler state) |
| `features/` | float16 embeddings of every split for both models, with the image ids they belong to |
| `metrics/` | Training histories, `training_summary.json`, per-model reports (classification, per-class, calibration, selective prediction, OOD, conformal, gate preview), confusion matrices, `summary.csv` / `summary.md` / `summary.json` |
| `predictions/` | Per-image predictions on test and both OOD test pools: top-3, conformal set, OOD scores, OOD flag, gate level |
| `plots/` | Training curves, confusion matrices, reliability diagrams, risk-coverage curves, OOD score histograms, conformal set-size histograms |
| `artifacts/` | `vision_model_v1.pt`, `vision_calib_v1.json`, `knn_bank_v1.npz` |

## Serving artifacts

`vision_model_v1.pt` stores the trained tensors only (48 LoRA tensors plus the head, ~8 MB), so
serving rebuilds the backbone and applies them:

1. Create the backbone named in the artifact (`hf-hub:imageomics/bioclip-2`) with open_clip.
2. If `lora` is not null, inject LoRA layers using its `config` and `modules` entries, then load
   `state_dict`.
3. Preprocess images with the `preprocess` block of `vision_calib_v1.json` (resize the shortest
   side, center crop, normalize), L2-normalize the embedding and apply the head.
4. Divide the logits by `temperature` for calibrated probabilities.
5. Compute the OOD score named in `ood.method` and compare it against `ood.thresholds`; scores are
   oriented so that higher means more in-distribution. The KNN score needs `knn_bank_v1.npz`.
6. Build the conformal prediction set with the `conformal` block (RAPS, always including the top-1
   class).

## Limitations

- Crop and image source coincide: rice comes only from field photos and corn only from lab photos,
  so telling rice from corn is close to trivial and the meaningful signal is disease separation
  within one crop.
- No domain-shift test set yet (Mendeley rice, PlantDoc, CD&S would fill this gap).
- Near-OOD detection is moderate: at the 95%-TPR threshold only about half of the unseen rice
  diseases are flagged, and `dead_heart` is the weakest case, so unseen rice diseases can still be
  answered confidently.
- Paddy Doctor has no leaf ids, so its leakage control relies on perceptual hashing alone.
- LoRA adapts the MLP projections rather than the attention q/v tensors, because open_clip fuses
  q/k/v inside `nn.MultiheadAttention`.
- Temperature, OOD thresholds and conformal quantiles are fit on these splits and would need
  re-fitting for a different image distribution.

## Citations

- Petchiammal A, Briskline Kiruba S, Murugan D, Pandarasamy Arjunan (2022). *Paddy Doctor: A Visual
  Image Dataset for Automated Paddy Disease Classification and Benchmarking*. IEEE Dataport.
  https://dx.doi.org/10.21227/hz4v-af08
- Mohanty S. P., Hughes D. P., Salathé M. (2016). *Using Deep Learning for Image-Based Plant Disease
  Detection*. Frontiers in Plant Science 7. https://doi.org/10.3389/fpls.2016.01419
- Gu J. et al. (2025). *BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive
  Learning*. https://arxiv.org/abs/2505.23883
