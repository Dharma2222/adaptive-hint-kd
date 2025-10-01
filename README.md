Here’s a clean **README.md** you can drop into your zip. It’s written for graders: what this is, how to run it, what to look at, and the results.

---

# Lightweight Knowledge Distillation with Adaptive Hints & Selective Layers

**Authors:** Nakul Patel [B01025608], Dharma Kevadiya[B01034333]
**Course Mini-Project**

## TL;DR (Abstract)

We distill a **5-layer CNN (\~1.3M params)** from a **ResNet-18 teacher (\~11M)** on **CIFAR-10**.
Two light tweaks make distillation simple and effective:

* **Adaptive-β:** start with strong hint loss, then **halve β** whenever KD loss plateaus (no extra networks).
* **Selective hints:** copy **only 3 teacher layers** chosen by activation variance (one quick scan).

Across 20–40 epochs, these heuristics deliver **+8–9 pp** over a no-hint student, with **lower memory** than matching all layers.

---

## 1) Problem & Motivation

Big CNNs are accurate but heavy; edge devices (phones, cams, drones) need **small, fast** models.
**Goal:** keep accuracy high while shrinking compute/memory using **knowledge distillation (KD)**.

---

## 2) Method (One Page)

**Teacher:** ResNet-18 (ImageNet weights) → fine-tuned **7 epochs** on CIFAR-10 (\~95% Top-1).
**Student:** 5-conv “ResNet-Lite” (\~1.3M params).

**Losses (combined as α·KD + β·Hint + γ·CE):**

* **Hint (MSE):** match student/teacher feature maps at selected layers.
* **KD (KL on logits, T=4):** match teacher’s soft probabilities.
* **CE:** standard cross-entropy with labels.

**Two Tweaks**

* **Adaptive-β:** β=250 initially; **halve** when validation KD loss doesn’t improve for **2 epochs**.
* **Selective hints:** pass 10 batches through the teacher, compute **activation variance** per block, keep **top-3** blocks.

---

## 3) Repo / Files in this zip

```
/README.md                      ← this file

/SGDmomentum_40/
   train_distill.py             ← local script (adaptive β + selective hints)
/distillation_scaling_40_epoch/  ← baseline student (KD+CE from start) + adaptive β + selective hints
   training_log_adaptive.csv
   training_log_selective.csv
   evaluation_results.csv
/distillation_scaling_20_epoch/  ← baseline student (KD+CE from start) + adaptive β + selective hints
   training_log_adaptive.csv
   training_log_selective.csv
   evaluation_results.csv
/figs/
   beta_timeline.png            ← β vs epochs
   feature_similarity.png       ← cosine sim vs epochs
   training_curves_20.png       ← acc/KD/hint (20 ep)
   training_curves_40.png       ← acc/KD/hint (40 ep)
```

> If you only submit notebooks: keep the same subfolder names; graders can open `colab_notebooks/*`.

---

## 4) How to run

### A) Colab (recommended for grading)

1. Open the desired notebook in `/colab_notebooks/`.
2. Run all cells (downloads CIFAR-10 automatically).
3. Plots and CSV logs will be created in `/results`.

### B) Local (optional)

```bash
pip install torch torchvision tqdm pandas numpy
python local/train_distill.py
```

* Uses CUDA if available, else CPU.
* Outputs checkpoints to `./checkpoints/` and a CSV log to `/results`.

---

## 5) Hyper-parameters (key ones)

* **Batch size:** 64
* **Optimizer:** Adam (lr=1e-3)
* **Temperature (KD):** T=4
* **β initial:** 250 (adaptive halving on KD plateau ≥2 epochs)
* **Selective scan:** 10 mini-batches; keep top-3 teacher blocks by variance
* **Epochs:** 20 and 40 tested

---

## 6) Results

### 20-Epoch comparison

| Method                                      |   Acc (%) |      Loss |    Params |
| ------------------------------------------- | --------: | --------: | --------: |
| Baseline (KD+CE from start)                 |     74.05 |     1.137 |     4.27M |
| **Static-β warm-up** (hint-only first 5 ep) | **79.50** | **0.840** |     4.27M |
| **+ Adaptive-β**                            | **81.37** | **0.702** |     4.27M |
| **+ Selective hints**                       | **80.88** | **0.732** | **4.20M** |

**Takeaway:** hint pretraining alone = **+5.5 pp** vs baseline; adaptive-β adds **+1.9 pp** more. Selective nearly matches adaptive with **\~1.5% fewer params**.

### 40-Epoch comparison

| Method                |   Acc (%) |      Loss |    Params |
| --------------------- | --------: | --------: | --------: |
| Baseline              |     73.92 |     1.161 |     4.27M |
| **+ Adaptive-β**      | **82.85** | **0.681** |     4.27M |
| **+ Selective hints** | **82.49** | **0.673** | **4.20M** |

**Takeaway:** distilled models gain **\~+1.5 pp** from 20→40 epochs; baseline **stagnates/overfits**.

---

## 7) What the curves show (figs/)

* **beta\_timeline.png:** β halves once early; then stays low (stable guidance).
* **feature\_similarity.png:** student/teacher feature cosine sim steadily rises (feature alignment working).
* **training\_curves\_20/40.png:**

  * **Accuracy:** adaptive > selective > baseline by epoch 20; gap persists at 40.
  * **KD loss:** drops faster after hint stage for adaptive/selective.
  * **Hint loss:** low and decreasing for hint-based models; baseline drifts.

---

## 8) Why it works (one-liners)

* **Hint loss** builds good features early (“learn how”).
* **KD loss** transfers teacher’s soft confidence (“which classes are similar”).
* **CE loss** anchors to true labels.
* **Adaptive-β** stops over-relying on hints once they stop helping.
* **Selective hints** remove noisy/low-value layers and save memory.

---

## 9) Limitations & Future Work

* Only CIFAR-10 tested; teacher–student gap still **\~12 pp**.
* Variance scan adds a small one-time cost (\~1 min).
* **Next:** try 1-D sensor data (e.g., crash pulses), add **INT8 quantization/pruning**, explore **dynamic re-selection** of hint layers mid-training.

---

## 10) Extra Experiments (quick notes)

* **Adam vs SGD:** Adam converged \~2× faster on the student; SGD+momentum needs more LR tuning.
* **LR search:** 5e-4 slowed convergence; 2e-3 unstable—1e-3 best trade-off.
* **Temperature:** T=2 reduced KD gains by \~0.5 pp; T=4 kept them strong.
* **β patience:** 1-epoch patience was noisy (+\~0.3 pp); **2-epoch** is stable.
* **Hint count:** 2 layers saved \~50% hint memory but cost \~0.7 pp accuracy.
* **Quantization:** 8-bit post-KD cut size \~4× with <0.5 pp accuracy loss.

---

## 11) Reproduce our numbers (quick path)

1. **Teacher fine-tune (7 ep)**: run the first cells of any notebook to load ImageNet weights and fine-tune on CIFAR-10.
2. **20 ep runs:** execute notebooks for baseline, static-β, adaptive-β, selective; copy the printed accuracy/loss into the results table.
3. **40 ep runs:** re-run adaptive-β and selective for 40 epochs; export logs to `/results`.
4. Compare CSV logs; regenerate plots from the plotting cells.


## 12) Environment

* Python ≥ 3.10, PyTorch ≥ 2.0, torchvision ≥ 0.15, numpy, pandas, tqdm.
* GPU: NVIDIA T4/RTX recommended; CPU works but slower.

---

## 13) License & Acknowledgements

* Code in this project is for academic use.
* Thanks to the authors of ResNet and CIFAR-10 datasets and to our course staff.

---

**Contact:** kevadiyadharma@gmail.com   
**Last updated:** 08/07/2025

---

