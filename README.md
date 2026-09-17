# Breast-Ultrasound Malignancy Risk Score

| | |
| --- | --- |
| Final rank | #19 |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 1 |
| Last submission | 2026-06-27 |

## Problem statement

### Worth the Biopsy? — Decision-Band Actionability of a Malignancy Risk Score

### Task

You are given de-identified **B-mode breast-ultrasound images**, each of a lesion that is
 benign or malignant. For each test image you output a single **calibrated risk** that the
 lesion is **malignant** (`risk` ∈ [0, 1]).

This is **not** scored by accuracy or AUC. You are scored by the **actionability** of your
 risk — how much real decision value it delivers when used to make the actual call (*biopsy*
 *this lesion, or not?*) across a whole **band of operating points**, and how well it holds up
 at its *weakest* operating point, not just on average.

---

### Data

Each image is a single **B-mode (grayscale) breast-ultrasound** frame showing one lesion,
 acquired during routine diagnostic exams and labelled **benign** or **malignant** by expert
 review. Every image is **de-identified** and then put through a strong per-image
 **obfuscation** — random crop, horizontal flip, brightness/contrast jitter, additive noise,
 down-scaling and lossy re-encoding — and assigned a fresh random id, so a served image cannot
 be traced back to any source while the lesion appearance that drives the label is preserved.
 All served images are 8-bit **RGB JPEG, 112×112 pixels**.

Everything is in `public/`:

```
public/
├── images/                 # all lesion images, <image_id>.jpg (train + test together)
├── train.csv               # labelled training rows
├── test.csv                # the image_ids to score
└── sample_submission.csv   # a valid, ready-to-overwrite example submission
```

`**images/**` — **927** images in total, one per lesion, named `<image_id>.jpg`, 8-bit RGB
 JPEG at **112×112**. The folder holds **both** the training and the test images.

`**train.csv**` — one row per training image (**570 rows**). Two columns:

- `**image_id**` — data type **string**. Opaque id; matches `images/<image_id>.jpg`.
- `**label**` — data type **integer** (target). **1** if the lesion is **malignant**, **0** if benign.

`**test.csv**` — one row per test image (**357 rows**). One column:

- `**image_id**` — data type **string**. The image to risk-score (its label is hidden).

`**sample_submission.csv**` — a valid, ready-to-overwrite example (**357 rows**); columns
 `image_id, risk`. It is a coarse demo baseline (format reference only).

**Dataset size:** **927 images total** — **570 labelled training** and **357 test**. Malignant
 prevalence is ≈ **37 %**. The split is by **patient**: every image of a given patient goes
 entirely to train **or** entirely to test (disjoint), so no patient appears on both sides.

---

### Submission format

A CSV with a header and **exactly one row per test `image_id**`:

```
image_id,risk
```

- Provide a `risk` for **every** `image_id` in `test.csv` — duplicate, unknown, or missing
    ids are **rejected**. `risk` is coerced numeric, NaN/inf → 0, clipped to [0, 1].
- Submit a probability in `[0, 1]`, not a 0/1 label — the metric uses the *value* of your
    risk at the operating points, not just its ranking.

**Example**

```
image_id,risk
scn-596-cplw1,0.08
scn-558-deid9,0.91
scn-627-f3my8,0.34
```

> **Note:** `image_id`s are **random** (`scn-<n>-<rand>`) and the images are **obfuscated**
>  — they cannot be matched to any public source. Recovering the malignancy label by matching
>  to the original dataset, or importing its withheld lesion-geometry, is **prohibited**
>  (`WHAT_NOT_TO_USE.md`); your risk must come from a model trained on the provided images.

---

### Scoring

At an operating point `p_t`, a decision is taken when `risk ≥ p_t`. For each `p_t` over the
 band `[0.20, 0.60]`, with `n` test images and prevalence `prev`:

```
yield(p_t)   = TP/n − (FP/n) · p_t/(1 − p_t)
trivial(p_t) = max( prev − (1 − prev) · p_t/(1 − p_t),  0 )        # best act-all / act-none policy
s(p_t)       = clip( (yield − trivial) / (prev − trivial), 0, 1 )  # standardised decision yield in [0,1]
```

The per-operating-point yields `s(p_t)` are folded into a single index that rewards delivering
 value **across the whole band** *and* **at the weakest operating point**, then warped through a
 steep, endpoint-matched response (the index uses cosine, logarithm, exponential, tangent and sine):

```
A     = Hann-cosine-windowed mean of s(p_t) over the band     # emphasise the pivotal centre
m     = −(1/T) · log( mean exp(−T · s(p_t)) )                  # soft worst operating point (→ min)
base  = 0.6·A + 0.4·m
o     = tan(W·base) / ( tan(W·base) + tan(W·(1−base)) )        # tangent-odds S-warp
score = sin( (π/2) · o )                                       # in [0, 1], higher is better
```

### What you may and may not use

This is a **machine-learning vision** challenge: learn from the training images, then output a
 malignancy risk per test image whose value is usable for the decision. The rules keep it about
 modelling the provided images, not reverse-engineering or label lookup.

### PROHIBITED — source lookup & label recovery (automatic disqualification)

The images are **obfuscated** (random crop, flip, brightness/contrast jitter, noise,
 down-scaling, lossy re-encoding) and carry **fresh random ids** (`scn-<n>-<rand>`), specifically
 so the served images cannot be traced back to any original source. You must **not** attempt to
 undo or circumvent this. In particular:

- **Matching / re-identifying** the images against the original public dataset or ANY external
    source — by pixel/byte comparison, perceptual or cryptographic hashing, nearest-neighbour
    search, reverse-image-search, EXIF/metadata, or learned-feature similarity — to recover the
    malignancy label.
- **Using labels obtained from an external source**: any public dataset, atlas, search engine,
    API, or model that was trained on, or returns, the original labels for these images. Your risk
    must be **inferred by a model trained only on the provided training data (the images and labels**
    **in `train.csv`), with generic pretrained weights permitted as a starting point**.
- **Cached / hard-coded / memorised labels** of any kind.
- **Parsing the `image_id**` or any id/order pattern to infer the label (ids are random).
- **Grader gaming**: exploiting the scoring arithmetic, the operating-point band, the
    act-all/act-none reference, validation, or numeric edge cases instead of modelling.
    (Act-all, act-none, a constant risk, or random risks score 0 by design — there is no shortcut.)
- **Reconstructing or using the segmentation masks / lesion geometry** from the source dataset
    (they are deliberately withheld). Lesion size is a known malignancy correlate; importing it
    from the source is label leakage, not modelling.
- Manually labelling the test images by a human (or an external service/API).

Submissions found to rely on any of the above are out of scope and may be rejected.

### Allowed

- **Any model that learns** the malignancy risk from the provided `train.csv` images — you
    choose the architecture, representation, training, and how you produce the risk value.
- General-purpose ML / vision libraries — `numpy`, `pandas`, `scikit-learn`,
    `PyTorch`/`TensorFlow`, `Pillow`, `scikit-image`, `opencv` and similar (generic pretrained
    weights are fine; a model that returns these images' original malignancy labels is not).

### The spirit of the task

The goal is a per-image malignancy risk, learned from the **provided images**, whose *value* is
 usable for the decision across plausible operating points. The score must come from modelling the
 provided images — never from recovering the labels by lookup or matching, or by importing the
 withheld lesion geometry.
