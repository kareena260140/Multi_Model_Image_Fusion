# 🧠 Brain Tumor Segmentation

A deep learning model that automatically detects and segments brain tumors from MRI scans.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-BraTS%202020-blue?style=flat)

---

## What does this project do?

It takes MRI brain scans as input and draws a mask over the tumor region — similar to how a radiologist would manually outline it, but done automatically by the model.

The model looks at 4 different types of MRI scans of the same brain at once (T1, T1CE, T2, FLAIR) because each one highlights different parts of the tumor.

---

## How to run it

**Step 1 — Clone the repo**
```bash
git clone https://github.com/kareena260140/brats_segmentation.git
cd brats_segmentation
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Set up Kaggle (to download the dataset)**
1. Go to [kaggle.com](https://kaggle.com) → Account → Create API Token
2. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`

**Step 4 — Run**
```bash
python main.py
```

That's it. The script will download the data, train the model, and save the results.

> **Using Google Colab?** Open `notebooks/run_pipeline.py` and paste each block into a Colab cell. A free GPU is enough.

---

## Project structure

```
brats_segmentation/
│
├── configs/config.py        ← change settings here (batch size, epochs, etc.)
├── data/
│   ├── loader.py            ← downloads and reads the dataset
│   └── preprocessing.py     ← cleans and prepares the images
├── models/
│   ├── blocks.py            ← small reusable pieces of the neural network
│   └── unet.py              ← the full model
├── training/
│   ├── losses.py            ← how the model measures its own mistakes
│   ├── callbacks.py         ← auto-saves best model, stops early if needed
│   └── trainer.py           ← runs the training loop
├── utils/visualize.py       ← plots results
└── main.py                  ← run this to start everything
```

---

## Dataset

Uses the **BraTS 2020** dataset from Kaggle — a standard benchmark for brain tumor segmentation with MRI scans from real patients.

- 4 MRI types per scan: T1, T1CE, T2, FLAIR
- Ground truth masks labeled by expert radiologists
- Download link: [kaggle.com/datasets/awsaf49/brats2020-training-data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)

---

## Results

| Metric | Score |
|---|---|
| Dice Score | ~0.87 |
| IoU | ~0.78 |
| Recall | ~0.89 |

Dice Score is the main one to look at — 1.0 is perfect, 0.0 is completely wrong. 0.87 means the model's predicted tumor region overlaps well with the real one.

---

## Requirements

- Python 3.9+
- TensorFlow 2.12+
- A GPU (Google Colab free tier works fine)

---

## References

- BraTS 2020 Dataset — Menze et al., IEEE TMI 2015
- Attention U-Net — Oktay et al., MIDL 2018
- CBAM — Woo et al., ECCV 2018
- Focal Tversky Loss — Abraham & Khan, ISBI 2019
