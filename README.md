# Face-based Identity Recognition (CSCI218)

This project compares two offline face recognition pipelines on the AT&T (ORL) dataset in Google Colab.

## Pipelines
- **Pipeline A (Baseline):** PCA (Eigenfaces-style) + Linear SVM  
- **Pipeline B (Winner):** Pretrained CNN embeddings (ResNet18 feature extractor) + Linear SVM
- **Pipeline C (New):** MLP (Multi-Layer Perceptron) on flattened pixels (with standardization)

## Dataset
- AT&T (ORL) Faces (40 subjects × 10 images each)

## Evaluation
- Identity-stratified split: **7 train / 3 test per subject**
- Metrics: Accuracy, Macro Precision/Recall/F1, Confusion Matrix
- Runtime: train + inference time
- Extra plots: per-seed comparisons, boxplots, per-class accuracy

## How to run (Google Colab)
1. Open the notebook: `T05_FaceRec_2Pipelines_Comparison.ipynb`
2. Run cells top → bottom
3. Upload the ORL dataset ZIP when prompted (must contain `s1/ ... s40/` with `.pgm` files)

## Outputs
- Results saved to `/content/results`
- Final download: `results.zip` (plots + CSV summaries)

## Results (CSVs)
- [runs_all_seeds.csv](runs_all_seeds.csv)
- [summary_mean_std.csv](summary_mean_std.csv)

