## FLODA: FLorence-2 Optimized for Deepfake Assessment

This repository provides a baseline using the **FLODA (FLorence-2 Optimized for Deepfake Assessment)** vision-language model for detecting AI-generated images in the [OpenFake](https://huggingface.co/datasets/CDL-AMLRT/OpenFake) dataset. 

### Run Inference + Evaluation
```bash
python baselines/Floda/eval.py \
  --in_csv data/OpenFake/test/metadata.csv \
  --out_csv baselines/Floda/floda_output_predictions.csv \
  --root_dir data/OpenFake \
  --device cuda
```

## Eval Results
Accuracy: 73.81%

| Class        | Precision | Recall  | F1-score |
|--------------|-----------|---------|----------|
| AI-generated | 0.7381    | 0.7383  | 0.7382   |
| Real         | 0.7382    | 0.7380  | 0.7381   |
