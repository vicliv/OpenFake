## FLODA: FLorence-2 Optimized for Deepfake Assessment

This repository provides a baseline using the **FakeVLM (Large Multimodal Model-Based
Synthetic Image Detection with Artifact Explanation)** vision-language model for detecting AI-generated images in the [OpenFake](https://huggingface.co/datasets/CDL-AMLRT/OpenFake) dataset. 

### Run Inference + Evaluation
```bash
python baselines/FakeVLM/eval.py \
  --in_csv data/OpenFake/test/metadata.csv \
  --out_csv baselines/FakeVLM/fakevlm_output_predictions.csv \
  --root_dir data/OpenFake \
  --device cuda
```

## Eval Results
Accuracy: 83.46%

| Class        | Precision | Recall  | F1-score |
|--------------|-----------|---------|----------|
| AI-generated | 0.8001    | 0.8921  | 0.8436   |
| Real         | 0.8781    | 0.7771  | 0.8245   |
