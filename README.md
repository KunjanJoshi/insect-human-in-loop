
# Uncertainty-Aware Agricultural Pest Identification (Human-in-the-Loop Prototype)

This project demonstrates a computer vision pipeline for automated agricultural pest identification using transfer learning (EfficientNet) and uncertainty-aware flagging for expert verification. The aim is to replicate a practical "human-in-the-loop" workflow where high-confidence predictions are accepted automatically and ambiguous cases are routed for manual review.

## Dataset
Kaggle: Agricultural Pests Image Dataset (`vencerlanz09/agricultural-pests-image-dataset`)

## Methods
- Model: EfficientNet-B0 (PyTorch + timm)
- Train/validation split: 80/20
- Uncertainty metric: prediction entropy  
  H(p) = - Σ pᵢ log(pᵢ)
- Workflow: top 15% most uncertain predictions are flagged for human verification

## Key Results
- Certain bucket (auto-accepted): **934 images**, **99.14% accuracy**
- Uncertain bucket (flagged): **165 images**, **73.33% accuracy**

## Outputs
- `outputs/best_model.pt`
- `outputs/confusion_matrix.png`
- `outputs/uncertain_predictions.csv`
- `outputs/certain_predictions.csv`
- `outputs/flagged_uncertain/`

## Tools
Python, PyTorch, torchvision, timm, scikit-learn, Google Colab
