# Advanced Features - Usage Guide

## 🚀 Quick Start

### 1. Install Additional Dependencies

```bash
pip install -r requirements_advanced.txt
```

### 2. Run the Demo

Test all new features with the demo script:

```bash
python demo_advanced_features.py
```

This will demonstrate:
- ✅ Grad-CAM visualization
- ✅ Comprehensive metrics (9 metrics)
- ✅ Advanced loss functions
- ✅ Uncertainty quantification
- ✅ Model ensemble
- ✅ Attention UNet
- ✅ Radiomics features

---

## 📚 Feature Documentation

### 1. Grad-CAM Visualization

**File**: `models/grad_cam.py`

**Purpose**: Visualize which regions the model focuses on

**Basic Usage**:
```python
from models.grad_cam import GradCAM

# Create Grad-CAM instance
grad_cam = GradCAM(model, target_layer=model.dec1)

# Generate heatmap
cam = grad_cam.generate_cam(input_tensor)

# Overlay on image
overlay = grad_cam.overlay_heatmap(image, cam, alpha=0.5)
```

**Multi-layer Visualization**:
```python
from models.grad_cam import MultiLayerGradCAM

# Multiple layers
layers = [model.dec1, model.dec2, model.dec3]
multi_cam = MultiLayerGradCAM(model, layers)

# Combined CAM
combined_cam = multi_cam.generate_multi_cam(input_tensor)
```

---

### 2. Comprehensive Metrics

**File**: `models/metrics.py`

**Purpose**: Clinical-grade evaluation metrics

**Compute All Metrics**:
```python
from models.metrics import SegmentationMetrics

metrics = SegmentationMetrics.compute_all_metrics(prediction, ground_truth)

print(f"Dice: {metrics['dice']:.4f}")
print(f"IoU: {metrics['iou']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
print(f"Hausdorff Distance: {metrics['hausdorff_95']:.2f}")
```

**Individual Metrics**:
```python
dice = SegmentationMetrics.dice_coefficient(pred, target)
sensitivity = SegmentationMetrics.sensitivity(pred, target)
hausdorff = SegmentationMetrics.hausdorff_distance(pred, target)
```

**ROC and PR Curves**:
```python
from models.metrics import compute_roc_curve, compute_pr_curve

fpr, tpr, roc_auc, _ = compute_roc_curve(y_true, y_pred_proba)
precision, recall, pr_auc, _ = compute_pr_curve(y_true, y_pred_proba)
```

---

### 3. Advanced Loss Functions

**File**: `models/losses.py`

**Purpose**: Better training with specialized losses

**Combo Loss** (Recommended):
```python
from models.losses import ComboLoss

criterion = ComboLoss(
    bce_weight=1.0,
    dice_weight=1.0,
    focal_weight=0.5
)

loss = criterion(pred_logits, target)
```

**Focal Tversky Loss** (For imbalanced data):
```python
from models.losses import FocalTverskyLoss

criterion = FocalTverskyLoss(
    alpha=0.3,  # FP weight
    beta=0.7,   # FN weight (higher = prioritize recall)
    gamma=1.5   # Focal parameter
)
```

**Factory Function**:
```python
from models.losses import get_loss_function

criterion = get_loss_function('focal_tversky', alpha=0.3, beta=0.7)
```

---

### 4. Uncertainty Quantification

**File**: `models/uncertainty.py`

**Purpose**: Estimate prediction confidence

**Monte Carlo Dropout**:
```python
from models.uncertainty import MCDropout

# Wrap model
mc_model = MCDropout(model, n_samples=10)

# Get prediction with uncertainty
mean_pred, uncertainty, samples = mc_model.predict_with_uncertainty(input_tensor)

# Visualize
from models.uncertainty import visualize_uncertainty
viz = visualize_uncertainty(image, mean_pred, uncertainty)
```

**Ensemble Uncertainty**:
```python
from models.uncertainty import EnsembleUncertainty

ensemble_unc = EnsembleUncertainty(models=[model1, model2, model3])
mean_pred, uncertainty, preds = ensemble_unc.predict_with_uncertainty(input_tensor)
```

**High Uncertainty Regions**:
```python
from models.uncertainty import get_high_uncertainty_regions

high_unc_mask, threshold = get_high_uncertainty_regions(
    uncertainty, 
    threshold_percentile=90
)
```

---

### 5. Model Ensemble

**File**: `models/ensemble.py`

**Purpose**: Combine multiple models for better accuracy

**Basic Ensemble**:
```python
from models.ensemble import ModelEnsemble

ensemble = ModelEnsemble(
    models=[model1, model2, model3],
    strategy='average'  # or 'voting', 'weighted', 'max'
)

prediction = ensemble.predict(input_tensor, device='cuda')
```

**With Confidence**:
```python
prediction, confidence = ensemble.predict_with_confidence(input_tensor)
```

**From Checkpoints**:
```python
ensemble = ModelEnsemble.from_checkpoints(
    model_class=UNet,
    checkpoint_paths=[
        'models/model1.pth',
        'models/model2.pth',
        'models/model3.pth'
    ]
)
```

**Learn Optimal Weights**:
```python
from models.ensemble import train_ensemble_weights

optimal_weights = train_ensemble_weights(
    models=[model1, model2, model3],
    val_loader=val_loader,
    n_iterations=100
)

ensemble = ModelEnsemble(models, weights=optimal_weights)
```

---

### 6. Attention UNet

**File**: `models/attention_unet.py`

**Purpose**: Advanced architecture with attention mechanisms

**Basic Usage**:
```python
from models.attention_unet import AttentionUNet

model = AttentionUNet(in_ch=1, out_ch=1, base=32)
output = model(input_tensor)
```

**With Attention Maps**:
```python
output, attention_maps = model.forward_with_attention(input_tensor)

# Access attention at different levels
for level, att_map in attention_maps.items():
    print(f"{level}: {att_map.shape}")
    # Visualize attention_map
```

---

### 7. Radiomics Features

**File**: `models/radiomics_extractor.py`

**Purpose**: Extract quantitative features from segmented regions

**Extract Features**:
```python
from models.radiomics_extractor import compute_radiomics_features

features = compute_radiomics_features(image, mask)

# Access features
print(f"Area: {features['area_pixels']}")
print(f"Compactness: {features['compactness']}")
print(f"Mean Intensity: {features['mean_intensity']}")
print(f"GLCM Contrast: {features['glcm_contrast']}")
```

**Generate Report**:
```python
from models.radiomics_extractor import format_radiomics_report

report = format_radiomics_report(features)
print(report)
```

**Available Features**:
- **Shape** (9): area, perimeter, compactness, eccentricity, solidity, extent, axes, aspect ratio
- **Intensity** (10): mean, std, min, max, median, range, skewness, kurtosis, energy, entropy
- **Texture** (6): GLCM contrast, dissimilarity, homogeneity, energy, correlation, ASM

---

## 🎓 Training with Advanced Features

### Use Advanced Training Script

```bash
# Train with Attention UNet and Combo Loss
python train_advanced.py \
    --architecture attention_unet \
    --loss combo \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3

# Train with standard UNet and Focal Tversky Loss
python train_advanced.py \
    --architecture unet \
    --loss focal_tversky \
    --epochs 50
```

**Available Options**:
- `--architecture`: `unet` or `attention_unet`
- `--loss`: `bce`, `dice`, `focal`, `tversky`, `focal_tversky`, `combo`
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--base_channels`: Base channels in UNet (default: 32)

---

## 📊 Integration Examples

### Example 1: Complete Evaluation Pipeline

```python
import torch
from models.metrics import SegmentationMetrics
from models.uncertainty import MCDropout
from models.grad_cam import GradCAM

# Load model and data
model.eval()

# 1. Get prediction with uncertainty
mc_model = MCDropout(model, n_samples=10)
mean_pred, uncertainty, _ = mc_model.predict_with_uncertainty(input_tensor)

# 2. Compute comprehensive metrics
metrics = SegmentationMetrics.compute_all_metrics(mean_pred, ground_truth)

# 3. Generate Grad-CAM
grad_cam = GradCAM(model, target_layer=model.dec1)
cam = grad_cam.generate_cam(input_tensor)

# 4. Print results
print("Metrics:", metrics)
print("Mean Uncertainty:", uncertainty.mean())
```

### Example 2: Ensemble with Uncertainty

```python
from models.ensemble import ModelEnsemble
from models.radiomics_extractor import compute_radiomics_features

# Create ensemble
ensemble = ModelEnsemble([model1, model2, model3])

# Get prediction with confidence
pred, confidence = ensemble.predict_with_confidence(input_tensor)

# Extract radiomics features
features = compute_radiomics_features(image, pred)

print(f"Confidence: {confidence.mean():.4f}")
print(f"Tumor Area: {features['area_pixels']:.2f} pixels")
```

---

## 🔬 Research Applications

### Ablation Study: Loss Functions

```python
losses = ['bce', 'dice', 'focal', 'tversky', 'combo']

for loss_name in losses:
    print(f"\nTraining with {loss_name} loss...")
    # Train model with this loss
    # Evaluate and compare results
```

### Architecture Comparison

```python
architectures = ['unet', 'attention_unet']

for arch in architectures:
    print(f"\nTraining {arch}...")
    # Train and evaluate
    # Compare metrics
```

### Uncertainty Analysis

```python
# Identify cases where model is uncertain
high_uncertainty_cases = []

for image, mask in test_loader:
    mean_pred, uncertainty, _ = mc_model.predict_with_uncertainty(image)
    
    if uncertainty.mean() > threshold:
        high_uncertainty_cases.append((image, mask, uncertainty))

# Analyze these cases
```

---

## 📈 Performance Tips

1. **Use GPU**: All modules support CUDA acceleration
2. **Batch Processing**: Process multiple images at once
3. **MC Dropout Samples**: 10-20 samples is usually sufficient
4. **Ensemble Size**: 3-5 models provides good balance
5. **Loss Selection**: `combo` works well for most cases

---

## 🐛 Troubleshooting

### Import Errors
```bash
pip install -r requirements_advanced.txt
```

### CUDA Out of Memory
- Reduce batch size
- Reduce MC dropout samples
- Use fewer ensemble models

### Slow Inference
- Use GPU if available
- Reduce uncertainty samples
- Use single model instead of ensemble for speed

---

## 📝 Citation

If you use these features in your research, consider citing the original papers:

- **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
- **Attention UNet**: Oktay et al., "Attention U-Net: Learning Where to Look"
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection"
- **Tversky Loss**: Salehi et al., "Tversky Loss Function for Image Segmentation"

---

## 🎉 Summary

You now have access to:
- ✅ 7 advanced modules
- ✅ 2,050+ lines of code
- ✅ 50+ new features
- ✅ State-of-the-art capabilities

Happy researching! 🚀
