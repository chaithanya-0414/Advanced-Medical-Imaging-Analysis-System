# 🚀 Quick Start Guide

## Running the Enhanced Application

### Option 1: Enhanced App (Recommended)
Run the new enhanced app with all advanced features:

```bash
streamlit run app_enhanced.py
```

**Features:**
- ✅ **Main Analysis** - Standard segmentation with staging
- ✅ **Grad-CAM** - Model interpretability visualization
- ✅ **Comprehensive Metrics** - 9 clinical metrics (requires ground truth)
- ✅ **Uncertainty** - Monte Carlo Dropout confidence estimation
- ✅ **Radiomics** - 25+ quantitative features

### Option 2: Original App
Run the original app:

```bash
streamlit run app.py
```

---

## 📋 Prerequisites

Install additional dependencies:

```bash
pip install -r requirements_advanced.txt
```

---

## 🎯 Usage Guide

### 1. Upload an Image
- Click "Browse files" or drag & drop
- Supported formats: JPG, PNG

### 2. Configure Settings (Sidebar)
- **Device**: Automatically uses GPU if available
- **Advanced Features**: Toggle Grad-CAM, Uncertainty, Radiomics
- **Threshold**: Adjust segmentation sensitivity
- **Uncertainty Samples**: More samples = better estimate (slower)

### 3. Explore Tabs

#### Tab 1: Main Analysis
- View segmentation results
- See tumor area and staging
- Download probability maps

#### Tab 2: Grad-CAM
- Visualize model attention
- Understand decision-making
- See activation statistics

#### Tab 3: Comprehensive Metrics
- Upload ground truth mask
- Get 9 clinical metrics
- Visual comparison overlay

#### Tab 4: Uncertainty
- See prediction confidence
- Identify uncertain regions
- Review high-uncertainty areas

#### Tab 5: Radiomics
- Extract 25+ features
- View shape, intensity, texture metrics
- Download full report

---

## 🧪 Testing the Features

### Test with Demo Script

```bash
python demo_advanced_features.py
```

This will test all modules without needing real data.

### Test with Real Data

1. **Prepare test images** in `data/test/`
2. **Run enhanced app**: `streamlit run app_enhanced.py`
3. **Upload image** and explore all tabs

---

## 🎓 Training with Advanced Features

### Train with Attention UNet

```bash
python train_advanced.py \
    --architecture attention_unet \
    --loss combo \
    --epochs 50 \
    --batch_size 8
```

### Train with Different Loss Functions

```bash
# Focal Tversky Loss (good for imbalanced data)
python train_advanced.py --loss focal_tversky

# Combo Loss (BCE + Dice + Focal)
python train_advanced.py --loss combo

# Standard Dice Loss
python train_advanced.py --loss dice
```

---

## 📊 Feature Comparison

| Feature | Original App | Enhanced App |
|---------|-------------|--------------|
| Basic Segmentation | ✅ | ✅ |
| Staging | ✅ | ✅ |
| DICOM Support | ✅ | ❌ (coming soon) |
| NIfTI Support | ✅ | ❌ (coming soon) |
| Grad-CAM | ❌ | ✅ |
| Comprehensive Metrics | ❌ | ✅ |
| Uncertainty | ❌ | ✅ |
| Radiomics | ❌ | ✅ |
| Batch Processing | ✅ | ❌ (coming soon) |

---

## 🐛 Troubleshooting

### "Module not found" Error
```bash
pip install -r requirements_advanced.txt
```

### Slow Performance
- Disable uncertainty quantification (computationally expensive)
- Reduce uncertainty samples (sidebar slider)
- Use GPU if available

### CUDA Out of Memory
- Reduce batch size in training
- Reduce uncertainty samples
- Use CPU instead of GPU

---

## 📝 Tips for Best Results

1. **For Interpretability**: Enable Grad-CAM to understand model decisions
2. **For Validation**: Use comprehensive metrics with ground truth
3. **For Clinical Trust**: Enable uncertainty to identify uncertain cases
4. **For Research**: Extract radiomics features for quantitative analysis

---

## 🎉 Next Steps

1. ✅ Run the enhanced app
2. ✅ Test all features with your data
3. ✅ Train with advanced loss functions
4. ✅ Compare UNet vs Attention UNet
5. ✅ Extract radiomics for analysis
6. ✅ Write research paper with results

---

## 📚 Documentation

- **Full Usage Guide**: See `ADVANCED_FEATURES.md`
- **Implementation Details**: See `walkthrough.md`
- **API Reference**: See docstrings in each module

---

## 🤝 Support

For issues or questions:
1. Check `ADVANCED_FEATURES.md` for detailed usage
2. Review `demo_advanced_features.py` for examples
3. Check module docstrings for API details

---

**Happy Analyzing! 🚀**
