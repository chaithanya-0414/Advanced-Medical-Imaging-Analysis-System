# 🫁 Advanced Medical Imaging Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A state-of-the-art deep learning system for automated lung tumor segmentation with advanced clinical analysis features including Grad-CAM visualization, uncertainty quantification, comprehensive metrics, and radiomics feature extraction.

![Demo](https://via.placeholder.com/800x400.png?text=Medical+Imaging+Analysis+Demo)

## ✨ Key Features

- 🎯 **Automated Tumor Segmentation** - Pixel-level identification using UNet/Attention UNet
- 🔍 **Grad-CAM Visualization** - Model interpretability and attention mapping
- 📊 **Comprehensive Metrics** - 9+ clinical metrics (Dice, IoU, Hausdorff, etc.)
- 🎲 **Uncertainty Quantification** - Monte Carlo Dropout for confidence estimation
- 🧬 **Radiomics Analysis** - 25+ quantitative tumor features
- 🛡️ **Foreign Object Detection** - Automatic handling of implants/metal artifacts
- 📄 **PDF Reporting** - Multi-page clinical reports with all analysis results
- 🗂️ **Multi-Format Support** - JPG/PNG, NIfTI (.nii), DICOM (.dcm)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/medical-imaging-analysis.git
cd medical-imaging-analysis

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Launch the enhanced application
streamlit run app_enhanced.py
```

Open your browser at `http://localhost:8502` and start analyzing!

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)
- 8GB+ RAM

See `requirements.txt` for complete dependency list.

## 🎓 Training Your Own Model

### Basic Training
```bash
python train.py
```

### Advanced Training with Custom Loss
```bash
python train_advanced.py --loss focal_tversky --model attention_unet --epochs 50
```

### Available Loss Functions
- `bce` - Binary Cross-Entropy
- `dice` - Dice Loss
- `focal` - Focal Loss
- `tversky` - Tversky Loss
- `focal_tversky` - Focal + Tversky
- `combo` - Combo Loss (BCE + Dice)
- `boundary` - Boundary Loss

## 📊 Model Performance

- **Dice Coefficient**: 0.85-0.92
- **IoU**: 0.75-0.85
- **Sensitivity**: 0.88-0.94
- **Specificity**: 0.96-0.99

## 🏗️ Project Structure

```
├── app.py                      # Basic Streamlit application
├── app_enhanced.py             # Advanced application with all features
├── train.py                    # Basic training script
├── train_advanced.py           # Advanced training with custom losses
├── main.py                     # Robust training with checkpointing
├── models/
│   ├── best_model.pth         # Trained model weights
│   ├── grad_cam.py            # Grad-CAM implementation
│   ├── metrics.py             # Clinical metrics
│   ├── uncertainty.py         # Uncertainty quantification
│   ├── radiomics_extractor.py # Radiomics features
│   └── report_generator.py    # PDF report generation
├── requirements.txt           # Python dependencies
├── ADVANCED_FEATURES.md       # Detailed feature documentation
├── QUICKSTART.md              # Quick start guide
├── PROJECT_SUMMARY.md         # Complete project documentation
└── PROJECT_SUMMARY.md         # Complete project documentation
```

## 📖 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- **[Advanced Features](ADVANCED_FEATURES.md)** - Detailed feature documentation
- **[Project Summary](PROJECT_SUMMARY.md)** - Complete technical documentation

## 🎯 Use Cases

- **Clinical Diagnosis** - Radiologist decision support
- **Research** - Quantitative tumor analysis
- **Education** - Medical imaging training
- **Screening** - Large-scale batch processing

## 🔬 Advanced Features

### Grad-CAM Visualization
Visualize which regions the AI model focuses on during prediction.

### Uncertainty Quantification
Estimate model confidence using Monte Carlo Dropout to flag cases needing expert review.

### Radiomics Analysis
Extract 25+ quantitative features including:
- Shape features (area, perimeter, compactness, etc.)
- Intensity features (mean, std, skewness, kurtosis, etc.)
- Texture features (GLCM: contrast, homogeneity, energy, etc.)

### Foreign Object Handling
Two strategies for handling metal implants and artifacts:
1. **Post-processing exclusion** - Subtract from tumor mask
2. **Pre-processing inpainting** - Remove before analysis

## 🛠️ Technologies Used

- **PyTorch** - Deep learning framework
- **Streamlit** - Web application framework
- **OpenCV** - Image processing
- **scikit-image** - Medical image processing
- **nibabel** - NIfTI file support
- **pydicom** - DICOM file support
- **FPDF** - PDF report generation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UNet architecture based on [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Attention UNet based on [Oktay et al., 2018](https://arxiv.org/abs/1804.03999)
- Grad-CAM implementation inspired by [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)

## 📧 Contact & Support

### Author
**Chaithanya**
- GitHub: [@chaithanya-0414](https://github.com/chaithanya-0414)
- Repository: [Advanced-Medical-Imaging-Analysis-System](https://github.com/chaithanya-0414/Advanced-Medical-Imaging-Analysis-System)

### Get Help
- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/chaithanya-0414/Advanced-Medical-Imaging-Analysis-System/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/chaithanya-0414/Advanced-Medical-Imaging-Analysis-System/discussions)
- **Pull Requests**: Contributions are welcome!

### Citation
If you use this project in your research or work, please cite:
```
@software{chaithanya2024medical_imaging,
  author = {Chaithanya},
  title = {Advanced Medical Imaging Analysis System},
  year = {2024},
  url = {https://github.com/chaithanya-0414/Advanced-Medical-Imaging-Analysis-System}
}
```

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Built with ❤️ for advancing medical imaging analysis**
