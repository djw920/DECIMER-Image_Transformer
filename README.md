<div align="center">

# 🧪 DECIMER Image Transformer 🖼️

### Deep Learning for Chemical Image Recognition using Efficient-Net V2 + Transformer

<p align="center">
  <img src="https://github.com/Kohulan/DECIMER-Image_Transformer/blob/master/DECIMER_V2.png?raw=true" alt="DECIMER Logo" width="600">
</p>

[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/Kohulan/DECIMER-Image_Transformer.svg?style=for-the-badge)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/issues/)
[![GitHub contributors](https://img.shields.io/github/contributors/Kohulan/DECIMER-Image_Transformer.svg?style=for-the-badge)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/graphs/contributors/)
[![tensorflow](https://img.shields.io/badge/TensorFlow-2.10.1-FF6F00.svg?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org)
[![Model Card](https://img.shields.io/badge/Model_Card-DECIMER-9cf.svg?style=for-the-badge)](https://zenodo.org/records/8300489)
[![DOI](https://zenodo.org/badge/293572361.svg)](https://zenodo.org/badge/latestdoi/293572361)
[![Documentation Status](https://readthedocs.org/projects/decimer-image-transformer/badge/?version=latest&style=for-the-badge)](https://decimer-image-transformer.readthedocs.io/en/latest/?badge=latest)
[![GitHub release](https://img.shields.io/github/release/Kohulan/DECIMER-Image_Transformer.svg?style=for-the-badge)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/releases/)
[![PyPI version fury.io](https://badge.fury.io/py/decimer.svg?style=for-the-badge)](https://pypi.python.org/pypi/decimer/)

</div>

---

## 📚 Table of Contents

- [📝 Abstract](#-abstract)
- [💡 Method and Model Changes](#-method-and-model-changes)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [🗂️ Batch Processing](#️-batch-processing)
- [✍️ Hand-drawn Model](#️-decimer---hand-drawn-model)
- [📄 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)
- [👨‍🔬 Author](#-author-kohulan)
- [🌐 Project Website](#-project-website)
- [🏛️ Research Group](#️-research-group)

---

## 📝 Abstract

<div align="center">
  <img src="https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif" width="350" align="right">
</div>

> The DECIMER 2.2 project tackles the OCSR (Optical Chemical Structure Recognition) challenge using cutting-edge computational intelligence methods. Our goal? To provide an automated, open-source software solution for chemical image recognition.
> 
> We've supercharged DECIMER with Google's TPU (Tensor Processing Unit) to handle datasets of over 1 million images with lightning speed!

---

## 💡 Method and Model Changes

<table>
  <tr>
    <td width="50%" align="center">
      <h3>🖼️ Image Feature Extraction</h3>
      <p>Now utilizing EfficientNet-V2 for superior image analysis</p>
    </td>
    <td width="50%" align="center">
      <h3>🔮 SMILES Prediction</h3>
      <p>Employing a state-of-the-art transformer model</p>
    </td>
  </tr>
</table>

### 🚀 Training Enhancements

1. **📦 TFRecord Files** - Lightning-fast data reading
2. **☁️ Google Cloud Buckets** - Efficient cloud storage solution
3. **🔄 TensorFlow Data Pipeline** - Optimized data loading
4. **⚡ TPU Strategy** - Harnessing the power of Google's TPUs

---

## ⚙️ Installation

```bash
# Create a conda wonderland
conda create --name DECIMER python=3.10.0 -y
conda activate DECIMER

# Equip yourself with DECIMER
pip install decimer
```

---

## 🚀 Usage

```python
from DECIMER import predict_SMILES

# Unleash the power of DECIMER
image_path = "path/to/your/chemical/masterpiece.jpg"
SMILES = predict_SMILES(image_path)
print(f"🎉 Decoded SMILES: {SMILES}")
```

---

## 🗂️ Batch Processing

`decimer_batch.py` converts an entire directory of chemical structure images to SMILES in one command, writing results to a CSV file. It includes structured logging, pre-flight image validation, configurable retries, and safe Ctrl-C handling that preserves partial results.

### Quick start

```bash
conda activate DECIMER

# Process all images in a folder; CSV saved alongside images
python decimer_batch.py ~/structures/

# Specify output path
python decimer_batch.py ~/structures/ -o ~/results/smiles.csv

# Write a rotating log file as well
python decimer_batch.py ~/structures/ -o ~/results/smiles.csv --log ~/logs/decimer.log
```

### All options

| Flag | Default | Description |
|---|---|---|
| `image_dir` | *(required)* | Directory containing chemical structure images |
| `-o / --output` | `<image_dir>/decimer_results_<timestamp>.csv` | Output CSV path |
| `-l / --log` | *(none)* | Rotating log file (5 MB × 3 backups) |
| `--ext` | `png jpg jpeg webp tif tiff` | Limit to specific extensions |
| `-r / --recursive` | off | Search subdirectories recursively |
| `--retries N` | `1` | Retry failed images up to N times |
| `--min-size BYTES` | `1024` | Skip files smaller than this (likely corrupt) |
| `-v / --verbose` | off | Show DEBUG-level messages on the console |

### Output CSV columns

| Column | Description |
|---|---|
| `filename` | Path relative to `image_dir` |
| `smiles` | Predicted SMILES string (empty on failure/skip) |
| `status` | `ok` \| `skipped: <reason>` \| `error: <reason>` |
| `elapsed_s` | Wall-clock seconds for that image |
| `timestamp` | ISO 8601 timestamp of completion |

Results are flushed to disk after every image — a crash or Ctrl-C mid-run preserves all completed rows.

### Logging

Without `--log`, only the console is used (INFO level by default). With `--log <file>`, a rotating file handler is added at DEBUG level, capturing full tracebacks for failed images:

```bash
# Verbose console output + persistent log file
python decimer_batch.py ~/structures/ --log ~/logs/decimer.log -v
```

Log format in file: `YYYY-MM-DD HH:MM:SS  LEVEL     message`

### Exit codes

| Code | Meaning |
|---|---|
| `0` | All images succeeded |
| `1` | Configuration or startup error |
| `2` | One or more images failed inference |
| `130` | Interrupted by Ctrl-C |

Non-zero exit codes make the script composable in shell pipelines and CI workflows.

### Running the tests

The test suite covers all error-handling paths and logging behaviour without loading the DECIMER model (all inference is mocked), so it runs in under a second:

```bash
conda activate DECIMER
pytest tests/test_decimer_batch.py -v
```

---

## ✍️ DECIMER - Hand-drawn Model

> 🌟 **New Feature Alert!** 🌟
> 
> Our latest model brings the magic of AI to hand-drawn chemical structures!
> 
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10781330.svg)](https://doi.org/10.5281/zenodo.10781330)

---

## 📄 Citation

If DECIMER helps your research, please cite:

1. Rajan K, et al. "DECIMER.ai - An open platform for automated optical chemical structure identification, segmentation and recognition in scientific publications." *Nat. Commun.* 14, 5045 (2023).
2. Rajan, K., et al. "DECIMER 1.0: deep learning for chemical image recognition using transformers." *J Cheminform* 13, 61 (2021).
3. Rajan, K., et al. "Advancements in hand-drawn chemical structure recognition through an enhanced DECIMER architecture," *J Cheminform* 16, 78 (2024).

---

## 🙏 Acknowledgements

- A big thank you to [Charles Tapley Hoyt](https://github.com/cthoyt) for his invaluable contributions!
- Powered by Google's TPU Research Cloud (TRC)

<p align="center">
  <img src="https://user-images.githubusercontent.com/30716951/220350828-913e6645-6a0a-403c-bcb8-160d061d4606.png" width="300">
</p>

---

## 👨‍🔬 Author: [Kohulan](https://kohulanr.com)

---

## 🌐 Project Website

Experience DECIMER in action at [decimer.ai](https://decimer.ai), brilliantly implemented by [Otto Brinkhaus](https://github.com/OBrink)!

---

<div align="center">

### 🎓 Maintained by the [Kohulan](https://www.kohulanr.com/#) @ Steinbeck Group

<a href="https://cheminf.uni-jena.de">
<img src="https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png" width="400" alt="Cheminformatics Group"/>
</a>

**[Natural Products Cheminformatics Research Group](https://cheminf.uni-jena.de)**  
Institute for Inorganic and Analytical Chemistry  
Friedrich Schiller University Jena, Germany

---
## ⭐ Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Kohulan/DECIMER-Image_Transformer&type=Date)](https://star-history.com/#Kohulan/DECIMER-Image_Transformer&Date)

</div>

---

<div align="center">

### 📊 Project Analytics

![Repobeats](https://repobeats.axiom.co/api/embed/bf532b7ac0d34137bdea8fbb82986828f86de065.svg "Repobeats analytics image")

</div>

<div align="center">
  
**Made with ❤️ and ☕ for the global chemistry community**

**© 2025 Kohulan @ Steinbeck Lab, Friedrich Schiller University Jena**
---
