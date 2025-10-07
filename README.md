# StarEmbed: Benchmarking Time Series Foundation Models on Astronomical Observations of Variable Starts

**StarEmbed is the first benchmark to test the state-of-the-art TSFMs on stellar time series observations ("light curves").**

A complete benchmark framework for astronomical time series. This repository includes tools for preprocessing raw light curves, generating embeddings (with TSFMs and Astromer), engineering handcrafted features, and evaluating performance on clustering, classification, and out-of-distribution detection.

| 🏠[**Benchmark Page**](https://hibb-bb.github.io/star-embed.github.io/) | [**🤗Huggingface Dataset**](https://huggingface.co/datasets/123anonymous123/StarEmbed) | [**📖Paper**](https://arxiv.org/abs/2406.19392) |
---

## **Directory Overview**

### **`datasets/`**
Raw light curve preprocessing and data preparation scripts  
→ *See `datasets/README.md` for detailed preprocessing workflows*

### **`model/`** 
Time series foundation model implementations and embedding generation
- **Astromer**: Transformer-based astronomical time series model
- **Chronos**: Amazon's forecasting foundation model  
- **Moirai**: Salesforce's universal time series model
- **`compute_avg_embeddings.py`**: Generate combined embeddings from multi-band data

### **`benchmark/`**
Evaluation pipeline with pre-computed embeddings
- **Classification**: kNN, Linear models, MLPs, Random Forest with HPO
- **Clustering**: K-Means, hierarchical clustering, t-SNE visualization  
→ *See `benchmark/README.md` for complete evaluation workflows*

### **`output/`**
Results storage for all experiments and evaluations

### **`bash_script/`**
SLURM job scripts and automation for high-performance computing


---

## **Quick Start**

1. **Preprocess data**: `datasets/` → Raw light curves to standardized format
2. **Generate embeddings**: `model/` → Extract features using TSFMs  
3. **Create combined embeddings**: `model/compute_avg_embeddings.py` → Multi-band aggregation
4. **Run evaluations**: `benchmark/` → Classification, clustering, visualization

---

## **Key Features**

✅ **First systematic benchmark** of TSFMs on astronomical data  
✅ **Multi-model support**: Astromer, Chronos, Moirai, traditional methods  
✅ **Optimized pipeline**: Pre-computed embeddings for 10x performance gain  
✅ **Comprehensive evaluation**: Classification, clustering, OOD detection  
✅ **HPC ready**: SLURM integration for large-scale experiments  

---

