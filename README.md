# BRTM-Sample Replication

This project is a minimal, self-contained replication of **Table 7** from the paper:  
📄 [_“From Reviews to Ratings: How to Leverage Text for Multiaspect Ranking”_ (INFORMS Information Systems Research, 2020)](https://pubsonline.informs.org/doi/10.1287/isre.2020.0981)

It includes the full pipeline for reproducing the original BRTM model, from data collection and preprocessing to training and evaluation.

---

## 📁 Project Structure

```
brtm_project/
├── brtm/                       # Core module (config, model, EM optimization, metrics)
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── tokenizer.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── var_em_gpu.py
│   │   ├── lbfgs_optimizer.py
│   │   └── lda_init.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       └── mrr_ndcg.py
├── Data/                       # Cleaned Airbnb dataset
├── Data_Crawl/                # Crawling and preprocessing tools
│   ├── airbnb_scraper/
│   └── crawl.py
├── brtm_outputs/              # Output directory for logs, figures, model artifacts
├── main_train.ipynb           # Main notebook: config → preprocess → train → evaluate
├── baseline_compare.ipynb     # Benchmark comparison with other models
├── train_v3.ipynb             # Scratch notebook for experimentation
├── requirements.txt           # Python dependency list
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Create a Virtual Environment

We recommend using `venv` or `conda`:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the Airbnb Dataset

- Download or symlink the Airbnb Amsterdam dataset into the `Data/` folder.  
- Ensure the directory contains the preprocessed files required for training (e.g., listings, reviews, calendars).

Alternatively, run the modified crawler in:

```bash
Data_Crawl/crawl.py
```

> **Note**: This module integrates `airbnb_scraper` and postprocessing logic.

---

## 🚀 Running the Model

1. Open `brtm/config.py` and **update all file paths** to match your local environment.
2. Launch the main training notebook:

```bash
jupyter notebook main_train.ipynb
```

This notebook will:
- Load and preprocess the Airbnb dataset
- Train the BRTM model using variational EM + LBFGS
- Evaluate using top-N hit rate and MRR/NDCG
- Save model weights and metrics to `brtm_outputs/`

---

## 🧪 Reproducibility

The codebase is focused on **Table 7 replication**, faithfully following the experiment protocol described in the INFORMS IS paper.

Feel free to explore the `baseline_compare.ipynb` notebook for additional evaluation.

---

## 📝 License

MIT License. For academic use only.

---

## ✨ Acknowledgments

Originally implemented and adapted from code and ideas presented in:  
🔗 [INFORMS 2020 | Cheng et al.](https://pubsonline.informs.org/doi/10.1287/isre.2020.0981)

