A minimal README explaining how to set up the virtual environment, download
(or symlink) the original Airbnb datasets, and run the reproduction script.

- **`brtm/`**  
  Core module containing the main functions and model implementations for BRTM.

- **`Data_Crawl/`**  
  Includes the modified Airbnb data crawling module `airbnb_scraper` and the integrated data crawling and preprocessing script `crawl.py`.

- **`Data/`**  
  Directory containing the collected and cleaned Airbnb datasets, ready for model training and evaluation.

- **`main_train.py`**  
  Main training script for running the full BRTM model pipeline.

- **`baseline_compare.py`**  
  Script for comparing BRTM with other baseline models.

- **`train_v3.ipynb`**  
  Experimental notebook used for prototyping and debugging during model development.

##  How to Run

1. Open `config.py` and localize all path settings according to your machine.
2. Run the `main_train.ipynb`


brtm_project/
├── brtm/                       
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── tokenizer.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── var_em_gpu.py
│   │   ├── lbfgs_optimizer.py
│   │   └── lda_init.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       └── mrr_ndcg.py
├── Data/
├── Data_Crawl/
├── brtm_outputs/
├── main_train.ipynb
├── baseline_compare.ipynb
├── README.md
└── requirements.txt