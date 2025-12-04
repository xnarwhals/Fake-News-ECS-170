# ECS170 Fake News Credibility

This repo contains our ECS170 final project: a fake-news credibility classifier with Streamlit web app. The project follows the proposal to build a baseline TF-IDF + Logistic Regression/SVM model, then a transformer (DistilBERT) model, plus explainability and a simple UI.

## Repo Layout
- `data/`: raw/processed data (e.g., `data/raw/standardized.csv`); add other datasets like FakeNewsNet/LIAR here.
- `notebooks/`: exploratory work (baseline and transformer experiments).
- `src/data/`: loading, cleaning, standardization, and URL scraping.
- `src/models/`: baseline TF-IDF+LogReg/SVM, transformer fine-tuning, explainability, and uncertainty helpers.
- `src/evaluation/`: metrics and calibration utilities.
- `src/ui/`: Streamlit app entry point.
- `tests/`: place minimal unit tests/smoke tests here.

## Live App (deploy)
https://fake-news-ecs-170-unk9x6vtxnwcbsu3f9ueyk.streamlit.app

## Quickstart (local dev)
1) Create a virtualenv with Python 3.10+ and install deps (`pip install -r requirements.txt` once added).
2) Put training data under `data/` (True.csv and Fake.csv go in `data/raw/`; they’re auto-used).
3) Train and save the baseline: `python3 scripts/train_baseline.py` (writes `models/baseline.joblib`).
4) Prototype in `notebooks/` (baseline vs transformer).
5) Run the Streamlit UI locally: `streamlit run src/ui/app.py` (it will load `models/baseline.joblib` or train from True/Fake if present).
