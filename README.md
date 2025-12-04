# ECS170 Fake News Credibility

This repo contains our ECS170 final project: a fake-news credibility classifier with both a report deliverable and an optional Streamlit web app. The project follows the proposal to build a baseline TF-IDF + Logistic Regression/SVM model, then a transformer (DistilBERT) model, plus explainability and a simple UI.

## Deliverables
- Report (3–4 pages, 12pt, single-spaced) **or** hosted web app. Name submissions `ECS170_FinalProject_GroupNumber.zip/pdf`.
- Report must emphasize Methodology and Results, include figures/diagrams, and briefly cover intro, background, discussion, conclusion, and contributions.
- Web app must be self-contained to use (just visit a URL) and include an info section (overview, methods, challenges, contributions, optional diagrams).

## Repo Layout
- `data/`: raw/processed data (e.g., `data/raw/standardized.csv`); add other datasets like FakeNewsNet/LIAR here.
- `notebooks/`: exploratory work (baseline and transformer experiments).
- `src/data/`: loading, cleaning, standardization, and URL scraping.
- `src/models/`: baseline TF-IDF+LogReg/SVM, transformer fine-tuning, explainability, and uncertainty helpers.
- `src/evaluation/`: metrics and calibration utilities.
- `src/ui/`: Streamlit app entry point.
- `tests/`: place minimal unit tests/smoke tests here.

## Live App (deploy)
- Deploy to Streamlit Community Cloud and share the link so users can just click and use the app:
  - Fork/clone to GitHub, then go to https://streamlit.io/cloud, choose this repo, set `src/ui/app.py` as the entry file.
  - Add secrets if you later fetch external models; current demo works without secrets.
  - The hosted URL will look like `https://<username>-<repo>-<branch>.streamlit.app`. Share that clickable link in your submission.
- Placeholder (replace with your deployed URL): https://streamlit.io/cloud

## Quickstart (local dev)
1) Create a virtualenv with Python 3.10+ and install deps (`pip install -r requirements.txt` once added).
2) Put training data under `data/` (or use the provided `data/raw/standardized.csv`).
3) Prototype in `notebooks/` (baseline vs transformer).
4) Save/export models to `models/` (or `artifacts/` if you add that) for the UI.
5) Run the Streamlit UI locally: `streamlit run src/ui/app.py`.

## What to include in the final submission
- Code + report or hosted app URL, zipped as `ECS170_FinalProject_GroupNumber.zip`.
- Figures/diagrams illustrating pipeline, experiments, and key results.
- Brief contribution notes (one sentence per team member).

## Status (scaffold)
- Stubs exist for data loading/cleaning, models, evaluation, and UI to match the proposal. Fill in training code, data paths, and figures as you finalize experiments.
