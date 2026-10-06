# Customer Support Query Understanding & Auto-Response

A two-stage NLP pipeline for turning noisy customer-support tickets into useful auto-responses.

## Pipeline

1. **Query extraction and normalization** removes greetings, signatures, quoted-thread wrappers, and boilerplate.
2. **Intent classification** predicts one of the Bitext support intents.
3. **Response retrieval** returns the reference response associated with the predicted intent.
4. **Model comparison** supports a custom TF-IDF/RNN baseline and optional DistilBERT/BERT fine-tuning.
5. **Streamlit deployment** shows the raw ticket, extracted query, predicted intent, confidence, top alternatives, and suggested reply.

## Dataset

Place `Bitext_Customer_Support_Dataset.csv` in `data/raw/`. The expected columns are:

```text
flags, instruction, category, intent, response
```

The repository intentionally does not commit the 18 MB source dataset. Add it locally or through Git LFS before running the project.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.baseline_model --data data/raw/Bitext_Customer_Support_Dataset.csv
streamlit run app/streamlit_app.py
```

The baseline uses TF-IDF plus a linear classifier, which is fast enough to run on a laptop. The deep-learning training scripts are optional and should be run in Colab or another GPU-enabled environment.

## Train transformer models

```bash
python -m src.transformer_models --model distilbert-base-uncased
python -m src.transformer_models --model bert-base-uncased
```

Record validation accuracy, test accuracy, inference time per query, model size, and training time in `reports/model_comparison.csv`. Do not use the expected reference figures as measured results; replace them with your own run measurements.

## Evaluation discipline

- Split before oversampling.
- Use stratification so all 27 intents remain represented.
- Keep the held-out test set untouched until final evaluation.
- Report accuracy, macro/weighted precision, recall, F1, and a confusion matrix.
- For a live chat deployment, compare latency and model size alongside accuracy.

## Repository structure

```text
customer-support-nlp/
├── app/streamlit_app.py
├── data/raw/README.md
├── notebooks/01_eda_and_experiments.md
├── src/
│   ├── baseline_model.py
│   ├── query_extraction.py
│   └── transformer_models.py
├── reports/
├── requirements.txt
└── README.md
```
