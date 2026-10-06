# EDA and experiment checklist

Use this checklist to build the final notebook:

1. Load the five expected columns and inspect nulls/duplicates.
2. Plot category and intent distributions.
3. Compare instruction length by intent.
4. Show top words per category after stopword removal.
5. Split train/validation/test with stratification before balancing.
6. Simulate noisy tickets with greetings, signatures, and quoted history.
7. Compare rule-based extraction with the Seq2Seq LSTM extractor.
8. Train three custom RNN/LSTM configurations.
9. Fine-tune DistilBERT and BERT-base on the same split.
10. Record accuracy, macro F1, inference ms/query, model size, and training time.
11. Plot the accuracy-versus-latency trade-off.
12. Demonstrate at least two raw-ticket-to-response examples.
