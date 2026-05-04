# BitcoinPredictionResearch-
Technical Indicator

## Local feature generation

To reproduce the same feature pipeline as the notebook (crawl → preprocess → engineer):

```bash
python generate_features.py
```

Outputs are written under `data/raw` and `data/processed`, with per-run crawl logs and
the engineered feature set saved as `data/processed/bitcoin_full_engineered_features.csv`.
