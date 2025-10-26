# Stock Price Prediction Using Data Augmentation with Generative Models

Machine learning research on using synthetic data to improve stock price forecasting accuracy and generalization.

## overview

This project investigates whether synthetic data generated through GANs can help machine learning models better predict stock prices. The main question: does training on both real and synthetic data lead to better generalization than using only real data?

We compare several deep learning approaches (LSTM, Quantum-LSTM, CycleGAN, WGAN) trained on hybrid datasets to find what actually works best for financial forecasting.

## the problem

Stock price prediction is hard because:
- Limited historical data available for training
- Markets are noisy and influenced by unpredictable events
- Models trained on past data often fail when market conditions change
- Rare events (crashes, surges) are underrepresented in training data

This research explores using synthetic data to address these challenges.

## approach

**Data:** Apple (AAPL) stock prices from Yahoo Finance, January 2020 - January 2023

**Generative Models (for synthetic data):**
- CycleGAN - generates realistic synthetic prices without paired examples
- Wasserstein GAN - produces high-quality diverse data for training

**Predictive Models:**
- LSTM - captures time-series patterns
- Quantum-enhanced LSTM - leverages quantum computing for better feature extraction

**Datasets tested:**
- Real data only
- Synthetic data only
- Hybrid (real + synthetic combined)

## key files

```
├── data/
│   ├── raw/                    # Original AAPL stock data
│   └── processed/              # Cleaned and normalized data
├── models/
│   ├── cyclegan.py            # CycleGAN implementation
│   ├── wgan.py                # Wasserstein GAN implementation
│   ├── lstm.py                # LSTM model
│   └── quantum_lstm.py        # Quantum-enhanced LSTM
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_synthetic_generation.ipynb
│   └── 03_model_training.ipynb
├── results/
│   ├── metrics/               # Performance metrics
│   └── visualizations/        # Charts and plots
└── README.md
```

## quick start

### requirements
- Python 3.8+
- PyTorch
- pandas, numpy, scikit-learn
- yfinance
- matplotlib, seaborn

### install

```bash
git clone https://github.com/vnmviv/Stock-Price-Prediction-Using-Data-Augmentation-with-Generative-Models.git
cd Stock-Price-Prediction-Using-Data-Augmentation-with-Generative-Models

pip install -r requirements.txt
```

### run

```bash
# Generate synthetic data
python models/cyclegan.py

# Train predictive models
python models/lstm.py

# View results
jupyter notebook notebooks/03_model_training.ipynb
```

## results

[Results to be updated with experimental findings]

Model performance comparison on test set:

| Model | RMSE | MAE | Generalization |
|-------|------|-----|-----------------|
| LSTM (real only) | - | - | - |
| LSTM (synthetic only) | - | - | - |
| LSTM (hybrid) | - | - | - |
| Quantum-LSTM (hybrid) | - | - | - |

## what we found

[Key findings to be added]

## how to use this repo

1. **Just exploring?** Start with `notebooks/01_data_exploration.ipynb`
2. **Want to train models?** Check out the notebooks in order (01 → 02 → 03)
3. **Want to use the code?** Import from the `models/` folder
4. **Have questions?** See the comments in each file

## references

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities
- Wasserstein GAN research on synthetic financial data
- LSTM time-series forecasting papers

## reproducibility

All experiments are designed to be reproducible:
- Fixed random seeds for deterministic results
- Data versioning with specific date ranges (2020-2023)
- Clear documentation of data preprocessing steps
- Git commits track all model versions

## limitations

This is research-stage code, not production-ready. Real trading systems need:
- Much more rigorous backtesting
- Transaction costs and slippage modeling
- Real-time data pipelines
- Risk management systems
- Regulatory compliance

## next steps

- Extend to multiple stocks and asset classes
- Test across different market regimes
- Implement real-time prediction pipeline
- Compare with classical financial models
- Validate on out-of-sample data from different time periods

## contact

Questions or suggestions? Feel free to open an issue or reach out.

---

if you find this work useful or want to collaborate, let me know!
