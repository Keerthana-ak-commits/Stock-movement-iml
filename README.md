# StockNet: Hybrid Stock Movement Prediction 📈

**StockNet** is a deep learning framework designed to predict the directional movement (Up/Down) of stock prices. It utilizes a hybrid architecture that fuses temporal market data (price history) with textual sentiment analysis (social media/tweets) to capture market dynamics more effectively than traditional time-series models.

##  Features

* **Hybrid Input:** Combines technical indicators (Open, High, Low, Close) with Natural Language Processing (NLP) on financial texts.
* **Deep Learning Architecture:**
    * **Text Encoder:** Bi-directional GRU with Attention Mechanism to create daily sentiment vectors.
    * **Temporal Encoder:** GRU-based sequence model to analyze price and sentiment trends over time (Lag window = 5 days).
* **Baseline Comparison:** Includes an ARIMAX (AutoRegressive Integrated Moving Average with Exogenous variables) implementation for statistical performance benchmarking.
* **Evaluation Metrics:** Tracks Accuracy, F1-Score, MCC (Matthews Correlation Coefficient), and ROC-AUC.

##  Tech Stack

* **Language:** Python 3.x
* **Deep Learning:** PyTorch
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Statistical Modeling:** Pmdarima
* **Visualization:** Matplotlib, Seaborn

##  Data Structure

The project expects a dataset named `merged_tweets_stock_data_nearest_date.csv` containing aligned stock prices and tweet text.

**Required CSV Columns:**
| Column Name | Description |
| :--- | :--- |
| `Ticker` | Stock symbol (e.g., AMZN, GOOGL) |
| `Nearest_Trading_Date` | Date of the trading day |
| `text` | Raw text content of the tweet/message |
| `Close` | Closing price of the stock |
| `High` | Highest price of the day |
| `Low` | Lowest price of the day |

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/stocknet.git](https://github.com/yourusername/stocknet.git)
    cd stocknet
    ```

2.  **Install Dependencies:**
    ```bash
    pip install torch pandas numpy scikit-learn matplotlib seaborn pmdarima
    ```

    *> **Note:** If you encounter issues with `pmdarima`, you may need to downgrade numpy:*
    ```bash
    pip install "numpy==1.23.5"
    ```

##  Model Architecture Details

The model processes data in two stages:

1.  **Message Processing (Text):**
    * Input: Batch of tweets.
    * Embedding Layer: Converts tokens to vectors.
    * Bi-GRU: Captures context in both directions.
    * **Attention Layer:** Aggregates all tweets for a single day into a specific "Daily Sentiment Vector."

2.  **Time-Series Prediction:**
    * Input: Concatenation of [Daily Sentiment Vector] + [Price Returns].
    * Main GRU: Looks back 5 days (Window Size) to predict the trend.
    * Output: Binary classification (1 = Price UP, 0 = Price DOWN).

##  Usage

1.  Ensure your dataset (`merged_tweets_stock_data_nearest_date.csv`) is in the root directory.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook Stocknet1.ipynb
    ```
3.  Run all cells to:
    * Preprocess data (Tokenization & Windowing).
    * Train the StockNet model.
    * Train the ARIMAX baseline.
    * Visualize Attention weights and Loss curves.

##  Performance

*Based on current testing benchmarks:*
* **Accuracy:** ~71%
* **ROC AUC:** ~0.79
* **MCC:** ~0.44
