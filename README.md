## EE 782: Advanced Topics in Machine Learning

### LSTM-based Stock Trading System

### Instructions:
1. Clone the repository using:
    ```bash
    git clone https://github.com/munish30monga/lstm_based_stock_trading_system
   ```
2. Navigate to the project directory: 
   ```bash
    cd lstm_based_stock_trading_system
   ```
3. Install the requirements using: 
    ```bash
    pip install -r requirements.txt
    ```
    or
    ```bash
    conda env create -f EE_782.yml
    ```
4. Give the appropriate command line argument, check the help using:
    ```bash
    python 22m2153.py --help
    ```

## Command-Line Arguments

- `--stocks`: List of stock tickers to train the model on.
- `--pred_stock`: Target stock to predict from the list of input stocks.
- `--save_plots`: Flag to determine if plots should be saved (True/False).
- `--batch_size`: Batch size for training.
- `--random_seed`: Seed for reproducibility.
- `--few_stocks`: Used to plot a limited number of stocks.
- `--analyse`: Analyze the stocks & plot day-by-day closing price series (True/False).
- `--start_date`: Start date for data (default: '2020-01-01').
- `--end_date`: End date for data (default: '2020-02-01').
- `--hidden_dim`: Hidden dimensions for LSTM.
- `--num_layers`: Number of layers in LSTM.
- `--seq_length`: Sequence length for LSTM model.
- `--pred_horizon`: Prediction horizon.
- `--predict`: Target column to predict (e.g., 'Close' or 'Open').
- `--epochs`: Number of epochs for training.
- `--learning_rate`: Learning rate for training.
- `--years_to_keep`: Number of years of data to retain for training.
- `--save_best`: Save the best model during training (True/False).
- `--load_best`: Load the best model (True/False).
- `--patience`: Number of epochs for early stopping.
- `--bid_ask_spread`: Bid-ask spread for trading.
- `--trade_commision`: Trade commission for trading.
- `--trade`: Flag to determine if trading should be done using the trained model (True/False).
- `--add_day_of_week`: Add the day of the week as an input feature (True/False).

## Example Usage

1. **Basic Usage with Single Stock:**
   
   ```bash
   python 22m2153.py --stocks AAPL --pred_stock AAPL
   ```
2. **Using Multiple Stocks:**
   
   ```bash
   python 22m2153.py --stocks AAPL CSCO BSX DRE --pred_stock AAPL
   ``` 
3. **Invoking Trading Module (use only with Single Stock):**
   
   ```bash
   python 22m2153.py --stocks AAPL --pred_stock AAPL --trade True --save_plots True
   ```  
4. **Changing Hyperparameters:**
   
   ```bash
   python 22m2153.py --stocks AAPL --pred_stock AAPL --seq_length 50 --pred_horizon 5 --add_day_of_week True 
   ```

### Owner:
[![Munish](https://img.shields.io/badge/22M2153-Munish_Monga-blue)](https://github.com/munish30monga)