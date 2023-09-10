# EE 782: Advanced Topics in Machine Learning
# Assignment 1: LSTM-based Stock Trading System
# NAME: Munish Monga
# ROLL NO: 22M2153
# Github Repo Link: https://github.com/munish30monga/lstm_based_stock_trading_system


import argparse
import pandas as pd                                             # for data handling  
import numpy as np                                              # for numerical computations               
import pathlib as pl                                            # for path handling
import matplotlib.pyplot as plt                                 # for plotting
import random                                                   # for random number generation
import mplfinance as mpl                                        # for candlestick plotting
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # for normalization
import seaborn as sns                                           # for plotting histograms
import torch                                                    # for deep learning functionality
import torch.nn as nn                                           # for LSTM model
from torch.utils.data import Dataset, DataLoader                # for data loading
from tqdm import tqdm                                           # for progress bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'         # set device to GPU if available

def create_df_dict(stocks):
    """
    Creates a dictionary of dataframes for each stock.
    
    Parameters:
    - stocks (list): A list of stock names.
    
    Returns:
    - df_dict (dict): A dictionary with stock names as keys and their corresponding dataframes as values.
    """
    data_dir = pl.Path('./dataset/sp500_tickers_A-D_1min_1pppix/') 
    df_dict = {}
    for stock in stocks:
        df = pd.read_csv(data_dir / f'{stock}_1min.txt', sep=',', header=None) # Reading the data
        df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']    # Naming the columns (see reference [1])
        df_dict[stock] = df
    return df_dict

def separate_datetime_dfs_dict(df_dict):
    """
    Separates the 'DateTime' column of each dataframe into separate 'Date' and 'Time' columns.
    
    Parameters:
    - df_dict (dict): Dictionary containing stock dataframes.
    
    Returns:
    - converted_dict (dict): Updated dictionary with separated 'Date' and 'Time' columns.
    """
    converted_dict = {}
    for stock, df in df_dict.items():
        df_copy = df.copy()
        df_copy['DateTime'] = pd.to_datetime(df_copy['DateTime'])
        df_copy['Date'] = df_copy['DateTime'].dt.date
        df_copy['Time'] = df_copy['DateTime'].dt.time
        converted_dict[stock] = df_copy
    return converted_dict

def get_latest_df_dict(df_dict, years_to_keep):
    """
    Filters each dataframe to keep only the latest data based on a given number of years.
    
    Parameters:
    - df_dict (dict): Dictionary containing stock dataframes.
    - years_to_keep (int, optional): Number of years of data to retain. Defaults to 10.
    
    Returns:
    - latest_df_dict (dict): Dictionary containing the latest data for each stock.
    """
    latest_df_dict = {}
    # df_dict_temp = separate_datetime_dfs_dict(df_dict)        # Ensure that the DateTime column is converted to Date and Time columns
    for stock, df in df_dict.items():
        df['DateTime'] = pd.to_datetime(df['DateTime'])       # Convert to datetime type
        latest_df = df[df['DateTime'] >= df['DateTime'].max() - pd.DateOffset(years=years_to_keep)]
        latest_df_dict[stock] = latest_df
    
    return latest_df_dict

def process_df_dict(latest_df_dict):
    """
    Processes the stock market data to exclude data outside trading hours and clean missing minutes or holidays.
    
    Parameters:
    - latest_df_dict (dict): Dictionary containing the latest data for each stock.
    
    Returns:
    - processed_df_dict (dict): Dictionary containing the processed data for each stock.
    """
    processed_df_dict = {}
    
    # Define trading hours
    trading_start = pd.to_datetime("9:30").time()           # 9:30 AM
    trading_end = pd.to_datetime("16:00").time()            # 4:00 PM
    
    for stock, df in latest_df_dict.items():
        
        # Exclude data outside of trading hours
        df = df[df['DateTime'].dt.time.between(trading_start, trading_end)]
        
        # Exclude data with missing minutes or days with holidays (assuming very low volume means holiday)
        volume_threshold = df['Volume'].quantile(0.05)     
        df = df[df['Volume'] > volume_threshold]
        
        processed_df_dict[stock] = df
        
    return processed_df_dict

def extend_df_dict(df_dict):
    """
    Adds a "DayOfWeek" column to each dataframe in the processed dictionary.
    
    Parameters:
    - df_dict (dict): Dictionary containing the processed data for each stock.
    
    Returns:
    - extended_df_dict (dict): Dictionary with dataframes containing an additional "DayOfWeek" column.
    """
    extended_df_dict = {}
    
    for stock, df in df_dict.items():
        # Extract day of the week from the DateTime column and add to the dataframe
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek + 1          # Mon-1 ... Fri-5
        extended_df_dict[stock] = df
        
    return extended_df_dict

def plot_day_by_day_CP(dfs_dict, combine_plots):
    dfs_dict = separate_datetime_dfs_dict(dfs_dict)     # Ensure that the DateTime column is converted to Date and Time columns
    stocks = list(dfs_dict.keys())
    n_stocks = len(stocks)
    
    if combine_plots:                           # If combine_plots is True, plot all the stocks in a single plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for stock in stocks:
            ax.plot(dfs_dict[stock]['Date'], dfs_dict[stock]['Close'], label=f'{stock} Closing Price')
        
        ax.set_title('Day-by-Day Closing Price Series')
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
        ax.xaxis.set_tick_params(rotation=45)
        
        plt.tight_layout()
        plt.show()
    else:
        nrows = (n_stocks + 1) // 2  # Calculate the number of rows needed for the given stocks

        # Initialize the subplots with 2 columns
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, 5 * nrows))
        
        # Flatten the axes for easy iteration, and then iterate only over the needed number of axes for the given stocks
        axes = axes.ravel()

        # Plot the data for each stock in stocks
        for i, stock in enumerate(stocks):
            ax = axes[i]
            
            # Plot the data using 'Date' for the x-axis and 'Close' for the y-axis
            ax.plot(dfs_dict[stock]['Date'], dfs_dict[stock]['Close'], label=f'{stock} Closing Price')
            
            # Set titles and labels
            ax.set_title(f'{stock} Day-by-Day Closing Price Series')
            ax.set_xlabel('Date')
            ax.set_ylabel('Closing Price')
            ax.legend()
            ax.grid(True)
            
            # Adjust the x-axis for better readability
            ax.xaxis.set_major_locator(plt.MaxNLocator(20))
            ax.xaxis.set_tick_params(rotation=45)

        # If the number of stocks is odd, remove the last unused subplot
        if n_stocks % 2 == 1:
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.show()
        
def analyse_stocks(stocks, disp_df = False, plot = False):
    df_dict = create_df_dict(stocks)
    latest_df_dict = get_latest_df_dict(df_dict, years_to_keep)
    df_dict = process_df_dict(latest_df_dict)
    if disp_df:
        for stock in stocks:
            print(f"Stock: {stock}")
            display(df_dict[stock])
    if plot:
        plot_day_by_day_CP(df_dict, combine_plots=True)
        
def scale_df_dict(df_dict):
    """
    Applies Min-Max Scaling and Scales the data in each dataframe between -1 and 1.
    
    Parameters:
    - df_dict (dict): Dictionary containing stock dataframes.
    
    Returns:
    - scaled_df_dict, scalers_dict (tuple): A tuple containing the scaled dataframes dictionary and a dictionary of scalers used for each column.
    """
    scaled_df_dict = {}
    scalers_dict = {}
    for stock, df in df_dict.items():
        scaled_df = pd.DataFrame()
        scaled_df['DateTime'] = df['DateTime']
        cols = list(df)[1:]
        for col in cols:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_df[col] = scaler.fit_transform(df[[col]].astype(float))
            scalers_dict[(stock, col)] = scaler
        scaled_df_dict[stock] = scaled_df
        
    return scaled_df_dict, scalers_dict

def train_test_df_split(df_dict):
    """
    Splits each dataframe into training, validation, and test sets.
    
    Parameters:
    - df_dict (dict): Dictionary containing stock dataframes.
    
    Returns:
    - train_df_dict, valid_df_dict, test_df_dict (tuple): A tuple containing dictionaries for training, validation, and test data for each stock.
    """    
    train_df_dict = {}
    valid_df_dict = {}
    test_df_dict = {}
    for stock, df in df_dict.items():
        df['DateTime'] = pd.to_datetime(df['DateTime'])  # Ensure the DateTime column is of datetime type

        # Find the date that is two years before the last date in the dataframe
        offset_date = df['DateTime'].iloc[-1] - pd.DateOffset(years=2)

        # Separate out the last two years for the test set
        mask = df['DateTime'] >= offset_date
        test_df = df[mask]

        # From the remaining data, split 10% for the validation set
        remaining_df = df[~mask]
        valid_length = int(0.1 * len(remaining_df))
        valid_df = remaining_df[-valid_length:]

        # The rest of the data is for the training set
        train_df = remaining_df[:-valid_length]

        train_df_dict[stock] = train_df
        valid_df_dict[stock] = valid_df
        test_df_dict[stock] = test_df

    return train_df_dict, valid_df_dict, test_df_dict

def df_to_tensors(df, seq_length, pred_horizon, predict = 'Close'): 
    """
    Converts a dataframe into input and target tensors for the LSTM model.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - seq_length (int): Sequence Length for LSTM.
    - pred_horizon (int): Prediction horizon.
    - predict (str, optional): Column name to predict. Defaults to 'Close'.
    
    Returns:
    - X, Y (tuple): Input and target tensors.
    """
    df = df.drop('DateTime', axis=1)
    df_as_np = np.array(df)
    predict_col_idx = list(df.columns).index(predict)
    X_list, Y_list = [], []
    for i in range(seq_length, len(df_as_np) - pred_horizon +1):
        X_list.append(df_as_np[i - seq_length:i, :df_as_np.shape[1]])
        Y_list.append(df_as_np[i + pred_horizon - 1:i + pred_horizon, predict_col_idx])       
    X_np, Y_np = np.array(X_list), np.array(Y_list)
    X, Y = torch.tensor(X_np, dtype=torch.float32), torch.tensor(Y_np, dtype=torch.float32)   
    return X, Y

# Defining a flexible LSTM model using PyTorch
class LSTM(nn.Module):
    """
    LSTM model for time series forecasting.
    
    Attributes:
    - input_dim (int): Number of input features.
    - hidden_dim (int): Number of hidden units.
    - num_layers (int): Number of LSTM layers.
    - output_dim (int): Number of output dimensions
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        
        # Hidden dimensions and number of layers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # Pass through the LSTM layers
        out, (hn,cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Only take the output from the final time step
        out = self.linear(out[:, -1, :])
        return out
    
class StockDataset(Dataset):
    """
    Dataset class for stock data.
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train_model(model, pred_stock, train_loader, valid_loader, num_epochs, learning_rate, save_best, patience):
    """
    Training Loop function
    
    Parameters:
    - model (LSTM): LSTM model to train.
    - train_loader (DataLoader): Training data loader.
    - valid_loader (DataLoader): Validation data loader.
    - num_epochs (int): Number of epochs for training.
    - learning_rate (float): Learning rate for optimization.
    - save_best (bool): Whether to save the best model.
    - patience (int): For early stopping.
    
    Returns:
    - tuple: Trained model, training losses per epoch, and validation losses per epoch.
    """
    # Transfer the model to the device
    model = model.to(device)
    
    # Define the loss function and the optimizer
    criterion = nn.MSELoss()  # Using Mean Squared Error Loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
     
    # Lists to store the average loss per epoch for training and validation
    train_losses = []
    valid_losses = []
    
    # Early stopping initialization
    epochs_no_improve = 0
    best_valid_loss = float('inf')
    best_model = None
    
    for epoch in range(1, num_epochs+1):
        model.train()  # Set the model to training mode
        running_train_loss = 0.0
        # Using tqdm for the training loop to display a progress bar
        pbar = tqdm(train_loader, total=len(train_loader), leave=False)
       
        for _, batch in enumerate(pbar):
            # Transfer sequences and labels to device
            X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            running_train_loss += loss.item()
            
            # Backward pass and optimization
            loss.backward()
            
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix(train_loss=loss.item())
        
        # Average training loss
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation loss
        model.eval()  # Set the model to evaluation mode
        running_valid_loss = 0.0
        for _, batch in enumerate(valid_loader):
            X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
            
            with torch.no_grad():
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                running_valid_loss += loss.item()
                
        # Average validation loss
        avg_valid_loss = running_valid_loss / len(valid_loader.dataset)
        valid_losses.append(avg_valid_loss)
        
        # Early stopping and best model saving logic
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model = model.state_dict()            
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Print epoch details
        print(f'''{"#"*100}
Epoch: [{epoch}/{num_epochs}] | Epoch Train Loss: {avg_train_loss} | Epoch Valid Loss: {avg_valid_loss}
{"#"*100}''')
        
        # Check early stopping condition
        if epochs_no_improve == patience:
            print(f"Early stopping!!, Since validation loss has not improved in the last {patience} epochs.")
            model.load_state_dict(best_model)  # Restore model to the best state
            break
        
    # Save the best model
    if save_best: 
        path = pl.Path("best_models")
        path.mkdir(parents=True, exist_ok=True)
        filename = f"Best_model_stock_{pred_stock}_epoch_{epoch}_valid_loss_{avg_valid_loss}.pt"
        print(f"{filename} saved in 'best_models' folder...") 
        torch.save(model.state_dict(), path / filename)

    return model, train_losses, valid_losses

def plot_losses(train_losses, valid_losses, title):
    """
    Plots training and validation losses per epoch.
    
    Parameters:
    - train_losses (list): Training losses per epoch.
    - valid_losses (list): Validation losses per epoch.
    """
    plots_dir = pl.Path('losses_plots')
    plots_dir.mkdir(exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plotting train losses
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
    
    # Plotting valid losses
    plt.plot(epochs, valid_losses, label='Validation Loss', marker='o', color='red')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = plots_dir / f"{title}.png"
    fig.savefig(plot_path, bbox_inches='tight')
    plt.show()
    
def descale_data(scaled_data, stock, column, scalers_dict):
    """
    Converts scaled data back to its original scale.
    
    Parameters:
    - scaled_data (array-like): Data that was scaled.
    - stock (str): Name of the stock.
    - column (str, optional): Column name of the data. Defaults to 'Close'.
    - scalers_dict (dict, optional): Dictionary of scalers used for each column. Defaults to scalers_dict.
    
    Returns:
    - array: Data in its original scale.
    """
    # Ensure scaled_data is a numpy array
    if isinstance(scaled_data, torch.Tensor):
        scaled_data_np = scaled_data.numpy()
    elif isinstance(scaled_data, pd.Series):
        scaled_data_np = scaled_data.values
    elif isinstance(scaled_data, list):
        scaled_data_np = np.array(scaled_data)
    else:
        scaled_data_np = scaled_data
    
    # Retrieve the appropriate scaler
    scaler = scalers_dict[(stock, column)]
    
    # Descale the data
    descaled_data = scaler.inverse_transform(scaled_data_np.reshape(-1, 1))
    
    return descaled_data

def create_pred_df_dict(df_dict, actual, predictions, seq_length, predict, pred_horizon):
    """
    Creates a dataframe for actual vs predicted values.
    
    Parameters:
    - df_dict (dict): Dictionary containing original dataframes.
    - actual (array): Actual values.
    - predictions (array): Predicted values.
    - seq_length (int, optional): Length of the sequence for LSTM. Defaults to 7.
    - predict (str, optional): Column name to predict. Defaults to 'Close'.
    - pred_horizon (int, optional): Prediction horizon. Defaults to 1.
    
    Returns:
    - dict: Dictionary containing dataframes with actual vs predicted values for each stock.
    """    
    pred_df_dict = {}
    
    for stock, df in df_dict.items():
        pred_df = pd.DataFrame()
        start_index = seq_length
        end_index = -pred_horizon + 1 if pred_horizon > 1 else None
        pred_df['DateTime'] = df['DateTime'].iloc[start_index:end_index]
        pred_df[f'{predict} (Actual)'] = actual[:, 0]
        pred_df[f'{predict} (Predicted)'] = predictions[:, 0]
        pred_df_dict[stock] = pred_df
        
    return pred_df_dict

def plot_actual_vs_predicted(df_dict, predict, title):
    """
    Plots actual vs predicted values for each stock.
    
    Parameters:
    - df_dict (dict): Dictionary containing dataframes with actual vs predicted values.
    - predict (str, optional): Column name to predict. Defaults to 'Close'.
    - title (str, optional): Plot title. 
    """

    plots_dir = pl.Path('predictions_plots')
    plots_dir.mkdir(exist_ok=True)
    df_dict = separate_datetime_dfs_dict(df_dict)
    stocks = list(df_dict.keys())
    for stock in stocks:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_dict[stock]['DateTime'], df_dict[stock][f'{predict} (Actual)'], label=f'{predict} Price (Actual)', color='blue')
        ax.plot(df_dict[stock]['DateTime'], df_dict[stock][f'{predict} (Predicted)'], label=f'{predict} Price (Predicted)', color='orange')
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
        ax.xaxis.set_tick_params(rotation=45)
        plt.tight_layout()
        plot_path = plots_dir / f"{title}.png"
        fig.savefig(plot_path, bbox_inches='tight')
        plt.show()
        
def test_model(model, test_loader):
    """
    Test function
    
    Parameters:
    - model (LSTM): Trained LSTM model.
    - test_loader (DataLoader): Test data loader.
    
    Returns:
    - tuple: Predicted values and test losses.
    """
    # Transfer the model to the device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # Define the loss function
    criterion = nn.MSELoss()

    # Lists to store the test losses and predictions
    test_losses = []
    predictions = []

    # Using tqdm for the test loop to display a progress bar
    pbar = tqdm(test_loader, total=len(test_loader), leave=False)

    with torch.no_grad(): # Ensure no gradients are computed
        for _, batch in enumerate(pbar):
            # Transfer sequences and labels to device
            X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            test_losses.append(loss.item())
            
            # Collect the predictions (move them to CPU for further analysis)
            batch_predictions = outputs.cpu().numpy().flatten()  # Flatten here
            predictions.extend(batch_predictions)
            
            # Update progress bar
            pbar.set_postfix(test_loss=loss.item())

    avg_test_loss = sum(test_losses) / len(test_loader.dataset)
    print(f'Average Test Loss: {avg_test_loss}')

    return predictions, test_losses

def load_best_model(model, stock):
    """
    Load the best model for a given stock from the 'best_models' folder.
    
    Parameters:
    - stock (str): Name of the stock.
    
    Returns:
    - model (LSTM): Trained LSTM model.
    """
    # Define the path to the 'best_models' folder
    models_path = pl.Path('best_models')
    
    # List all files in the 'best_models' folder and filter out model files for the given stock
    stock_models = [f for f in models_path.iterdir() if f.name.startswith(f"Best_model_stock_{stock}")]
    
    # Check if there are any models for the stock
    if not stock_models:
        print(f"No models found for stock: {stock}")
        return None
    
    # Sort the models based on validation loss (assuming naming convention holds)
    stock_models = sorted(stock_models, key=lambda x: float(x.name.split("_valid_loss_")[1].split(".pt")[0]))
        
    # Load the best model (with the lowest validation loss)
    best_model_path = stock_models[0]
    model.load_state_dict(torch.load(best_model_path))
        
    print(f"Loading best model for stock {stock} from {best_model_path}...")
    return model

def create_tema_df_dict(pred_df_dict, column):
    tema_df_dict = {}
    for stock, df in pred_df_dict.items():
        df = df.copy()
        df.set_index('DateTime', inplace=True)
        
        # Resampling the data to daily frequency for better visualization
        if column == "Open":
            df_daily = df.resample('D').first()
        else:  # Assuming 'Close' or other column
            df_daily = df.resample('D').last()

        # Drop NaN rows which might occur if there's no data for some days
        df_daily.dropna(inplace=True)

        # Calculate the EMAs on the resampled data
        df_daily['Short_EMA'] = df_daily[column].ewm(span=9, adjust=False).mean()
        df_daily['Middle_EMA'] = df_daily[column].ewm(span=21, adjust=False).mean()
        df_daily['Long_EMA'] = df_daily[column].ewm(span=55, adjust=False).mean()

        tema_df_dict[stock] = df_daily.reset_index()
    return tema_df_dict

def generate_signals(tema_df):
    buy_signals = []
    sell_signals = []
    
    position = 0  # 0: no position, 1: holding the stock
    
    for i in range(1, len(tema_df)):
        # Buy Signal
        if tema_df['Short_EMA'].iloc[i] > tema_df['Middle_EMA'].iloc[i] and \
           tema_df['Short_EMA'].iloc[i-1] <= tema_df['Middle_EMA'].iloc[i-1] and \
           tema_df['Middle_EMA'].iloc[i] > tema_df['Long_EMA'].iloc[i] and \
           position == 0:
            buy_signals.append(tema_df.index[i])
            position = 1

        # Sell Signal
        elif (tema_df['Short_EMA'].iloc[i] < tema_df['Middle_EMA'].iloc[i] or \
              tema_df['Middle_EMA'].iloc[i] < tema_df['Long_EMA'].iloc[i]) and \
              position == 1:
            sell_signals.append(tema_df.index[i])
            position = 0

    # Ensure that for every buy signal, there's a corresponding sell signal
    if len(buy_signals) > len(sell_signals):
        sell_signals.append(tema_df.index[-1])  # Add the last date as a sell signal

    return buy_signals, sell_signals

def buy_and_sell(tema_df_dict):
    buy_dict = {}
    sell_dict = {}
    
    for stock, df in tema_df_dict.items():
        buy_signals, sell_signals = generate_signals(df)
        buy_dict[stock] = buy_signals
        sell_dict[stock] = sell_signals

    return buy_dict, sell_dict

def plot_tema(tema_df_dict, buy_dict, sell_dict, predict, plot_title):
    # Ensure the directory exists
    plots_dir = pl.Path('tema_plots')
    plots_dir.mkdir(exist_ok=True)

    tema_df_dict = separate_datetime_dfs_dict(tema_df_dict)
    for stock, df in tema_df_dict.items():
        fig, ax = plt.subplots(figsize=(15, 10))
        
        ax.plot(df['Date'], df[predict], label=f'{predict} Price', color='blue')
        ax.plot(df['Date'], df['Short_EMA'], label='Short EMA', color='orange')
        ax.plot(df['Date'], df['Middle_EMA'], label='Middle EMA', color='green')
        ax.plot(df['Date'], df['Long_EMA'], label='Long EMA', color='red')
        
        # Extract buy and sell dates for this specific stock
        buy_dates = [date for date in buy_dict[stock] if date in df.index]
        sell_dates = [date for date in sell_dict[stock] if date in df.index]
        
        # Increase the marker size using the `s` parameter
        ax.scatter(df.loc[buy_dates, 'Date'], df.loc[buy_dates, predict], marker='^', color='green', label='Buy Signal', s=220)
        ax.scatter(df.loc[sell_dates, 'Date'], df.loc[sell_dates, predict], marker='v', color='red', label='Sell Signal', s=220)
        
        title = f"Triple Exponential Moving Average Plot for {stock}-{predict}"
        ax.set_title(title, fontsize=20)
        ax.set_xlabel('Date', fontsize=16)
        ax.set_ylabel('Closing Price', fontsize=16)
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
        ax.xaxis.set_tick_params(rotation=45)
        plt.tight_layout()
        # Save the plot to the "tema_plots" folder
        save_path = plots_dir / (plot_title + ".png")
        plt.savefig(save_path)
        plt.show()
        
def calculate_trade_cost(tema_df_dict, buy_dict, sell_dict, predict, bid_ask_spread, trade_commission):
    list_of_trades = []
    profitable_buy_dict = {}
    profitable_sell_dict = {}

    for stock, df in tema_df_dict.items():
        total_profit = 0.0
        profitable_buy_signals = []
        profitable_sell_signals = []

        buy_signals = buy_dict[stock]
        sell_signals = sell_dict[stock]

        for i in range(len(buy_signals)):
            buy_price = df[predict].loc[buy_signals[i]]
            sell_price = df[predict].loc[sell_signals[i]]

            # Calculate the cost due to bid-ask spread
            spread_cost_buy = bid_ask_spread / 100 * buy_price
            spread_cost_sell = bid_ask_spread / 100 * sell_price

            # Calculate the net profit or loss for this trade (excluding trade commission for now)
            trade_profit = sell_price - buy_price - spread_cost_buy - spread_cost_sell - 2 * trade_commission

            # Append to the list of trades
            trade_details = {
                'Stock Name': stock,
                'Buy Price': buy_price,
                'Sell Price': sell_price,
                'Profit/Loss': trade_profit
            }
            list_of_trades.append(trade_details)

            total_profit += trade_profit

            # If the trade was profitable, add it to the profitable trades dictionary
            if trade_profit > 0:
                profitable_buy_signals.append(buy_signals[i])
                profitable_sell_signals.append(sell_signals[i])

        profitable_buy_dict[stock] = profitable_buy_signals
        profitable_sell_dict[stock] = profitable_sell_signals

    return list_of_trades, profitable_buy_dict, profitable_sell_dict

def trading_module(pred_df_dict_valid, pred_df_dict_test, pred_stock, hyperparameters, plot = False):
    # Create tema_df_dict for validation and test sets
    tema_df_dict_valid = create_tema_df_dict(pred_df_dict_valid, column = f"{hyperparameters['predict']} (Actual)")
    tema_df_dict_test = create_tema_df_dict(pred_df_dict_test, column = f"{hyperparameters['predict']} (Predicted)")
    
    # Generate buy and sell signals for validation and test sets
    buy_signals_valid, sell_signals_valid = generate_signals(tema_df_dict_valid[pred_stock])
    buy_signals_test, sell_signals_test = generate_signals(tema_df_dict_test[pred_stock])
    
    # Create buy and sell dictionaries for validation and test sets
    buy_dict_valid, sell_dict_valid = buy_and_sell(tema_df_dict_valid)
    buy_dict_test, sell_dict_test = buy_and_sell(tema_df_dict_test)
    
    # Get list of trades and profitable buy/sell dictionaries for validation and test sets
    list_of_trades_valid, profitable_buy_dict_valid, profitable_sell_dict_valid = calculate_trade_cost(tema_df_dict_valid, buy_dict_valid, sell_dict_valid, predict=f"{hyperparameters['predict']} (Actual)", bid_ask_spread=hyperparameters['bid_ask_spread'], trade_commission=hyperparameters['trade_commision'])
    list_of_trades_test, profitable_buy_dict_test, profitable_sell_dict_test = calculate_trade_cost(tema_df_dict_test, buy_dict_test, sell_dict_test, predict=f"{hyperparameters['predict']} (Predicted)", bid_ask_spread=hyperparameters['bid_ask_spread'], trade_commission=hyperparameters['trade_commision'])
        
    # Print buy and sell signals for validation set
    print(f"Buy and sell signals for '{pred_stock}' on Validation Data:")
    print(f"Buy signals: {buy_signals_valid}")
    print(f"Sell signals: {sell_signals_valid}")
    print(f"Profitable Buy Signals: {profitable_buy_dict_valid[pred_stock]}")
    print(f"Profitable Sell Signals: {profitable_sell_dict_valid[pred_stock]}")
    print()
    
    # Print buy and sell signals for test set
    print(f"Buy and sell signals for '{pred_stock}' on Test Data:")
    print(f"Buy signals: {buy_signals_test}")
    print(f"Sell signals: {sell_signals_test}")
    print(f"Profitable Buy Signals: {profitable_buy_dict_test[pred_stock]}")
    print(f"Profitable Sell Signals: {profitable_sell_dict_test[pred_stock]}")
    print()
    
    if plot:
        print(f"Saving Triple EMA Crossover Plots for '{pred_stock}' in 'tema_plots' folder...")
        plot_title = f"Triple EMA Crossover Plot for '{pred_stock}' on Validation Data"
        plot_tema(tema_df_dict_valid, buy_dict_valid, sell_dict_valid, predict=f"{hyperparameters['predict']} (Actual)", plot_title=plot_title)
        plot_title = f"Triple EMA Crossover Plot for '{pred_stock}' on Test Data"
        plot_tema(tema_df_dict_test, buy_dict_test, sell_dict_test, predict=f"{hyperparameters['predict']} (Predicted)", plot_title=plot_title)
        plot_title = f"Triple EMA Crossover Plot for '{pred_stock}' on Validation Data (Profitable Trades Only)"
        plot_tema(tema_df_dict_valid, profitable_buy_dict_valid, profitable_sell_dict_valid, predict=f"{hyperparameters['predict']} (Actual)", plot_title=plot_title)
        plot_title = f"Triple EMA Crossover Plot for '{pred_stock}' on Test Data (Profitable Trades Only)"
        plot_tema(tema_df_dict_test, profitable_buy_dict_test, profitable_sell_dict_test, predict=f"{hyperparameters['predict']} (Predicted)", plot_title=plot_title)
        
    return list_of_trades_valid, list_of_trades_test, profitable_buy_dict_valid, profitable_sell_dict_valid, profitable_buy_dict_test, profitable_sell_dict_test

def align_df_dict(df_dict):
    """
    Aligns the DateTime values across multiple stock dataframes.
    
    Parameters:
    - df_dict (dict): Dictionary containing the latest data for each stock.
    
    Returns:
    - aligned_df_dict (dict): Dictionary where all stocks have the same DateTime values.
    """
    # Extract DateTime sequences for each stock
    datetime_lists = [set(df['DateTime'].values) for _, df in df_dict.items()]
    
    # Find the common DateTime values across all stocks
    common_datetimes = set.intersection(*datetime_lists)
    
    aligned_df_dict = {}
    for stock, df in df_dict.items():
        aligned_df = df[df['DateTime'].isin(common_datetimes)]
        aligned_df_dict[stock] = aligned_df.sort_values(by='DateTime')  # Ensure sorted order
        
    return aligned_df_dict

def multi_stock_df_to_tensors(df_dict, seq_length, pred_horizon, target_stock, predict='Close'):
    """
    Converts aligned dataframes of multiple stocks into input and target tensors for the LSTM model.
    
    Parameters:
    - df_dict (dict of pd.DataFrame): Dictionary of aligned dataframes, one for each stock.
    - seq_length (int): Sequence Length for LSTM.
    - pred_horizon (int): Prediction horizon.
    - target_stock (str): The stock we're aiming to predict.
    - predict (str, optional): Column name to predict. Defaults to 'Close'.
    
    Returns:
    - X, Y (tuple): Input and target tensors.
    """
    
    # Extract and combine data for all stocks (excluding DateTime columns)
    data_arrays = [df.drop('DateTime', axis=1).values for _, df in df_dict.items()]
    combined_data = np.concatenate([arr[..., np.newaxis] for arr in data_arrays], axis=-1)
    
    # Identify the target column index for the stock we're predicting
    target_col_idx = list(df_dict[target_stock].columns).index(predict)-1       # -1 since we dropped the DateTime column
    target_stock_idx = list(df_dict.keys()).index(target_stock)
    
    # Sequence extraction logic
    X_list, Y_list = [], []
    for i in range(seq_length, len(combined_data) - pred_horizon + 1):
        X_list.append(combined_data[i - seq_length:i])
        Y_list.append([combined_data[i + pred_horizon - 1, target_col_idx, target_stock_idx]])

    X_np, Y_np = np.array(X_list), np.array(Y_list)
    X, Y = torch.tensor(X_np, dtype=torch.float32), torch.tensor(Y_np, dtype=torch.float32)

    return X, Y

class MultiStockLSTM(nn.Module):
    """
    LSTM model for Multiple Stocks
    
    Attributes:
    - input_dim (int): Flattened dimension for number of input features multiplied by number of stocks.
    - hidden_dim (int): Number of hidden units.
    - num_layers (int): Number of LSTM layers.
    - output_dim (int): Number of output dimensions.
    """
    def __init__(self, input_dim, num_stocks, hidden_dim, num_layers, output_dim):
        super(MultiStockLSTM, self).__init__()
        
        # Hidden dimensions and number of layers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Flattened input dimension
        self.flattened_dim = input_dim * num_stocks             
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.flattened_dim, hidden_dim, num_layers, batch_first=True)
        
        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        # Reshape input data to flatten the last two dimensions
        x = x.view(x.size(0), x.size(1), -1)
        
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # Pass through the LSTM layers
        out, _ = self.lstm(x, (h0, c0))
        
        # Only take the output from the final time step
        out = self.linear(out[:, -1, :])
        return out
    
 
def main(stocks, pred_stock, save_plots=False, **kwargs):
    # Default hyperparameters
    hyperparameters = {
        'batch_size': 128,
        'random_seed': 42,
        'few_stocks': 4,
        'analyse': False,
        'start_date': '2020-01-01',
        'end_date': '2020-02-01',
        'hidden_dim': 32,
        'num_layers': 1,
        'seq_length': 10,
        'pred_horizon': 1,
        'predict': 'Close',
        'epochs': 10,
        'learning_rate': 0.01,
        'years_to_keep': 10,
        'save_best': False,
        'load_best': False,
        'patience': 4,
        'bid_ask_spread': 0.02,
        'trade_commision': 0.1,
        'trade': False,
        'add_day_of_week': False
    }
    multi_stock = False
    if len(stocks) > 1:
        multi_stock = True
    hyperparameters.update(kwargs)
    print(f'{"#"*100}\nLSTM-based Stock Trading System\n{"#"*100}')
    if multi_stock:
        print(f"List of Stocks: {stocks}")
    print(f"Predicting Stock: {pred_stock}")
    print(f"Hyperparameters: {hyperparameters}")
    # Override default hyperparameterss with provided arguments
    for key, value in kwargs.items():
        if key in hyperparameters:
            hyperparameters[key] = value
    
    # Analyse stocks if required
    if hyperparameters['analyse']:
        analyse_stocks(stocks, disp_df=True, plot=True, years_to_keep=hyperparameters['years_to_keep'])
    
    # Creating df_dict, scaling, and making X and Y's
    df_dict = create_df_dict(stocks)
    latest_df_dict = get_latest_df_dict(df_dict, hyperparameters['years_to_keep'])
    processed_df_dict = process_df_dict(latest_df_dict)
    if multi_stock:
        aligned_df_dict = align_df_dict(processed_df_dict)
        scaled_df_dict, scalers_dict = scale_df_dict(aligned_df_dict)
    elif hyperparameters['add_day_of_week']:
        print(f"Adding day of week as an input feature to {pred_stock}...")
        extended_df_dict = extend_df_dict(processed_df_dict)
        scaled_df_dict, scalers_dict = scale_df_dict(extended_df_dict)
    else:
        scaled_df_dict, scalers_dict = scale_df_dict(processed_df_dict)
    train_df_dict, valid_df_dict, test_df_dict = train_test_df_split(scaled_df_dict)
    
    # Making X and Y's
    if multi_stock:
        X_train, Y_train = multi_stock_df_to_tensors(train_df_dict, seq_length=hyperparameters['seq_length'], pred_horizon=hyperparameters['pred_horizon'], target_stock=pred_stock,predict=hyperparameters['predict'])
        X_valid, Y_valid = multi_stock_df_to_tensors(valid_df_dict, seq_length=hyperparameters['seq_length'], pred_horizon=hyperparameters['pred_horizon'], target_stock=pred_stock, predict=hyperparameters['predict'])
        X_test, Y_test = multi_stock_df_to_tensors(test_df_dict, seq_length=hyperparameters['seq_length'], pred_horizon=hyperparameters['pred_horizon'], target_stock=pred_stock, predict=hyperparameters['predict'])
    else:
        X_train, Y_train = df_to_tensors(train_df_dict[pred_stock], seq_length=hyperparameters['seq_length'], pred_horizon=hyperparameters['pred_horizon'], predict=hyperparameters['predict'])
        X_valid, Y_valid = df_to_tensors(valid_df_dict[pred_stock], seq_length=hyperparameters['seq_length'], pred_horizon=hyperparameters['pred_horizon'], predict=hyperparameters['predict'])
        X_test, Y_test = df_to_tensors(test_df_dict[pred_stock], seq_length=hyperparameters['seq_length'], pred_horizon=hyperparameters['pred_horizon'], predict=hyperparameters['predict'])
    
    # Getting training, validation, and test data loaders
    train_dataset = StockDataset(X_train, Y_train)
    valid_dataset = StockDataset(X_valid, Y_valid)
    test_dataset = StockDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=16, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=hyperparameters['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
    
    # Training the model
    print("Training model...")
    if multi_stock:
        model = MultiStockLSTM(input_dim=X_train.shape[2], num_stocks=X_train.shape[3], hidden_dim=hyperparameters['hidden_dim'], num_layers=hyperparameters['num_layers'], output_dim=Y_train.shape[1]).to(device)
    else:
        model = LSTM(input_dim=X_train.shape[2], hidden_dim=hyperparameters['hidden_dim'], num_layers=hyperparameters['num_layers'], output_dim=Y_train.shape[1]).to(device)
    if hyperparameters['load_best']:
        trained_model = load_best_model(model, pred_stock)
    else:
        trained_model, train_losses, valid_losses = train_model(model, pred_stock, train_loader, valid_loader, num_epochs=hyperparameters['epochs'], learning_rate=hyperparameters['learning_rate'], save_best=hyperparameters['save_best'], patience=hyperparameters['patience'])
    
    # Getting predictions and descaling on the validation sets
    with torch.no_grad():
        Y_valid_pred = trained_model(X_valid.to(device)).to('cpu').numpy()
    Y_valid_descaled = descale_data(scaled_data=Y_valid, stock=pred_stock, column=hyperparameters['predict'], scalers_dict=scalers_dict)
    Y_valid_pred_descaled = descale_data(scaled_data=Y_valid_pred, stock=pred_stock, column=hyperparameters['predict'], scalers_dict=scalers_dict)
    
    # Getting predictions on the validation set
    with torch.no_grad():
        Y_valid_pred = trained_model(X_valid.to(device)).to('cpu').numpy()
    Y_valid_descaled = descale_data(scaled_data=Y_valid, stock=pred_stock, column=hyperparameters['predict'], scalers_dict=scalers_dict)
    Y_valid_pred_descaled = descale_data(scaled_data=Y_valid_pred, stock=pred_stock, column=hyperparameters['predict'], scalers_dict=scalers_dict)
    print(f"Predictions on the validation set for {pred_stock}:")
    pred_df_dict_valid = create_pred_df_dict(valid_df_dict, Y_valid_descaled, Y_valid_pred_descaled, seq_length=hyperparameters['seq_length'], predict=hyperparameters['predict'], pred_horizon=hyperparameters['pred_horizon'])
    print(pred_df_dict_valid[pred_stock])
        
    # Getting predictions on the test set
    Y_test_pred, _ = test_model(trained_model, test_loader)
    Y_test_descaled = descale_data(scaled_data=Y_test, stock=pred_stock, column=hyperparameters['predict'], scalers_dict=scalers_dict)
    Y_test_pred_descaled = descale_data(scaled_data=Y_test_pred, stock=pred_stock, column=hyperparameters['predict'], scalers_dict=scalers_dict)
    print(f"Predictions on the test set for {pred_stock}:")
    pred_df_dict_test = create_pred_df_dict(test_df_dict, Y_test_descaled, Y_test_pred_descaled, seq_length=hyperparameters['seq_length'], predict=hyperparameters['predict'], pred_horizon=hyperparameters['pred_horizon'])
    print(pred_df_dict_test[pred_stock])
    
    if hyperparameters['trade']:
        if multi_stock:
            print("Trading not supported for multiple stocks!")
        else:
            print(f"Trading using the trained model for {pred_stock}...")
            list_of_trades_valid, list_of_trades_test, profitable_buy_dict_valid, profitable_sell_dict_valid, profitable_buy_dict_test, profitable_sell_dict_test = trading_module(pred_df_dict_valid, pred_df_dict_test, pred_stock, hyperparameters, save_plots)
    
    if save_plots:
        if not hyperparameters['load_best']:
            print(f"Saving Training and Validation Losses Plot for {pred_stock} in 'losses_plots' folder...")
            plot_title = f"Training and Validation Losses for {pred_stock}"
            plot_losses(train_losses, valid_losses, title=plot_title)
        print(f"Saving Actual vs. Predicted '{hyperparameters['predict']}' Prices Plot for {pred_stock} in 'predictions_plots' folder...")
        plot_title = f"Actual vs. Predicted '{hyperparameters['predict']}' Prices for {pred_stock} on Validation Set"
        plot_actual_vs_predicted(pred_df_dict_valid, predict=hyperparameters['predict'], title=plot_title)
        plot_title = f"Actual vs. Predicted '{hyperparameters['predict']}' Prices for {pred_stock} on Test Set"
        plot_actual_vs_predicted(pred_df_dict_test, predict=hyperparameters['predict'], title=plot_title)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an LSTM model for stock prediction')
    
    # Required arguments
    parser.add_argument('--stocks', nargs='+', help='Stocks to train the model on')
    parser.add_argument('--pred_stock', type=str, help='Target stock to predict from the list of input stocks')
    
    # Optional arguments
    parser.add_argument('--save_plots', type=bool, help='Plot actual vs predicted values')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--random_seed', type=int, help='Seed for reproducibility')
    parser.add_argument('--few_stocks', type=int, help='Used in Q1 to plot few stocks')
    parser.add_argument('--analyse', type=bool, help='Analyse the stocks & plot day-by-day closing price series')
    parser.add_argument('--start_date', type=str, help="Start date for data (default: '2020-01-01')")
    parser.add_argument('--end_date', type=str, help="End date for data (default: '2020-02-01')")
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimensions for LSTM')
    parser.add_argument('--num_layers', type=int, help='Number of layers in LSTM')
    parser.add_argument('--seq_length', type=int, help='Sequence length for LSTM model')
    parser.add_argument('--pred_horizon', type=int, help='Prediction horizon')
    parser.add_argument('--predict', type=str, help="Predict the 'Close' or 'Open' price")
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training')
    parser.add_argument('--years_to_keep', type=int, help='Number of years to keep for training')
    parser.add_argument('--save_best', type=bool, help='Save the best model')
    parser.add_argument('--load_best', type=bool, help='Load the best model')
    parser.add_argument('--patience', type=int, help='Patience for early stopping')
    parser.add_argument('--bid_ask_spread', type=float, help='Bid-ask spread for trading')
    parser.add_argument('--trade_commision', type=float, help='Trade commission for trading')
    parser.add_argument('--trade', type=bool, help='Trade using the trained model')
    parser.add_argument('--add_day_of_week', type=bool, help='Add day of week as an input feature')
    args = parser.parse_args()
    hyperparams = {key: value for key, value in vars(args).items() if value is not None and key not in ['stocks', 'pred_stock','save_plots']}
    main(args.stocks, args.pred_stock, args.save_plots, **hyperparams)

