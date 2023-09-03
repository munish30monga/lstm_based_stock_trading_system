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
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Date'] = df['DateTime'].dt.date
        df['Time'] = df['DateTime'].dt.time
        converted_dict[stock] = df
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
    for stock, df in df_dict.items():
        df['DateTime'] = pd.to_datetime(df['DateTime'])       # Convert to datetime type
        
        # Calculate the earliest DateTime value based on years_to_keep from the last date
        offset_date = df['DateTime'].max() - pd.DateOffset(years=years_to_keep)
        
        # If the earliest date in the dataframe is newer than offset_date, return the entire dataframe
        if df['DateTime'].min() > offset_date:
            print(f"Keeping all data for {stock} since it does not have {years_to_keep} years of data.")
            latest_df = df
        else:
            latest_df = df[df['DateTime'] >= offset_date]
            
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
        test_offset_date = df['DateTime'].iloc[-1] - pd.DateOffset(years=2)
        
        # Separate out the last two years for the test set
        mask_test = df['DateTime'] >= test_offset_date
        test_df = df[mask_test]
        
        # For the validation set, separate the data two years before the test set's start date
        valid_offset_date = test_df['DateTime'].iloc[0] - pd.DateOffset(years=2)
        mask_valid = (df['DateTime'] >= valid_offset_date) & (df['DateTime'] < test_df['DateTime'].iloc[0])
        valid_df = df[mask_valid]
        
        # The rest of the data is for the training set
        mask_train = df['DateTime'] < valid_df['DateTime'].iloc[0]
        train_df = df[mask_train]
        
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

def train_model(model, train_loader, valid_loader, num_epochs, learning_rate, regularize, weight_decay):
    """
    Training Loop function
    
    Parameters:
    - model (LSTM): LSTM model to train.
    - train_loader (DataLoader): Training data loader.
    - valid_loader (DataLoader): Validation data loader.
    - num_epochs (int): Number of epochs for training.
    - learning_rate (float): Learning rate for optimization.
    - regularize (bool): Whether to apply regularization.
    
    Returns:
    - tuple: Trained model, training losses per epoch, and validation losses per epoch.
    """
    # Transfer the model to the device
    model = model.to(device)
    
    # Define the loss function and the optimizer
    criterion = nn.MSELoss()  # Using Mean Squared Error Loss for regression
    if regularize:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
     
    # Lists to store the average loss per epoch for training and validation
    train_losses = []
    valid_losses = []
    
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
        
        # Print epoch details
        print(f'''{"#"*100}
Epoch: [{epoch}/{num_epochs}] | Epoch Train Loss: {avg_train_loss} | Epoch Valid Loss: {avg_valid_loss}
{"#"*100}''')
    
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
    - save_plot (bool, optional): If True, saves the plots in 'predictions_plots' folder. Defaults to False.
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
        
def main(stock, plot=False, **kwargs):
    # Default hyperparameters
    hyperparameters = {
        'batch_size': 128,
        'random_seed': 42,
        'few_stocks': 4,
        'start_date': '2020-01-01',
        'end_date': '2020-02-01',
        'hidden_dim': 32,
        'num_layers': 1,
        'seq_length': 10,
        'pred_horizon': 1,
        'predict': 'Close',
        'epochs': 10,
        'learning_rate': 0.01,
        'weight_decay': 0.03,
        'regularize': False,
        'years_to_keep': 10
    }
    hyperparameters.update(kwargs)
    print(f'{"#"*100}\nLSTM-based Stock Trading System\n{"#"*100}')
    print(f"Stock Name: {stock}")
    print(f"Hyperparameters: {hyperparameters}")
    # Override default hyperparameterss with provided arguments
    for key, value in kwargs.items():
        if key in hyperparameters:
            hyperparameters[key] = value
    
    # Your code for creating df_dict, scaling, and making X and Y's
    df_dict = create_df_dict([stock])
    latest_df_dict = get_latest_df_dict(df_dict, hyperparameters['years_to_keep'])
    processed_df_dict = process_df_dict(latest_df_dict)
    scaled_df_dict, scalers_dict = scale_df_dict(processed_df_dict)
    train_df_dict, valid_df_dict, test_df_dict = train_test_df_split(scaled_df_dict)
    
    # Making X and Y's
    X_train, Y_train = df_to_tensors(train_df_dict[stock], seq_length=hyperparameters['seq_length'], pred_horizon=hyperparameters['pred_horizon'], predict=hyperparameters['predict'])
    X_valid, Y_valid = df_to_tensors(valid_df_dict[stock], seq_length=hyperparameters['seq_length'], pred_horizon=hyperparameters['pred_horizon'], predict=hyperparameters['predict'])
    X_test, Y_test = df_to_tensors(test_df_dict[stock], seq_length=hyperparameters['seq_length'], pred_horizon=hyperparameters['pred_horizon'], predict=hyperparameters['predict'])
    
    # Getting training, validation, and test data loaders
    train_dataset = StockDataset(X_train, Y_train)
    valid_dataset = StockDataset(X_valid, Y_valid)
    test_dataset = StockDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=16, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=hyperparameters['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
    
    # Training the model
    print("Training model...")
    model = LSTM(input_dim=X_train.shape[2], hidden_dim=hyperparameters['hidden_dim'], num_layers=hyperparameters['num_layers'], output_dim=Y_train.shape[1]).to(device)
    trained_model, train_losses, valid_losses = train_model(model, train_loader, valid_loader, num_epochs=hyperparameters['epochs'], learning_rate=hyperparameters['learning_rate'], regularize=hyperparameters['regularize'], weight_decay=hyperparameters['weight_decay'])
    
    # Getting predictions and descaling on the validation sets
    with torch.no_grad():
        Y_valid_pred = trained_model(X_valid.to(device)).to('cpu').numpy()
    Y_valid_descaled = descale_data(scaled_data=Y_valid, stock=stock, column=hyperparameters['predict'], scalers_dict=scalers_dict)
    Y_valid_pred_descaled = descale_data(scaled_data=Y_valid_pred, stock=stock, column=hyperparameters['predict'], scalers_dict=scalers_dict)
    
    # Getting predictions and descaling on the test sets
    print("Getting predictions on the test set...")
    Y_test_pred, test_losses = test_model(trained_model, test_loader)
    Y_test_descaled = descale_data(scaled_data=Y_test, stock=stock, column=hyperparameters['predict'], scalers_dict=scalers_dict)
    Y_test_pred_descaled = descale_data(scaled_data=Y_test_pred, stock=stock, column=hyperparameters['predict'], scalers_dict=scalers_dict)
    
    # Print validation and test dataframes
    print(f"Predictions on the validation set for {stock}:")
    pred_df_dict_valid = create_pred_df_dict(valid_df_dict, Y_valid_descaled, Y_valid_pred_descaled, seq_length=hyperparameters['seq_length'], predict=hyperparameters['predict'], pred_horizon=hyperparameters['pred_horizon'])
    print(pred_df_dict_valid[stock])
    print(f"Predictions on the test set for {stock}:")
    pred_df_dict_test = create_pred_df_dict(test_df_dict, Y_test_descaled, Y_test_pred_descaled, seq_length=hyperparameters['seq_length'], predict=hyperparameters['predict'], pred_horizon=hyperparameters['pred_horizon'])
    print(pred_df_dict_test[stock])
    
    if plot:
        print("Training and validation loss plots are saved in the 'losses_plots' folder.")
        plot_losses(train_losses, valid_losses, title=f"Training and Validation Losses for {stock}")
        print("Actual vs. Predicted plots are saved in the 'predictions_plots' folder.")
        plot_actual_vs_predicted(pred_df_dict_valid, predict=hyperparameters['predict'], title=f"Actual vs. Predicted '{hyperparameters['predict']}' Prices for {stock} on Validation Set")
        plot_actual_vs_predicted(pred_df_dict_test, predict=hyperparameters['predict'], title=f"Actual vs. Predicted '{hyperparameters['predict']}' Prices for {stock} on Test Set")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an LSTM model for stock prediction')
    
    # Required argument
    parser.add_argument('stock', type=str, help='The stock to train the model on')
    
    # Optional arguments
    parser.add_argument('--plot', action='store_true', help='Plot actual vs predicted values')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--random_seed', type=int, help='Seed for reproducibility')
    parser.add_argument('--few_stocks', type=int, help='Used in Q1 to plot few stocks')
    parser.add_argument('--start_date', type=str, help="Start date for data (default: '2020-01-01')")
    parser.add_argument('--end_date', type=str, help="End date for data (default: '2020-02-01')")
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimensions for LSTM')
    parser.add_argument('--num_layers', type=int, help='Number of layers in LSTM')
    parser.add_argument('--seq_length', type=int, help='Sequence length for LSTM model')
    parser.add_argument('--pred_horizon', type=int, help='Prediction horizon')
    parser.add_argument('--predict', type=str, help="Predict the 'Close' or 'Open' price")
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for regularization')
    parser.add_argument('--regularize', type=bool, help='To regularize the model or not')
    parser.add_argument('--years_to_keep', type=int, help='Number of years to keep for training')
    args = parser.parse_args()
    hyperparams = {key: value for key, value in vars(args).items() if value is not None and key not in ['stock', 'plot']}
    main(args.stock, args.plot, **hyperparams)

