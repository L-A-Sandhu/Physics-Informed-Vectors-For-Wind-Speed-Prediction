import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Generator, Tuple
from tcn import TCN
import json
import pickle
import os
import random
import logging
import warnings
import shutil
from io import StringIO
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This suppresses TensorFlow logs
tf.get_logger().setLevel('ERROR')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Time Series Prediction with LSTM/TCN')
    parser.add_argument('--model_type', type=str, default='LSTM', choices=['LSTM', 'TCN','JOINT'],
                        help='Type of the model to use: LSTM or TCN')      
    parser.add_argument('--folder_path', type=str, default='', 
                        help='Path to the CSV folder containing the data')
    parser.add_argument('--folder_list', nargs='+', default='', 
                        help='Paths to folders containing CSV files to be merged')
    parser.add_argument('--side_folder_list', nargs='+', default='', 
                        help='Paths to folders containing CSV files to be merged for side data')
    parser.add_argument('--num_rows', type=int, default=None, 
                        help='Number of rows to load from the CSV file')
    parser.add_argument('--n_past', type=int, default=3, 
                        help='Number of past time steps to consider')
    parser.add_argument('--n_future', type=int, default=1, 
                        help='Number of future time steps to predict')
    parser.add_argument('--lemda', type=float, default=0.5, 
                        help='Lambda value for custom loss function')
    parser.add_argument('--n_units', nargs='+', default=[256], type=int, 
                        help='Number of units in each LSTM layer')
    parser.add_argument('--num_epochs', type=int, default=1000, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size for training')
    parser.add_argument('--initial_lr', type=float, default=0.0001, 
                        help='Initial learning rate')
    parser.add_argument('--inp', nargs='+', type=str, default=['train_custom'], 
                        help='List of actions to perform: train_custom, train_mse, test_custom, test_mse, infer')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Patience for early stopping')
    parser.add_argument('--save_metrics', action='store_true', default=False, 
                        help='Save training metrics to a JSON file')
    return parser.parse_args()

def save_losses(train_losses, val_losses, checkpoint_dir):
    # Ensure the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Define the file path for saving the losses
    filepath = os.path.join(checkpoint_dir, 'losses.pkl')
    
    # Save the losses as a dictionary
    with open(filepath, 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

    print(f"Losses saved to {filepath}")


def load_data(folder_path, rows):
    all_dataframes = []
    for root, _, files in os.walk(folder_path):
        existing_csv_file = os.path.join(root, "merged_data.csv")
        if os.path.exists(existing_csv_file):
            dataframe = pd.read_csv(existing_csv_file)
            all_dataframes.append(dataframe)
        else:
            csv_files = [file for file in files if file.endswith(".csv")]
            if csv_files:
                dataframes = [pd.read_csv(os.path.join(root, csv_file), skiprows=2, nrows=rows) for csv_file in csv_files]
                concatenated_dataframe = pd.concat(dataframes, ignore_index=True)
                concatenated_dataframe.sort_values(by=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
                concatenated_dataframe.to_csv(os.path.join(root, "merged_data.csv"), index=False)
                all_dataframes.append(concatenated_dataframe)
    return pd.concat(all_dataframes, ignore_index=True) if all_dataframes else None

def load_data_from_folders(folder_paths, num_rows=None):
    all_dataframes = []
    for folder_path in folder_paths:
        existing_csv_file = os.path.join(folder_path, "merged_data_folder.csv")
        if os.path.exists(existing_csv_file):
            all_dataframes.append(pd.read_csv(existing_csv_file))
        else:
            csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
            if csv_files:
                dataframes = [pd.read_csv(os.path.join(folder_path, csv_file), skiprows=2, nrows=num_rows) for csv_file in csv_files]
                merged_dataframe = pd.concat(dataframes, ignore_index=True)
                merged_dataframe.sort_values(by=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
                merged_dataframe.to_csv(os.path.join(folder_path, "merged_data_folder.csv"), index=False)
                all_dataframes.append(merged_dataframe)
    return pd.concat(all_dataframes, ignore_index=True) if all_dataframes else None

def create_side_by_side_dataframe(folder_paths, num_rows=None):
    paths_needing_merge = [folder_path for folder_path in folder_paths if not os.path.exists(os.path.join(folder_path, "merged_data_folder.csv"))]

    # If there are any paths needing merge, call the second function
    if paths_needing_merge:
        load_data_from_folders(paths_needing_merge, num_rows)

    all_dataframes = [pd.read_csv(os.path.join(folder_path, "merged_data_folder.csv")) for folder_path in folder_paths if os.path.exists(os.path.join(folder_path, "merged_data_folder.csv"))]

    return pd.concat(all_dataframes, axis=1) if all_dataframes else pd.DataFrame()
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dense, Dropout
from tensorflow.keras import layers
def load_and_prepare_model(model_path, output_removed=True):
    model = load_model(model_path)
    if output_removed:
        model = Model(inputs=model.input, outputs=model.layers[-2].output)
    return model
def custom_transformer_block(lstm_features, tcn_features, num_heads=8, ff_dim=2048, dropout_rate=0.2):
    # Dynamically adjust d_model to match the dimensionality of lstm_features
    d_model = lstm_features.shape[-1]
    
    # Ensure lstm_features and tcn_features have a sequence length dimension
    lstm_features_reshaped = tf.expand_dims(lstm_features, axis=1)  # Adding sequence length dimension
    tcn_features_reshaped = tf.expand_dims(tcn_features, axis=1)
    
    # Cross-attention: LSTM as key, TCN as query
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(query=tcn_features_reshaped, key=lstm_features_reshaped, value=lstm_features_reshaped)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + tcn_features_reshaped)
    
    # Feed-Forward Network
    ffn_output = Dense(ff_dim, activation='relu')(attention_output)
    ffn_output = Dense(d_model)(ffn_output)  # Second linear transformation to match the dimensionality of lstm_features
    ffn_output = Dropout(dropout_rate)(ffn_output)
    transformer_output = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
    
    # Removing the sequence length dimension before returning, if necessary
    transformer_output = tf.squeeze(transformer_output, axis=1)
    
    return transformer_output


def make_joint_model(NN=1, lstm_path="", tcn_path="", custom=True, lstm_trainable=True, tcn_trainable=True):
    # Load models
    model_type = 'model_custom' if custom else 'model_mse'
    lstm_model = load_and_prepare_model(f"{lstm_path}/checkpoint/LSTM/{model_type}")
    tcn_model = load_and_prepare_model(f"{tcn_path}/checkpoint/TCN/{model_type}")
    
    # Verify input shapes
    if lstm_model.input_shape != tcn_model.input_shape:
        raise ValueError("Input shapes of LSTM and TCN models do not match.")
    
    # Prepare input layer
    input_layer = Input(shape=lstm_model.input_shape[1:])
    
    # Process LSTM and TCN models
    lstm_features = lstm_model(input_layer)
    tcn_features = tcn_model(input_layer)
    print("LSTM features shape:", lstm_features.shape)
    print("TCN features shape:", tcn_features.shape)

    # Concatenating features is no longer needed as we use cross-attention
    x = lstm_features
    for _ in range(NN):
        x = custom_transformer_block(lstm_features, tcn_features, num_heads=8, ff_dim=1024, dropout_rate=0.2)
    
    # Set models' trainable flag
    lstm_model.trainable = lstm_trainable
    tcn_model.trainable = tcn_trainable
    
    # Assuming the output layer from LSTM and TCN has the same shape and can be represented by a Dense layer
    output_layer_shape = lstm_model.layers[-1].output_shape[-1]  # Assuming both models have the same output shape
    output_layer = Dense(output_layer_shape, activation='linear')(x)  # Change activation according to your needs
    
    # Create the joint model
    joint_model = Model(inputs=input_layer, outputs=output_layer)
    
    return joint_model





def preprocess_data(data, A, n_past, n_future, split_ratio=0.8, n_infer=200):
    target_column = 'Wind Speed'
    #wind_speed_data = data[target_column].values[:-1]
    wind_speed_data = data[target_column].values[1:]
    data_modified = data.iloc[:-1].copy()
    A_modified = A.iloc[:-1].copy()
    A_numpy = A_modified.values
    print(np.shape(A_numpy))
    print(np.shape(wind_speed_data))
    X = np.dot(np.linalg.pinv(A_numpy), wind_speed_data)
    data_modified['Estimated Wind Speed'] = np.dot(A_numpy, X)
    min_vals, max_vals = data_modified.min(), data_modified.max()
    data_modified = data_modified.apply(lambda x: (x - min_vals[x.name]) / (max_vals[x.name] - min_vals[x.name]) if min_vals[x.name] != max_vals[x.name] else 1)
    
    def create_datasets(data_modified, n_past, n_future):
        x, x_with_estimate, y_actual, y_estimated = [], [], [], []
        for i in range(len(data_modified) - n_past - n_future + 1):
            x.append(data_modified.iloc[i: i + n_past].drop('Estimated Wind Speed', axis=1).values)
            x_with_estimate.append(data_modified.iloc[i: i + n_past].values)
            y_actual.append(data_modified[target_column].iloc[i + n_past: i + n_past + n_future].values)
            y_estimated.append(data_modified['Estimated Wind Speed'].iloc[i + n_past: i + n_past + n_future].values)
        return np.array(x), np.array(x_with_estimate), np.array(y_actual), np.array(y_estimated)

    x_train, x_train_with_estimate, y_train_actual, y_train_estimated = create_datasets(data_modified, n_past, n_future)
    
    x_train_infer = x_train[-n_infer:]
    y_train_actual_infer = y_train_actual[-n_infer:]
    y_train_estimated_infer = y_train_estimated[-n_infer:]
    x_train_with_estimate_infer = x_train_with_estimate[-n_infer:]

    x_train = x_train[:-n_infer]
    y_train_actual = y_train_actual[:-n_infer]
    y_train_estimated = y_train_estimated[:-n_infer]
    x_train_with_estimate = x_train_with_estimate[:-n_infer]

    indices = np.random.permutation(x_train.shape[0])
    split_idx = int(len(indices) * split_ratio)
    return (x_train[indices[:split_idx]], x_train[indices[split_idx:]],
            x_train_with_estimate[indices[:split_idx]], x_train_with_estimate[indices[split_idx:]],
            y_train_actual[indices[:split_idx]], y_train_actual[indices[split_idx:]],
            y_train_estimated[indices[:split_idx]], y_train_estimated[indices[split_idx:]],
            x_train_infer, y_train_actual_infer, y_train_estimated_infer,x_train_with_estimate_infer, min_vals, max_vals)

def data_generator(x_data: np.ndarray, y_data_actual: np.ndarray, y_data_estimated: np.ndarray, batch_size: int) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    data_size = len(x_data)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    for start_idx in range(0, data_size, batch_size):
        end_idx = min(start_idx + batch_size, data_size)
        batch_indices = indices[start_idx:end_idx]
        yield x_data[batch_indices], y_data_actual[batch_indices], y_data_estimated[batch_indices]

def create_tcn_model(input_shape, n_units_list, n_future):
    # Assuming necessary imports and setup are done above
    model = Sequential()
    # Add the first TCN layer outside the loop to specify input_shape
    model.add(TCN(input_shape=input_shape,
                  nb_filters=n_units_list[0],
                  kernel_size=3,  # Example kernel_size, adjust as needed
                  dilations=[1, 2, 4, 8, 16],
                  padding='causal',
                  return_sequences=(len(n_units_list) > 1)))
    model.add(Dropout(0.2))
    for nb_filters in n_units_list[1:]:
        model.add(TCN(nb_filters=nb_filters,
                      kernel_size=3,  # Example kernel_size, adjust as needed
                      dilations=[1, 2, 4, 8, 16],
                      padding='causal',
                      return_sequences=(nb_filters != n_units_list[-1])))
        model.add(Dropout(0.2))
    model.add(Dense(n_future, activation='linear'))
    return model

def create_lstm_model(input_shape, n_units_list, n_future):
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    model = Sequential()
    for i, n_units in enumerate(n_units_list):
        is_return_sequences = i < len(n_units_list) - 1
        if i == 0:
            model.add(LSTM(units=n_units, activation="tanh", return_sequences=is_return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units=n_units, activation="tanh", return_sequences=is_return_sequences))
        model.add(Dropout(0.2))
    model.add(Dense(units=n_future))
    return model
def custom_loss(y_true, y_pred, y_estimated, lambda_value):
    # Cast all tensors to float64
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    y_estimated = tf.cast(y_estimated, tf.float64)
    lambda_value = tf.cast(lambda_value, tf.float64)
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    additional_term = tf.reduce_mean(tf.square(y_pred - y_estimated), axis=-1)
    return mse + lambda_value * additional_term, mse
def get_model_summary(model):
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
def draw_box(contents, width=80):
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    left_padding = max((terminal_width - width) // 2, 0)
    padding_str = ' ' * left_padding
    bold_start = '\033[1m'
    bold_end = '\033[0m'
    top_bottom_border = bold_start + "+" + "-" * (width - 2) + "+" + bold_end
    padded_contents = [content.center(width - 4) for content in contents.split('\n')]
    print("\n" * 5)
    print(padding_str + top_bottom_border)
    for line in padded_contents:
        print(padding_str + "|" + line + "|")
    print(padding_str + top_bottom_border)
def train_model(model, x_train, y_train_actual, y_train_estimated, initial_lambda_value, num_epochs, batch_size, initial_lr, checkpoint_dir, patience, use_custom_loss=True, decay_epoch=10):
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    checkpoint_dir = os.path.join(checkpoint_dir, 'model_custom' if use_custom_loss else 'model_mse')
    optimizer = Adam(learning_rate=initial_lr)
    best_loss = np.inf
    patience_counter = 0
    train_losses, val_losses = [], []
    # Adjust decay rate
    k = -np.log(0.01) / decay_epoch
    model_summary = get_model_summary(model)
    current_file_path = os.path.abspath(__file__)
    for epoch in range(num_epochs):
        lambda_value = initial_lambda_value * np.exp(-k * epoch) if epoch < decay_epoch else 0
        train_loss, val_loss = train_epoch(model, optimizer, x_train, y_train_actual, y_train_estimated, lambda_value, batch_size, use_custom_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        box_content = f"Checkpoint directory: {checkpoint_dir}\nModel Summary:\n{model_summary}" + \
              f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n" + \
              f"Custom Loss: {use_custom_loss}\n" + \
              "Current investigation\n" + \
              f" {current_file_path}"
        clear_screen()
        draw_box(box_content)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            model.save(os.path.join(checkpoint_dir), save_format='tf')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                draw_box("Early stopping due to no improvement")
                break
    plot_training_validation_loss(train_losses, val_losses, checkpoint_dir)
    save_losses(train_losses, val_losses, checkpoint_dir)

def train_epoch(model, optimizer, x_train, y_train_actual, y_train_estimated, lambda_value, batch_size, use_custom_loss):
    epoch_train_loss, epoch_val_loss = [], []
    for x_batch, y_actual_batch, y_estimated_batch in data_generator(x_train, y_train_actual, y_train_estimated, batch_size):
        with tf.GradientTape() as tape:
            y_pred_batch = model(x_batch, training=True)
            if use_custom_loss:
                loss, mse_1_train = custom_loss(y_actual_batch, y_pred_batch, y_estimated_batch, lambda_value)
            else:
                loss = tf.keras.losses.mean_squared_error(y_actual_batch, y_pred_batch)
                mse_1_train=loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_train_loss.append(tf.reduce_mean(mse_1_train).numpy())
    return np.mean(epoch_train_loss), calculate_validation_loss(model, x_train, y_train_actual, y_train_estimated, lambda_value, use_custom_loss)
def calculate_validation_loss(model, x_val, y_val_actual, y_val_estimated, lambda_value, use_custom_loss):
    val_preds = model.predict(x_val)
    if use_custom_loss:
        val_loss, mse_1_val = custom_loss(y_val_actual, val_preds, y_val_estimated, lambda_value)
    else:
        val_loss = tf.keras.losses.mean_squared_error(y_val_actual, val_preds)
        mse_1_val = val_loss  
    return np.mean(mse_1_val.numpy())

def save_preprocessed_data(X_train, X_test, X_train_with_estimate, X_test_with_estimate,
                           y_train_actual, y_test_actual, y_train_estimated, y_test_estimated,
                           x_train_infer, y_train_actual_infer, y_train_estimated_infer, x_train_with_estimate_infer,
                           min_vals, max_vals, directory='./data'):
    """
    Saves preprocessed data arrays and json files to the specified directory.

    Parameters:
    - X_train, X_test, ...: Numpy arrays of processed data.
    - min_vals, max_vals: Dictionary of minimum and maximum values to be saved as JSON.
    - directory: The path to the directory where files will be saved.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save arrays to the disk
    np.save(os.path.join(directory, 'X_train.npy'), X_train)
    np.save(os.path.join(directory, 'X_test.npy'), X_test)
    np.save(os.path.join(directory, 'X_train_with_estimate.npy'), X_train_with_estimate)
    np.save(os.path.join(directory, 'X_test_with_estimate.npy'), X_test_with_estimate)
    np.save(os.path.join(directory, 'y_train_actual.npy'), y_train_actual)
    np.save(os.path.join(directory, 'y_test_actual.npy'), y_test_actual)
    np.save(os.path.join(directory, 'y_train_estimated.npy'), y_train_estimated)
    np.save(os.path.join(directory, 'y_test_estimated.npy'), y_test_estimated)
    np.save(os.path.join(directory, 'x_train_infer.npy'), x_train_infer)
    np.save(os.path.join(directory, 'y_train_actual_infer.npy'), y_train_actual_infer)
    np.save(os.path.join(directory, 'y_train_estimated_infer.npy'), y_train_estimated_infer)
    np.save(os.path.join(directory, 'x_train_with_estimate_infer.npy'), x_train_with_estimate_infer)

    # Ensure the min_vals and max_vals are in a serializable format
    # Convert Pandas Series to a Python dict if necessary
    if isinstance(min_vals, pd.Series):
        min_vals = min_vals.to_dict()
    if isinstance(max_vals, pd.Series):
        max_vals = max_vals.to_dict()

    # Save min_vals and max_vals as JSON
    with open(os.path.join(directory, 'min_vals.json'), 'w') as f:
        json.dump(min_vals, f)
    with open(os.path.join(directory, 'max_vals.json'), 'w') as f:
        json.dump(max_vals, f)




def plot_training_validation_loss(train_losses, val_losses, checkpoint_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'training_validation_loss.png'))

def evaluate_model(model_path, x_test, y_test, batch_size, max_value, min_value):
    model = load_model(model_path)
    predictions = model.predict(x_test, batch_size=batch_size)
    predictions_denormalized = denormalize(predictions, min_value, max_value)
    y_test_denormalized = denormalize(y_test, min_value, max_value)
    mse = mean_squared_error(y_test_denormalized, predictions_denormalized)
    mae = mean_absolute_error(y_test_denormalized, predictions_denormalized)
    r2 = r2_score(y_test_denormalized, predictions_denormalized)
    model_size = os.path.getsize(model_path)
    test_metrics = {'mse': mse, 'mae': mae, 'r2_score': r2, 'model_size_bytes': model_size}
    with open(model_path + '_test_metrics.json', 'w') as json_file:
        json.dump(test_metrics, json_file)

def denormalize(data, min_value, max_value):
    return data * (max_value - min_value) + min_value
def main():
    args = parse_arguments()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.folder_path:
        data = load_data(args.folder_path, args.num_rows)
    elif args.folder_list:
        data = load_data_from_folders(args.folder_list, args.num_rows)
    else:
        raise ValueError("No valid data source specified.")
    model_path = f'./checkpoint/{args.model_type}/'
    side_merge_data = create_side_by_side_dataframe(args.side_folder_list, args.num_rows)
    (X_train, X_test, X_train_with_estimate, X_test_with_estimate,
     y_train_actual, y_test_actual, y_train_estimated, y_test_estimated,
     x_train_infer, y_train_actual_infer, y_train_estimated_infer,x_train_with_estimate_infer,
     min_vals, max_vals) = preprocess_data(data, side_merge_data, args.n_past, args.n_future)
    save_preprocessed_data(X_train, X_test, X_train_with_estimate, X_test_with_estimate,
                       y_train_actual, y_test_actual, y_train_estimated, y_test_estimated,
                       x_train_infer, y_train_actual_infer, y_train_estimated_infer, x_train_with_estimate_infer,
                       min_vals, max_vals)

    model_path = f'./checkpoint/{args.model_type}/'
    for action in args.inp:
        if action == 'train_custom':
            if args.model_type == 'LSTM':
                model = create_lstm_model(input_shape=(X_train_with_estimate.shape[1], X_train_with_estimate.shape[2]), n_units_list=args.n_units, n_future=args.n_future)
            elif args.model_type == 'TCN':
                model = create_tcn_model(input_shape=(X_train_with_estimate.shape[1], X_train_with_estimate.shape[2]), n_units_list=args.n_units, n_future=args.n_future)
            elif args.model_type == 'JOINT':
                model=make_joint_model(NN=1, lstm_path="./", tcn_path="./", custom=True, lstm_trainable=True, tcn_trainable=True)
            else:
                raise ValueError("Invalid model type specified.")
            train_model(model, X_train_with_estimate, y_train_actual, y_train_estimated, args.lemda, args.num_epochs, args.batch_size, args.initial_lr, model_path, args.patience)
        elif action == 'train_mse':
            if args.model_type == 'LSTM':
                model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), n_units_list=args.n_units, n_future=args.n_future)
            elif args.model_type == 'TCN':
                model = create_tcn_model(input_shape=(X_train.shape[1], X_train.shape[2]), n_units_list=args.n_units, n_future=args.n_future)
            elif args.model_type == 'JOINT':
                model=make_joint_model(NN=1, lstm_path="./", tcn_path="./", custom=False, lstm_trainable=True, tcn_trainable=True)
            else:
                raise ValueError("Invalid model type specified.")
            train_model(model, X_train, y_train_actual, y_train_estimated, args.lemda, args.num_epochs, args.batch_size, args.initial_lr, model_path, args.patience, use_custom_loss=False)
        
        elif action == 'test_custom':
            model_path_custom=model_path+'/model_custom'
            evaluate_model(model_path_custom, X_test_with_estimate, y_test_actual, args.batch_size, max_vals['Wind Speed'], min_vals['Wind Speed'])
        elif action == 'test_mse':
            model_path_mse=model_path+'/model_mse'
            evaluate_model(model_path_mse, X_test, y_test_actual, args.batch_size, max_vals['Wind Speed'], min_vals['Wind Speed'])
        elif action == 'infer':
            perform_inference(model_path, x_train_infer,x_train_with_estimate_infer, y_train_actual_infer, min_vals['Wind Speed'], max_vals['Wind Speed'])
def perform_inference(model_path, x_train_infer,x_train_with_estimate_infer, y_train_actual_infer, min_wind_speed, max_wind_speed):
    path_custom=os.path.join(model_path, 'model_custom')
    model_custom = load_model(path_custom)
    path_mse=os.path.join(model_path, 'model_mse')
    model_mse = load_model(path_mse)
    predicted_custom = model_custom.predict(x_train_with_estimate_infer)
    predicted_mse = model_mse.predict(x_train_infer)
    compare_predictions(predicted_custom, predicted_mse, y_train_actual_infer, min_wind_speed, max_wind_speed,model_path)

def compare_predictions(predicted_custom, predicted_mse, y_train_actual, min_val, max_val,model_path):
    predicted_custom_denorm = denormalize(predicted_custom, min_val, max_val)
    predicted_mse_denorm = denormalize(predicted_mse, min_val, max_val)
    y_actual_denorm = denormalize(y_train_actual, min_val, max_val)
    plot_predictions(predicted_custom_denorm, predicted_mse_denorm, y_actual_denorm, model_path+'predicted_vs_actual_wind_speed.png')
    plot_errors(predicted_custom_denorm, predicted_mse_denorm, y_actual_denorm, model_path+'prediction_errors.png')


import pickle
def plot_predictions(predicted_custom, predicted_mse, actual, filename):
    pickle_filename = os.path.splitext(filename)[0] + '.pickle'
    fig = plt.figure(figsize=(12, 6))
    plt.plot(predicted_custom, label='Predicted Wind Speed (with PIV)', color='blue')
    plt.plot(predicted_mse, label='Predicted Wind Speed (Without PIV)', color='green')
    plt.plot(actual, label='Actual Wind Speed', color='orange')
    plt.title('Actual vs Predicted Wind Speed', fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.xlabel('Time Steps', fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.ylabel('Wind Speed', fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.legend(prop={'size': 16, 'weight': 'bold'})
    plt.savefig(filename)

    # Save the figure for later use with pickle
    with open(pickle_filename, 'wb') as f:
        pickle.dump(fig, f)
    
    plt.close(fig)  



def plot_errors(predicted_custom, predicted_mse, actual, filename):
    error_custom = np.abs(actual - predicted_custom)
    error_mse = np.abs(actual - predicted_mse)
    plt.figure(figsize=(12, 6))
    plt.plot(error_custom, label='Error (Custom Loss)', color='blue')
    plt.plot(error_mse, label='Error (MSE)', color='green')
    plt.title('Error in Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(filename)

if __name__ == "__main__":
    main()
