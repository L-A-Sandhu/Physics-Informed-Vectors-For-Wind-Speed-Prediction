import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, Flatten, Dense, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Function to load data
def load_data(city, dataset_type):
    X = []
    y = []
    label_folder_base = f"./data_final/{city}/{dataset_type}/"

    for label in [0, 1]:
        label_folder = os.path.join(label_folder_base, str(label))
        for file in os.listdir(label_folder):
            if file.endswith('.npy'):
                data = np.load(os.path.join(label_folder, file))
                X.append(data)
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {dataset_type} data for {city}: X shape = {X.shape}, y shape = {y.shape}")
    return X, y


# Function to split data into Input_1 (first 11 time steps) and Input_2 (12th time step)
def split_data(X):
    input_1 = X[:, :11, :, :, :]  # First 11 time steps
    input_2 = X[:, 11:, :, :, :]  # Last time step
    return input_1, input_2


# Build the complete model
def build_model(input_shape_1, input_shape_2):
    # Input 1
    input_1 = Input(shape=input_shape_1, name="Input_1")
    convlstm_1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', return_sequences=True)(input_1)
    convlstm_3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', return_sequences=False)(convlstm_1)
    output_3 = Flatten(name="Output_3")(convlstm_3)

    # Input 2
    input_2 = Input(shape=input_shape_2, name="Input_2")
    convlstm_2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', return_sequences=False)(input_2)
    output_2 = Flatten(name="Output_2")(convlstm_2)

    # Align Output_3 to match Output_2 shape
    aligned_output_3 = Dense(672, activation='linear', name="Aligned_Output_3")(output_3)

    # Compute MSE loss
    mse_loss = Subtract(name="MSE_Loss")([aligned_output_3, output_2])

    # Build the model
    model = Model(inputs=[input_1, input_2], outputs=[aligned_output_3, output_2, mse_loss])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # Compile with MSE loss
    model.summary()
    return model


# Phase 1: Train ConvLSTM_2
def train_convlstm_2(model, X_train, Y_train):
    X_train_normal = X_train[Y_train == 0]  # Use only normal data
    _, input_2 = split_data(X_train_normal)
    target_output_2 = input_2.reshape(input_2.shape[0], -1)  # Flatten the 12th time step

    # Extract ConvLSTM_2 sub-model
    convlstm_2_model = Model(inputs=model.get_layer("Input_2").input,
                             outputs=model.get_layer("Output_2").output)
    convlstm_2_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train ConvLSTM_2
    convlstm_2_model.fit(
        input_2, target_output_2,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint("convlstm_2_model.h5", save_best_only=True)
        ]
    )
    print("Phase 1: ConvLSTM_2 training completed.")
    return convlstm_2_model


# Save ConvLSTM_2 features for Input_2
def save_convlstm_2_features(convlstm_2_model, X, Y, filename):
    _, input_2 = split_data(X[Y == 0])  # Use only `Y=0` data
    features = convlstm_2_model.predict(input_2)
    np.save(filename, features)
    print(f"Features saved to {filename}")
    return features


# Phase 2: Train ConvLSTM_3
def train_convlstm_3(model, X_train, Y_train, convlstm_2_features):
    X_train_normal = X_train[Y_train == 0]  # Use only normal data
    input_1, _ = split_data(X_train_normal)

    # Load precomputed ConvLSTM_2 features
    target_features = np.load(convlstm_2_features)

    # Extract ConvLSTM_3 sub-model
    convlstm_3_model = Model(inputs=model.get_layer("Input_1").input,
                             outputs=model.get_layer("Aligned_Output_3").output)
    convlstm_3_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train ConvLSTM_3
    convlstm_3_model.fit(
        input_1, target_features,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint("convlstm_3_model.h5", save_best_only=True)
        ]
    )
    print("Phase 2: ConvLSTM_3 training completed.")
    return convlstm_3_model


# Phase 3: Fine-tune the full model (optional)
def fine_tune_full_model(model, X_train, Y_train):
    X_train_normal = X_train[Y_train == 0]
    input_1, input_2 = split_data(X_train_normal)
    target_output_2 = input_2.reshape(input_2.shape[0], -1)  # Flatten the 12th time step
    dummy_output_3 = np.zeros_like(target_output_2)
    dummy_mse_loss = np.zeros_like(target_output_2)

    # Fine-tune the full model
    model.fit(
        [input_1, input_2],
        [dummy_output_3, target_output_2, dummy_mse_loss],
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint("fine_tuned_model.h5", save_best_only=True)
        ]
    )
    print("Phase 3: Full model fine-tuning completed.")


# Main function to process a city
def process_city(city):
    print(f"Processing city: {city}")

    # Load data
    X_train, Y_train = load_data(city, 'train')
    X_valid, Y_valid = load_data(city, 'valid')
    X_test, Y_test = load_data(city, 'test')
    X_train = X_train[:, :, 4:]  # Keep columns 4 onward
    X_valid = X_valid[:, :, 4:]
    X_test = X_test[:, :, 4:]    # Reshape data to match input shape
    # Reshape data
    X_train = X_train.reshape(-1, 12, 5, 9, 1)
    X_valid = X_valid.reshape(-1, 12, 5, 9, 1)
    X_test = X_test.reshape(-1, 12, 5, 9, 1)

    # Input shapes
    input_shape_1 = (11, 5, 9, 1)
    input_shape_2 = (1, 5, 9, 1)

    # Build model
    model = build_model(input_shape_1, input_shape_2)

    # Phase 1: Train ConvLSTM_2
    convlstm_2_model = train_convlstm_2(model, X_train, Y_train)
    convlstm_2_features_file = f"./Results/{city}_convlstm_2_features.npy"
    save_convlstm_2_features(convlstm_2_model, X_train, Y_train, convlstm_2_features_file)

    # Phase 2: Train ConvLSTM_3
    train_convlstm_3(model, X_train, Y_train, convlstm_2_features_file)

    # Phase 3: Fine-tune the full model (optional)
    fine_tune_full_model(model, X_train, Y_train)

    print(f"Processing completed for city: {city}")


# Run the process for multiple cities
if __name__ == "__main__":
    os.makedirs("./Results", exist_ok=True)
    process_city("AKTAU")
    process_city("Esbjerg")

