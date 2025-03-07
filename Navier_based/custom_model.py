import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
















import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from attention import Attention  # Importing your custom attention layer

class VolatilityAwareWindPredictor:
    def __init__(self, input_shape, output_steps, min_val, max_val):
        self.input_shape = input_shape  # (Time, Features, Locations)
        self.output_steps = output_steps
        self.min_val = min_val
        self.max_val = max_val
        self.scaler = lambda x: x * (max_val - min_val) + min_val

    def build_custom_model(self):
        # Custom model without attention (using ConvLSTM and Hybrid Loss)
        inputs = layers.Input(shape=self.input_shape)  # (Time, Features, Locations)
        
        # Add extra dimension at the end of the input (e.g., Time x Features x Locations x 1)
        x = layers.Reshape((-1, self.input_shape[1], self.input_shape[2], 1))(inputs)

        # ConvLSTM Network
        x = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), 
                              activation='tanh', return_sequences=True, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), 
                              activation='tanh', return_sequences=True, padding="same")(x)

 


        # Reshape x to (batch_size, 24, 1440)
        x = layers.Reshape((self.input_shape[0], -1))(x)  # Flatten the features and locations, keep time as it is

        # Concatenate memberships_train with x
        # Assuming memberships_train has shape (batch_size, 3) for each example
        memberships = layers.Input(shape=(3,), name='memberships')  # 3 values for each example
        memberships_repeated = layers.Lambda(lambda x: tf.tile(x[:, tf.newaxis, :], [1, 24, 1]))(memberships)
        x = layers.Concatenate(axis=-1)([x, memberships_repeated])
        
        # Custom attention layer (PP will be shaped (batch_size, 24, 1443))
        PP = Attention(units=32)(x)  # Apply custom attention
        
        # Output Layer
        outputs = layers.Dense(self.output_steps)(PP)

        # Model definition
        self.model = models.Model(inputs=[inputs, memberships], outputs=outputs)
        
        # Custom loss with gradient clipping
        optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)
        self.model.compile(optimizer=optimizer, loss=self._hybrid_loss)
        return self.model

    def _hybrid_loss(self, y_true, y_pred):
        print("y_true shape:", y_true.shape)  # Print shape of y_true
        print("y_pred shape:", y_pred.shape)  # Print shape of y_pred

        # Assuming y_true contains both the target values and the membership weights
        # Here, we assume that y_true's last 3 columns are the membership weights
        y_true_values = y_true[:, :-3]  # Get the actual target values (shape [?, 10])
        memberships = y_true[:, -3:]    # Extract the membership weights (shape [?, 3])

        print("y_true_values shape:", y_true_values.shape)  # Print shape of y_true_values
        print("memberships shape:", memberships.shape)  # Print shape of memberships

        # Split memberships into individual weights (low, medium, high)
        low = memberships[:, 0]
        medium = memberships[:, 1]
        high = memberships[:, 2]

        # Print the individual membership weights
        print("low shape:", low.shape)
        print("medium shape:", medium.shape)
        print("high shape:", high.shape)
        
        # Compute the loss components: MAE, MSE, ACL
        mae = K.mean(K.abs(y_true_values - y_pred), axis=-1)
        mse = K.mean(K.pow(K.abs(y_true_values - y_pred), 2), axis=-1)
        acl = K.mean(K.pow(K.abs(y_true_values - y_pred), 2.2), axis=-1)

        print("mae shape:", mae.shape)
        print("mse shape:", mse.shape)
        print("acl shape:", acl.shape)
        
        # Combine the loss components with their respective membership weights
        loss = low * mae + medium * mse + high * acl
        print("loss shape:", loss.shape)  # Print shape of the final loss
        
        return loss
    def build_baseline_model(self):
        # Simple ConvLSTM model trained with MSE loss
        inputs = layers.Input(shape=self.input_shape)
        
        # Add extra dimension at the end of the input (e.g., Time x Features x Locations x 1)
        x = layers.Reshape((-1, self.input_shape[1], self.input_shape[2], 1))(inputs)
        
        # ConvLSTM Network (Simple ConvLSTM without attention)
        x = layers.ConvLSTM2D(64, (3, 3), activation='tanh', return_sequences=True)(x)
        x = layers.ConvLSTM2D(32, (3, 3), activation='tanh')(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(self.output_steps)(x)
        
        self.baseline_model = models.Model(inputs=inputs, outputs=outputs)
        self.baseline_model.compile(optimizer='adam', loss='mse')
        return self.baseline_model


    def train(self, X_train, y_train, X_val, y_val, memberships_train, memberships_val, model_type='custom', epochs=1000, patience=10, batch_size=32):

        print("Training started...")
        print(f"X_train shape: {X_train.shape}, Batch size: {batch_size}")

        # Create checkpoint directory
        checkpoint_dir = f"./checkpoint/{model_type}/"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Configure callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
            monitor='loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )

        # Prepare targets based on model type
        if model_type == 'custom':
            # Combine target values (y_train) and memberships (memberships_train) into a single tensor for custom model
            y_train_combined = tf.concat([y_train, memberships_train], axis=-1) if memberships_train is not None else y_train
            y_val_combined = tf.concat([y_val, memberships_val], axis=-1) if memberships_val is not None else y_val
            return self.model.fit(
                [X_train, memberships_train], y_train_combined,
                validation_data=([X_val, memberships_val], y_val_combined),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, checkpoint],
                verbose=1
            )            
        else:
            # For baseline model, just use y_train and y_val without concatenating memberships
            y_train_combined = y_train
            y_val_combined = y_val
            return self.baseline_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, checkpoint],
                verbose=1
            )

    def evaluate(self, X_test, y_test, memberships_test=None, model_type='custom'):
        print("Evaluation started...")
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        if model_type == 'custom':
            y_test_combined = y_test
            y_pred = self.model.predict([X_test, memberships_test])
        else:
            y_test_combined = y_test
            y_pred = self.baseline_model.predict(X_test)

        # Fix scaler usage (changed from __call__ to inverse_transform)
        y_test_unscaled = self.scaler(y_test_combined)
        y_pred_unscaled = self.scaler(y_pred)

        # Horizon detection (only add this block)
        results = {}
        if y_test_unscaled.ndim == 2 and y_test_unscaled.shape[1] > 1:  # Check for multiple horizons
            num_horizons = y_test_unscaled.shape[1]
            for h in range(num_horizons):
                results[f'MSE_{h+1}'] = mean_squared_error(y_test_unscaled[:, h], y_pred_unscaled[:, h])
                results[f'MAE_{h+1}'] = mean_absolute_error(y_test_unscaled[:, h], y_pred_unscaled[:, h])
                results[f'R2_{h+1}'] = r2_score(y_test_unscaled[:, h], y_pred_unscaled[:, h])
            
            # Add averages (only these 3 new lines)
            results['MSE_avg'] = np.mean([v for k,v in results.items() if 'MSE' in k])
            results['MAE_avg'] = np.mean([v for k,v in results.items() if 'MAE' in k])
            results['R2_avg'] = np.mean([v for k,v in results.items() if 'R2' in k])
        else:  # Keep original behavior
            results.update({
                'MSE': mean_squared_error(y_test_unscaled, y_pred_unscaled),
                'MAE': mean_absolute_error(y_test_unscaled, y_pred_unscaled),
                'R2': r2_score(y_test_unscaled, y_pred_unscaled)
            })

        return results
    def preprocess_data(self, X_train, X_val, X_test):
        print("Preprocessing data...")
        print("X_train shape before reshape:", X_train.shape)
        print("X_val shape before reshape:", X_val.shape)
        print("X_test shape before reshape:", X_test.shape)

        # Add an extra dimension to the input data (for ConvLSTM compatibility)
        X_train = layers.Reshape((-1, self.input_shape[1], self.input_shape[2], 1))(X_train)
        X_val = layers.Reshape((-1, self.input_shape[1], self.input_shape[2], 1))(X_val)
        X_test = layers.Reshape((-1, self.input_shape[1], self.input_shape[2], 1))(X_test)

        print("X_train shape after reshape:", X_train.shape)
        print("X_val shape after reshape:", X_val.shape)
        print("X_test shape after reshape:", X_test.shape)

        return X_train, X_val, X_test
