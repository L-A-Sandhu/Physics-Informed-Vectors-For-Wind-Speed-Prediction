import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from cascaded_covlstm import ConvLSTM2D_Code  # Import your custom layer
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from attention import Attention  # Importing your custom attention layer

class VolatilityAwareWindPredictor:
    def __init__(self, input_shape, output_steps, min_val, max_val):
        self.input_shape = input_shape  # (Time, Features, Locations)
        self.output_steps = output_steps
        self.min_val = min_val
        self.max_val = max_val
        self.lambda_entropy = 0.3
        self.scaler = lambda x: x * (max_val - min_val) + min_val

    def build_custom_model(self):
        # Primary spatiotemporal input (e.g., shape: (Time, Features, Locations))
        inputs = layers.Input(shape=self.input_shape, name="spatiotemporal_input")
        
        # Membership input (one-hot encoded, shape: (3,))
        memberships = layers.Input(shape=(3,), name="memberships")
        
        # Convert one-hot memberships to integer labels for conditional gating
        membership_labels = layers.Lambda(lambda x: tf.argmax(x, axis=1), name="membership_labels")(memberships)
        
        # Reshape the spatiotemporal input to add a channel dimension: (Time, Features, Locations, 1)
        x = layers.Reshape(
            (-1, self.input_shape[1], self.input_shape[2], 1),
            name="reshape_input"
        )(inputs)
        
        # First custom ConvLSTM layer with conditional gates using membership labels
        x = ConvLSTM2D_Code(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            num_gate_copies=3
        )(x, label_values=membership_labels)
        
        # Batch normalization for stability
        # x = layers.BatchNormalization(name="batch_norm")(x)
        
        # Second custom ConvLSTM layer
        x = ConvLSTM2D_Code(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            num_gate_copies=3
        )(x, label_values=membership_labels)
        
        # Reshape output to collapse spatial dimensions (resulting shape: (Time, -1))
        x = layers.Reshape((self.input_shape[0], -1), name="reshape_conv_output")(x)
        
        # Attention layer: processes the temporal features directly
        attention_output = Attention(units=128, name="attention_layer")(x)
        
        # Concatenate the attention output with the membership values
        # Note: memberships is a (batch_size, 3) tensor; here we assume the attention output is also a flat vector per sample.
        combined = layers.Concatenate(axis=-1, name="concat_membership")([attention_output, memberships])
        
        # Dense layer with 16 units
        x = layers.Dense(64, activation='relu', name="dense_64")(combined)
        
        # Final Dense output layer (prediction for self.output_steps)
        outputs = layers.Dense(self.output_steps, name="final_output")(x)
        
        # Create and compile the model
        self.model = models.Model(
            inputs=[inputs, memberships], 
            outputs=outputs
        )

        # Custom loss with gradient clipping
        optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)
        self.model.compile(optimizer=optimizer, loss=self._hybrid_loss)
        return self.model





#Attention Removed 
    # def build_custom_model(self):
    #     # Primary spatiotemporal input (e.g., shape: (Time, Features, Locations))
    #     inputs = layers.Input(shape=self.input_shape, name="spatiotemporal_input")
        
    #     # Membership input (one-hot encoded, shape: (3,))
    #     memberships = layers.Input(shape=(3,), name="memberships")
        
    #     # Convert one-hot memberships to integer labels for conditional gating
    #     membership_labels = layers.Lambda(lambda x: tf.argmax(x, axis=1), name="membership_labels")(memberships)
        
    #     # Reshape the spatiotemporal input to add a channel dimension: (Time, Features, Locations, 1)
    #     x = layers.Reshape(
    #         (-1, self.input_shape[1], self.input_shape[2], 1),
    #         name="reshape_input"
    #     )(inputs)
        
    #     # First custom ConvLSTM layer with conditional gates using membership labels
    #     x = ConvLSTM2D_Code(
    #         filters=64,
    #         kernel_size=(3, 3),
    #         padding="same",
    #         return_sequences=True,
    #         num_gate_copies=3
    #     )(x, label_values=membership_labels)
        
    #     # Batch normalization for stability
    #     # x = layers.BatchNormalization(name="batch_norm")(x)
        
    #     # Second custom ConvLSTM layer
    #     x = ConvLSTM2D_Code(
    #         filters=32,
    #         kernel_size=(3, 3),
    #         padding="same",
    #         return_sequences=True,
    #         num_gate_copies=3
    #     )(x, label_values=membership_labels)
        
    #     # Reshape output to collapse spatial dimensions (resulting shape: (Time, -1))
    #     # x = layers.Reshape((self.input_shape[0], -1), name="reshape_conv_output")(x)
    #     x = layers.Reshape((-1, ), name="reshape_conv_output")(x)
    #     # Attention layer: processes the temporal features directly
    #     # attention_output = Attention(units=128, name="attention_layer")(x)
        
    #     # Concatenate the attention output with the membership values
    #     # Note: memberships is a (batch_size, 3) tensor; here we assume the attention output is also a flat vector per sample.
    #     combined = layers.Concatenate(axis=-1, name="concat_membership")([x, memberships])
        
    #     # Dense layer with 16 units
    #     x = layers.Dense(64, activation='relu', name="dense_64")(combined)
        
    #     # Final Dense output layer (prediction for self.output_steps)
    #     outputs = layers.Dense(self.output_steps, name="final_output")(x)
        
    #     # Create and compile the model
    #     self.model = models.Model(
    #         inputs=[inputs, memberships], 
    #         outputs=outputs
    #     )

    #     # Custom loss with gradient clipping
    #     optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)
    #     self.model.compile(optimizer=optimizer, loss=self._hybrid_loss)
    #     return self.model











    # def build_custom_model(self):
    #     # Custom model with conditional ConvLSTM gates
    #     inputs = layers.Input(shape=self.input_shape)  # (Time, Features, Locations)
        
    #     # Add channel dimension
    #     x = layers.Reshape((-1, self.input_shape[1], self.input_shape[2], 1))(inputs)

    #     # Get membership labels input (assuming one-hot encoded)
    #     memberships = layers.Input(shape=(3,), name='memberships')
        
    #     # Convert one-hot memberships to integer labels
    #     membership_labels = layers.Lambda(lambda x: tf.argmax(x, axis=1))(memberships)

    #     # Custom ConvLSTM Network with conditional gates
    #     x = ConvLSTM2D_Code(
    #         filters=64,
    #         kernel_size=(3, 3),
    #         padding="same",
    #         return_sequences=True,
    #         num_gate_copies=3  # Should match number of membership categories
    #     )(x, label_values=membership_labels)
        
    #     x = layers.BatchNormalization()(x)
        
    #     x = ConvLSTM2D_Code(
    #         filters=32,
    #         kernel_size=(3, 3),
    #         padding="same",
    #         return_sequences=True,
    #         num_gate_copies=3
    #     )(x, label_values=membership_labels)

    #     # Reshape and concatenate memberships
    #     x = layers.Reshape((self.input_shape[0], -1))(x)
    #     memberships_repeated = layers.Lambda(
    #         lambda x: tf.tile(x[:, tf.newaxis, :], [1, 24, 1])
    #     )(memberships)
    #     x = layers.Concatenate(axis=-1)([x, memberships_repeated])

    #     # Attention and output layers
    #     PP = Attention(units=32)(x)
    #     outputs = layers.Dense(self.output_steps)(PP)

    #     # Create model with dual inputs
    #     self.model = models.Model(
    #         inputs=[inputs, memberships], 
    #         outputs=outputs
    #     )

    #     # Custom loss with gradient clipping
    #     optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)
    #     self.model.compile(optimizer=optimizer, loss=self._hybrid_loss)
    #     return self.model
    def compute_target_variance(self, y_true):
        mean = tf.reduce_mean(y_true)
        variance = tf.reduce_mean(tf.square(y_true - mean))
        return variance

    def _hybrid_loss(self, y_true, y_pred):
        # Extract target values and fuzzy memberships
        y_true_values = y_true[:, :-3]  # Actual targets
        memberships = y_true[:, -3:]    # Shape: [batch_size, 3]

        low = memberships[:, 0]
        medium = memberships[:, 1]
        high = memberships[:, 2]
        total = low + medium + high + 1e-9

        # Compute base loss terms
        mae = K.mean(K.abs(y_true_values - y_pred), axis=-1)
        mse = K.mean(K.square(y_true_values - y_pred), axis=-1)
        rmse = K.sqrt(mse)
        target_var = self.compute_target_variance(y_true)

        # Original fuzzy-weighted loss
        loss = (low/total) * mae + (medium/total) * rmse + (high/total) * (mse / target_var)

        # --- Entropy Regularization ---
        epsilon = 1e-7
        entropy = - (low * K.log(low + epsilon) + 
                   medium * K.log(medium + epsilon) + 
                   high * K.log(high + epsilon))
          # Tunable
        loss += self.lambda_entropy * entropy

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


    def train(self, X_train, y_train, X_val, y_val, memberships_train, memberships_val, model_type='custom', epochs=1000, patience=10, batch_size=128):

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
