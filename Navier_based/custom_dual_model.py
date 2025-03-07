import random
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from attention import Attention  # Importing your custom attention layer
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
import os

os.environ['PYTHONHASHSEED'] = str(seed)
from tensorflow.keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True
class VolatilityAwareWindPredictor:
    def __init__(self, input_shape, output_steps, min_val, max_val,patience_1,patience_2):
        self.input_shape = input_shape  # (Time, Features, Locations)
        self.output_steps = output_steps
        self.min_val = min_val
        self.max_val = max_val
        self.patience_1=patience_1
        self.patience_2=patience_2
        self.lambda_entropy = 0.2
        self.scaler = lambda x: x * (max_val - min_val) + min_val


    def build_custom_model(self, label):
        # Custom model without attention (using ConvLSTM and Hybrid Loss)
        inputs = layers.Input(shape=self.input_shape, name=f'{label}_input')  # (Time, Features, Locations)
        
        # Add extra dimension at the end of the input (e.g., Time x Features x Locations x 1)
        x = layers.Reshape((-1, self.input_shape[1], self.input_shape[2], 1), name=f'{label}_reshape_input')(inputs)

        # Use the custom ConvLSTM2D layer (instead of the standard Keras ConvLSTM2D)
        x = ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            activation='tanh',
            return_sequences=True,
            padding="same",
            data_format='channels_last',  # Ensure to match the data format if required
            name=f'{label}_conv_lstm1'
        )(x)

        x = ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            activation='tanh',
            return_sequences=True,
            padding="same",
            data_format='channels_last',
            name=f'{label}_conv_lstm2'
        )(x)

        # Reshape x to (batch_size, 24, 1440)
        x = layers.Reshape((self.input_shape[0], -1), name=f'{label}_reshape_flatten')(x)  # Flatten the features and locations, keep time as it is

        # Concatenate memberships_train with x
        memberships = layers.Input(shape=(3,), name=f'{label}_memberships')  # 3 values for each example

        # Custom attention layer (PP will be shaped (batch_size, 24, 1443))
        PP = Attention(units=64, name=f'{label}_attention')(x)  # Apply custom attention
        PP = layers.Concatenate(axis=-1, name=f'{label}_concatenate')([PP, memberships])
        
        # Output Layer
        outputs = layers.Dense(self.output_steps, name=f'{label}_output')(PP)

        # Model definition
        self.model = models.Model(inputs=[inputs, memberships], outputs=outputs)

        # Custom loss with gradient clipping
        optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)

        # Pass the label to the hybrid loss function
        self.model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: self._hybrid_loss(y_true, y_pred))

        return self.model
    def compute_target_variance(self, y_true):
        mean = tf.reduce_mean(y_true)
        variance = tf.reduce_mean(tf.square(y_true - mean))
        return variance

    # def _hybrid_loss(self, y_true, y_pred):
    #     print("y_true shape:", y_true.shape)  # Print shape of y_true
    #     print("y_pred shape:", y_pred.shape)  # Print shape of y_pred

    #     # Assuming y_true contains both the target values and the membership weights
    #     # Here, we assume that y_true's last 3 columns are the membership weights
    #     y_true_values = y_true[:, :-3]  # Get the actual target values (shape [?, 10])
    #     memberships = y_true[:, -3:]    # Extract the membership weights (shape [?, 3])

    #     print("y_true_values shape:", y_true_values.shape)  # Print shape of y_true_values
    #     print("memberships shape:", memberships.shape)  # Print shape of memberships

    #     # Split memberships into individual weights (low, medium, high)
    #     low = memberships[:, 0]
    #     medium = memberships[:, 1]
    #     high = memberships[:, 2]
    #     target_var=self.compute_target_variance(y_true)
    #     # Print the individual membership weights
    #     print("low shape:", low.shape)
    #     print("medium shape:", medium.shape)
    #     print("high shape:", high.shape)
    #     total = low + medium + high + 1e-9
    #     # Compute the loss components: MAE, MSE, ACL
    #     mae = K.mean(K.abs(y_true_values - y_pred), axis=-1)
    #     mse = K.mean(K.pow(K.abs(y_true_values - y_pred), 2), axis=-1)
    #     acl = K.mean(K.pow(K.abs(y_true_values - y_pred), 2.05), axis=-1)
    #     rmse = K.sqrt(K.mean(K.pow(K.abs(y_true_values - y_pred), 2), axis=-1))
    #     print("mae shape:", mae.shape)
    #     print("mse shape:", mse.shape)
    #     print("acl shape:", acl.shape)
        
    #     # Combine the loss components with their respective membership weights
    #     loss = (low/total) * mae + (medium/total) *(rmse) + (( high/total) * (mse/target_var))


    #     print("loss shape:", loss.shape)  # Print shape of the final loss
        
    #     return loss
########################################################################
#######################Entropy based####################################
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
###############################################################################
##############################################################################
##############################################################################

########################################################################
####################### Sobolev-regularized loss####################################



    # def _hybrid_loss(self, y_true, y_pred):
    #     # --- Common Components ---
    #     y_true_values = y_true[:, :-3]
    #     memberships = y_true[:, -3:]
    #     low = memberships[:, 0]
    #     medium = memberships[:, 1]
    #     high = memberships[:, 2]
    #     total = low + medium + high + 1e-9

    #     # Base loss terms
    #     mae = K.mean(K.abs(y_true_values - y_pred), axis=-1)
    #     mse = K.mean(K.square(y_true_values - y_pred), axis=-1)
    #     rmse = K.sqrt(mse)
    #     target_var = self.compute_target_variance(y_true_values)

    #     # Fuzzy-weighted loss
    #     loss = (low/total) * mae + (medium/total) * rmse + (high/total) * (mse/target_var)

    #     # --- Sobolev Regularization ---
    #     if hasattr(self, 'model'):
    #         lambda_sobolev = 0.01  # Sobolev regularization weight

    #         # Get gradients for each loss component
    #         grad_mae = K.gradients(mae, self.model.trainable_weights)
    #         grad_rmse = K.gradients(rmse, self.model.trainable_weights)
    #         grad_mse = K.gradients(mse / target_var, self.model.trainable_weights)
            
    #         # Clip gradients to avoid large values (gradient normalization)
    #         def clip_gradients(gradients, clip_value=1.0):
    #             return [K.clip(g, -clip_value, clip_value) if g is not None else None for g in gradients]

    #         grad_mae = clip_gradients(grad_mae)
    #         grad_rmse = clip_gradients(grad_rmse)
    #         grad_mse = clip_gradients(grad_mse)

    #         # Calculate sum of squares for each gradient list
    #         sum_mae_sq = sum([K.sum(K.square(g)) for g in grad_mae if g is not None])
    #         sum_rmse_sq = sum([K.sum(K.square(g)) for g in grad_rmse if g is not None])
    #         sum_mse_sq = sum([K.sum(K.square(g)) for g in grad_mse if g is not None])

    #         sobolev_penalty = (
    #             low * sum_mae_sq +
    #             medium * sum_rmse_sq +
    #             high * sum_mse_sq
    #         )

    #         # Apply Sobolev penalty to loss with gradient norm stabilization
    #         loss += lambda_sobolev * sobolev_penalty

    #     return loss

###############################################################################
##############################################################################
##############################################################################


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
            monitor='val_loss',
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

    def pred(self, X_test, y_test, memberships_test=None, model_type='custom'):
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
        return y_test_unscaled, y_pred_unscaled
    def evaluate(self, y_test_unscaled, y_pred_unscaled):        
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
    # def preprocess_data(self, X_train, X_val, X_test):
    #     print("Preprocessing data...")
    #     print("X_train shape before reshape:", X_train.shape)
    #     print("X_val shape before reshape:", X_val.shape)
    #     print("X_test shape before reshape:", X_test.shape)

    #     # Add an extra dimension to the input data (for ConvLSTM compatibility)
    #     X_train = layers.Reshape((-1, self.input_shape[1], self.input_shape[2], 1))(X_train)
    #     X_val = layers.Reshape((-1, self.input_shape[1], self.input_shape[2], 1))(X_val)
    #     X_test = layers.Reshape((-1, self.input_shape[1], self.input_shape[2], 1))(X_test)

    #     print("X_train shape after reshape:", X_train.shape)
    #     print("X_val shape after reshape:", X_val.shape)
    #     print("X_test shape after reshape:", X_test.shape)

    #     return X_train, X_val, X_test



    def preprocess_data(self, X_train, X_val, X_test, mem_train, mem_val, mem_test, Y_train, Y_val, Y_test):
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

        # Ensure that the number of examples in X_train matches mem_train and similarly for val and test
        assert X_train.shape[0] == mem_train.shape[0], "Number of examples in X_train and mem_train do not match"
        assert X_val.shape[0] == mem_val.shape[0], "Number of examples in X_val and mem_val do not match"
        assert X_test.shape[0] == mem_test.shape[0], "Number of examples in X_test and mem_test do not match"

        # Identify indexes where High (column 2) is greater than Low (column 0) for MH
        MH_train_idx = np.where(mem_train[:, 2] > mem_train[:, 0])[0]  # Indices where High > Low
        MH_val_idx = np.where(mem_val[:, 2] > mem_val[:, 0])[0]
        MH_test_idx = np.where(mem_test[:, 2] > mem_test[:, 0])[0]

        # Convert the indices to TensorFlow tensors of type int32
        MH_train_idx = tf.convert_to_tensor(MH_train_idx, dtype=tf.int32)
        MH_val_idx = tf.convert_to_tensor(MH_val_idx, dtype=tf.int32)
        MH_test_idx = tf.convert_to_tensor(MH_test_idx, dtype=tf.int32)

        # Use NumPy's setdiff1d to find the complementary indices for LH
        LM_train_idx = np.setdiff1d(np.arange(X_train.shape[0]), MH_train_idx.numpy())
        LM_val_idx = np.setdiff1d(np.arange(X_val.shape[0]), MH_val_idx.numpy())
        LM_test_idx = np.setdiff1d(np.arange(X_test.shape[0]), MH_test_idx.numpy())

        # Convert the indices back to TensorFlow tensors
        LM_train_idx = tf.convert_to_tensor(LM_train_idx, dtype=tf.int32)
        LM_val_idx = tf.convert_to_tensor(LM_val_idx, dtype=tf.int32)
        LM_test_idx = tf.convert_to_tensor(LM_test_idx, dtype=tf.int32)

        # Split the datasets into MH and LM based on the above indices
        X_MH_train = tf.gather(X_train, MH_train_idx, axis=0)
        X_LM_train = tf.gather(X_train, LM_train_idx, axis=0)
        
        X_MH_val = tf.gather(X_val, MH_val_idx, axis=0)
        X_LM_val = tf.gather(X_val, LM_val_idx, axis=0)
        
        X_MH_test = tf.gather(X_test, MH_test_idx, axis=0)
        X_LM_test = tf.gather(X_test, LM_test_idx, axis=0)

        # Split Y_train, Y_val, Y_test based on the corresponding indices
        Y_MH_train = tf.gather(Y_train, MH_train_idx, axis=0)
        Y_LM_train = tf.gather(Y_train, LM_train_idx, axis=0)
        
        Y_MH_val = tf.gather(Y_val, MH_val_idx, axis=0)
        Y_LM_val = tf.gather(Y_val, LM_val_idx, axis=0)
        
        Y_MH_test = tf.gather(Y_test, MH_test_idx, axis=0)
        Y_LM_test = tf.gather(Y_test, LM_test_idx, axis=0)

        # Now split mem_train, mem_val, and mem_test into mem_MH and mem_LH
        mem_MH_train = tf.gather(mem_train, MH_train_idx, axis=0)
        mem_LM_train = tf.gather(mem_train, LM_train_idx, axis=0)
        
        mem_MH_val = tf.gather(mem_val, MH_val_idx, axis=0)
        mem_LM_val = tf.gather(mem_val, LM_val_idx, axis=0)
        
        mem_MH_test = tf.gather(mem_test, MH_test_idx, axis=0)
        mem_LM_test = tf.gather(mem_test, LM_test_idx, axis=0)

        return (X_train, X_val, X_test, 
                X_MH_train, X_MH_val, X_MH_test,
                X_LM_train, X_LM_val, X_LM_test,
                Y_MH_train, Y_MH_val, Y_MH_test,
                Y_LM_train, Y_LM_val, Y_LM_test,
                mem_MH_train, mem_MH_val, mem_MH_test,
                mem_LM_train, mem_LM_val, mem_LM_test)

