import tensorflow as tf
import numpy as np

# Assume the updated custom ConvLSTM2D layer is defined as in the previous code.
# from C_ConvLSTM2D_custom import ConvLSTM2D_Code  # Uncomment if importing from a file.

# For this test, we'll use the code directly.
class ConvLSTM2D_Code(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding="valid",
                 data_format=None,
                 dilation_rate=1,
                 activation="tanh",
                 recurrent_activation="sigmoid",
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 recurrent_initializer="orthogonal",
                 bias_initializer="zeros",
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 seed=None,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 num_gate_copies=1,
                 **kwargs):
        super(ConvLSTM2D_Code, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
        self.padding = padding.upper()  # "VALID" or "SAME"
        self.data_format = data_format  # we assume channels_last if None
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, (tuple, list)) else (dilation_rate, dilation_rate)
        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = tf.keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = tf.keras.constraints.get(recurrent_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.seed = seed
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.num_gate_copies = num_gate_copies

    def build(self, input_shape):
        # input_shape: (batch, time, height, width, channels)
        if self.data_format is None or self.data_format == 'channels_last':
            batch, time, height, width, channels = input_shape
        else:
            raise NotImplementedError("Currently only channels_last is supported.")
        
        # Shared convolution kernel for input-to-hidden:
        kernel_shape = self.kernel_size + (channels, self.filters * 4)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Create multiple copies for recurrent weights:
        recurrent_kernel_shape = (self.num_gate_copies,) + self.kernel_size + (self.filters, self.filters * 4)
        self.recurrent_kernel = self.add_weight(shape=recurrent_kernel_shape,
                                                initializer=self.recurrent_initializer,
                                                name='recurrent_kernel',
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint)
        if self.use_bias:
            bias_shape = (self.num_gate_copies, self.filters * 4)
            self.bias = self.add_weight(shape=bias_shape,
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if self.unit_forget_bias:
                bias_value = tf.keras.backend.get_value(self.bias)
                bias_value[:, self.filters:2*self.filters] = 1.0
                tf.keras.backend.set_value(self.bias, bias_value)
        else:
            self.bias = None
        
        self.built = True

    def call(self, inputs, label_values, training=None, initial_state=None):
        """
        inputs: Tensor of shape (batch, time, height, width, channels)
        label_values: Tensor of shape (batch,) containing integer labels in [0, num_gate_copies-1]
                      which indicate which copy of recurrent weights to use for each sample.
        """
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        time_steps = inputs.shape[1] if inputs.shape[1] is not None else input_shape[1]
        height = inputs.shape[2]
        width = inputs.shape[3]
        
        if initial_state is None:
            h = tf.zeros((batch_size, height, width, self.filters))
            c = tf.zeros((batch_size, height, width, self.filters))
        else:
            h, c = initial_state
        
        outputs = []
        time_indices = range(time_steps)
        if self.go_backwards:
            time_indices = reversed(list(time_indices))
        
        for t in time_indices:
            x_t = inputs[:, t, :, :, :]  # (batch, height, width, channels)
            # Shared input-to-hidden convolution:
            z_input = tf.nn.conv2d(x_t, self.kernel, strides=self.strides, padding=self.padding,
                                   dilations=self.dilation_rate, data_format="NHWC")
            
            # Hidden-to-hidden (recurrent) convolution per sample:
            def recurrent_conv_per_sample(args):
                h_sample, label_val = args
                label_val = tf.cast(label_val, tf.int32)
                kernel_sample = self.recurrent_kernel[label_val]  # shape: (kernel_h, kernel_w, filters, filters*4)
                z_rec = tf.nn.conv2d(h_sample[None, ...], kernel_sample,
                                     strides=(1, 1),
                                     padding=self.padding,
                                     dilations=self.dilation_rate,
                                     data_format="NHWC")
                z_rec = tf.squeeze(z_rec, axis=0)
                if self.use_bias:
                    bias_sample = self.bias[label_val]
                    z_rec = tf.nn.bias_add(z_rec, bias_sample, data_format="NHWC")
                return z_rec

            z_recurrent = tf.map_fn(
                recurrent_conv_per_sample,
                (h, label_values),
                dtype=tf.float32
            )
            
            # Sum both contributions:
            z = z_input + z_recurrent
            
            # Split into gates: [input gate, forget gate, cell candidate, output gate]
            z_channels = self.filters
            i = self.recurrent_activation(z[..., :z_channels])
            f = self.recurrent_activation(z[..., z_channels:2*z_channels])
            c_candidate = self.activation(z[..., 2*z_channels:3*z_channels])
            o = self.recurrent_activation(z[..., 3*z_channels:])
            
            c = f * c + i * c_candidate
            h = o * self.activation(c)
            
            if self.return_sequences:
                outputs.append(h)
        
        if self.return_sequences:
            output = tf.stack(outputs, axis=1)
        else:
            output = h
        
        if self.return_state:
            return [output, h, c]
        else:
            return output

    def compute_output_shape(self, input_shape):
        batch, time, height, width, channels = input_shape
        if self.padding == "SAME":
            out_height, out_width = height, width
        else:
            kh, kw = self.kernel_size
            out_height = height - kh + 1
            out_width = width - kw + 1
        
        if self.return_sequences:
            output_shape = (batch, time, out_height, out_width, self.filters)
        else:
            output_shape = (batch, out_height, out_width, self.filters)
        
        if self.return_state:
            return [output_shape, (batch, out_height, out_width, self.filters), (batch, out_height, out_width, self.filters)]
        else:
            return output_shape

# ---------------------
# Testing the custom layer

# Parameters for dummy data
batch_size = 4
time_steps = 10
height = 28
width = 28
channels = 3

# Create dummy input data (e.g., images over time)
dummy_inputs = tf.random.normal((batch_size, time_steps, height, width, channels))
# Create dummy label values for each sample; here we use 3 unique copies (0, 1, 2)
dummy_labels = tf.constant([0, 1, 2, 0], dtype=tf.int32)

# Instantiate the custom ConvLSTM2D layer with multiple gate copies.
num_gate_copies = 3
conv_lstm_layer = ConvLSTM2D_Code(filters=16,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  return_sequences=True,
                                  num_gate_copies=num_gate_copies)

# Run a forward pass:
output = conv_lstm_layer(dummy_inputs, label_values=dummy_labels)
print("Output shape:", output.shape)
