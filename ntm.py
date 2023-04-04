# Credit: this code is derived from https://github.com/MarkPKCollier/NeuralTuringMachine/blob/master/ntm.py
# With substantial changes to enable TF2 and Keras compatibility, eager execution, and model saving

import tensorflow as tf  # For neural network
import collections       # For storage container
import numpy as np       # For math operations

# Structure for storing the NTM state
NTMControllerState = collections.namedtuple('NTMControllerState',
                                            ('controller_state', 'read_vector_list', 'w_list', 'M'))


# Controller state cell, original paper suggested Dense or LSTM
def _single_cell(num_units):
    return tf.keras.layers.LSTMCell(num_units)


# Expand the input tensor x along a specified axis by repeating it N times.
def expand(x, axis, N):
    shape = list(x.shape)
    shape.insert(axis, 1)
    tile_shape = [1] * len(shape)
    tile_shape[axis] = N
    x = tf.tile(tf.reshape(x, shape), tile_shape)
    return x


# Layers for learned parameter initialization
def init_layer(units):
    return tf.keras.layers.Dense(units, use_bias=False)


# Linear initializer for the controller
def create_linear_initializer(input_size):
    return tf.keras.initializers.TruncatedNormal(stddev=1.0 / np.sqrt(input_size))


# Connect the learned initialization vector in the computational graph
def wire_li(layer):
    wired = layer(tf.ones([1, 1]))
    return wired


# Cell class
class NTMCell(tf.keras.layers.Layer):
    def __init__(self, controller_layers, controller_units, memory_size, memory_vector_dim, read_head_num,
                 write_head_num, addressing_mode='content_and_location', shift_range=1, output_dim=None, clip_value=20,
                 init_mode='constant',  # Best recommended init mode is 'constant'
                 reuse=None,  # For compatibility with original script
                 name=None, **kwargs):  # For saving model

        # Initialize the parent component
        parent = super(NTMCell, self)
        parent.__init__(name=name)

        # Controller capacity
        self.controller_layers = controller_layers
        self.controller_units = controller_units

        # Memory storage capacity
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim

        # Memory access capacity
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.shift_range = shift_range
        self.addressing_mode = addressing_mode

        # Recurrent output dimension
        self.output_dim = output_dim

        # Safety feature: clip the outputs to this value to prevent exploding gradients
        self.clip_value = clip_value

        # Memory initialization mode
        self.init_mode = init_mode

        # Derived parameters and internal layers
        self.num_heads = self.read_head_num + self.write_head_num
        self.num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1

        # Initialize internal controller layers
        self.controller = tf.keras.layers.StackedRNNCells(
            [_single_cell(num_units=self.controller_units) for _ in range(self.controller_layers)])
        self.read_layers = [init_layer(self.memory_vector_dim) for _ in range(self.read_head_num)]
        self.w_layers = [init_layer(self.memory_size) for _ in range(self.num_heads)]

        # Initialize the head parameters layer
        self.parameters_layer = tf.keras.layers.Dense(
            self.num_parameters_per_head * self.num_heads + self.memory_vector_dim * 2 * self.write_head_num,
            kernel_initializer=create_linear_initializer(self.controller_units))

        # Cannot initialize output_layer here because, which depends on the input configuration
        self.output_layer = None

        # Initialize the memory
        if self.init_mode == 'learned':
            self.memory_layer = init_layer(self.memory_size * self.memory_vector_dim)
        elif self.init_mode == 'random':
            self.memory_layer = self.add_weight(
                name='init_M',
                shape=[self.memory_size, self.memory_vector_dim],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5),
                trainable=False)
        elif self.init_mode == 'constant':
            self.memory_layer = self.add_weight(
                name='init_M',
                shape=[self.memory_size, self.memory_vector_dim],
                initializer=tf.constant_initializer(1e-6),
                trainable=False)
        parent.__init__(**kwargs)

    # To enable model saving
    def get_config(self):
        config = {
            **super(NTMCell, self).get_config(),
            'controller_layers': self.controller_layers,
            'controller_units': self.controller_units,
            'memory_size': self.memory_size,
            'memory_vector_dim': self.memory_vector_dim,
            'read_head_num': self.read_head_num,
            'write_head_num': self.write_head_num,
            'addressing_mode': self.addressing_mode,
            'shift_range': self.shift_range,
            'output_dim': self.output_dim,
            'clip_value': self.clip_value,
            'init_mode': self.init_mode
        }
        return config

    # Instrument the class to the amount of parameters contributed by each individual component
    def params_count(self):
        print("C:", self.controller.count_params(), end=", ")
        print("P:", self.parameters_layer.count_params(), end=", ")
        all_w_params = 0
        for layer in self.w_layers:
            all_w_params += layer.count_params()
        print("W:", all_w_params, end=", ")
        all_r_params = 0
        for layer in self.read_layers:
            all_r_params += layer.count_params()
        print("R:", all_r_params, end=", ")
        print("O:", self.output_layer.count_params(), end=", ")
        print("M(n):", self.memory_size * self.memory_vector_dim)

    # For Keras TF2 main call
    def call(self, x, prev_state):
        return self(x, NTMControllerState(*prev_state))

    # Exposing this for TF1 compatibility
    # X - external input, prev_state - previous state
    def __call__(self, x, prev_state):
        prev_state = NTMControllerState(*prev_state)  # TF 1 and 2 compatibility
        prev_read_vector_list = prev_state.read_vector_list

        controller_input = tf.concat([x] + prev_read_vector_list, axis=1)

        # Connect the controller
        controller_output, controller_state = self.controller(controller_input, prev_state.controller_state)

        parameters = self.parameters_layer(controller_output)

        # Gradient clipping with norm to prevent exploding
        parameters = tf.clip_by_value(parameters, -self.clip_value, self.clip_value)

        # Parameters for read/write heads
        head_parameter_list = tf.split(parameters[:, :self.num_parameters_per_head * self.num_heads], self.num_heads,
                                       axis=1)
        erase_add_list = tf.split(parameters[:, self.num_parameters_per_head * self.num_heads:],
                                  2 * self.write_head_num, axis=1)

        prev_w_list = prev_state.w_list
        prev_M = prev_state.M

        w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            beta = tf.nn.softplus(head_parameter[:, self.memory_vector_dim])
            g = tf.sigmoid(head_parameter[:, self.memory_vector_dim + 1])
            s = tf.nn.softmax(
                head_parameter[:, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)]
            )
            gamma = tf.nn.softplus(head_parameter[:, -1]) + 1
            w = self.addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i])
            w_list.append(w)

        # Reading (Sec 3.1)
        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []

        for i in range(self.read_head_num):
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], axis=2) * prev_M, axis=1)
            read_vector_list.append(read_vector)

        # Writing (Sec 3.2)
        write_w_list = w_list[self.read_head_num:]
        M = prev_M
        for i in range(self.write_head_num):
            w = tf.expand_dims(write_w_list[i], axis=2)
            erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)
            add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)
            M = M * (tf.ones(tf.shape(M), dtype=self.dtype) - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)

        # Determine shape of recurrent output and wire it up
        if not self.output_layer:
            if self.output_dim:
                output_dim = self.output_dim
            else:
                output_dim = tf.shape(x)[1]

            o2o_initializer = create_linear_initializer(
                self.controller_units + self.memory_vector_dim * self.read_head_num)
            self.output_layer = tf.keras.layers.Dense(output_dim, kernel_initializer=o2o_initializer)

        # Form the output layer
        inner_output = tf.concat([controller_output] + read_vector_list, axis=1)
        ntm_output = self.output_layer(inner_output)
        ntm_output = tf.clip_by_value(ntm_output, -self.clip_value, self.clip_value)

        return ntm_output, NTMControllerState(
            controller_state=controller_state, read_vector_list=read_vector_list, w_list=w_list, M=M)

    def addressing(self, k, beta, g, s, gamma, prev_M, prev_w):
        # Sec 3.3.1 Focusing by Content

        # Cosine Similarity
        k_expanded = tf.expand_dims(k, axis=1)  # Expand along the second dimension
        K = tf.keras.losses.cosine_similarity(k_expanded, prev_M, axis=-1)

        # Calculating w^c
        K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keepdims=True)  # eq (5)

        if self.addressing_mode == 'content':  # Limited addressing mode, otherwise use content + location
            return w_c

        # Sec 3.3.2 Focusing by Location
        g = tf.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w  # eq (7)

        s = tf.concat([s[:, :self.shift_range + 1],
                       tf.zeros([tf.shape(s)[0], self.memory_size - (self.shift_range * 2 + 1)], dtype=self.dtype),
                       s[:, -self.shift_range:]], axis=1)

        # Circular convolution
        t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
        s_matrix = tf.stack(
            [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)],
            axis=1
        )
        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)  # eq (8)
        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keepdims=True)  # eq (9)

        return w

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        # Wire all layers into the graph
        controller_init_state = self.controller.get_initial_state(inputs, batch_size, dtype)
        read_vector_list = [expand(tf.tanh(tf.squeeze(wire_li(layer))), axis=0, N=batch_size) for layer in
                            self.read_layers]
        w_list = [expand(tf.nn.softmax(tf.squeeze(wire_li(layer))), axis=0, N=batch_size) for layer in self.w_layers]

        # Connect and initialize memory
        if self.init_mode == 'learned':
            inner_M = tf.tanh(
                tf.reshape(tf.squeeze(self.memory_layer(tf.ones([1, 1])), [self.memory_size, self.memory_vector_dim])))
        elif self.init_mode == 'random':
            inner_M = tf.tanh(self.memory_layer)
        elif self.init_mode == 'constant':
            inner_M = self.memory_layer

        M = expand(inner_M, axis=0, N=batch_size)

        return NTMControllerState(
            controller_state=controller_init_state,
            read_vector_list=read_vector_list,
            w_list=w_list,
            M=M)

    @property
    def state_size(self):
        # Sum of sizes of all internal states
        return NTMControllerState(
            controller_state=sum(sum(x) for x in self.controller.state_size),
            read_vector_list=[self.memory_vector_dim for _ in range(self.read_head_num)],
            w_list=[self.memory_size for _ in range(self.read_head_num + self.write_head_num)],
            M=tf.TensorShape([self.memory_size * self.memory_vector_dim]))

    @property
    def output_size(self):
        # Size of the output layer
        return self.output_dim
