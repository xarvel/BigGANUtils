import tensorflow as tf

from tensorflow.keras.layers import Layer, Reshape

from .SNConv2D import SNConv2D

class SelfAttention(Layer):
    def __init__(self, kernel_initializer=None, conv2d=SNConv2D, sn_epsilon=1e-12, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.conv2d = conv2d
        self.kernel_initializer = kernel_initializer
        self.sn_epsilon = sn_epsilon

    def get_config(self):
        config = super().get_config()
        config.update({
            # "conv2d": self.conv2d,
            # "kernel_initializer": self.kernel_initializer,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        batch, height, width, in_channels = input_shape

        self.theta = self.conv2d(filters=in_channels // 8, kernel_size=1, padding='valid', use_bias=False,
                                 kernel_initializer=self.kernel_initializer, sn_epsilon=self.sn_epsilon)
        self.phi = self.conv2d(filters=in_channels // 8, kernel_size=1, padding='valid', use_bias=False,
                               kernel_initializer=self.kernel_initializer, sn_epsilon=self.sn_epsilon)
        self.g = self.conv2d(filters=in_channels // 2, kernel_size=1, padding='valid', use_bias=False,
                             kernel_initializer=self.kernel_initializer, sn_epsilon=self.sn_epsilon)
        self.o = self.conv2d(filters=in_channels, kernel_size=1, padding='valid', use_bias=False,
                             kernel_initializer=self.kernel_initializer, sn_epsilon=self.sn_epsilon)

        self.reshape_theta = Reshape((height * width, in_channels // 8))
        self.reshape_phi = Reshape(((height * width) // 4, in_channels // 8))
        self.reshape_g = Reshape(((height * width) // 4, in_channels // 2,))
        self.reshape_o = Reshape((height, width, in_channels // 2))

        # Learnable gain parameter
        self.gamma = self.add_weight(
            name="gamma",
            shape=(),
            initializer="zeros",
            dtype=tf.float32,
            trainable=True,
        )
        self.built = True

    @tf.function
    def call(self, x):
        # Apply convs
        theta = self.theta(x)
        phi = tf.nn.max_pool2d(self.phi(x), [2, 2], strides=2, padding='VALID')
        g = tf.nn.max_pool2d(self.g(x), [2, 2], strides=2, padding='VALID')
        # Perform reshapes
        theta = self.reshape_theta(theta)
        phi = self.reshape_phi(phi)
        g = self.reshape_g(g)
        # Matmul and softmax to get attention maps
        beta = tf.nn.softmax(tf.matmul(theta, phi, transpose_b=True), -1)
        # Attention map times g path
        o = self.o(self.reshape_o(tf.matmul(g, beta, transpose_a=True, transpose_b=True)))
        return self.gamma * o + x