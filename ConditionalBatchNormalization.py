import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization, Layer

from .SNDense import SNDense

class ConditionalBatchNormalization(Layer):
    def __init__(self, kernel_initializer=None, batch_normalization=BatchNormalization, dense=SNDense,  sn_epsilon=1e-12, bn_epsilon=1e-3, **kwargs):
        super(ConditionalBatchNormalization, self).__init__(**kwargs)

        self.kernel_initializer = kernel_initializer
        self.dense = dense
        self.bn_epsilon = bn_epsilon
        self.sn_epsilon = sn_epsilon
        self.batch_normalization = batch_normalization

    def get_config(self):
        config = super().get_config()
        config.update({
            # "kernel_initializer": self.kernel_initializer,
            # "dense": self.dense
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        batch, height, width, channels = input_shape

        self.linear_gamma = self.dense(
            sn_epsilon=self.sn_epsilon,
            units=channels,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name='linear_gamma'
        )
        self.linear_beta = self.dense(
            sn_epsilon=self.sn_epsilon,
            units = channels,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name='linear_beta'
        )
        self.batchnorm = self.batch_normalization(name='batch_norm', epsilon=self.bn_epsilon)

    # @tf.function
    def call(self, x, c, training=None):
        x = self.batchnorm(x, training=training)
        gamma = self.linear_gamma(c, training=training)
        beta = self.linear_beta(c, training=training)

        return x * gamma[:, None, None] + beta[:, None, None]
