import tensorflow as tf
from tensorflow.keras.layers import ReLU, AveragePooling2D, Layer

from .SNConv2D import SNConv2D

class DBlock(Layer):
    def __init__(
            self,
            out_channels,
            downsample=True,
            preactivation=True,
            kernel_initializer=None,
            conv2d=SNConv2D,
            sn_epsilon=1e-12,
            **kwargs
    ):

        super(DBlock, self).__init__(**kwargs)

        self.sn_epsilon = sn_epsilon
        self.conv2d = conv2d
        self.out_channels = out_channels
        self.hidden_channels = out_channels
        self.kernel_initializer = kernel_initializer
        self.activation1 = ReLU()
        self.activation2 = ReLU()
        self.conv33_1 = self.conv2d(filters=self.hidden_channels,
                                    kernel_size=3,
                                    padding='same',
                                    sn_epsilon=self.sn_epsilon,
                                    kernel_initializer=self.kernel_initializer)
        self.conv33_2 = self.conv2d(filters=self.out_channels,
                                    kernel_size=3,
                                    padding='same',
                                    sn_epsilon=self.sn_epsilon,
                                    kernel_initializer=self.kernel_initializer)
        self.av_pool_1 = AveragePooling2D(padding='valid')
        self.av_pool_2 = AveragePooling2D(padding='valid')
        self.av_pool_3 = AveragePooling2D(padding='valid')

        self.downsample = downsample
        self.preactivation = preactivation

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "downsample": self.downsample,
            "preactivation": self.preactivation,
            "kernel_initializer": self.kernel_initializer,
            "conv2d": self.conv2d,
            "sn_epsilon": self.sn_epsilon
        })
        return config

    def build(self, input_shape):
        batch, height, width, channels = input_shape
        self.in_channels = channels
        self.learnable_sc = True if (self.in_channels != self.out_channels) or self.downsample else False

        if self.learnable_sc:
            self.conv_sc = self.conv2d(filters=self.out_channels,
                                       kernel_size=1,
                                       padding='valid',
                                       sn_epsilon=self.sn_epsilon,
                                       kernel_initializer=self.kernel_initializer)

    def shortcut(self, x, training=None):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x, training=training)
            if self.downsample:
                x = self.av_pool_1(x)
        else:
            if self.downsample:
                x = self.av_pool_1(x)
            if self.learnable_sc:
                x = self.conv_sc(x, training=training)
        return x

    @tf.function
    def call(self, x, training=None):
        if self.preactivation:
            h = self.activation1(x)
        else:
            h = x

        h = self.conv33_1(h, training=training)
        h = self.conv33_2(self.activation2(h), training=training)

        if self.downsample:
            h = self.av_pool_3(h)

        return h + self.shortcut(x, training=training)
