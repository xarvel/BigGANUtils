from tensorflow.keras.layers import ReLU, UpSampling2D, Layer, BatchNormalization

from .ConditionalBatchNormalization import ConditionalBatchNormalization
from .SNConv2D import SNConv2D
from .SNDense import SNDense


# @tf.RegisterGradient("ResizeBilinearGrad")
# def _ResizeBilinearGrad_grad(op, grad):
#     up = tf.image.resize(grad, tf.shape(op.inputs[0])[1:-1])
#     return up, None
#
#
# @tf.RegisterGradient("ResizeNearestNeighborGrad")
# def ResizeNearestNeighborGrad(op, grad):
#     up = tf.image.resize(grad, tf.shape(op.inputs[0])[1:-1])
#     return up, None


class GBlock(Layer):
    def __init__(
            self,
            out_channels,
            batch_normalization=BatchNormalization,
            kernel_initializer=None,
            conv2d=SNConv2D,
            bn_epsilon=1e-3,
            sn_epsilon=1e-12,
            dense=SNDense,
            **kwargs
    ):
        super(GBlock, self).__init__(**kwargs)

        self.batch_normalization = batch_normalization
        self.sn_epsilon = sn_epsilon
        self.bn_epsilon = bn_epsilon
        self.conv2d = conv2d
        self.dense = dense
        self.out_channels = out_channels
        self.kernel_initializer = kernel_initializer

        self.cbn_1 = ConditionalBatchNormalization(
            kernel_initializer=self.kernel_initializer,
            dense=self.dense,
            name='cbn_1',
            bn_epsilon=self.bn_epsilon,
            sn_epsilon=self.sn_epsilon,
            batch_normalization=self.batch_normalization
        )
        self.cbn_2 = ConditionalBatchNormalization(
            kernel_initializer=self.kernel_initializer,
            dense=self.dense,
            name='cbn_2',
            bn_epsilon=self.bn_epsilon,
            sn_epsilon=self.sn_epsilon,
            batch_normalization=self.batch_normalization
        )
        self.up_sample_1 = UpSampling2D(name='up_sample_1')
        self.up_sample_2 = UpSampling2D(name='up_sample_2')
        self.relu_1 = ReLU(name='relu_1')
        self.relu_2 = ReLU(name='relu_2')
        self.conv33_1 = self.conv2d(
            filters=self.out_channels,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            name='conv33_1',
            sn_epsilon=self.sn_epsilon
        )
        self.conv33_2 = self.conv2d(
            filters=self.out_channels,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            name='conv33_2',
            sn_epsilon=self.sn_epsilon
        )
        self.upsample = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels
            # "kernel_initializer": self.kernel_initializer,
            # "conv2d": self.conv2d,
            # "dense": self.dense,
            # "bn_epsilon": self.bn_epsilon,
            # "sn_epsilon": self.sn_epsilon
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        x_shape, label_shape = input_shape
        batch, height, width, channels = x_shape
        self.in_channels = channels
        self.learnable_sc = self.in_channels != self.out_channels

        if self.learnable_sc:
            self.conv_sc = self.conv2d(
                filters=self.out_channels,
                kernel_size=1,
                padding='valid',
                kernel_initializer=self.kernel_initializer,
                name='conv_sc',
                sn_epsilon=self.sn_epsilon
            )

    # @tf.function
    def call(self, inputs, training=None):
        x, label = inputs
        h = self.relu_1(self.cbn_1(x, label, training=training))
        if self.upsample:
            h = self.up_sample_1(h)
            x = self.up_sample_2(x)

        h = self.conv33_1(h, training=training)
        h = self.relu_2(self.cbn_2(h, label, training=training))
        h = self.conv33_2(h, training=training)

        if self.learnable_sc:
            x = self.conv_sc(x, training=training)

        return h + x
