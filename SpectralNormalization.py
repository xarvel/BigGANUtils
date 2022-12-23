import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import Wrapper


class SpectralNormalization(Wrapper):
    """
    Attributes:
       layer: tensorflow keras layers (with kernel attribute)
    """

    def __init__(self, layer, epsilon=1e-12, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        """Build `Layer`"""

        if not self.layer.built:
            self.layer.build(input_shape)

            if hasattr(self.layer, "kernel"):
                self.w = self.layer.kernel
            elif hasattr(self.layer, "embeddings"):
                self.w = self.layer.embeddings
            else:
                raise AttributeError(
                    "{} object has no attribute 'kernel' nor "
                    "'embeddings'".format(type(self.layer).__name__)
                )

            self.w_shape = self.w.shape.as_list()
            self.u = self.add_weight(
                shape=tuple([1, self.w_shape[-1]]),
                initializer=k.initializers.TruncatedNormal(stddev=0.02),
                name='sn_u',
                trainable=False,
                dtype=tf.float32)

        super(SpectralNormalization, self).build()

    # @tf.function
    def call(self, inputs):
        """Call `Layer`"""
        # Recompute weights for each forward pass
        self._compute_weights()
        output = self.layer(inputs)
        return output

    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        eps = self.epsilon
        _u = tf.identity(self.u)

        _v = tf.matmul(_u, tf.transpose(w_reshaped))
        _v = _v / tf.maximum(tf.reduce_sum(_v ** 2) ** 0.5, eps)
        _u = tf.matmul(_v, w_reshaped)
        _u = _u / tf.maximum(tf.reduce_sum(_u ** 2) ** 0.5, eps)

        self.u.assign(_u)
        sigma = tf.matmul(tf.matmul(_v, w_reshaped), tf.transpose(_u))

        weights = self.w / sigma

        if hasattr(self.layer, 'kernel'):
            self.layer.kernel = weights
        elif hasattr(self.layer, 'embeddings'):
            self.layer.embeddings = weights


    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())