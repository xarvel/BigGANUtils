import matplotlib.pyplot as plt
import tensorflow as tf

from .model_to_img import model_to_img


def sample_single_image_mnist(
        noise: tf.Tensor,
        generator: tf.keras.Model,
        label: int
):
    z = tf.expand_dims(noise, axis=0)
    label = tf.constant([label], dtype=tf.int64)
    generated_image = generator([z, label], training=False)
    generated_image_show = tf.squeeze(generated_image, axis=3)
    print(generated_image.shape)
    plt.axis('off')
    plt.imshow(model_to_img(generated_image_show[0]), cmap='gray')

    return generated_image[0]
