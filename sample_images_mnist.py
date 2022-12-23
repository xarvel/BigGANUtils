import os

import matplotlib.pyplot as plt
import tensorflow as tf
from .model_to_img import model_to_img

num_classes = 10

def sample_images_mnist(
        image_size: int,
        noise: tf.Tensor,
        generator: tf.keras.Model,
        save_dir: str = 'samples',
        step: int = 0,
        save: bool = False,
        show: bool = True,
        zoom: int = 1
):
    labels = tf.range(num_classes)
    z = tf.repeat(tf.expand_dims(noise, axis=0), num_classes, axis=0)
    gen_imgs = generator([z, labels])
    gen_imgs = tf.squeeze(gen_imgs, axis=3)
    gen_imgs = model_to_img(gen_imgs)

    fig, axs = plt.subplots(1, num_classes)
    fig.subplots_adjust(
        wspace=0.0,
        hspace=0.0
    )
    px = 1 / plt.rcParams['figure.dpi']

    fig.set_figheight(zoom * image_size * px)
    fig.set_figwidth(zoom * image_size * num_classes * px)

    for i in range(num_classes):
        axs[i].imshow(gen_imgs[i], cmap='gray')
        axs[i].axis('off')

    # Create folder if not exits
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if save:
        plt.savefig(save_dir + '/image_at_step_{:06d}.png'.format(step), bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)
