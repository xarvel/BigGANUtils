import math
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from .model_to_img import model_to_img

def sample_images(
        image_size: int,
        num_classes: int,
        noise: tf.Tensor,
        save_dir: str = 'samples',
        epoch: int = 0,
        generator: tf.keras.Model = None,
        save: bool = False,
        show: bool = True,
        zoom: int = 1
):
    square_size = int(math.sqrt(num_classes))
    rows = square_size
    cols = square_size

    z = tf.repeat(tf.expand_dims(noise, axis=0), num_classes, axis=0)
    labels = tf.range(num_classes)
    gen_imgs = model_to_img(generator([z, labels]))

    fig, axs = plt.subplots(rows, cols)
    fig.subplots_adjust(
        wspace=0.0,
        hspace=0.0
    )
    px = 1 / plt.rcParams['figure.dpi']

    fig.set_figheight(zoom * image_size * rows * px)
    fig.set_figwidth(zoom * image_size * cols * px)

    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(gen_imgs[j * square_size + i])

            axs[i, j].axis('off')

    # Create folder if not exits
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if save:
        plt.savefig(save_dir + '/image_at_epoch_{:04d}.jpeg'.format(epoch), bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)
