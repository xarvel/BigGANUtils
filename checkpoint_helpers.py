import tensorflow as tf
import os

def checkpoint_helpers(
        generator_optimizer,
        discriminator_optimizer,
        generator,
        discriminator,
        dir
):
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )

    local_device_option = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    latest_checkpoint = tf.train.latest_checkpoint(dir)

    def restore():
        return checkpoint.restore(latest_checkpoint, options=local_device_option)

    def save():
        checkpoint_prefix = os.path.join(dir, "checkpoint")
        checkpoint.save(file_prefix=checkpoint_prefix, options=local_device_option)

    return restore, save

