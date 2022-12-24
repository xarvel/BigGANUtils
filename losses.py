import tensorflow as tf

def discriminator_loss(logits_real: tf.Tensor, logits_fake: tf.Tensor) -> tf.Tensor:
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - logits_real))
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + logits_fake))

    return real_loss, fake_loss
def generator_loss(logits_fake: tf.Tensor) -> tf.Tensor:
    loss = -tf.reduce_mean(logits_fake)

    return loss