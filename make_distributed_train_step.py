import tensorflow as tf

def make_distributed_train_step(
        generator,
        discriminator,
        discriminator_loss,
        generator_loss,
        generator_optimizer,
        discriminator_optimizer,
        batch_size,
        latent_dim,
        strategy
):
    @tf.function
    def train_step_distributed(iterator):
        def train_step(inputs):
            images, labels = inputs
            noise = tf.random.normal([batch_size, latent_dim])
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator([noise, labels], training=True)
                gen_predictions = discriminator([generated_images, labels], training=True)
                real_predictions = discriminator([images, labels], training=True)
                disc_loss = discriminator_loss(real_predictions, gen_predictions)
                gen_loss = generator_loss(gen_predictions)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_weights)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_weights)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_weights))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_weights))

            return disc_loss, gen_loss

        disc_loss, gen_loss = strategy.run(train_step, args=(next(iterator),))

        disc_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, disc_loss, axis=None)
        gen_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, gen_loss, axis=None)

        return disc_loss, gen_loss

    return train_step_distributed
