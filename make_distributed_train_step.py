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
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    @tf.function
    def train_step_distributed(iterator):
        def train_step(inputs):
            images, labels = inputs
            noise = tf.random.normal([batch_size, latent_dim])
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.watch(generator.trainable_weights + discriminator.trainable_weights)

                generated_images = generator([noise, labels], training=True)
                gen_predictions = discriminator(generated_images, labels, training=True)
                real_predictions = discriminator(images, labels, training=True)
                disc_loss_real, disc_loss_fake = discriminator_loss(real_predictions, gen_predictions)
                disc_loss_real = disc_loss_real / global_batch_size
                disc_loss_fake = disc_loss_fake / global_batch_size

                disc_loss = disc_loss_real + disc_loss_fake
                gen_loss = generator_loss(gen_predictions)
                gen_loss = gen_loss / global_batch_size

            gradients_of_generator = tape.gradient(gen_loss, generator.trainable_weights)
            gradients_of_discriminator = tape.gradient(disc_loss, discriminator.trainable_weights)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_weights))
            discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, discriminator.trainable_weights)
            )

            return disc_loss_real, disc_loss_fake, gen_loss

        disc_loss_real, disc_loss_fake, gen_loss = strategy.run(train_step, args=(next(iterator),))

        disc_loss_real = strategy.reduce(tf.distribute.ReduceOp.MEAN, disc_loss_real, axis=None)
        disc_loss_fake = strategy.reduce(tf.distribute.ReduceOp.MEAN, disc_loss_fake, axis=None)
        gen_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, gen_loss, axis=None)

        return disc_loss_real, disc_loss_fake, gen_loss

    return train_step_distributed
