import matplotlib.pyplot as plt

def sample_analytics(
        dir,
        generator_lr,
        discriminator_lr,
        filters,
        batch_size,
        meta,
        save=False,
        show=True
):
    history = meta['history']

    with plt.xkcd():
        fig, axs = plt.subplots(figsize=(10, 10))
        plt.plot(history['gen_loss'], label='Generator loss')
        plt.plot(history['disc_loss_fake'], label='Discriminator loss fake')
        plt.plot(history['disc_loss_real'], label='Discriminator loss real')
        plt.title('Training process')

        plt.figtext(.7, .16, '\n'.join([
            'Epoch = %s' % meta['epoch'],
            "Generator LR = %s" % generator_lr,
            'Discriminator LR = %s' % discriminator_lr,
            'Filters = %s' % filters,
            'Batch size = %s' % batch_size
        ]))

        fig.legend()

        if save:
            fig.savefig(dir + '/analytics_at_epoch_{:04d}.png'.format(meta['epoch']))

        if show:
            plt.show()
        else:
            plt.close(fig)