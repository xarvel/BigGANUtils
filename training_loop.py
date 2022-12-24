from tqdm import tqdm
import time

def training_loop(
        start_epoch,
        epochs,
        steps_per_epochs,
        train_iterator,
        train_step,
        on_step_end,
        on_epoch_end
):
    for epoch in range(start_epoch, epochs + 1):
        start = time.time()
        print('Epoch: {}/{}'.format(epoch, epochs))

        pbar = tqdm(range(steps_per_epochs))
        for step in pbar:
            disc_loss_real, disc_loss_fake, gen_loss = train_step(train_iterator)

            pbar.set_postfix({
                'disc_loss_real': round(float(disc_loss_real), 4),
                'disc_loss_fake': round(float(disc_loss_fake), 4),
                'gen_loss': round(float(gen_loss), 4)
            })

            pbar.set_description("Current step %s" % (epoch * steps_per_epochs) + step)
            on_step_end(disc_loss_real, disc_loss_fake, gen_loss)

        on_epoch_end(epoch)

        print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))