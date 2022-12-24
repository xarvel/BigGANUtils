import json
import os

from .checkpoint_helpers import checkpoint_helpers


def checkpoint_with_history(
        generator_optimizer,
        discriminator_optimizer,
        generator,
        discriminator,
        dir
):
    restore_checkpoint, save_checkpoint = checkpoint_helpers(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
        dir=dir
    )

    defaultMeta = {
        "history": {
            "disc_loss_real": [],
            "disc_loss_fake": [],
            "gen_loss": []
        },
        "epoch": 1
    }
    meta = defaultMeta

    META_FILE = dir + '/meta.json'

    def restore_checkpoint_with_history():
        try:
            with open(META_FILE) as f:
                meta = json.load(f)
                meta['epoch'] += 1
        except:
            meta = defaultMeta
            pass

        status = restore_checkpoint()

        return meta

    def save_checkpoint_with_history():
        print('Saving checkpoint');

        # Create folder if not exits
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

        with open(META_FILE, 'w') as f:
            json.dump(meta, f)

        save_checkpoint()

    return restore_checkpoint_with_history, save_checkpoint_with_history
