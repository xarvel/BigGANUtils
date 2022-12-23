import glob
import IPython
import imageio
from pygifsicle import optimize

def combine_samples(dir, filename='image_at_epoch_*.png'):
    gif_file = dir + '/samples.gif'

    with imageio.get_writer(gif_file, mode='I') as writer:
        filenames = glob.glob(dir + '/' + filename)
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    optimize(gif_file)  # optimize gif size
    return IPython.display.Image(filename=gif_file, embed=True)
