from .SpectralNormalization import SpectralNormalization
from tensorflow.keras.layers import Conv2D

def SNConv2D(*args, sn_epsilon=1e-12, **kwargs):
    return SpectralNormalization(Conv2D(*args, **kwargs), name=kwargs.get('name'), epsilon=sn_epsilon)

