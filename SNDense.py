from .SpectralNormalization import SpectralNormalization
from tensorflow.keras.layers import Dense

def SNDense(*args, sn_epsilon=1e-12, **kwargs):
    return SpectralNormalization(Dense(*args, **kwargs), name=kwargs.get('name'), epsilon=sn_epsilon)

