from tensorflow.keras.layers import Embedding
from .SpectralNormalization import SpectralNormalization

def SNEmbedding(*args, sn_epsilon=1e-12, **kwargs):
    return SpectralNormalization(Embedding(*args, **kwargs), name=kwargs.get('name'), epsilon=sn_epsilon)