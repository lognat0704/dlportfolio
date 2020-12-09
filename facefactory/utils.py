import dnnlib.tflib as tflib
import pickle

def load_stylegan(model_file_path):
    tflib.init_tf()
    
    with open(model_file_path, 'rb') as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)
    
    return generator_network, discriminator_network, Gs_network