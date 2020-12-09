# coding: utf-8
import os
from tqdm import tqdm
import numpy as np
import dnnlib.tflib as tflib

from encoder.perceptual_model import load_images
from keras.models import load_model
#rom keras.preprocessing import image
import efficientnet
import logging

logger = logging.getLogger()

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

class Latent_generator:
    def __init__(self, model_file, model_type):
        tflib.init_tf()
        self.encoder = load_model(model_file)
        self.model_type = model_type
        logger.info('Latent_generator is loaded')
       
    def predict(self, src_img, dlatent_dir) :
        # Hyper parameters
        BATCH_SIZE = 1 
        ref_images= [src_img]
        ref_images = list(filter(os.path.isfile, ref_images))
            
        if dlatent_dir:
            os.makedirs(dlatent_dir, exist_ok=True)
            
        dlatent_dict = {}
                   
        if self.model_type == 'effnet':
            from efficientnet import preprocess_input as preprocess_input_by_model
        if self.model_type == 'resnet':
            from keras.applications.resnet50 import preprocess_input as preprocess_input_by_model
            
        for images_batch in tqdm(split_to_batches(ref_images, BATCH_SIZE), total=len(ref_images)//BATCH_SIZE):
            names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
            
            dlatents = self.encoder.predict(preprocess_input_by_model(load_images(images_batch,img_size=256)))
            
            if dlatent_dir:
                for dlatent, img_name in zip(dlatents, names):
                    np.save(os.path.join(dlatent_dir, f'{img_name}.npy'), dlatent)
                    logger.info(f'1x512 latent generated and save to {os.path.join(dlatent_dir, img_name)}.npy')
             
            for i,name in enumerate(names):
                dlatent_dict[name] = dlatents[i]
        
        return dlatent_dict
                       
    
        
