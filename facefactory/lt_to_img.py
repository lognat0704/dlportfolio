import os
import os.path
import random
import PIL.Image
import numpy as np
import uuid
import dnnlib.tflib as tflib
from encoder.generator_model import Generator

import logging

logger = logging.getLogger()

class Image_generator:
    def __init__(self, Gs_network):
        tflib.init_tf()
        self.Gs_network = Gs_network
       
    def run(self, src_lt, images_dir, img_name, flag, noise_flag = False, truncation_psi = 0.7, truncation_layers = 8):
        """
        return: numpy arrays with shape (batch, h, w, c)
             
        """
        ## src_lt can be a npy file or a np array, both have to be shape of (1, 512)
        if src_lt is not None: 
            ## image file name, if not given in images_dir, is drived from src_lt file name if available, otherwise uuid
            if isinstance(src_lt, str): ## npy file
                src_lt = np.load(src_lt)
                if img_name is None and images_dir is not None:
                    img_name = os.path.splitext(os.path.basename(src_lt))[0]
            else:
                if img_name is None and images_dir is not None:
                    img_name = uuid.uuid4().hex
            
            assert len(src_lt.shape) in (2, 3), "Input dimension is not 2 or 3"
            
            if truncation_psi is not None and truncation_layers is not None:
                use_truncation = True
            else:
                use_truncation = False
                truncation_psi = 1.0
                truncation_layers = 0
            
            ## add batch dimension if necessary
            if len(src_lt.shape) == 2:
                src_lt = np.expand_dims(src_lt, axis = 0)
            
            assert src_lt.shape[1:] in ((1,512), (18,512)), "Input shape w/o batch is NOT (1, 512) or (18, 512)"
                            
            if src_lt.shape[1:] == (1, 512): ## single layer, tilt to 18 layers
                dlatents = np.tile(src_lt, (1, 18, 1))
            else:
                dlatents = src_lt
                
            if use_truncation: ## use truncation
                coeff = np.ones_like(dlatents)
                coeff[:,:truncation_layers, :] *= truncation_psi
                dlatents = dlatents * coeff
                            
            for layer, layer_dl in enumerate(dlatents[0]):
                if np.sum(np.abs(layer_dl) > 1):
                    logger.debug(f'layer {layer} has {np.sum(np.abs(layer_dl) >= 1)} dimensions beyond 1 or -1')
            #dlatents[dlatents > 1] = 1
            #dlatents[dlatents < -1] = -1
    
            synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
            raw_images = self.Gs_network.components.synthesis.run(dlatents, randomize_noise=noise_flag, **synthesis_kwargs)
            # import tensorflow as tf
            # print([n.name for n in tf.get_default_graph().as_graph_def().node if '_Run/concat' in n.name])
            # raw_images = np.squeeze(raw_images, axis = 0)  
            
            if images_dir:
                img = PIL.Image.fromarray(np.squeeze(raw_images, axis = 0), 'RGB')
                #img.save(os.path.join(generated_images_dir, f'{lt_name}.png'), 'PNG')
                img.save(os.path.join(images_dir, f'{img_name}.png'), 'PNG')
                #print('The image is saved to ' + os.path.join(images_dir, f'{img_name}.png'))
                logger.info(f'The image is saved to {os.path.join(images_dir, img_name)}.png')

        if src_lt is None:
                        
            w = 1024
            h= 1024
            
            ## randomly choose gender
            gender = random.choice(['male', 'female'])
            ## randomly choose 2 npy files, given gender
            person_src = np.tile(np.load(os.path.join('latents', gender, random.choice(os.listdir(os.path.join('latents', gender))))), (18, 1))
            person_dst = np.tile(np.load(os.path.join('latents', gender, random.choice(os.listdir(os.path.join('latents', gender))))), (18, 1))
            
            #candidates = np.load(np.random.choice(['candidates_m.npy', 'candidates_w.npy'], 1)[0])
            #a,b = np.random.choice(7, 2, replace = False)
                
            #person_src = np.tile(candidates[a][1], (18, 1))
            #person_dst = np.tile(candidates[b][1], (18, 1))
            
            if flag == 0: ## random perturbation
                dlatents = np.array([candidates[a][1] + np.random.normal(0,0.2, (1, 512))]* 18 +
                             [candidates[a][1] + np.random.normal(0,0.2, (1, 512))]* 18 +  
                             [candidates[b][1] + np.random.normal(0,0.2, (1, 512))]* 18 +
                             [candidates[b][1] + np.random.normal(0,0.2, (1, 512))]* 18).reshape(4, 18, 512)
        
            if flag == 1: ## layer swap
                person_dst_top = np.copy(person_dst)
                person_dst_mid = np.copy(person_dst)
                person_dst_btn = np.copy(person_dst)
                person_dst_half = np.copy(person_dst)
                    
                person_dst_top[0:4] = person_src[0:4]
                person_dst_mid[4:8] = person_src[4:8]
                person_dst_btn[8:12] = person_src[8:12]
                person_dst_half[0:9] = person_src[0:9]
                    
                dlatents = np.array([person_dst_top]+[person_dst_mid]+[person_dst_btn]+[person_dst_half])
                
            if flag == 2: ## linear combination
                dlatents = np.array([alpha * person_src + (1-alpha) * person_dst for alpha in [0.2, 0.4, 0.6, 0.8]])
            
            synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
            raw_images = self.Gs_network.components.synthesis.run(dlatents, randomize_noise=noise_flag, **synthesis_kwargs)
            
            if images_dir:
                #names = ['face_'+str(a)+ '_' + str(b) + '_' + str(int(i)) for i in np.linspace(1, 4, 4)]
                #generated_images = row_images
                #for img_array, img_name in zip(generated_images, names):
                #for img_array, img_name in zip(raw_images, names):
                for img in raw_images:
                    img = PIL.Image.fromarray(img, 'RGB')
                    if img_name is None:
                        img_name = uuid.uuid4().hex
                    img.save(os.path.join(images_dir, f'{img_name}.png'), 'PNG')  
                    #print('Face image from the random latent is saved to ' + os.path.join(images_dir, f'{img_name}.png'))
                    logger.info(f'Face image from the random latent is saved to {os.path.join(images_dir, img_name)}.png')
                    
        return raw_images