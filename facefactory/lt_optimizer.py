# coding: utf-8

import os
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
#import dnnlib
import dnnlib.tflib as tflib
#import config
#import tensorflow as tf
#from encoder.generator_model import Generator 
from encoder.perceptual_model import PerceptualModel
# from encoder.generator_model_disc import Generator as Generator_disc
from encoder.perceptual_model_disc import PerceptualDiscriminatorModel
from encoder.generator_unified import Generator as MyGenerator

import logging

logger = logging.getLogger()

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
class Latent_optimizer:
    def __init__(self, discriminator_network, Gs_network, dlatent_df = 1):
        tflib.init_tf()
        self.discriminator_network = discriminator_network
        self.Gs_network = Gs_network
        self.dlatent_df = dlatent_df
        logger.info('FFHQ_Latent_optimizer is loaded')
        
        ## construct the generator model, dlatent_df controls number of distinct layers (1 or 18)
        self.generator = MyGenerator(self.Gs_network, 1, randomize_noise=False, dlatent_df = dlatent_df)
        
                       
    def optimize(self, src_lt, ref_img_src, dlatent_dir, max_iter=1200, multi_style_loss=False, disc_loss=True, clip = False, min_loss_improvement = 0.01, max_waiting_iter = 50):
        """
        return a dict with image file name as key and a dlatent_dfx512 np array of latent as value
        """
          
        ##ref_lt: make it batch_size(1)*18*512
        if isinstance(src_lt, str):
            src_lt = np.load(src_lt)
        if src_lt.shape != (18,512):
            src_lt = np.array([src_lt[0]]*18)     
        ref_lt = np.expand_dims(src_lt, axis=0)
        
        ##ref_img
        ref_images = [ref_img_src]
        batch_size = 1
        dlatent_dict = {}
         
        if multi_style_loss:
            img_size = 256 ## because VGG requires 256 as input dimension
            logger.info(f'Use VGG style loss, and set img size to {img_size} because the VGG requires it')
                        
            self.perceptual_model_vgg = PerceptualModel(img_size, layer=9, batch_size=1)
            self.perceptual_model_vgg.build_perceptual_model(self.generator.generated_image)
            
#             generator = Generator(self.Gs_network, batch_size = 1, randomize_noise=False, dlatent_df = 1)
              
#             ##ref_img
#             perceptual_model = PerceptualModel(256, layer=9, batch_size=1)
#             perceptual_model.build_perceptual_model(generator.generated_image)
        
               
#         def generate_image(latent_vector):
#             latent_vector = latent_vector.reshape((1, 18, 512))
#             self.generator.set_dlatents(latent_vector)
#             img_array = self.generator.generate_images()[0]
#             img = PIL.Image.fromarray(img_array, 'RGB')
#             return img.resize((256, 256))
        
        
            for images_batch in tqdm(split_to_batches(ref_images, batch_size), total=len(ref_images)//batch_size):
                names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
                self.perceptual_model_vgg.set_reference_images(images_batch)
                self.generator.set_dlatents(ref_lt) 
                op = self.perceptual_model_vgg.optimize(self.generator.dlatent_variable, max_iter, learning_rate=0.005)
                pbar = tqdm(op, leave=False, total=max_iter)
                for loss in pbar:
                    pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
                print(' '.join(names), ' loss:', loss)    
                generated_dlatents = self.generator.get_dlatents()
                
                for dlatent, img_name in zip(generated_dlatents, names):
                    ## clip each element between -2 and 2
                    if clip:
                        dlatent[dlatent > 1] = 2
                        dlatent[dlatent < -1] = -2
                        
                    dlatent_dict[img_name] = dlatent
                    if dlatent_dir:
                        np.save(os.path.join(dlatent_dir, f'{img_name}_{loss:.3f}_{self.dlatent_df}x512.npy'), dlatent)
                        logger.info(f'{self.dlatent_df}x512 latent generated and saved to ' + os.path.join(dlatent_dir, f'{img_name}_{best_loss:.3f}.npy'))
            
            return dlatent_dict
                
                
        if disc_loss:
            img_size = 1024 ## because discriminator requires 1024 as input dimension
            logger.info(f'Use discriminator loss and set img size to {img_size} because the discriminator requires it')
            
            self.perceptual_model_disc = PerceptualDiscriminatorModel(img_size, batch_size=1)
            self.perceptual_model_disc.build_perceptual_model(self.discriminator_network, 
                                              self.generator.generator_output, ## image tensor
                                              self.generator.generated_image, ## image 
                                              self.generator.dlatent_variable)
            
            
            for images_batch in tqdm(split_to_batches(ref_images, batch_size), total=len(ref_images)//batch_size):
                names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
     
                self.perceptual_model_disc.set_reference_images(images_batch)
                #self.generator.set_dlatents(ref_lt[0,0].reshape((1,512)))
                self.generator.set_dlatents(ref_lt) 
                op = self.perceptual_model_disc.optimize(max_iter, min_loss_improvement, max_waiting_iter)
                pbar = tqdm(op, leave=False, total=max_iter, mininterval = 5.0)
            
                best_loss = None
                init_loss = None
                next_loss = None
                best_dlatent = None
                best_iter = None
                dlatent_frames = []
            
                for it, loss_dict in enumerate(pbar):
                    pbar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v) for k, v in loss_dict.items()]))
                    curr_loss = loss_dict['loss']
                                        
                    if it == 0: ## initialization
                        init_loss = curr_loss
                        best_loss = curr_loss
                        best_dlatent = self.generator.get_dlatents()
                        best_iter = 1 
                        next_loss = best_loss - abs(best_loss) * min_loss_improvement
                        
                    elif curr_loss < best_loss: ## if loss drops
                        
                        ## captures best loss and dlatent                        
                        best_loss = curr_loss
                        best_dlatent = self.generator.get_dlatents()
                        self.generator.stochastic_clip_dlatents()
                        
                        if best_loss < next_loss: ## if loss drops big enough, reset iteration watch
                            best_iter = it + 1 
                            next_loss = best_loss - abs(best_loss) * min_loss_improvement
                        
                        elif it - best_iter >= max_waiting_iter: ## if no big drop happens after waiting long enough
                            break
                
                logger.info(" ".join(names) + " Loss drops from {:.4f} to {:.4f} at iteration {}".format(init_loss, best_loss, best_iter+1))
                
                for dlatent, img_name in zip(best_dlatent, names):
                    dlatent = dlatent.reshape(-1,512)
#                     if clip:
#                         dlatent[dlatent > 1] = 1
#                         dlatent[dlatent < -1] = -1
                    
                    dlatent_dict[img_name] = dlatent
                    if dlatent_dir:
                        dlatent_fn = f'{img_name}_{best_loss:.3f}_{self.dlatent_df}x512_disc.npy'
                        np.save(os.path.join(dlatent_dir, dlatent_fn), dlatent)
                        logger.info(f'{self.dlatent_df}x512 latent generated and saved to ' + os.path.join(dlatent_dir, dlatent_fn))
                
            return dlatent_dict
