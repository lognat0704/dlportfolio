# coding: utf-8
import numpy as np
import os
import logging

logger = logging.getLogger()

             
class Latent_mixer:
    def __init__(self):
        logger.info('Latent_mixer_loaded')
        
    def lt_direction_mix(self, src_lt, mix_dl_dir, direction, coeff, n = 1):     
        """
        src_lt: 2 dimension array with shape 1x512 or 18x512, or a npy file containing such array
        return value: 3 dimension array [n, 18, 512]
        """
        if isinstance(src_lt, str):
            src_name = os.path.splitext(os.path.basename(src_lt))[0]
            src_lt = np.load(src_lt)
        if src_lt.shape != (18,512):
            src_lt = np.array([src_lt[0]]*18)
            logger.info("This latent is not (18,512), and its automatically update to (18,512)")
            
        if direction in ['smile', 'gender', 'age', 'pose']:
            
            #direction_vector = np.load(os.path.join('lt_directions', direction + '.npy')) ## 18 * 512
            # ## make every row of direction vector with norm = 1
            # direction_vector = direction_vector / np.linalg.norm(direction_vector[0])
            
            direction_vector = np.load(os.path.join('lt_directions/InterFaceGAN', 'stylegan_ffhq_'+ direction + '_w_boundary' + '.npy')) ## 1 * 512
            direction_vector = np.tile(direction_vector, (18, 1))
            
            # dot = np.dot(src_lt[0], direction_vector[0]) ## assuming both src_lt and direction_vector have identical 18 layers
            dot = np.array([np.dot(src_lt[i], direction_vector[i]) for i in range(18)])
            #dot = 0
                              
        shrink = 0.9
        new_lt = []
        
        for i in range(n):
            if coeff > 0: 
                new_lt.append(src_lt + max(coeff*(shrink**i) - np.mean(dot), 0) * direction_vector)
                #new_lt.append(src_lt + np.matmul(np.diag(np.clip(coeff*(shrink**i) - dot, a_min=0, a_max=None)), direction_vector))
            else: 
                new_lt.append(src_lt + min(coeff*(shrink**i) - np.mean(dot), 0) * direction_vector)
                #new_lt.append(src_lt + np.matmul(np.diag(np.clip(coeff*(shrink**i) - min(dot), a_min=None, a_max=0)), direction_vector))
            #logger.info(f'dot product between the latent and {direction} vector: {dot:.2f} --> {np.dot(new_lt[i][0], direction_vector[0]):.2f}')
        new_lt = np.asarray(new_lt)
        
        if mix_dl_dir:
            np.save(os.path.join(mix_dl_dir, f'{src_name}_{direction}_{coeff}.npy'), new_lt)
            #print('Add direction and save to')
            #print(os.path.join(mix_dl_dir, f'{src_name}_{direction_name}_{coeff}.npy'))
            logger.info('Add direction and save to ' + os.path.join(mix_dl_dir, f'{src_name}_{direction}_{coeff}.npy'))
        return new_lt
        
    def lt_swap(self,src_lt, dst_lt, mix_dl_dir):
        src_name = os.path.splitext(os.path.basename(src_lt))[0]
        dst_name = os.path.splitext(os.path.basename(dst_lt))[0]
        src_lt = np.array([np.load(src_lt)[0]]*18)
        dst_lt = np.array([np.load(dst_lt)[0]]*18) 
        person_src = src_lt
        person_dst = dst_lt
    
        person_dst_top = np.copy(person_dst)
        person_dst_mid = np.copy(person_dst)
        person_dst_btn = np.copy(person_dst)
        person_dst_half = np.copy(person_dst)
    
        person_dst_top[0:4] = person_src[0:4]
        person_dst_mid[4:8] = person_src[4:8]
        person_dst_btn[8:12] = person_src[8:12]
        person_dst_half[0:9] = person_src[0:9]
    
        dlatents = np.array([person_dst_top]+[person_dst_mid]+[person_dst_btn]+[person_dst_half])
        if mix_dl_dir:
            for i,lt in enumerate(dlatents):
                np.save(os.path.join(mix_dl_dir, f'{src_name}_{dst_name}_{i}.npy'), lt)
                print('lt_swap and save to')
                print(os.path.join(mix_dl_dir, f'{src_name}_{dst_name}_{i}.npy'))
        return dlatents
            
            
    def lt_linear(self,src_lt, dst_lt, mix_dl_dir):
        src_name = os.path.splitext(os.path.basename(src_lt))[0]
        dst_name = os.path.splitext(os.path.basename(dst_lt))[0]
        src_lt = np.array([np.load(src_lt)[0]]*18)
        dst_lt = np.array([np.load(dst_lt)[0]]*18) 
        person_src = src_lt
        person_dst = dst_lt
        dlatents = np.array([(1 - i) * person_src + i * person_dst for i in np.linspace(0, 1, 5)])
    
        if mix_dl_dir:
            for i,lt in enumerate(dlatents[1:-1]):
                np.save(os.path.join(mix_dl_dir, f'{src_name}_{dst_name}_{i}.npy'), lt)
                #print('lt_linear and save to')
                #print(os.path.join(mix_dl_dir, f'{src_name}_{dst_name}_{i}.npy'))
                logger.info(f'lt_linear and save to {mix_dl_dir}/{src_name}_{dst_name}_{i}.npy')
        return dlatents