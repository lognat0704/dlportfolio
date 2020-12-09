import os
import argparse
import time
import PIL.Image
import datetime
from neuralgym import Config, get_gpus
import logging
import numpy as np

start = time.time()
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger()
#get_gpus(1)

config = Config('face_entertain_config.yml')


parser = argparse.ArgumentParser(description='Face Entertainment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_img', help='input image path')
parser.add_argument('--output_path', help='output file path')

args, other_args = parser.parse_known_args()

img_name = os.path.splitext(os.path.basename(args.input_img))[0]

## create folders if necessary
os.makedirs(args.output_path, exist_ok=True)
os.makedirs('temp', exist_ok=True)

## load stylegan model
from utils import load_stylegan
_, discriminator_network, Gs_network = load_stylegan(config.model_path)
logger.info('GAN model loaded')

from lt_to_img import Image_generator
img_generator = Image_generator(Gs_network)

from lt_mixer import Latent_mixer
lt_mix = Latent_mixer()

from img_filter import Align_img
align_img = Align_img('data/shape_predictor_68_face_landmarks.dat')
    
aligned_file = align_img.landmarksDetector(args.input_img, None, config.face_id)

if aligned_file is None:
    raise SystemError('No face detected')

if config.latent_file == '' or config.latent_file is None:
    
    from img_to_lt import Latent_generator
    latent_g_effnet = Latent_generator(model_file = 'data/finetuned_effnet.h5.b3', model_type = 'effnet')
    
    dlatent = latent_g_effnet.predict(aligned_file, dlatent_dir=None)
    pre_opt_dlatent = list(dlatent.values())[0]
    
    if config.optimal_flag:
        from lt_optimizer import Latent_optimizer
        optimizer = Latent_optimizer(discriminator_network, Gs_network, dlatent_df = config.dlatent_df)
        
        dlatent = optimizer.optimize(src_lt = list(dlatent.values())[0], ref_img_src = aligned_file, 
                                 dlatent_dir='temp', max_iter = config.max_iter, 
                                 multi_style_loss=False, disc_loss=True, clip = True, 
                                 min_loss_improvement = config.min_loss_improvement,
                                 max_waiting_iter = config.max_waiting_iter)
else: 
    dlatent = {}
    pre_opt_dlatent = None
    logger.info(f'loading latent from {config.latent_file}')
    dlatent[os.path.splitext(os.path.basename(aligned_file))[0]] = np.load(config.latent_file)
    

## to show pictures after transformation
canvas = []
size = 512
d = 20

## to show pictures of original and replicated
if pre_opt_dlatent is not None:
    replica = PIL.Image.new('RGB', (size*3 + d*4, size*1 + d*2), 'white')
else:
    replica = PIL.Image.new('RGB', (size*2 + d*3, size*1 + d*2), 'white')
## the orginal
replica.paste(PIL.Image.open(aligned_file).resize((size, size),PIL.Image.ANTIALIAS), (d, d))
## the effnet output
if pre_opt_dlatent is not None:
     replica.paste(PIL.Image.fromarray(img_generator.run(pre_opt_dlatent, None, None, flag = 0, noise_flag = 0, truncation_psi=0.7, truncation_layers=8)[0]).resize((size,size), PIL.Image.ANTIALIAS), (size + d*2, d))

for k, v in dlatent.items():
    
    ## save optimzation output
    if pre_opt_dlatent is not None:
        col = size*2 + d*3
    else:
        col = size*1 + d*2
    replica.paste(PIL.Image.fromarray(img_generator.run(v, None, None, flag = 0, noise_flag = 0, truncation_psi=0.7, truncation_layers=8)[0]).resize((size,size), PIL.Image.ANTIALIAS), (col, d))
    
#     ## each manipulation outputs only 2 images, arranged vertically. so it's 2*3 output
#     ## initialize canvas
#     for i in range(config.n_try):
#         canvas.append(PIL.Image.new('RGB', (size*3 + d*4, size*2 + d*3), 'white'))

#     ## age
#     img_np = img_generator.run(lt_mix.lt_direction_mix(src_lt=v, mix_dl_dir=None, direction='age', coeff=config.new_dot['old'], n = config.n_try), None, None, flag = 0)
#     for i in range(config.n_try):
#         canvas[i].paste(PIL.Image.fromarray(img_np[i]).resize((size, size),PIL.Image.ANTIALIAS), (d, d))
    
#     img_np = img_generator.run(lt_mix.lt_direction_mix(src_lt=v, mix_dl_dir=None, direction='age', coeff= config.new_dot['young'], n = config.n_try), None, None, flag = 0)
#     for i in range(config.n_try):
#         canvas[i].paste(PIL.Image.fromarray(img_np[i]).resize((size, size),PIL.Image.ANTIALIAS), (d, size + d*2))
    
#     ## smile
#     img_np = img_generator.run(lt_mix.lt_direction_mix(src_lt=v, mix_dl_dir=None, direction='smile', coeff=config.new_dot['cool'], n = config.n_try), None, None, flag = 0)
#     for i in range(config.n_try):
#         canvas[i].paste(PIL.Image.fromarray(img_np[i]).resize((size, size),PIL.Image.ANTIALIAS), (size + d*2, d))
         
#     img_np = img_generator.run(lt_mix.lt_direction_mix(src_lt=v, mix_dl_dir=None, direction='smile', coeff= config.new_dot['smile'], n = config.n_try), None, None, flag = 0)
#     for i in range(config.n_try):
#         canvas[i].paste(PIL.Image.fromarray(img_np[i]).resize((size, size),PIL.Image.ANTIALIAS), (size + d*2, size + d*2))
       
#     ## gender
#     img_np = img_generator.run(lt_mix.lt_direction_mix(src_lt=v, mix_dl_dir=None, direction='gender', coeff=config.new_dot['female'], n = config.n_try), None, None, flag = 0, noise_flag = True)
#     for i in range(config.n_try):
#         canvas[i].paste(PIL.Image.fromarray(img_np[i]).resize((size, size),PIL.Image.ANTIALIAS), (size*2 + d*3, d))
        
#     img_np = img_generator.run(lt_mix.lt_direction_mix(src_lt=v, mix_dl_dir=None, direction='gender', coeff=config.new_dot['male'], n = config.n_try), None, None, flag = 0, noise_flag = True)
#     for i in range(config.n_try):
#         canvas[i].paste(PIL.Image.fromarray(img_np[i]).resize((size, size),PIL.Image.ANTIALIAS), (size*2 + d*3, size + d*2))
    
    
    ## each manipulation outputs a series of 6 images, arranged horizontally. so it's 3*6 output
    ## initialize canvas
    size = 256
    row = 4
    step = 6
    
    for i in range(config.n_try):
        canvas.append(PIL.Image.new('RGB', (size*step + d*2, size*row + d*(row+1)), 'white'))

    ## age
    for col, coeff in enumerate(np.linspace(config.new_dot['young'], config.new_dot['old'], 6)):
        img_np = img_generator.run(lt_mix.lt_direction_mix(src_lt=v, mix_dl_dir=None, direction='age', coeff=coeff, n = config.n_try), None, None, flag = 0)
        for i in range(config.n_try):
            canvas[i].paste(PIL.Image.fromarray(img_np[i]).resize((size, size),PIL.Image.ANTIALIAS), (d + col * size, d))

    ## smile
    for col, coeff in enumerate(np.linspace(config.new_dot['cool'], config.new_dot['smile'], 6)):
        img_np = img_generator.run(lt_mix.lt_direction_mix(src_lt=v, mix_dl_dir=None, direction='smile', coeff=coeff, n = config.n_try), None, None, flag = 0)
        for i in range(config.n_try):
            canvas[i].paste(PIL.Image.fromarray(img_np[i]).resize((size, size),PIL.Image.ANTIALIAS), (d + col * size, size + d*2))
    
    ## gender
    for col, coeff in enumerate(np.linspace(config.new_dot['female'], config.new_dot['male'], 6)):
        img_np = img_generator.run(lt_mix.lt_direction_mix(src_lt=v, mix_dl_dir=None, direction='gender', coeff=coeff, n = config.n_try), None, None, flag = 0)
        for i in range(config.n_try):
            canvas[i].paste(PIL.Image.fromarray(img_np[i]).resize((size, size),PIL.Image.ANTIALIAS), (d + col * size, size*2 + d*3))

    ## pose
    for col, coeff in enumerate(np.linspace(config.new_dot['pose_l'], config.new_dot['pose_r'], 6)):
        img_np = img_generator.run(lt_mix.lt_direction_mix(src_lt=v, mix_dl_dir=None, direction='pose', coeff=coeff, n = config.n_try), None, None, flag = 0)
        for i in range(config.n_try):
            canvas[i].paste(PIL.Image.fromarray(img_np[i]).resize((size, size),PIL.Image.ANTIALIAS), (d + col * size, size*3 + d*4))


#file_suffix = str(datetime.datetime.now()).replace('-', '').replace(' ', '').replace(':', '').replace('.', '')
file_suffix = img_name

for i in range(config.n_try):
    if i == 0:
        canvas[i].save(os.path.join(args.output_path, 'canvas.png'))
    canvas[i].save(os.path.join(args.output_path, 'canvas_' + str(i) + '_' + file_suffix + '.png'))

replica.save(os.path.join(args.output_path, 'replica.png'))
replica.save(os.path.join(args.output_path, 'replica_' + file_suffix + '.png'))

end = time.time()
logger.info(f'total time used: {end - start:.2f} seconds')

   