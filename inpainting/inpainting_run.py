import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='image.png', type=str,
                    help='The filename of the original image with or without the mask.')
parser.add_argument('--mask', default='image_mask.png', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='model', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    #ng.get_gpus(1)
    args = parser.parse_args()

    model = InpaintCAModel()
    original_image = cv2.imread(args.image) ## input image without mask
    original_mask = cv2.imread(args.mask)
    mask_mean = np.mean(original_mask[:,:,0] > 127.5) * 100
    print(f'mask mean: {mask_mean:.3}%. if this number > 20%, quality may deteriorate')
    print(f'original input image shape: {original_image.shape}, original mask shape: {original_mask.shape}')
    
    h, w, _ = original_image.shape
    
    ## step 1 - resize to 256 * 256 while keeping the aspect ratio, pad 0 on bottom or right accordingly
    size = 256
    ratio = min(size/h, size/w)
    image = cv2.resize(original_image, (0, 0), fx = ratio, fy = ratio)
    mask = cv2.resize(original_mask, (0, 0),  fx = ratio, fy = ratio)
    h, w, _ = image.shape
    image = np.pad(image,((0,max(0, size-h)),(0,max(0, size-w)),(0,0)), 'constant', constant_values = 0) 
    mask = np.pad(mask, ((0,max(0, size-h)),(0,max(0, size-w)),(0,0)), 'constant', constant_values = 0) 
    
    assert image.shape == mask.shape ## so mask is 3 channel
    assert (mask[:,:,0] == mask[:,:,1]).all() & (mask[:,:,0] == mask[:,:,2]).all()
    
    
    input_image = np.concatenate([np.expand_dims(image, 0), np.expand_dims(mask, 0)], axis=3) 

    ## step 2 - run the model on the 256 * 256 image and mask
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image) ## input image is normalized and masked inside this function
        output = (output + 1.) * 127.5
        #output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded')
        result = sess.run(output)
        
    ## step 3 - post processing, drop padded area, resize back to original size and only fullfill the masked area with predicted area
    out_img = cv2.resize(result[0][:h, :w, :], (0,0), fx = 1/ratio, fy = 1/ratio)
    ## after resizing out_img shape may not be idential to original_img due to rounding error. the folloing step fix it
    out_img = out_img[-original_image.shape[0]:,-original_image.shape[1]:]
    print(f'out_img shape: {out_img.shape}')
    assert out_img.shape == original_image.shape
    out_img[original_mask[:,:,0] <= 127.5, :] = original_image[original_mask[:,:,0] <= 127.5, :]
    cv2.imwrite(args.output, out_img)
    print('Inpainting is done')

