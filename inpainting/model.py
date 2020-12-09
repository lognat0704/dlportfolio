import urllib
from urllib.request import urlopen
import utils
import cv2
import numpy as np
import tensorflow as tf
from inpaint_model import InpaintCAModel
import base64


tools = utils.Tool()

class Image_inpainting:
    
    #def __init__(self):
    #    print('Image_inpainting is loaded')
        
    def json_image_checker(self, json_image):
        try:
            urllib.request.urlopen(json_image)
        except urllib.error.HTTPError as err:
            return False
        
        if tools.isBase64(json_image):
            print('Acceptable BASE64')
            jpg_recovered = base64.b64decode(json_image)
            f = open("image.png", "wb")
            f.write(jpg_recovered)
            f.close() 
            return True
        
        elif urlopen(json_image).info()['Content-type'].endswith(("jpg","jpeg")):
            resource = urlopen(json_image)
            output = open("image.png","wb")
            output.write(resource.read())
            output.close()
            return True
        
        elif urlopen(json_image).info()['Content-type'].endswith("png"):
            im = Image.open(BytesIO(requests.get(json_image).content))
            bg = Image.new("RGB", im.size, (255,255,255))
            bg.paste(im, (0,0), im)
            bg.save("image.png", quality=100)
            return True 
        
        else:
            return False
        
    def json_mask_checker(self, json_mask):
        try:
            urllib.request.urlopen(json_mask)
        except urllib.error.HTTPError as err:
            return False
        
        if tools.isBase64(json_mask):
            print('Acceptable BASE64')
            jpg_recovered = base64.b64decode(json_mask)
            f = open("image_mask.png", "wb")
            f.write(jpg_recovered)
            f.close() 
            return True
        
        elif urlopen(json_mask).info()['Content-type'].endswith(("jpg","jpeg")):
            resource = urlopen(json_mask)
            output = open("image_mask.png","wb")
            output.write(resource.read())
            output.close()
            return True
        
        elif urlopen(json_mask).info()['Content-type'].endswith("png"):
            im = Image.open(BytesIO(requests.get(json_mask).content))
            bg = Image.new("RGB", im.size, (255,255,255))
            bg.paste(im, (0,0), im)
            bg.save("image_mask", quality=100)
            return True 
        
        else:
            return False
        
    def fixing(self):
        retJSON = {}
        dict_key ='Img_link'
        model = InpaintCAModel()
        original_image = cv2.imread('image.png') ## input image without mask
        original_mask = cv2.imread('image_mask.png')
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
                var_value = tf.contrib.framework.load_variable('model', from_name)
                assign_ops.append(tf.assign(var, var_value))
                
            sess.run(assign_ops)
            print('Model loaded')
            result = sess.run(output)
            print('Run finished')
        
            
        out_img = cv2.resize(result[0][:h, :w, :], (0,0), fx = 1/ratio, fy = 1/ratio)
        ## after resizing out_img shape may not be idential to original_img due to rounding error. the folloing step fix it
        out_img = out_img[-original_image.shape[0]:,-original_image.shape[1]:]
        print(f'out_img shape: {out_img.shape}')
        assert out_img.shape == original_image.shape
        out_img[original_mask[:,:,0] <= 127.5, :] = original_image[original_mask[:,:,0] <= 127.5, :]
        cv2.imwrite('output.png', out_img)
        dict_item = base64.b64encode(open('output.png', 'rb').read())
        dict_item = dict_item.decode('utf-8')
        retJSON[dict_key] = dict_item
        
        return retJSON
        
