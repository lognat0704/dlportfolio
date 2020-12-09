import os
import tempfile
#from keras.utils import get_file
import PIL.Image
import logging
import numpy as np
import dlib
import scipy.ndimage



logger = logging.getLogger()

# LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

class Align_img:
    def __init__(self, landmarks_model_path):
        #print('Align_img_loaded!')
        #self.landmarks_detector = LandmarksDetector(landmarks_model_path)
        
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.landmarks_detector = dlib.shape_predictor(landmarks_model_path)
        
        logger.info('Align_img_loaded!')
    
    def get_landmarks(self, image):
        
        ## for unknown reason, dlib.load_rgb_image can't open jpg files. use Image.open instead
        #img = dlib.load_rgb_image(image)
        
        img = np.array(PIL.Image.open(image))
        logger.info(f'input image size: {img.shape}')
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.landmarks_detector(img, detection).parts()]
            yield face_landmarks
               
    def landmarksDetector(self, raw_img, output_img_path=None, face_id=None):        
        """
        return: a list of aligned face image paths
        """
        
        # RAW_IMAGES_DIR = 'raw_images'
                
        ## for image other than PNG, convert it to PNG and save it to temp folder 
        ## why?? is conversion to PNG really necessary?
#         if os.path.splitext(os.path.basename(raw_img))[1] !='.png':
#             im = Image.open(raw_img)
#             raw_img = os.path.join(tempfile.gettempdir(),os.path.splitext(os.path.basename(raw_img))[0])+'.png'
#             im.save(raw_img,'png',quality = 100)
            
        raw_img_name = os.path.basename(raw_img)
        aligned_face_path_list = []
        
#         if self.landmarks_detector.get_landmarks(raw_img) is None:
#             logger.info('no face detected!')
#             return None
        
        face_area=[]
        img_name = os.path.splitext(os.path.basename(raw_img))[0]
        
        if output_img_path is None: ## if no output img path specified, use system temp folder
            output_img_path = tempfile.gettempdir()
    
        
        #for i, face_landmarks in enumerate(self.landmarks_detector.get_landmarks(raw_img), start=1):
        for i, face_landmarks in enumerate(self.get_landmarks(raw_img)):
            #aligned_face_path = os.path.join(output_img_path, 'aligned_' + str(i) + '_' + os.path.basename(raw_img))   
            aligned_face_path = os.path.join(output_img_path, img_name + '_aligned_' + str(i) + '.jpg')   
            aligned_face_path_list.append(aligned_face_path)
            face_area.append(image_align(raw_img, aligned_face_path, face_landmarks))
        
        if len(face_area) == 0:
            logger.warning('no face detected!')
            return None
        
        if face_id is None: ## if face is chosen automatically
            if len(face_area) > 1: ## multiple faces, choose the biggest one
                face_id = face_area.index(max(face_area))
            else: ## only one face
                face_id = 0
            
        logger.warning(f'{len(face_area)} face(s) detected and #{face_id} chosen. please check aligned images if necessary')
        
        return aligned_face_path_list[face_id]
    


def image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True):
    
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    lm = np.array(face_landmarks)
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise
    
    face_h = np.max(lm[:,0]) - np.min(lm[:,0])
    face_w = np.max(lm[:,1]) - np.min(lm[:,1])
    
    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle, qsize is the length of one side of the oriented crop rectangle, determined by length(eye_to_eye) and length(eye_to_mouth), quad is np array with four points of the oriented crop rectangle
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
        
    img = PIL.Image.open(src_file)
        
    import math
    #output_size = 2**np.clip(math.ceil(math.log2(min(face_w, face_h))), 8, 10)
    output_size = 1024
    transform_size = output_size * 4
       
    logger.info(f'cropped face size: {face_h} * {face_w}, aligned face image size: {output_size} * {output_size}')
    
    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        print(shrink)
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    ## border helps to create the crop rectangle, which contains the oriented crop rectangle. it is the distance between the oriented crop rectangle points and crop rectangle edges
    border = max(int(np.rint(qsize * 0.1)), 3) ## at least 10% of qsize
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2] ## coordinates of the oriented crop rectangle after cropping

    # Pad.
    ## if the oriented crop rectangle,enlarged by border, goes beyond actual crop rectangle, pad is added
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3))) ## pad is at least qsize * 0.3
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2] ## coordinates of the oriented crop rectangle after cropping

    # Transform.
    ## transform the oriented crop rectangle (quadrilateral) to a normal rectangle and only keep data inside the oriented crop rectangle
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    img.save(dst_file, 'PNG')
    
    return face_w * face_h
