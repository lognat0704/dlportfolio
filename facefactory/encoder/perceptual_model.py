import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import keras.backend as K
import clr
import numpy as np
import gc

def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    #preprocessed_images = preprocess_input(loaded_images)
    return loaded_images


class PerceptualModel:
    def __init__(self, img_size, layer=9, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        print(K.tensorflow_backend._get_available_gpus())
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size

        self.perceptual_model = None
        self.loss = 0
        
    def build_perceptual_model(self, generated_image_tensor):
        
        def tf_custom_l1_loss(img1,img2):
            return tf.reduce_mean(tf.abs(img2-img1), axis=None)
        
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        generated_image = preprocess_input(tf.image.resize_images(generated_image_tensor, (self.img_size, self.img_size), method=1))
        
        ## add-on multi layers perceptual loss##
        self.ref_img_features = []
        self.features_weight = []
        
        self.perceptual_model = Model(vgg16.input, [vgg16.layers[1].output,vgg16.layers[2].output,vgg16.layers[8].output, vgg16.layers[12].output])    
        for i, generated_img_features in enumerate(self.perceptual_model(generated_image)):    
            self.ref_img_features.append(tf.get_variable('ref_img_features_%s'%i, shape=generated_img_features.shape, dtype='float32', initializer=tf.initializers.zeros()))
            self.features_weight.append(tf.get_variable('features_weight_%s'%i, shape=generated_img_features.shape, dtype='float32', initializer=tf.initializers.zeros()))
            self.sess.run([self.features_weight[i].initializer, self.features_weight[i].initializer])
            self.loss += 0.4 * tf_custom_l1_loss(self.features_weight[i] * self.ref_img_features[i], self.features_weight[i] * generated_img_features)/4
            

    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = preprocess_input(load_images(images_list, self.img_size))
        for i ,image_features in enumerate(self.perceptual_model.predict_on_batch(loaded_image)):
            weight_mask = np.ones(self.features_weight[i].shape)
            self.sess.run(tf.assign(self.features_weight[i], weight_mask)) 
            self.sess.run(tf.assign(self.ref_img_features[i], image_features))
            

    def optimize(self, vars_to_optimize, iterations=100, learning_rate=1.):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        
        ## vanilla SGD
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = clr.cyclic_learning_rate(global_step=global_step,
                                                                  learning_rate_min=0.005,
                                                                  max_lr=learning_rate,
                                                                  step_size=100,
                                                                  mode='triangular2'))
        
        
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize], global_step=global_step)
        
        ##clr.cyclic_learning_rate(global_step=global_step,learning_rate_min=0.12, max_lr=learning_rate, step_size=250., mode='exp_range')       
        
        ## Adam
        # optimizer = tf.traein.AdamOptimizer(learning_rate=learning_rate)
        # min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        # adam_vars = [v for v in tf.global_variables() if 'beta' in v.name or 'Adam' in v.name]
        # #print(adam_vars)
        # 
        # init_op = tf.variables_initializer(adam_vars)
        # self.sess.run([init_op])
        
        min_loss = 100
        min_loss_iter = 0
        
        for it in range(iterations):
            assign_op = global_step.assign(it+1)      
            self.sess.run(assign_op)
           
            _, loss = self.sess.run([min_op, self.loss])         
            if(min_loss - loss > 0.005):
                min_loss = loss
                min_loss_iter = it
            if(it - min_loss_iter >= 1500):
                print('early stop!\n')
                break
            yield loss

