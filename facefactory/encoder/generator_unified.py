import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial
#import tensorflow.keras.models
#from keras.preprocessing import image
#from keras.models import Model
#from keras.applications.vgg16 import VGG16, preprocess_input
#import clr
#import gc 


def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))


def create_variable_for_generator(name, batch_size, dlatent_df):
    return tf.tile(tf.get_variable('learnable_dlatents',
                           shape=(batch_size, dlatent_df, 512), 
                           dtype='float32',
                           initializer=tf.initializers.random_normal()),
                   [1, int(18/dlatent_df), 1])


class Generator:
    def __init__(self, model, batch_size, randomize_noise=False, dlatent_df=18):
        self.batch_size = batch_size
        self.dlatent_df = dlatent_df
        
        self.initial_dlatents = np.zeros((self.batch_size, 18, 512))
        
        ## has to specify custom_scope in order to keep tensor name fix, change dnnlib/tflib/network.py accordingly
        model.components.synthesis.run(self.initial_dlatents, randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                             custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size, dlatent_df=dlatent_df),
                                       partial(create_stub, batch_size=batch_size)],
                             custom_scope = '/_opt', 
                             structure='fixed')
        
        # print([n.name for n in tf.get_default_graph().as_graph_def().node if '_Run/concat' in n.name])
        
        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()

        self.dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents' in v.name)
        self.set_dlatents(self.initial_dlatents)
        
        try:
            self.generator_output = tf.reshape(self.graph.get_tensor_by_name('G_synthesis_1/_Run/_opt/concat:0'), shape = [self.batch_size, 3, 1024, 1024])
        except KeyError:
            # If we loaded only Gs and didn't load G or D, then scope "G_synthesis_1" won't exist in the graph.
            self.generator_output = tf.reshape(self.graph.get_tensor_by_name('G_synthesis/_Run/_opt/concat:0'), shape = [self.batch_size, 3, 1024, 1024])
        ## for unknown reason self.generator_output mighe be <-1 or >1, but convert_images_to_uint8 assumes range [-1, 1]
        ## clip it
        self.generator_output = tf.clip_by_value(self.generator_output, -0.5/127.5-1, -0.5/127.5+1)
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)
        
        # Implement stochastic clipping similar to what is described in https://arxiv.org/abs/1702.04782
        # (Slightly different in that the latent space is normal gaussian here and was uniform in [-1, 1] in that paper,
        # so we clip any vector components outside of [-2, 2]. It seems fine, but I haven't done an ablation check.)
        clipping_mask = tf.logical_or(self.dlatent_variable > 2.0, self.dlatent_variable < -2.0)
        clipped_values = tf.where(clipping_mask, tf.random_normal(shape=self.dlatent_variable.shape), self.dlatent_variable)
        self.stochastic_clip_op = tf.assign(self.dlatent_variable, clipped_values)
        


    def reset_dlatents(self):
        self.set_dlatents(self.initial_dlatents)

    def set_dlatents(self, dlatents):
        assert (dlatents.shape == (self.batch_size, 18, 512))
        self.sess.run(tf.assign(self.dlatent_variable, dlatents[:, :self.dlatent_df, :]))

    def get_dlatents(self):
        return self.sess.run(self.dlatent_variable)
    
    def stochastic_clip_dlatents(self):
        self.sess.run(self.stochastic_clip_op)

#     def generate_images(self, dlatents=None):
#         if dlatents:
#             self.set_dlatents(dlatents)
#         return self.sess.run(self.generated_image_uint8)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    #preprocessed_images = preprocess_input(loaded_images)
    return loaded_images


class PerceptualVggModel:
    def __init__(self, img_size, layer=9, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size
        
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(vgg16.input, [vgg16.layers[1].output,vgg16.layers[2].output,vgg16.layers[8].output, vgg16.layers[12].output])   

        self.loss = 0
        
    def build_perceptual_model(self, generated_image_tensor):
        
        def tf_custom_l1_loss(img1,img2):
            return tf.reduce_mean(tf.abs(img2-img1), axis=None)
        
        
        generated_image = preprocess_input(tf.image.resize_images(generated_image_tensor, (self.img_size, self.img_size), method=1))
        
        ## add-on multi layers perceptual loss##
        self.ref_img_features = []
        self.features_weight = []
        
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




# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
        loaded_images = np.vstack(loaded_images)
        
        # Reimplement tflib.convert_images_from_uint8 in numpy (with bugfix):
        preprocessed_images = np.copy(loaded_images).astype(dtype=np.float32)
        preprocessed_images = np.transpose(preprocessed_images, axes=(0, 3, 1, 2))
        drange = [-1,1]
        preprocessed_images = (preprocessed_images - drange[0]) * ((drange[1] - drange[0]) / 255) + drange[0]
        # ("+ drange[0]" at the end is a fix of a bug in tflib.convert_images_from_uint8())
        
        # NHWC --> NCHW:
        # preprocessed_images = tflib.convert_images_from_uint8(loaded_images, nhwc_to_nchw=True)
        # preprocessed_images = np.transpose(np.copy(loaded_images), axes=(0, 3, 1, 2))
        return loaded_images, preprocessed_images

class PerceptualDiscriminatorModel:
    def __init__(self, img_size, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        self.img_size = img_size
        self.batch_size = batch_size
        
        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None
        
    def build_perceptual_model(self, discriminator_network, generator_output_tensor, generated_image_tensor, vars_to_optimize,
                               initial_learning_rate=0.005, learning_rate_decay_steps=70, learning_rate_decay_rate=0.66):
        
        def generated_image_tensor_fn(name):
            return generator_output_tensor
        
        discriminator_network.run(
            np.zeros((self.batch_size, 3, 1024, 1024)), None,
            minibatch_size=self.batch_size,
            custom_inputs=[generated_image_tensor_fn, partial(create_stub, batch_size=self.batch_size)],
            structure='fixed')
        
        self.graph = tf.get_default_graph()
        
        # Learning rate
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        incremented_global_step = tf.assign_add(global_step, 1)
        self._reset_global_step = tf.assign(global_step, 0)
        self.learning_rate = tf.train.exponential_decay(initial_learning_rate, incremented_global_step,
                                        learning_rate_decay_steps, learning_rate_decay_rate, staircase=True)
        
        self.sess.run([self._reset_global_step])
        
        self.discriminator_input = self.graph.get_tensor_by_name("D/_Run/D/images_in:0") # (?, 3, 1024, 1024)
        
        # Pull out a tensor from the discriminator net at each level of detail, including the raw image in:
        tensor_name_list = [
            "D/_Run/D/images_in:0",								# (1, 3, 1024, 1024)
            "D/_Run/D/1024x1024/Conv0/LeakyReLU/IdentityN:0",	# (1, 16, 1024, 1024)
            "D/_Run/D/512x512/Conv0/LeakyReLU/IdentityN:0",		# (1, 32, 512, 512)
            "D/_Run/D/256x256/Conv0/LeakyReLU/IdentityN:0",		# (1, 64, 256, 256)
            "D/_Run/D/128x128/Conv0/LeakyReLU/IdentityN:0",		# (1, 128, 128, 128)
            "D/_Run/D/64x64/Conv0/LeakyReLU/IdentityN:0",		# (1, 256, 64, 64)
            "D/_Run/D/32x32/Conv0/LeakyReLU/IdentityN:0",		# (1, 512, 32, 32)
            "D/_Run/D/16x16/Conv0/LeakyReLU/IdentityN:0",		# (1, 512, 16, 16)
            "D/_Run/D/8x8/Conv0/LeakyReLU/IdentityN:0",			# (1, 512, 8, 8)
            "D/_Run/D/4x4/Conv/LeakyReLU/IdentityN:0",			# (1, 512, 4, 4)
            "D/_Run/D/4x4/Dense0/LeakyReLU/IdentityN:0",		# (1, 512) 
        ]
            
        # Just mash them all together, unweighted, into one n-dimensional vector.
        # (I spent hours picking and choosing combinations of layers, doing weighted averages to accommodate different layers being different sizes, 
        # even sinusoidally shuffling the weights during training while lerping toward even weighting... none of it worked as well as this.)
        tensors = [tf.reshape(self.graph.get_tensor_by_name(t), shape=[-1]) for t in tensor_name_list]
        self.discriminator_output = tf.concat(tensors, axis=0)
            
        # Image
        generated_image = tf.image.resize_images(generated_image_tensor, (self.img_size, self.img_size), method=1)
        self.reference_image = tf.get_variable('ref_img', shape=generated_image.eval().shape, dtype='float32', initializer=tf.initializers.zeros())
        self._assign_reference_image_ph = tf.placeholder(tf.float32, name="assign_ref_img_ph")
        self._assign_reference_image = tf.assign(self.reference_image, self._assign_reference_image_ph)
            
        # Perceptual image features
        generated_img_features = self.discriminator_output
        self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.eval().shape,
                                   dtype='float32', initializer=tf.initializers.zeros())
        self._assign_reference_img_feat_ph = tf.placeholder(tf.float32, name="assign_ref_img_feat_ph")
        self._assign_reference_img_feat = tf.assign(self.ref_img_features, self._assign_reference_img_feat_ph)
            
        # Feature weights
        self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.eval().shape,
                                  dtype='float32', initializer=tf.initializers.zeros())
            
        self._assign_features_weight_ph = tf.placeholder(tf.float32, name="assign_features_weight_ph")
        self._assign_features_weight = tf.assign(self.features_weight, self._assign_features_weight_ph)
        self.sess.run([self.features_weight.initializer])
            
        # Loss
        self.loss = tf.losses.mean_squared_error(self.features_weight * self.ref_img_features, self.features_weight * generated_img_features)
            
        # Also report L2 loss even though we don't optimize based on this op 
        # (though we do include the raw image tensor in the tensors that we pulled from the discriminator).
        # Pixel values are [0, 255] but feature values are ~[-1, 1], so divide by 128^2:
            
        self.l2_loss = tf.losses.mean_squared_error(self.reference_image, generated_image) / 128**2
            
        # Optimizer
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        self._init_optimizer_vars = tf.variables_initializer(optimizer.variables())
    
    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image, preprocessed_images = load_images(images_list, self.img_size)
        image_features = self.sess.run([self.discriminator_output], {self.discriminator_input:preprocessed_images})
        image_features = image_features[0]
            
        # in case if number of images less than actual batch size, can be optimized further
        weight_mask = np.ones(self.features_weight.shape)
        if len(images_list) != self.batch_size:
            raise NotImplementedError("We don't support image lists not divisible by batch size.")
            features_space = list(self.features_weight.shape[1:])
            existing_features_shape = [len(images_list)] + features_space
            empty_features_shape = [self.batch_size - len(images_list)] + features_space
            
            existing_examples = np.ones(shape=existing_features_shape)
            empty_examples = np.zeros(shape=empty_features_shape)
            weight_mask = np.vstack([existing_examples, empty_examples])
            image_features = np.vstack([image_features, np.zeros(empty_features_shape)])
        
        self.sess.run([self._assign_features_weight], {self._assign_features_weight_ph: weight_mask})
        self.sess.run([self._assign_reference_img_feat], {self._assign_reference_img_feat_ph: image_features})
        self.sess.run([self._assign_reference_image], {self._assign_reference_image_ph: loaded_image})
     
    def optimize(self, iterations):
        self.sess.run([self._init_optimizer_vars, self._reset_global_step])
        fetch_ops = [self.train_op, self.loss, self.l2_loss, self.learning_rate]
        
        for _ in range(iterations):
            _, loss, l2_loss, lr = self.sess.run(fetch_ops)
            yield {"loss":loss, "l2_loss":l2_loss, "lr": lr}

 
 
            

     