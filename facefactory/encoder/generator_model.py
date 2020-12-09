import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial


def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))


def create_variable_for_generator(name, batch_size, dlatent_df):
    return tf.tile(tf.get_variable('learnable_dlatents_multi_style_loss',
                           shape=(batch_size, dlatent_df, 512), 
                           dtype='float32',
                           initializer=tf.initializers.random_normal()),
                   [1, int(18/dlatent_df), 1])


class Generator:
    def __init__(self, model, batch_size, randomize_noise=False, dlatent_df=18):
        self.batch_size = batch_size
        self.dlatent_df = dlatent_df
        
        self.initial_dlatents = np.zeros((self.batch_size, 18, 512))
        #self.initial_dlatents  = np.array([np.load('dlatent_avg.npy')]*self.batch_size)
        #self.initial_dlatents  = ref_lt
        
        ## has to specify custom_scope in order to keep tensor name fix, change dnnlib/tflib/network.py accordingly
        model.components.synthesis.run(self.initial_dlatents, randomize_noise=randomize_noise, minibatch_size=self.batch_size,
                             custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size, dlatent_df=dlatent_df),
                                       partial(create_stub, batch_size=batch_size)],
                             custom_scope = '/_vgg', 
                             structure='fixed')
        
        print([n.name for n in tf.get_default_graph().as_graph_def().node if '_Run/concat' in n.name])
        
        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()

        print([v for v in tf.global_variables() if 'learnable_dlatents_multi_style_loss' in v.name])
        self.dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents_multi_style_loss' in v.name)
                
        self.set_dlatents(self.initial_dlatents)
        
        self.generator_output = tf.reshape(self.graph.get_tensor_by_name('G_synthesis_1/_Run/_vgg/concat:0'), shape = [self.batch_size, 3, 1024, 1024])
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)

    def reset_dlatents(self):
        self.set_dlatents(self.initial_dlatents)

    def set_dlatents(self, dlatents):
        assert (dlatents.shape == (self.batch_size, 18, 512))
        self.sess.run(tf.assign(self.dlatent_variable, dlatents[:, :self.dlatent_df, :]))

    def get_dlatents(self):
        return self.sess.run(self.dlatent_variable)

    def generate_images(self, dlatents=None):
        if dlatents:
            self.set_dlatents(dlatents)
        return self.sess.run(self.generated_image_uint8)
