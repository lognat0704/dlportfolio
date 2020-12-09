""" common model for DCGAN """
import logging

import cv2
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.gan_ops import gan_wgan_loss, gradients_penalty
from neuralgym.ops.gan_ops import random_interpolates

from inpaint_ops import gen_conv, gen_deconv, dis_conv
from inpaint_ops import random_bbox, bbox2mask, local_patch
from inpaint_ops import spatial_discounting_mask
from inpaint_ops import resize_mask_like, contextual_attention

from inpaint_ops import gated_conv, gated_deconv, dis_sn_conv, gan_sn_pgan_loss, random_ff_mask

logger = logging.getLogger()

class InpaintCAModel(Model):
    def __init__(self):
        super().__init__('InpaintCAModel')

    def build_inpaint_net(self, x, mask, config=None, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x, ones_x*mask], axis=3)

        # two stage network
        # cnum = 32 ## deepfill v1
        cnum = 24 ## deepfill v2
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv, gated_conv, gated_deconv],
                          training=training, padding=padding):
            # stage1
            x = gated_conv(x, cnum, 5, 1, name='conv1') # gen_conv(x, cnum, 5, 1, name='conv1')
            x = gated_conv(x, 2*cnum, 3, 2, name='conv2_downsample') # gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='conv3') # gen_conv(x, 2*cnum, 3, 1, name='conv3')
            x = gated_conv(x, 4*cnum, 3, 2, name='conv4_downsample') # gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample')
            x = gated_conv(x, 4*cnum, 3, 1, name='conv5') # gen_conv(x, 4*cnum, 3, 1, name='conv5')
            x = gated_conv(x, 4*cnum, 3, 1, name='conv6') # gen_conv(x, 4*cnum, 3, 1, name='conv6')
            mask_s = resize_mask_like(mask, x)
            
            ## dilated gated conv
            x = gated_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous') # gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous') # gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous') # gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous') # gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
            
            x = gated_conv(x, 4*cnum, 3, 1, name='conv11') # gen_conv(x, 4*cnum, 3, 1, name='conv11')
            x = gated_conv(x, 4*cnum, 3, 1, name='conv12') # gen_conv(x, 4*cnum, 3, 1, name='conv12')
            x = gated_deconv(x, 2*cnum, name='conv13_upsample') # gen_deconv(x, 2*cnum, name='conv13_upsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='conv14') # gen_conv(x, 2*cnum, 3, 1, name='conv14')
            x = gated_deconv(x, cnum, name='conv15_upsample') # gen_deconv(x, cnum, name='conv15_upsample')
            x = gated_conv(x, cnum//2, 3, 1, name='conv16') # gen_conv(x, cnum//2, 3, 1, name='conv16')
            
            x = gen_conv(x, 3, 3, 1, activation=tf.tanh, name='conv17') ## https://github.com/JiahuiYu/generative_inpainting/issues/182
            x = tf.clip_by_value(x, -1., 1.)
            x_stage1 = x
            # return x_stage1, None, None

            # stage2, paste result as input
            # x = tf.stop_gradient(x)
            x = x*mask + xin*(1.-mask)
            x.set_shape(xin.get_shape().as_list())
            # conv branch
            xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
            x = gated_conv(xnow, cnum, 5, 1, name='xconv1') # gen_conv(xnow, cnum, 5, 1, name='xconv1')
            x = gated_conv(x, cnum, 3, 2, name='xconv2_downsample') # gen_conv(x, cnum, 3, 2, name='xconv2_downsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='xconv3') # gen_conv(x, 2*cnum, 3, 1, name='xconv3')
            x = gated_conv(x, 2*cnum, 3, 2, name='xconv4_downsample') # gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')
            x = gated_conv(x, 4*cnum, 3, 1, name='xconv5') # gen_conv(x, 4*cnum, 3, 1, name='xconv5')
            x = gated_conv(x, 4*cnum, 3, 1, name='xconv6') # gen_conv(x, 4*cnum, 3, 1, name='xconv6')
            x = gated_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous') # gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous') # gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous') # gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous') # gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')
            x_hallu = x
            # attention branch
            x = gated_conv(xnow, cnum, 5, 1, name='pmconv1') # gen_conv(xnow, cnum, 5, 1, name='pmconv1')
            x = gated_conv(x, cnum, 3, 2, name='pmconv2_downsample') # gen_conv(x, cnum, 3, 2, name='pmconv2_downsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='pmconv3') # gen_conv(x, 2*cnum, 3, 1, name='pmconv3')
            x = gated_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample') # gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')
            x = gated_conv(x, 4*cnum, 3, 1, name='pmconv5') # gen_conv(x, 4*cnum, 3, 1, name='pmconv5')
            x = gated_conv(x, 4*cnum, 3, 1, name='pmconv6',activation=tf.nn.relu) # gen_conv(x, 4*cnum, 3, 1, name='pmconv6',activation=tf.nn.relu)
            x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
            x = gated_conv(x, 4*cnum, 3, 1, name='pmconv9') # gen_conv(x, 4*cnum, 3, 1, name='pmconv9')
            x = gated_conv(x, 4*cnum, 3, 1, name='pmconv10') # gen_conv(x, 4*cnum, 3, 1, name='pmconv10')
            pm = x
            x = tf.concat([x_hallu, pm], axis=3)

            x = gated_conv(x, 4*cnum, 3, 1, name='allconv11') # gen_conv(x, 4*cnum, 3, 1, name='allconv11')
            x = gated_conv(x, 4*cnum, 3, 1, name='allconv12') # gen_conv(x, 4*cnum, 3, 1, name='allconv12')
            x = gated_deconv(x, 2*cnum, name='allconv13_upsample') # gen_deconv(x, 2*cnum, name='allconv13_upsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='allconv14') # gen_conv(x, 2*cnum, 3, 1, name='allconv14')
            x = gated_deconv(x, cnum, name='allconv15_upsample') # gen_deconv(x, cnum, name='allconv15_upsample')
            x = gated_conv(x, cnum//2, 3, 1, name='allconv16') # gen_conv(x, cnum//2, 3, 1, name='allconv16')
            
            x = gen_conv(x, 3, 3, 1, activation=tf.tanh, name='allconv17') ## https://github.com/JiahuiYu/generative_inpainting/issues/182
            x_stage2 = tf.clip_by_value(x, -1., 1.)
        return x_stage1, x_stage2, offset_flow

    def build_wgan_local_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator_local', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)
            x = dis_conv(x, cnum*2, name='conv2', training=training)
            x = dis_conv(x, cnum*4, name='conv3', training=training)
            x = dis_conv(x, cnum*8, name='conv4', training=training)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_global_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator_global', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)
            x = dis_conv(x, cnum*2, name='conv2', training=training)
            x = dis_conv(x, cnum*4, name='conv3', training=training)
            x = dis_conv(x, cnum*4, name='conv4', training=training)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_discriminator(self, batch_local, batch_global,
                                 reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.build_wgan_local_discriminator(
                batch_local, reuse=reuse, training=training)
            dglobal = self.build_wgan_global_discriminator(
                batch_global, reuse=reuse, training=training)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global
        
    ## currently last conv use leaky_relu as activation. is it necessary?
    def build_SNGAN_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator',reuse=reuse):
            cnum=64
            
            x = dis_sn_conv(x,cnum,name='conv1',training=training)
            x = dis_sn_conv(x,2*cnum,name='conv2',training=training)
            x = dis_sn_conv(x,4*cnum,name='conv3',training=training)
            x = dis_sn_conv(x,4*cnum,name='conv4',training=training)
            x = dis_sn_conv(x,4*cnum,name='conv5',training=training)
            x = dis_sn_conv(x,4*cnum,name='conv6',training=training)
            
        return x


    def build_graph_with_losses(self, batch_data, mask, config, training=True, summary=False, reuse=False):
        batch_pos = batch_data / 127.5 - 1.## normalize value to  [-1, 1]
        
        # generate mask, 1 represents masked point
        #bbox = random_bbox(config)
        #mask = bbox2mask(bbox, config, name='mask_c')
        
        ## generate free-form mask
        # mask = random_ff_mask(config)
        
        #mask = mask[:,:,:,0:1]/255. ## because data_datapipe uses cv2.imread which turns grayscale image to 3 channels
        mask = tf.cast(mask[:,:,:,0:1] > 127.5, tf.float32) ## make mask 0 or 1, for unknown reason, mask image (256*256) contains values other than 0 or 255
                
        ## set value inside mask to 0
        batch_incomplete = batch_pos*(1.-mask)
        
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=reuse, training=training,
            padding=config.PADDING)
        batch_predicted = x2
        #logger.info('Set batch_predicted to x2.')
        losses = {}
        
        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        
        #l1_alpha = config.COARSE_L1_ALPHA
        losses['ae_loss'] = tf.reduce_mean(tf.abs(batch_pos - x1)) + 2*tf.reduce_mean(tf.abs(batch_pos - x2))
        
        if summary:
            viz_img = [batch_pos, batch_incomplete, x1, x2, batch_complete]
            if offset_flow is not None:
                viz_img.append(
                    resize(offset_flow, scale=4,
                           func=tf.image.resize_nearest_neighbor))
            images_summary(
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_complete', config.VIZ_MAX_OUT)

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        ones_gan_batch = tf.ones_like(batch_pos_neg)[:, :, :, 0:1]
        gan_batch = tf.concat([batch_pos_neg, ones_gan_batch, mask * ones_gan_batch], axis = 3)
        
        # SN-PatchGAN:
        pos_neg = self.build_SNGAN_discriminator(gan_batch, reuse=reuse, training=training)
        pos, neg = tf.split(pos_neg, 2)
           
        ## gan loss
        # g_loss, d_loss = gan_hinge_loss(pos, neg) # https://github.com/JiahuiYu/neuralgym/blob/dev/neuralgym/ops/gan_ops.py
        g_loss_global, d_loss_global = gan_sn_pgan_loss(pos, neg, summary = summary)
        losses['g_loss'] = g_loss_global
        losses['d_loss'] = d_loss_global
            
        if summary and not config.PRETRAIN_COARSE_NETWORK:
            scalar_summary('convergence/d_loss', losses['d_loss'])
            scalar_summary('convergence/g_loss', losses['g_loss'])
 
        if summary and not config.PRETRAIN_COARSE_NETWORK:
            # summary the magnitude of gradients from different losses w.r.t. predicted image
            #gradients_summary(losses['g_loss'], batch_predicted, name='dev/g_loss_to_predicted')
            gradients_summary(losses['g_loss'], x1, name='dev/g_loss_to_x1')
            gradients_summary(losses['g_loss'], x2, name='dev/g_loss_to_x2')
            #gradients_summary(losses['d_loss'], batch_predicted, name='dev/d_loss_to_predicted')
            gradients_summary(losses['d_loss'], x1, name='dev/d_loss_to_x1')
            gradients_summary(losses['d_loss'], x2, name='dev/d_loss_to_x2')
            gradients_summary(losses['ae_loss'], x1, name='dev/ae_loss_to_x1')
            gradients_summary(losses['ae_loss'], x2, name='dev/ae_loss_to_x2')
        
        losses['g_loss'] = losses['ae_loss'] + 0.25 * losses['g_loss'] ## ref: https://github.com/JiahuiYu/generative_inpainting/issues/267
                
        if summary and not config.PRETRAIN_COARSE_NETWORK:
            scalar_summary('losses/ae_loss', losses['ae_loss'])
            scalar_summary('losses/g_ae_loss', losses['g_loss'])
        
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        
        ## inpaint-net use tf.layer.conv2d, which doesn't explicitly define weights. 
        ## need to run this line to include those weights in tensorboard
        # for var in tf.trainable_variables():
        for var in g_vars:
            tf.summary.histogram(var.name, var)
        
        return g_vars, d_vars, losses

    def build_infer_graph(self, batch_data, config, bbox=None, name='val'):
        """
        """
        config.MAX_DELTA_HEIGHT = 0
        config.MAX_DELTA_WIDTH = 0
        #if bbox is None:
        #    bbox = random_bbox(config)
        #mask = bbox2mask(bbox, config, name=name+'mask_c')
        if bbox is None:
            mask = random_ff_mask(config, name = name + 'ff_mask')
            
        batch_pos = batch_data / 127.5 - 1.
        edges = None
        batch_incomplete = batch_pos*(1.-mask)
        # inpaint
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=True,
            training=False, padding=config.PADDING)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        # apply mask and reconstruct
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # global image visualization
        viz_img = [batch_pos, batch_incomplete, batch_complete]
        if offset_flow is not None:
            viz_img.append(
                resize(offset_flow, scale=4,
                       func=tf.image.resize_nearest_neighbor))
        images_summary(
            tf.concat(viz_img, axis=2),
            name+'_raw_incomplete_complete', config.VIZ_MAX_OUT)
        return batch_complete

    def build_static_infer_graph(self, batch_data, config, name):
        """
        """
        # generate mask, 1 represents masked point
        #bbox = (tf.constant(config.HEIGHT//2), tf.constant(config.WIDTH//2),
        #        tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
        bbox = None
        return self.build_infer_graph(batch_data, config, bbox, name)


    def build_server_graph(self, batch_data, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis=3) ## author wrote axis = 2, should it be axis = 3?
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32) ## make mask 0 or 1
        
        batch_pos = batch_raw / 127.5 - 1.## normalize input img [-1,1]
        batch_incomplete = batch_pos * (1. - masks) ## make masked area all 0s
        
        # inpaint
        x1, x2, flow = self.build_inpaint_net(
            batch_incomplete, masks, reuse=reuse, training=is_training,
            config=None)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
        return batch_complete
