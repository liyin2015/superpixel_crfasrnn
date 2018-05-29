"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
import high_dim_filter_loader
custom_module = high_dim_filter_loader.custom_module
import time


def _diagonal_initializer(shape):
    return np.eye(shape[0], shape[1], dtype=np.float32)


def _potts_model_initializer(shape):
    return -1 * _diagonal_initializer(shape)


class CrfRnnLayer(Layer):
    """ Implements the CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, bil_rate = 0.5, theta_alpha_seg = None, **kwargs): #add theta_alpha_seg
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_alpha_seg = theta_alpha_seg #to add sp-pairwise
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        self.spatial_norm_vals = None #to modularize
        self.bilateral_norm_vals = None
        self.bilateral_outs = []
        self.bil_rate = bil_rate #ratio of wegiths
        super(CrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights of the spatial kernel
        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_classes, self.num_classes),
                                                   initializer=_diagonal_initializer,
                                                   trainable=True)

        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_classes, self.num_classes),
                                                     initializer=_diagonal_initializer,
                                                     trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    initializer=_potts_model_initializer,
                                                    trainable=True)

        super(CrfRnnLayer, self).build(input_shape)
    def filtering_norming(self, imgs):
        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
        all_ones = np.ones((c, h, w), dtype=np.float32)
        # Prepare filter normalization coefficients, they are tensors
        self.spatial_norm_vals = custom_module.high_dim_filter(all_ones, imgs[0], bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        self.bilateral_norm_vals = [custom_module.high_dim_filter(all_ones, imgs[0], bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)] #add original image
        for i in range(1, len(imgs)):
            theta_alpha_seg = self.theta_alpha_seg if self.theta_alpha_seg is not None else self.theta_alpha
            self.bilateral_norm_vals.append(custom_module.high_dim_filter(all_ones, imgs[i], bilateral=True,
                                                            theta_alpha=theta_alpha_seg,
                                                            theta_beta=self.theta_beta)) # add segmented image
        
    def bilateral_filtering(self, softmax_out, imgs):
#         bilateral_outs = []
        self.bilateral_outs = []
        self.bilateral_outs.append(custom_module.high_dim_filter(softmax_out, imgs[0], bilateral=True,
                                                          theta_alpha=self.theta_alpha,
                                                          theta_beta=self.theta_beta))
        if len(imgs) > 1: #we have segmentations
            for i in range(1,  len(imgs)):
                theta_alpha_seg = self.theta_alpha_seg if self.theta_alpha_seg is not None else self.theta_alpha
                self.bilateral_outs.append(custom_module.high_dim_filter(softmax_out, imgs[i], bilateral=True,
                                                          theta_alpha=theta_alpha_seg,
                                                          theta_beta=self.theta_beta))

        self.bilateral_outs = [out / norm for (out, norm) in zip(self.bilateral_outs, self.bilateral_norm_vals)]


    def call(self, inputs):
        start_time = time.time()
        unaries = tf.transpose(inputs[0][0, :, :, :], perm=(2, 0, 1))
        rgb = tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1))
        segs = [tf.transpose(inputs[i][0, :, :, :], perm=(2, 0, 1)) for i in range(2,len(inputs))]

        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]

        # Prepare filter normalization coefficients, they are tensors
        self.filtering_norming([rgb]+segs)
        q_values = unaries

        for i in range(self.num_iterations):
            softmax_out = tf.nn.softmax(q_values, 0)

            # Spatial filtering
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                        theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / self.spatial_norm_vals

            # Bilateral filtering
            self.bilateral_filtering(softmax_out, [rgb]+segs)

            # Weighting filter outputs
            
            message_passing = tf.matmul(self.spatial_ker_weights,
                                         tf.reshape(spatial_out, (c, -1)))
            ratios = [1.0] + [self.bil_rate]*len(segs)
            message_passing += tf.add_n([tf.matmul(self.bilateral_ker_weights*ratios[i],
                                         tf.reshape(self.bilateral_outs[i], (c, -1))) for i in range(len(segs)+1)])

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w))
            q_values = unaries - pairwise
        elapsed_time = time.time() - start_time
        print(elapsed_time)

        return tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))
    


    def compute_output_shape(self, input_shape):
        return input_shape
