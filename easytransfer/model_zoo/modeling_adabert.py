# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf


EP = np.asarray(
    [0.09987373, 0.10367607, 0.10762545, 0.09987373, 0.10367607, 0.10762545, 0.09442003, 0.09442003, 0.09440472,
     0.09440472], dtype=np.float32)


def get_gumbel(shape, eps=1e-8):
    """Generate samples from a gumbel distribution."""

    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def get_logits(z, eps=1e-8):
    shifted_z = z - tf.reduce_max(z, axis=-1, keepdims=True)
    noralizer = tf.log(tf.reduce_sum(tf.exp(shifted_z)) + eps)
    return shifted_z - noralizer


class AdaBERTStudent(object):
    """A TensorFlow-based implementation of AdaBERT
    Attributes:
        global_step (op): calculate the global step.
        loss (op): calculate the loss
        update (op): update the model as well as the arch.
        acc (op): calculate the classification accuracy.
    """

    def __init__(self,
                 inputs,
                 is_training,
                 vocab_size,
                 is_pair_task,
                 num_classes,
                 Kmax=8,
                 num_intermediates=3,
                 emb_size=128,
                 seq_len=8,
                 keep_prob=0.9,
                 temp_decay_steps=18000,
                 model_opt_lr=5e-4,
                 arch_opt_lr=1e-4,
                 model_l2_reg=3e-4,
                 arch_l2_reg=1e-3,
                 loss_gamma=0.8,
                 loss_beta=4.0,
                 pretrained_word_embeddings=None,
                 pretrained_pos_embeddings=None,
                 given_arch=None):
        """The constructor of AdaBERTStudent class
        Arguments:
            inputs (list): a list of tensors corresponding to the fields ["ids", "mask", "seg_ids", "prob_logits", "labels"]
            is_training (Tensor): bool, ()
            vocab_size (int): the cardinality of vocabulary.
            is_pair_task (bool): single sentence or paired sentences task
            num_classes (int): the number of categories.
            Kmax (int): the maximal number of cells.
            num_intermediates (int): the number of intermediate nodes in
                in each cell.
            emb_size (int): the dimension of embeddings.
            seq_len (int): the length of each word sequence.
            keep_prob (float): the probability of no drop out.
            temp_decay_steps (int): the number of steps for annealing temperature.
            model_opt_lr (float): the learning rate of the optimizer for
                model parameters.
            arch_opt_lr (float): the learning rate of the optimizer for arch
                parameters.
            model_l2_reg (float): the weights of l2 regularization for
                model parameters.
            arch_l2_reg (float): the weights of l2 regularization for arch
                parameters.
            loss_gamma (float): the weight for balancing L_CE and L_KD
            loss_beta (float): the weight of L_E
            pretrained_word_embeddings (np.array): the pretrained word embeddings with shape (vocab_size, 768)
            pretrained_pos_embeddings (np.array): the pretrained position embeddings with shape (512, 768)
            given_arch (dict): describe the neural architecture.
        """
        self.Kmax = Kmax
        self.num_intermediates = num_intermediates
        self.emb_size = emb_size
        assert seq_len <= 512, "Sequence length should be equal or less than 512, but given {}".format(seq_len)
        self.seq_len = seq_len
        self.keep_prob = keep_prob
        self.temp_decay_steps = temp_decay_steps
        self.card_of_o = len(EP)
        self.vocab_size = vocab_size
        self.is_pair_task = is_pair_task
        self.num_classes = num_classes
        self.model_opt_lr = model_opt_lr
        self.arch_opt_lr = arch_opt_lr
        self.model_l2_reg = model_l2_reg
        self.arch_l2_reg = arch_l2_reg
        self.loss_gamma = loss_gamma
        self.loss_beta = loss_beta
        self.pretrained_word_embeddings = pretrained_word_embeddings
        self.pretrained_pos_embeddings = pretrained_pos_embeddings
        self.given_arch = given_arch

        self._build_graph(inputs, is_training)

    def _build_graph(self, inputs, is_training):
        """Create the computation graph
        """

        # input tensors
        word_ids = inputs[0]
        masks = inputs[1]
        seg_ids = inputs[2]
        if len(inputs) == 5:
            prob_logits = inputs[3]
            labels = inputs[4]
        else:
            labels = inputs[3]

        self.global_step = tf.train.get_or_create_global_step()
        if self.given_arch is None:
            # this is the search procedure
            # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            decay_rate = np.e ** ((100 / self.temp_decay_steps) * np.log(0.2))
            tf.logging.info("########## decay_rate={} ##########".format(decay_rate))
            temperature = tf.train.exponential_decay(learning_rate=1.0,
                                                     global_step=self.global_step,
                                                     decay_steps=100,
                                                     decay_rate=decay_rate,
                                                     staircase=True)

        # Determine the batch size dynamically
        dy_batch_size = tf.shape(labels)[0]
        if self.is_pair_task:
            num_words = tf.reduce_sum(masks, -1)
            sec_sent_words = tf.reduce_sum(seg_ids, -1)
            # (bs,)
            first_sent_words = num_words - sec_sent_words
            # (bs, seq_len)
            pos_ids = tf.tile(tf.expand_dims(tf.range(self.seq_len, dtype=tf.int64), 0), [dy_batch_size, 1])
            first_sent_pos_ids = pos_ids * (1 - seg_ids) * masks
            sec_sent_pos_ids = (pos_ids - tf.expand_dims(first_sent_words, -1)) * seg_ids
            pos_ids = first_sent_pos_ids + sec_sent_pos_ids
        else:
            # (bs, seq_len)
            pos_ids = tf.tile(tf.expand_dims(tf.range(self.seq_len), 0), [dy_batch_size, 1])
        # Embedding layer
        # just one in cpu memory
        with tf.device("cpu"):
            if self.pretrained_word_embeddings is not None:
                if self.given_arch is None:
                    # this is the search procedure
                    high_dimensional_word_embeddings = tf.get_variable(
                        name="hd_wemb",
                        shape=self.pretrained_word_embeddings.shape,
                        initializer=tf.constant_initializer(self.pretrained_word_embeddings))
                    high_dimensional_pos_embeddings = tf.get_variable(
                        name="hd_pemb",
                        shape=self.pretrained_pos_embeddings.shape,
                        initializer=tf.constant_initializer(self.pretrained_pos_embeddings))
                    h0_word = tf.nn.embedding_lookup(high_dimensional_word_embeddings, word_ids)
                    h0_pos = tf.nn.embedding_lookup(high_dimensional_pos_embeddings, pos_ids)
                    h0 = h0_word + h0_pos
                else:
                    wemb = tf.get_variable(
                        name="wemb",
                        shape=self.pretrained_word_embeddings.shape,
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(self.pretrained_word_embeddings))
                    pemb = tf.get_variable(
                        name="pemb",
                        shape=self.pretrained_pos_embeddings.shape,
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(self.pretrained_pos_embeddings))
                    h0_word = tf.nn.embedding_lookup(wemb, word_ids)
                    h0_pos = tf.nn.embedding_lookup(pemb, pos_ids)
                    h0 = h0_word + h0_pos
            else:
                wemb = tf.get_variable(
                    name="wemb",
                    shape=(self.vocab_size, self.emb_size),
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
                pemb = tf.get_variable(
                    name="pemb",
                    shape=(512, self.emb_size),
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
                h0_word = tf.nn.embedding_lookup(wemb, word_ids)
                h0_pos = tf.nn.embedding_lookup(pemb, pos_ids)
                h0 = h0_word + h0_pos
        if self.pretrained_word_embeddings is not None and self.given_arch is None:
            compress_transformation = tf.get_variable(
                name="compress_transformation",
                shape=(self.pretrained_word_embeddings.shape[1], self.emb_size),
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            h0 = tf.matmul(h0, tf.tile(tf.expand_dims(compress_transformation, 0), [dy_batch_size, 1, 1]))
            # compute the low-dimensional embeddings
            ld_wemb = tf.matmul(high_dimensional_word_embeddings, compress_transformation)
            ld_pemb = tf.matmul(high_dimensional_pos_embeddings, compress_transformation)
            self.ld_embs = [ld_wemb, ld_pemb]
        else:
            self.ld_embs = [wemb, pemb]
        if self.is_pair_task:
            h0_first = h0 * tf.expand_dims(tf.cast(masks * (1 - seg_ids), tf.float32), -1)
            h0_second = h0 * tf.expand_dims(tf.cast(masks * seg_ids, tf.float32), -1)
        else:
            h0_first = h0 * tf.expand_dims(tf.cast(masks, tf.float32), -1)
            h0_second = h0_first
        with tf.variable_scope("preprocessing", reuse=tf.AUTO_REUSE) as scope:
            h0_first = tf.contrib.layers.layer_norm(
                h0_first,
                reuse=scope.reuse,
                begin_norm_axis=2,
                scope=scope)
            h0_first = tf.nn.dropout(
                h0_first, keep_prob=self.keep_prob)
        with tf.variable_scope("preprocessing", reuse=tf.AUTO_REUSE) as scope:
            h0_second = tf.contrib.layers.layer_norm(
                h0_second,
                reuse=scope.reuse,
                begin_norm_axis=2,
                scope=scope)
            h0_second = tf.nn.dropout(
                h0_second, keep_prob=self.keep_prob)

        if self.given_arch is None:
            # this is the search procedure
            arch_params = dict()
            with tf.variable_scope("arch_params", reuse=tf.AUTO_REUSE):
                for dest in range(2, 2 + self.num_intermediates):
                    for src in range(0, dest):
                        alpha = tf.get_variable(
                            name="alpha{}to{}".format(src, dest),
                            shape=(self.card_of_o,),
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.01))
                        alpha_logits = get_logits(alpha)
                        gumbel_noise = get_gumbel([self.card_of_o])
                        y_o = tf.nn.softmax(
                            (alpha_logits + tf.cast(is_training, tf.float32) * gumbel_noise) / temperature)
                        y_hard = tf.cast(tf.equal(y_o, tf.reduce_max(y_o, 0, keep_dims=True)),
                                         y_o.dtype)
                        y_o = tf.stop_gradient(y_hard - y_o) + y_o
                        arch_params[(src, dest)] = y_o

        prev_prev_out = h0_first
        prev_out = h0_second
        # each tensor has shape (bs, emb_size)
        cell_states = list()
        # consider embeddings as 0-th layer
        cell_states.append(tf.reduce_mean(0.5 * (prev_prev_out + prev_out), 1))
        for l in range(self.Kmax):
            if self.given_arch:
                cell_state = self.build_cell(
                    prev_prev_out, prev_out, l, is_training, None, self.given_arch)
            else:
                # this is the search procedure
                cell_state = self.build_cell(
                    prev_prev_out, prev_out, l, is_training, arch_params, self.given_arch)
            # each (bs, emb_size)
            cell_states.append(tf.reduce_mean(cell_state, 1))
            prev_prev_out = prev_out
            prev_out = cell_state
        if self.given_arch is None:
            # this is the search procedure
            with tf.variable_scope("arch_params", reuse=tf.AUTO_REUSE):
                alphaN = tf.get_variable(
                    name="alphaN",
                    shape=(self.Kmax + 1,),
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer(stddev=0.01))
                alphaN_logits = get_logits(alphaN)
                gumbel_noise = get_gumbel([self.Kmax + 1])
                y_o = tf.nn.softmax((alphaN_logits + tf.cast(is_training, tf.float32) * gumbel_noise) / temperature)
                y_hard = tf.cast(tf.equal(y_o,
                                          tf.reduce_max(y_o, 0, keep_dims=True)),
                                 y_o.dtype)
                y_o = tf.stop_gradient(y_hard - y_o) + y_o
                # use at least one cell and keep BP viable
                # sampled_N = tf.argmax(y_o, output_type=tf.int32) + 1
                sampled_N = tf.reduce_sum(y_o * tf.range(self.Kmax + 1, dtype=y_o.dtype))

        # L_{CE}
        optional_ce_loss = list()
        layerwise_logits = list()
        for l in range(len(cell_states)):
            att_cell_weights = tf.get_variable(
                name="l{}_att_cell_ws".format(l + 1),
                shape=(l + 1,),
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            att_cell_weights = tf.nn.softmax(att_cell_weights)
            # (l+1+1, bs, emb_size)
            cell_states_tensor = tf.stack(cell_states[:l + 1])
            # reshaped to (l+1+1, 1, 1) s.t. appropriately broadcast with (l+1+1, bs, emb_size)
            final_representation = tf.reduce_sum(
                tf.reshape(att_cell_weights, [-1, 1, 1]) * cell_states_tensor,
                0)
            final_representation = tf.nn.tanh(final_representation)
            final_representation = tf.nn.dropout(
                final_representation, keep_prob=self.keep_prob)
            with tf.variable_scope("output_layer", reuse=tf.AUTO_REUSE):
                logits = tf.layers.dense(inputs=final_representation,
                                         units=self.num_classes,
                                         activation=None,
                                         name="fc")
                layerwise_logits.append(logits)
            # CSV_Reader returns labels of shape (batch_size,)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(labels, self.num_classes), logits=logits)
            optional_ce_loss.append(tf.reduce_mean(losses))
        optional_ce_loss = tf.stack(optional_ce_loss)
        if self.given_arch is None:
            self.logits = tf.reduce_sum(
                tf.reshape(y_o, [-1, 1, 1]) * tf.stack(layerwise_logits),
                0)
            L_CE = tf.reduce_sum(y_o * optional_ce_loss)
        else:
            self.logits = logits
            L_CE = optional_ce_loss[-1]

        if self.given_arch is None:
            # this is the search procedure
            # L_{KD}
            # (bs, 13, #classes)
            teacher_prob_logits = tf.reshape(prob_logits, [-1, 13, self.num_classes])
            # (13, bs, #classes)
            teacher_prob_logits = tf.transpose(teacher_prob_logits, perm=[1, 0, 2])
            # (13, bs, #classes)
            teacher_prob_lbs = tf.nn.softmax(teacher_prob_logits)
            optional_kd_loss = list()
            for l in range(len(layerwise_logits)):
                included_prob_ces = list()
                delta_l = 12 // (1 + l)
                for m in range(l + 1):
                    # calculate cross-entropy losses of shape (bs,)
                    prob_ces = tf.nn.softmax_cross_entropy_with_logits(labels=teacher_prob_lbs[m * delta_l, :, :],
                                                                       logits=layerwise_logits[m])
                    included_prob_ces.append(tf.reduce_mean(prob_ces))
                optional_kd_loss.append((1.0 / float(len(included_prob_ces))) * tf.add_n(included_prob_ces))
            optional_kd_loss = tf.stack(optional_kd_loss)
            L_KD = tf.reduce_sum(y_o * optional_kd_loss)

            # L_{E}
            per_edge_les = list()
            for alpha in arch_params.values():
                per_edge_les.append(tf.reduce_sum(alpha * EP))
            L_E = (tf.cast(sampled_N, tf.float32) / self.Kmax) * (1.0 / float(
                np.sum([indegree for indegree in range(2, 2 + self.num_intermediates)]))) * tf.add_n(per_edge_les)

            # Interpolated
            self.loss = (1.0 - self.loss_gamma) * L_CE + self.loss_gamma * L_KD + self.loss_beta * L_E
        else:
            self.loss = L_CE

        self.model_params = [var for var in tf.trainable_variables()
                             if "arch_params" not in var.name]
        print("There are {} model vars".format(len(self.model_params)))
        self.arch_params = [var for var in tf.trainable_variables()
                            if "arch_params" in var.name]
        print("There are {} arch vars".format(len(self.arch_params)))
        num_var_floats = 0
        for var in tf.trainable_variables():
            num_var_floats += np.prod(var.shape.as_list())
        print("Trainable variables are of {} MBs".format(4 * num_var_floats / 1024 / 1024))
        model_reg_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in self.model_params])
        if self.given_arch is None:
            # this is the search procedure
            arch_reg_loss = tf.add_n(
                [tf.nn.l2_loss(var) for var in self.arch_params])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        max_device_idx = -1
        for op in update_ops:
            last_semicomma_idx = op.device.rfind(':')
            if last_semicomma_idx >= 0:
                # TO DO: any other cases? not end with CPU:i or GPU:i
                device_idx = int(op.device[last_semicomma_idx + 1:])
                max_device_idx = max(device_idx, max_device_idx)
        print("Max device index is {}".format(max_device_idx))
        if max_device_idx >= 0:
            update_ops = [op for op in update_ops if op.device.endswith(str(max_device_idx))]
        print("There are {} ops in GraphKeys.UPDATE_OPS.".format(len(update_ops)))
        # This trick can reduce the size of GraphDef drastically.
        update_ops = tf.group(*update_ops)

        # op_param_lr = tf.train.cosine_decay(self.model_opt_init_lr,
        #                                    self.global_step,
        #                                    10000,
        #                                    alpha=self.model_opt_final_lr / self.model_opt_init_lr)
        model_opt = tf.train.AdamOptimizer(self.model_opt_lr, name="model_opt")
        if self.given_arch is None:
            # this is the search procedure
            arch_opt = tf.train.AdamOptimizer(self.arch_opt_lr, name="arch_opt")

        with tf.control_dependencies([update_ops]):
            # As there are batch normalization layers, we need to depend on
            # their updates first
            model_grads_and_vars = [(grad, var) for grad, var in \
                                    model_opt.compute_gradients(
                                        self.loss + self.model_l2_reg * model_reg_loss,
                                        var_list=self.model_params) \
                                    if grad is not None]
            print("There are {} model parameters having gradients.".format(len(model_grads_and_vars)))
            clipped_model_grads, _ = tf.clip_by_global_norm(
                [tp[0] for tp in model_grads_and_vars], 5.0)
            if self.given_arch is None:
                # this is the search procedure
                model_update = model_opt.apply_gradients(
                    zip(clipped_model_grads,
                        [tp[1] for tp in model_grads_and_vars]))
            else:
                model_update = model_opt.apply_gradients(
                    zip(clipped_model_grads,
                        [tp[1] for tp in model_grads_and_vars]),
                    global_step=self.global_step)
        if self.given_arch is None:
            # this is the search procedure
            with tf.control_dependencies([model_update]):
                # Update model parameters before updating arch parameters
                # according to DARTS.
                arch_grads_and_vars = [(grad, var) for grad, var in \
                                       arch_opt.compute_gradients(
                                           self.loss + self.arch_l2_reg * arch_reg_loss,
                                           var_list=self.arch_params) \
                                       if grad is not None]
                print("There are {} arch parameters having gradients".format(len(arch_grads_and_vars)))
                clipped_arch_grads, _ = tf.clip_by_global_norm(
                    [tp[0] for tp in arch_grads_and_vars], 5.0)
                # actually arch update
                self.update = arch_opt.apply_gradients(
                    zip(clipped_arch_grads,
                        [tp[1] for tp in arch_grads_and_vars]),
                    global_step=self.global_step)
        else:
            self.update = model_update

        self.predictions = tf.argmax(self.logits, -1)
        self.acc = tf.metrics.accuracy(labels, self.predictions)

    def build_cell(self,
                   input0,
                   input1,
                   index,
                   is_training,
                   arch_params,
                   given_arch=None):
        """Create the computation graph for a cell
        """

        inputs = [input0, input1]
        with tf.variable_scope("c{}".format(index)):
            for i in range(2, 2 + self.num_intermediates):
                inputs.append(
                    self.build_node(inputs, i, arch_params, given_arch, is_training))
            att_node_weights = tf.get_variable(
                name="att_node_weights",
                shape=(self.num_intermediates,),
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.01))
            att_node_weights = tf.nn.softmax(att_node_weights)
            # (#inter, bs, seq_len, emb_dim)
            intermediates = tf.stack(inputs[-self.num_intermediates:])
            att_node_weights = tf.reshape(att_node_weights, [-1, 1, 1, 1])
            # (bs, seq_len, emb_dim)
            output = tf.reduce_sum(att_node_weights * intermediates, 0)
        return output

    def build_node(self,
                   inputs,
                   index,
                   arch_params,
                   given_arch,
                   is_training):
        """Create the computation graph for a node."""

        states = list()
        with tf.variable_scope("node{}".format(index)):
            for src, h in enumerate(inputs):
                if given_arch is not None and (src, index) not in given_arch:
                    tf.logging.info("excluded edge {}-{}".format(src, index))
                    continue
                states.append(
                    self.build_edge(h, arch_params[(src, index)] if arch_params else given_arch[(src, index)],
                                    is_training, src, index))
        return tf.add_n(states)

    def build_edge(self,
                   h_last,
                   alpha,
                   is_training,
                   src,
                   tgt):
        """Create the computation graph for an edge."""

        with tf.variable_scope("edge{}to{}".format(src, tgt)):
            h_last_activation = tf.nn.relu(h_last)
            NHWC_h0 = tf.reshape(h_last_activation,
                                 [-1, 1, self.seq_len, self.emb_size])

            if isinstance(alpha, int):
                # given arch
                if alpha == 0:
                    return self.build_cnn3(h_last_activation, is_training)
                elif alpha == 1:
                    return self.build_cnn5(h_last_activation, is_training)
                elif alpha == 2:
                    return self.build_cnn7(h_last_activation, is_training)
                elif alpha == 3:
                    return self.build_dilated_cnn3(NHWC_h0, is_training)
                elif alpha == 4:
                    return self.build_dilated_cnn5(NHWC_h0, is_training)
                elif alpha == 5:
                    return self.build_dilated_cnn7(NHWC_h0, is_training)
                elif alpha == 6:
                    return self.build_max_pool(NHWC_h0)
                elif alpha == 7:
                    return self.build_avg_pool(NHWC_h0)
                elif alpha == 8:
                    return tf.identity(h_last)
                else:
                    return tf.zeros_like(h_last)

            # enumerate all candidate ops
            h1_cnn3 = self.build_cnn3(h_last_activation, is_training)
            h1_cnn5 = self.build_cnn5(h_last_activation, is_training)
            h1_cnn7 = self.build_cnn7(h_last_activation, is_training)
            h1_dila_cnn3 = self.build_dilated_cnn3(NHWC_h0, is_training)
            h1_dila_cnn5 = self.build_dilated_cnn5(NHWC_h0, is_training)
            h1_dila_cnn7 = self.build_dilated_cnn7(NHWC_h0, is_training)
            h1_max_pool = self.build_max_pool(NHWC_h0)
            h1_mean_pool = self.build_avg_pool(NHWC_h0)
            h1_res = tf.identity(h_last)
            h1_skip = tf.zeros_like(h_last)
            # (|O|, bs, seq_len, emb_dim)
            h = tf.stack([h1_cnn3, h1_cnn5, h1_cnn7, h1_dila_cnn3, \
                          h1_dila_cnn5, h1_dila_cnn7, \
                          h1_max_pool, h1_mean_pool, \
                          h1_res, h1_skip])
            op_weights = tf.reshape(alpha, [-1, 1, 1, 1])
            # (bs, seq_len, emb_dim)
            h = tf.reduce_sum(h * op_weights, 0)
        return h

    def build_cnn3(self, x, is_training):
        conv3 = tf.get_variable(
            name="conv3",
            shape=(3, self.emb_size, self.emb_size),
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        h1_cnn3 = tf.nn.conv1d(x,
                               filters=conv3,
                               stride=1,
                               padding='SAME')
        h1_cnn3 = tf.squeeze(
            tf.layers.batch_normalization(
                tf.expand_dims(h1_cnn3, 1),
                momentum=0.9,
                training=is_training,
                name="cnn3bn"),
            [1])
        return h1_cnn3

    def build_cnn5(self, x, is_training):
        conv5 = tf.get_variable(
            name="conv5",
            shape=(5, self.emb_size, self.emb_size),
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        h1_cnn5 = tf.nn.conv1d(x,
                               filters=conv5,
                               stride=1,
                               padding='SAME')
        h1_cnn5 = tf.squeeze(
            tf.layers.batch_normalization(
                tf.expand_dims(h1_cnn5, 1),
                momentum=0.9,
                training=is_training,
                name="cnn5bn"),
            [1])
        return h1_cnn5

    def build_cnn7(self, x, is_training):
        conv7 = tf.get_variable(
            name="conv7",
            shape=(7, self.emb_size, self.emb_size),
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        h1_cnn7 = tf.nn.conv1d(x,
                               filters=conv7,
                               stride=1,
                               padding='SAME')
        h1_cnn7 = tf.squeeze(
            tf.layers.batch_normalization(
                tf.expand_dims(h1_cnn7, 1),
                momentum=0.9,
                training=is_training,
                name="cnn7bn"),
            [1])
        return h1_cnn7

    def build_dilated_cnn3(self, x, is_training):
        dila_conv3 = tf.get_variable(
            name="dila_conv3",
            shape=(1, 3, self.emb_size),
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        h1_dila_cnn3 = tf.nn.dilation2d(
            x,
            filter=dila_conv3,
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
        h1_dila_cnn3 = tf.squeeze(
            tf.layers.batch_normalization(
                h1_dila_cnn3, momentum=0.9, training=is_training, name="dila_cnn3bn"),
            [1])
        return h1_dila_cnn3

    def build_dilated_cnn5(self, x, is_training):
        dila_conv5 = tf.get_variable(
            name="dila_conv5",
            shape=(1, 5, self.emb_size),
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        h1_dila_cnn5 = tf.nn.dilation2d(
            x,
            filter=dila_conv5,
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
        h1_dila_cnn5 = tf.squeeze(
            tf.layers.batch_normalization(
                h1_dila_cnn5, momentum=0.9, training=is_training, name="dila_cnn5bn"),
            [1])
        return h1_dila_cnn5

    def build_dilated_cnn7(self, x, is_training):
        dila_conv7 = tf.get_variable(
            name="dila_conv7",
            shape=(1, 7, self.emb_size),
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        h1_dila_cnn7 = tf.nn.dilation2d(
            x,
            filter=dila_conv7,
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
        h1_dila_cnn7 = tf.squeeze(
            tf.layers.batch_normalization(
                h1_dila_cnn7, momentum=0.9, training=is_training, name="dila_cnn6bn"),
            [1])
        return h1_dila_cnn7

    def build_max_pool(self, x):
        h1_max_pool = tf.nn.max_pool(x,
                                     ksize=[1, 1, 3, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME')
        h1_max_pool = tf.squeeze(h1_max_pool, [1])
        return h1_max_pool

    def build_avg_pool(self, x):
        h1_mean_pool = tf.nn.avg_pool(x,
                                      ksize=[1, 1, 3, 1],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
        h1_mean_pool = tf.squeeze(h1_mean_pool, [1])
        return h1_mean_pool
