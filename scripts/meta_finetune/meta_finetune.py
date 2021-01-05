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
import sys
sys.path.append("../..")
import tensorflow as tf
from easytransfer.losses import softmax_cross_entropy
from easytransfer import base_model
from easytransfer import Config
from easytransfer import layers, FLAGS
from easytransfer import model_zoo
from easytransfer import preprocessors
from easytransfer.datasets import CSVReader, OdpsTableReader
from easytransfer.evaluators import classification_eval_metrics


# algo defined parameters
tf.app.flags.DEFINE_string("layer_indexes", "", help='layer_indexes for domain corruption')
tf.app.flags.DEFINE_float("domain_weight", 0, help='domain_weight')
tf.app.flags.DEFINE_string("domains", "", help='domains')
tf.app.flags.DEFINE_boolean("do_sent_pair", False, "do sentence pair classification, False for sentence classification")

aFLAGS = tf.app.flags.FLAGS


def weighted_softmax_cross_entropy(labels, depth, logits, weights):
     with tf.variable_scope("mft"):
        labels = tf.squeeze(labels)
        weights = tf.squeeze(weights)
        one_hot_labels = tf.one_hot(labels, depth=depth, dtype=tf.float32)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels,
                                            logits=logits, weights=weights)
        return loss


class Application(base_model):
    def __init__(self, **kwargs):
        super(Application, self).__init__(**kwargs)
        self.weights = None
        self.domains = None
        self.domain_logits = None


    def build_logits(self, features, mode=None):

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path, user_defined_config=self.config)
        bert_backbone = model_zoo.get_pretrained_model(self.config.pretrain_model_name_or_path)
        dense = layers.Dense(self.num_labels, kernel_initializer=layers.get_initializer(0.02), name='dense')

        input_ids, input_mask, segment_ids, label_ids, domains, weights = preprocessor(features)
        
        self.domains = domains
        self.weights = weights
        hidden_size = bert_backbone.config.hidden_size
        self.domain_logits = dict()

        bert_model = bert_backbone.bert
        embedding_output = bert_model.embeddings([input_ids, segment_ids], training=is_training)
        attention_mask = layers.get_attn_mask_bert(input_ids, input_mask)
        encoder_outputs = bert_model.encoder([embedding_output, attention_mask], training=is_training)
        encoder_outputs = encoder_outputs[0]
        pooled_output = bert_model.pooler(encoder_outputs[-1][:, 0])

        if mode == tf.estimator.ModeKeys.TRAIN:
            pooled_output = tf.nn.dropout(pooled_output, keep_prob=0.9)

        with tf.variable_scope("mft", reuse=tf.AUTO_REUSE):
            # add domain network
            logits = dense(pooled_output)
            domains = tf.squeeze(domains)
            
            domain_embedded_matrix = tf.get_variable("domain_projection", [num_domains, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
            domain_embedded = tf.nn.embedding_lookup(domain_embedded_matrix, domains)

            for layer_index in layer_indexes:
                content_tensor = tf.reduce_mean(encoder_outputs[layer_index], axis=1)
                content_tensor_with_domains = domain_embedded + content_tensor

                domain_weights = tf.get_variable("domain_weights", [num_domains, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
                domain_bias = tf.get_variable("domain_bias", [num_domains], initializer=tf.zeros_initializer())

                current_domain_logits = tf.matmul(content_tensor_with_domains, domain_weights, transpose_b=True)
                current_domain_logits = tf.nn.bias_add(current_domain_logits, domain_bias)

                self.domain_logits["domain_logits_"+str(layer_index)] = current_domain_logits
        return logits, label_ids


    def build_loss(self, logits, labels):
        cls_loss = weighted_softmax_cross_entropy(labels, self.num_labels, logits, self.weights)
        total_domain_loss = 0
        for layer_index in layer_indexes:
            shuffle_domain_labels = tf.random_shuffle(self.domains)
            current_domain_logits = self.domain_logits["domain_logits_"+str(layer_index)]
            domain_loss = softmax_cross_entropy(shuffle_domain_labels, num_domains, current_domain_logits)
            total_domain_loss += domain_loss
        total_domain_loss = total_domain_loss/len(layer_indexes)
        return cls_loss + domain_weight * total_domain_loss


    def build_eval_metrics(self, logits, labels):
        return classification_eval_metrics(logits, labels, self.num_labels)



def main(_):
    domain_str = aFLAGS.domains
    global num_domains, layer_indexes, domain_weight, do_sent_pair
    num_domains = len(domain_str.split(","))
    layer_indexes_str = aFLAGS.layer_indexes
    layer_indexes = [int(x) for x in layer_indexes_str.split(",")]
    domain_weight = aFLAGS.domain_weight
    do_sent_pair = aFLAGS.do_sent_pair

    app = Application()

    if "PAI" in tf.__version__:
        train_reader = OdpsTableReader(input_glob=app.train_input_fp, is_training=True, input_schema=app.input_schema, batch_size=app.train_batch_size)
        eval_reader = OdpsTableReader(input_glob=app.eval_input_fp, is_training=False, input_schema=app.input_schema, batch_size=app.eval_batch_size)
        app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)
    else:
        train_reader = CSVReader(input_glob=app.train_input_fp, is_training=True, input_schema=app.input_schema, batch_size=app.train_batch_size)
        eval_reader = CSVReader(input_glob=app.eval_input_fp, is_training=False, input_schema=app.input_schema, batch_size=app.eval_batch_size)
        app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)

    

if __name__ == '__main__':
    tf.app.run()
