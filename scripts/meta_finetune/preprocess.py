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
import os
import numpy as np
import tensorflow as tf
from easytransfer import Config
from easytransfer import base_model
from easytransfer import layers
from easytransfer import model_zoo
from easytransfer import preprocessors
from easytransfer import FLAGS
from easytransfer.datasets import CSVReader, OdpsTableReader
import scipy.spatial.distance as distance


tf.app.flags.DEFINE_string("domains", "", "Domain names, seperated by comma")
tf.app.flags.DEFINE_string("classes", "", "Class names, seperated by comma")
tf.app.flags.DEFINE_boolean("do_sent_pair", False, "Do sentence pair classification, False for sentence classification")
aFLAGS = tf.app.flags.FLAGS


class SentClsApplication(base_model):
    def __init__(self, **kwargs):
        super(SentClsApplication, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path)
        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)
        input_ids, input_mask, segment_ids, label_ids, texts, domains, labels = preprocessor(features)
        outputs = model([input_ids, input_mask, segment_ids], mode=mode)
        pooled_output = outputs[1]
        ret = {"pooled_output": pooled_output, "text": texts, "domain": domains, "label": labels}
        return ret

    def build_predictions(self, output):
        predictions = dict()
        predictions["pool_output"] = output["pooled_output"]
        predictions["text"] = output["text"]
        predictions["domain"] = output["domain"]
        predictions["label"] = output["label"]
        return predictions


class SentPairClsApplication(base_model):
    def __init__(self, **kwargs):
        super(SentPairClsApplication, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path)
        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

        input_ids, input_mask, segment_ids, label_ids, texts1, texts2, domains, labels = preprocessor(features)
        outputs = model([input_ids, input_mask, segment_ids], mode=mode)
        pooled_output = outputs[1]
        ret = {"pooled_output": pooled_output, "text1": texts1, "text2": texts2, "domain": domains, "label": labels}
        return ret

    def build_predictions(self, output):
        predictions = dict()
        predictions["pool_output"] = output["pooled_output"]
        predictions["text1"] = output["text1"]
        predictions["text2"] = output["text2"]
        predictions["domain"] = output["domain"]
        predictions["label"] = output["label"]
        return predictions


def compute_weight(domain, label, current_embedding, centroid_embeddings):
    key_name = domain.decode('utf-8') + "\t" + label.decode('utf-8')
    current_centroid = centroid_embeddings[key_name]
    other_centroids = list()
    for current_key in centroid_embeddings.keys():
        items = current_key.split("\t")
        current_domain = items[0]
        current_label = items[1]
        if not (current_domain == domain.decode('utf-8')) and (current_label==label.decode('utf-8')):
            other_centroids.append(centroid_embeddings[current_key])
    other_centroids = np.array(other_centroids)
    other_centroid_mean = np.mean(other_centroids, axis=0)
    first_cos_sim = 1 - distance.cosine(current_embedding, current_centroid)
    second_cos_sim = 1 - distance.cosine(current_embedding, other_centroid_mean)
    return (first_cos_sim + second_cos_sim) / 2


def main(_):

    # load domain and class dict
    domains = aFLAGS.domains.split(",")
    classes = aFLAGS.classes.split(",")

    if aFLAGS.do_sent_pair:
        app = SentPairClsApplication()
    else:
        app = SentClsApplication()

    # create empty centroids
    domain_class_embeddings = dict()
    for domain_name in domains:
        for class_name in classes:
            key_name = domain_name + "\t" + class_name
            domain_class_embeddings[key_name] = list()

    # for training data
    if "PAI" in tf.__version__:
        predict_reader = OdpsTableReader(input_glob=app.predict_input_fp, is_training=False, input_schema=app.input_schema, batch_size=app.predict_batch_size)
    else:
        predict_reader = CSVReader(input_glob=app.predict_input_fp, is_training=False, input_schema=app.input_schema, batch_size=app.predict_batch_size)

    # do inference for training data
    temp_output_data = list()

    for output in app.run_predict(reader=predict_reader, checkpoint_path=app.predict_checkpoint_path):
        current_size = len(output["pool_output"])
        for i in range(current_size):
            if FLAGS.do_sent_pair:
                pool_output = output["pool_output"][i]
                text1 = output["text1"][i]
                text2 = output["text2"][i]
                domain = output["domain"][i]
                label = output["label"][i]
                key_name = domain.decode('utf-8') + "\t" + label.decode('utf-8')
                domain_class_embeddings[key_name].append(pool_output)
                temp_output_data.append((text1, text2, domain, label, pool_output))
            else:
                pool_output = output["pool_output"][i]
                text = output["text"][i]
                domain = output["domain"][i]
                label = output["label"][i]
                key_name = domain.decode('utf-8') + "\t" + label.decode('utf-8')
                domain_class_embeddings[key_name].append(pool_output)
                temp_output_data.append((text, domain, label, pool_output))
            
    # compute centroids
    centroid_embeddings = dict()
    for key_name in domain_class_embeddings:
        domain_class_data_embeddings = np.array(domain_class_embeddings[key_name])
        centroid_embeddings[key_name] = np.mean(domain_class_data_embeddings, axis=0)

    # output files for meta fine-tune
    if "PAI" in tf.__version__:
        #write odps tables
        records = []
        if aFLAGS.do_sent_pair:
            for text1, text2, domain, label, embeddings in temp_output_data:
                weight = compute_weight(domain, label, embeddings, centroid_embeddings)
                tup = (text1, text2, str(domains.index(domain)), label, np.around(weight, decimals=5))
                records.append(tup)
        else:
            for text, domain, label, embeddings in temp_output_data:
                weight = compute_weight(domain, label, embeddings, centroid_embeddings)
                tup = (text, str(domains.index(domain)), label, np.around(weight, decimals=5))
                records.append(tup)

        with tf.python_io.TableWriter(FLAGS.outputs) as writer:
            if aFLAGS.do_sent_pair:
                indices = list(x for x in range(0, 5))
            else:
                indices = list(x for x in range(0, 4))
            writer.write(records, indices)
    else:
        #write to local file
        with open(FLAGS.outputs, 'w+') as f:
            if aFLAGS.do_sent_pair:
                for text1, text2, domain, label, embeddings in temp_output_data:
                    weight = compute_weight(domain, label, embeddings, centroid_embeddings)
                    f.write(text1 + '\t' + text2 + '\t' + str(domains.index(domain)) + '\t' + label + '\t' + np.around(weight, decimals=5).astype('str') + '\n')
            else:
                for text, domain, label, embeddings in temp_output_data:
                    weight = compute_weight(domain, label, embeddings, centroid_embeddings)
                    f.write(text + '\t' + str(domains.index(domain)) + '\t' + label + '\t' + np.around(weight, decimals=5).astype('str') + '\n')
        
if __name__ == '__main__':
    tf.app.run()
    
