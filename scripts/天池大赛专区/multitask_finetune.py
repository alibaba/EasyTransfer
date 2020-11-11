import os
import tensorflow as tf
from easytransfer import base_model, FLAGS
from easytransfer import layers
from easytransfer import model_zoo
from easytransfer import preprocessors
from easytransfer.datasets import TFRecordReader
from easytransfer.losses import softmax_cross_entropy
from sklearn.metrics import classification_report
import numpy as np

class MultiTaskTFRecordReader(TFRecordReader):
    def __init__(self, input_glob, batch_size, is_training=False,
                 **kwargs):

        super(MultiTaskTFRecordReader, self).__init__(input_glob, batch_size, is_training, **kwargs)
        self.task_fps = []
        with tf.gfile.Open(input_glob, 'r') as f:
            for line in f:
                line = line.strip()
                self.task_fps.append(line)

    def get_input_fn(self):
        def input_fn():
            num_datasets = len(self.task_fps)
            datasets = []
            for input_glob in self.task_fps:
                dataset = tf.data.TFRecordDataset(input_glob)
                dataset = self._get_data_pipeline(dataset, self._decode_tfrecord)
                datasets.append(dataset)

            choice_dataset = tf.data.Dataset.range(num_datasets).repeat()
            return tf.data.experimental.choose_from_datasets(datasets, choice_dataset)

        return input_fn

class Application(base_model):
    def __init__(self, **kwargs):
        super(Application, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path)

        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

        global_step = tf.train.get_or_create_global_step()

        tnews_dense = layers.Dense(15,
                     kernel_initializer=layers.get_initializer(0.02),
                     name='tnews_dense')

        ocemotion_dense = layers.Dense(7,
                             kernel_initializer=layers.get_initializer(0.02),
                             name='ocemotion_dense')

        ocnli_dense = layers.Dense(3,
                             kernel_initializer=layers.get_initializer(0.02),
                             name='ocnli_dense')

        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)

        outputs = model([input_ids, input_mask, segment_ids], mode=mode)
        pooled_output = outputs[1]

        if mode == tf.estimator.ModeKeys.TRAIN:
            pooled_output = tf.nn.dropout(pooled_output, keep_prob=0.9)

        logits = tf.case([(tf.equal(tf.mod(global_step, 3), 0), lambda: tnews_dense(pooled_output)),
                          (tf.equal(tf.mod(global_step, 3), 1), lambda: ocemotion_dense(pooled_output)),
                          (tf.equal(tf.mod(global_step, 3), 2), lambda: ocnli_dense(pooled_output)),
                          ], exclusive=True)

        if mode == tf.estimator.ModeKeys.PREDICT:
            ret = {
                "tnews_logits": tnews_dense(pooled_output),
                "ocemotion_logits": ocemotion_dense(pooled_output),
                "ocnli_logits": ocnli_dense(pooled_output),
                "label_ids": label_ids
            }
            return ret

        return logits, label_ids

    def build_loss(self, logits, labels):
        global_step = tf.train.get_or_create_global_step()
        return tf.case([(tf.equal(tf.mod(global_step, 3), 0), lambda : softmax_cross_entropy(labels, 15, logits)),
                      (tf.equal(tf.mod(global_step, 3), 1), lambda : softmax_cross_entropy(labels, 7, logits)),
                      (tf.equal(tf.mod(global_step, 3), 2), lambda : softmax_cross_entropy(labels, 3, logits)),
                      ], exclusive=True)

    def build_predictions(self, output):
        tnews_logits = output['tnews_logits']
        ocemotion_logits = output['ocemotion_logits']
        ocnli_logits = output['ocnli_logits']

        tnews_predictions = tf.argmax(tnews_logits, axis=-1, output_type=tf.int32)
        ocemotion_predictions = tf.argmax(ocemotion_logits, axis=-1, output_type=tf.int32)
        ocnli_predictions = tf.argmax(ocnli_logits, axis=-1, output_type=tf.int32)

        ret_dict = {
            "tnews_predictions": tnews_predictions,
            "ocemotion_predictions": ocemotion_predictions,
            "ocnli_predictions": ocnli_predictions,
            "label_ids": output['label_ids']
        }
        return ret_dict

def main(_):
    FLAGS.mode = "train"
    app = Application()
    train_reader = MultiTaskTFRecordReader(input_glob=app.train_input_fp,
                                           is_training=True,
                                           input_schema=app.input_schema,
                                           batch_size=app.train_batch_size)

    app.run_train(reader=train_reader)
    FLAGS.mode = "predict"
    app = Application()
    predict_reader = MultiTaskTFRecordReader(input_glob=app.predict_input_fp,
                                           is_training=False,
                                           input_schema=app.input_schema,
                                           batch_size=app.predict_batch_size)

    ckpts = set()
    with tf.gfile.GFile(os.path.join(app.config.model_dir, "checkpoint"), mode='r') as reader:
        for line in reader:
            line = line.strip()
            line = line.replace("oss://", "")
            ckpts.add(int(line.split(":")[1].strip().replace("\"", "").split("/")[-1].replace("model.ckpt-", "")))

    best_macro_f1 = 0
    best_ckpt = None
    for ckpt in sorted(ckpts):
        checkpoint_path = os.path.join(app.config.model_dir, "model.ckpt-" + str(ckpt))
        tf.logging.info("checkpoint_path is {}".format(checkpoint_path))
        all_tnews_preds = []
        all_tnews_gts = []
        all_ocemotion_preds = []
        all_ocemotion_gts = []
        all_ocnli_preds = []
        all_ocnli_gts = []
        for i, output in enumerate(app.run_predict(reader=predict_reader, checkpoint_path=checkpoint_path)):
            label_ids = np.squeeze(output['label_ids'])
            if i%3 ==0:
                tnews_predictions = output['tnews_predictions']
                all_tnews_preds.extend(tnews_predictions.tolist())
                all_tnews_gts.extend(label_ids.tolist())
            elif i%3==1:
                ocemotion_predictions = output['ocemotion_predictions']
                all_ocemotion_preds.extend(ocemotion_predictions.tolist())
                all_ocemotion_gts.extend(label_ids.tolist())
            elif i%3==2:
                ocnli_predictions = output['ocnli_predictions']
                all_ocnli_preds.extend(ocnli_predictions.tolist())
                all_ocnli_gts.extend(label_ids.tolist())

        tnews_report = classification_report(all_tnews_gts, all_tnews_preds, digits=4)
        tnews_macro_avg_f1 = float(tnews_report.split()[-8])

        ocemotion_report = classification_report(all_ocemotion_gts, all_ocemotion_preds, digits=4)
        ocemotion_macro_avg_f1 = float(ocemotion_report.split()[-8])

        ocnli_report = classification_report(all_ocnli_gts, all_ocnli_preds, digits=4)
        ocnli_macro_avg_f1 = float(ocnli_report.split()[-8])

        macro_f1 = (tnews_macro_avg_f1 + ocemotion_macro_avg_f1 + ocnli_macro_avg_f1)/3.0
        if macro_f1 >= best_macro_f1:
            best_macro_f1 = macro_f1
            best_ckpt = ckpt

    tf.logging.info("best ckpt {}, best best_macro_f1 {}".format(best_ckpt, best_macro_f1))

if __name__ == "__main__":
    tf.app.run()
