import sys
import tensorflow as tf
from easytransfer import base_model, Config
from easytransfer import layers
from easytransfer import model_zoo
from easytransfer.datasets import CSVReader
from easytransfer.losses import softmax_cross_entropy
from easytransfer import preprocessors
from easytransfer.evaluators import classification_eval_metrics, matthew_corr_metrics
import os

_app_flags = tf.app.flags
_app_flags.DEFINE_string("task_name", default=None, help='task_name')
_app_flags.DEFINE_string("pretrain_model_name_or_path", default=None, help='pretrain_model_name_or_path')
_app_flags.DEFINE_string("train_input_fp", default=None, help='train_input_fp')
_app_flags.DEFINE_string("eval_input_fp", default=None, help='eval_input_fp')
_app_flags.DEFINE_integer("train_batch_size", default=128, help='train_batch_size')
_app_flags.DEFINE_integer("eval_batch_size", default=128, help='eval_batch_size')
_app_flags.DEFINE_float("num_epochs", default=1, help='num_epochs')
_app_flags.DEFINE_string("model_dir", default='', help='model_dir')
_app_flags.DEFINE_float("learning_rate", 1e-4, "learning_rate")
_APP_FLAGS = _app_flags.FLAGS

class Application(base_model):
    def __init__(self, **kwargs):
        super(Application, self).__init__(**kwargs)
        self.user_defined_config = kwargs["user_defined_config"]

    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path,
                                                      user_defined_config=self.user_defined_config)

        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

        dense = layers.Dense(self.num_labels,
                             kernel_initializer=layers.get_initializer(0.02),
                             name='dense')

        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
        outputs = model([input_ids, input_mask, segment_ids], mode=mode)
        pooled_output = outputs[1]

        if mode == tf.estimator.ModeKeys.TRAIN:
            pooled_output = tf.nn.dropout(pooled_output, keep_prob=0.9)

        logits = dense(pooled_output)
        return logits, label_ids

    def build_loss(self, logits, labels):
        return softmax_cross_entropy(labels, self.num_labels, logits)

    def build_eval_metrics(self, logits, labels):
        if _APP_FLAGS.task_name == "CoLA":
            return matthew_corr_metrics(logits, labels)
        else:
            return classification_eval_metrics(logits, labels, self.num_labels)

if __name__ == "__main__":

    config_json = {
        "worker_hosts": "localhost",
        "task_index": 1,
        "job_name": "chief",
        "num_gpus": 1,
        "num_workers": 1,
        "preprocess_config": {
            "input_schema": None,
            "sequence_length": 128,
            "first_sequence": None,
            "second_sequence": None,
            "label_name": "label",
            "label_enumerate_values": None,
        },
        "model_config": {
            "pretrain_model_name_or_path": None,
            "num_labels": None
        },
        "train_config": {
            "keep_checkpoint_max": 11,
            "save_steps": None,
            "optimizer_config": {
                "optimizer": "adam",
                "weight_decay_ratio": 0.01,
                "warmup_ratio": 0.1,
            },
            "distribution_config": {
                "distribution_strategy": None,
            }
        },
        "evaluate_config": {
            "eval_batch_size": 8
        }
    }

    for arg in sys.argv[1:]:
        key = arg.split("=")[0].replace("--", "")
        val = arg.split("=")[1]
        if key == 'train_input_fp' or key == "train_batch_size" \
                or key == "model_dir" or key == 'num_epochs':
            config_json['train_config'][key] = val
        elif key == "eval_input_fp":
            config_json['evaluate_config'][key] = val
        elif key == "learning_rate" or key == 'warmup_ratio' or key == 'weight_decay_ratio':
            config_json['train_config']['optimizer_config'][key] = val
        elif key == 'pretrain_model_name_or_path':
            config_json['model_config'][key] = val
        elif key == 'num_gpus':
            config_json[key] = int(val)
        elif key == 'task_name':
            if val == "TNEWS":
                config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config']['label_enumerate_values'] = "115,114,108,109,116,110,113,112,102,103,100,101,106,107,104"
                config_json['model_config']['num_labels'] = 15

            elif val == "AFQMC":
                config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1,sent2:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config']['second_sequence'] = "sent2"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "0,1"
                config_json['model_config']['num_labels'] = 2

            elif val == "IFLYTEK":
                config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = ",".join([str(idx) for idx in range(119)])
                config_json['model_config']['num_labels'] = 119

            elif val == "CMNLI":
                config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1,sent2:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config']['second_sequence'] = "sent2"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "entailment,neutral,contradiction"
                config_json['model_config']['num_labels'] = 3

            elif val == "CSL":
                config_json['preprocess_config']['input_schema'] = "label:str:1,sent1:str:1,sent2:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config']['second_sequence'] = "sent2"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "0,1"
                config_json['model_config']['num_labels'] = 2

            elif val == "QQP":
                config_json['preprocess_config']['input_schema'] = "idx:str:1,xx1:str:1,xx2:str:1,sent1:str:1,sent2:str:1,label:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config']['second_sequence'] = "sent2"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "0,1"
                config_json['model_config']['num_labels'] = 2

            elif val == "SST-2":
                config_json['preprocess_config']['input_schema'] = "sent1:str:1,label:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "0,1"
                config_json['model_config']['num_labels'] = 2

            elif val == "CoLA":
                config_json['preprocess_config']['input_schema'] = "idx:str:1,label:str:1,xx:str:1,sent1:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "0,1"
                config_json['model_config']['num_labels'] = 2

            elif val == "MRPC":
                config_json['preprocess_config']['input_schema'] = "label:str:1,xx:str:1,xx2:str:1,sent1:str:1,sent2:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config']['second_sequence'] = "sent2"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "0,1"
                config_json['model_config']['num_labels'] = 2

            elif val == "RTE":
                config_json['preprocess_config']['input_schema'] = "idx:str:1,sent1:str:1,sent2:str:1,label:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config']['second_sequence'] = "sent2"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "not_entailment,entailment"
                config_json['model_config']['num_labels'] = 2

            elif val == "BoolQ":
                config_json['preprocess_config']['input_schema'] = "idx:str:1,sent1:str:1,sent2:str:1,label:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config']['second_sequence'] = "sent2"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "True,False"
                config_json['model_config']['num_labels'] = 2

            elif val == "WiC":
                config_json['preprocess_config']['input_schema'] = "idx:str:1,sent1:str:1,sent2:str:1,label:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config']['second_sequence'] = "sent2"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "True,False"
                config_json['model_config']['num_labels'] = 2

            elif val == "WSC":
                config_json['preprocess_config']['input_schema'] = "idx:str:1,sent1:str:1,label:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "True,False"
                config_json['model_config']['num_labels'] = 2

            elif val == "COPA":
                config_json['preprocess_config']['input_schema'] = "idx:str:1,sent1:str:1,sent2:str:1,label:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config']['second_sequence'] = "sent2"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "0,1"
                config_json['model_config']['num_labels'] = 2

            elif val == "CB":
                config_json['preprocess_config']['input_schema'] = "idx:str:1,sent1:str:1,sent2:str:1,label:str:1"
                config_json['preprocess_config']['first_sequence'] = "sent1"
                config_json['preprocess_config']['second_sequence'] = "sent2"
                config_json['preprocess_config'][
                    'label_enumerate_values'] = "neutral,entailment,contradiction"
                config_json['model_config']['num_labels'] = 3

    config = Config(mode="train_and_evaluate_on_the_fly", config_json=config_json)
    app = Application(user_defined_config=config)
    train_reader = CSVReader(input_glob=app.train_input_fp,
                             is_training=True,
                             input_schema=app.input_schema,
                             batch_size=app.train_batch_size)

    app.run_train(reader=train_reader)

    eval_reader = CSVReader(input_glob=app.eval_input_fp,
                            is_training=False,
                            input_schema=app.input_schema,
                            batch_size=app.eval_batch_size)
    ckpts = set()
    with tf.gfile.GFile(os.path.join(app.config.model_dir, "checkpoint"), mode='r') as reader:
        for line in reader:
            line = line.strip()
            line = line.replace("oss://", "")
            ckpts.add(int(line.split(":")[1].strip().replace("\"", "").split("/")[-1].replace("model.ckpt-", "")))

    if _APP_FLAGS.task_name != "CoLA":
        # early stopping
        best_acc = 0
        best_ckpt = None
        for ckpt in sorted(ckpts):
            checkpoint_path = os.path.join(app.config.model_dir, "model.ckpt-" + str(ckpt))
            tf.logging.info("checkpoint_path is {}".format(checkpoint_path))
            eval_results = app.run_evaluate(reader=eval_reader, checkpoint_path=checkpoint_path)
            acc = eval_results['py_accuracy']
            if acc > best_acc:
                best_ckpt = ckpt
                best_acc = acc
        tf.logging.info("best ckpt {}, best acc {}".format(best_ckpt, best_acc))

    else:
        #early stopping
        best_matthew_corr = 0
        best_ckpt = None
        for ckpt in sorted(ckpts):
            checkpoint_path = os.path.join(app.config.model_dir, "model.ckpt-"+str(ckpt))
            tf.logging.info("checkpoint_path is {}".format(checkpoint_path))
            eval_results = app.run_evaluate(reader=eval_reader, checkpoint_path=checkpoint_path)
            matthew_corr = eval_results['matthew_corr']
            if matthew_corr > best_matthew_corr:
                best_ckpt = ckpt
                best_matthew_corr = matthew_corr
        tf.logging.info("best ckpt {}, best matthew_corr {}".format(best_ckpt, best_matthew_corr))
