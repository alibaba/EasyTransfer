.. _tutorial-quick-start:

=============================
Quick Start
=============================

1. 数据准备
-----------------------------

假如你有一个用 :code:`\t` 分隔的 :code:`.csv` 文件如下

.. code-block:: python

    上课时学生手机响个不停，老师一怒之下把手机摔了，家长拿发票让老师赔，大家怎么看待这种事？ 	108
    商赢环球股份有限公司关于延期回复上海证券交易所对公司2017年年度报告的事后审核问询函的公告 商赢环球股份有限公司,年度报告,商赢环球,赢环球股份有限公司,事后审核问询函,上海证券交易所	104
    通过中介公司买了二手房，首付都付了，现在卖家不想卖了。怎么处理？ 	106
    2018年去俄罗斯看世界杯得花多少钱？ 莫斯科,贝加尔湖,世界杯,俄罗斯,Hour	112
    剃须刀的个性革新，雷明登天猫定制版新品首发 剃须刀,绝地求生,定制版,战狼2,红海行动,天猫定制版三防,雷明登,维克托	109
    再次证明了“无敌是多么寂寞”——逆天的中国乒乓球队！ 世乒赛,张怡宁,许昕,兵乓球,乒乓球	103
    三农盾SACC-全球首个推出：互联网+区块链+农产品的电商平台 湖南省,区块链,物联网,集中化,SACC三农盾	109
    重做or新英雄？其实重做对暴雪来说同样重要 暴雪,重做,新英雄,黑百合,英雄联盟	116


使用下面的odps命令在odps上创建两张表 clue_tnews_train和clue_tnews_dev

.. code-block:: bash
    
    #!/usr/bin/env bash

    odpscmd="/Users/jerry/Develops/odpscmd/bin/odpscmd"
    config="/Users/jerry/Develops/odpscmd/conf/odps_config_pai_exp_dev_tn_hz.ini"

    ${odpscmd} --config="${config}" -e "drop table if exists clue_tnews_train;"
    ${odpscmd} --config="${config}" -e "create table clue_tnews_train(
        content STRING label STRING) lifecycle 365;"

    ${odpscmd} --config="${config}" -e "tunnel upload train.csv clue_tnews_train -fd='\t' -h=false;"

    ${odpscmd} --config="${config}" -e "drop table if exists clue_tnews_dev;"
    ${odpscmd} --config="${config}" -e "create table clue_tnews_dev(
        content STRING label STRING) lifecycle 365;"

    ${odpscmd} --config="${config}" -e "tunnel upload train.csv clue_tnews_dev -fd='\t' -h=false;"



2. 编辑配置文件
-----------------------------

配置文件详情请见 `<http://gitlab.alibaba-inc.com/PAI-TL/ATP-tutorials/blob/master/tutorials-easytransfer/quick_start/config.json>`_

.. code-block:: json
    
    {
      "preprocess_config": {
        "input_schema": "content:str:1,label:str:1",
        "first_sequence": "content",
        "second_sequence": null,
        "sequence_length":64,
        "label_name": "label",
        "label_enumerate_values": "100,101,102,103,104,106,107,108,109,110,112,113,114,115,116",
        "output_schema": "content,label,predictions"
      },
      "model_config": {
        "pretrain_model_name_or_path": "google-bert-base-zh",
        "num_labels": 15
      },
      "train_config": {
        "train_input_fp": null,
        "train_batch_size": 32,
        "save_steps": 100,
        "num_epochs": 1,
        "model_dir": "oss://tongruntwzjk/test_quick_start",
        "optimizer_config": {
          "optimizer": "adam",
          "weight_decay_ratio": 0,
          "warmup_ratio": 0.1,
          "learning_rate": 1e-5
        },
        "distribution_config": {
          "distribution_strategy": "MirroredStrategy"
        }
      },
      "evaluate_config": {
        "eval_input_fp": null,
        "eval_batch_size": 8
      },
      "predict_config": {
        "predict_checkpoint_path": "oss://tongruntwzjk/test_quick_start/model.ckpt-200",
        "predict_input_fp": null,
        "predict_batch_size": 32,
        "predict_output_fp": null
      },
      "export_config": {
        "input_tensors_schema": "input_ids:int:64,input_mask:int:64,segment_ids:int:64,label_ids:int:1",
        "receiver_tensors_schema": "input_ids:int:64,input_mask:int:64,segment_ids:int:64",
        "export_dir_base": "oss://tongruntwzjk/test_quick_start/cache_export",
        "checkpoint_path": "oss://tongruntwzjk/test_quick_start/model.ckpt-200"
      }
    }


3. 搭建网络
-----------------------------
该前端代码展示了仅用一个文件就可以完成训练/评估/预测/导出这个四个功能，用户在使用的时候可以以此为基础进行修改。

代码详情请见：`<http://gitlab.alibaba-inc.com/PAI-TL/ATP-tutorials/blob/master/tutorials-easytransfer/quick_start/main_finetune.py>`_

.. code-block:: python


    from ez_transfer import base_model
    from ez_transfer import preprocessors
    from ez_transfer import model_zoo
    from ez_transfer.datasets import OdpsTableReader,OdpsTableWriter
    from ez_transfer.losses import softmax_cross_entropy
    from ez_transfer import FLAGS
    from ez_transfer import layers
    from ez_transfer.utils.eval_metrics import PyEvaluator
    import tensorflow as tf

    class Application(base_model):
        def __init__(self, **kwargs):
            super(Application, self).__init__(**kwargs)

        def build_logits(self, features, mode=None):
            preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path)
            model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

            dense = layers.Dense(self.num_labels,
                                 kernel_initializer=layers.get_initializer(0.02),
                                 name='dense')

            input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
            outputs = model([input_ids, input_mask, segment_ids], mode=mode)
            pooled_output = outputs[1]

            logits = dense(pooled_output)
            return logits, label_ids

        def build_loss(self, logits, labels):
            return softmax_cross_entropy(labels, depth=self.config.num_labels, logits=logits)

        def build_eval_metrics(self, logits, labels):
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            info_dict = {
                "predictions": predictions,
                "labels": labels,
            }
            evaluator = PyEvaluator()
            labels = [i for i in range(self.num_labels)]
            metric_dict = evaluator.get_metric_ops(info_dict, labels)
            ret_metrics = evaluator.evaluate(labels)
            tf.summary.scalar("eval accuracy", ret_metrics['py_accuracy'])
            tf.summary.scalar("eval F1 micro score", ret_metrics['py_micro_f1'])
            tf.summary.scalar("eval F1 macro score", ret_metrics['py_macro_f1'])
            return metric_dict

        def build_predictions(self, output):
            logits, _ = output
            predictions = dict()
            predictions["predictions"] = tf.argmax(logits, axis=-1, output_type=tf.int32)
            return predictions

    def train_and_evaluate_on_the_fly():
        app = Application()
        train_reader = OdpsTableReader(input_glob=app.train_input_fp,
                                       is_training=True,
                                       input_schema=app.input_schema,
                                       batch_size=app.train_batch_size)

        eval_reader = OdpsTableReader(input_glob=app.eval_input_fp,
                                      input_schema=app.input_schema,
                                      batch_size=app.eval_batch_size)

        app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)

    def predict_on_the_fly():
        app = Application()
        pred_reader = OdpsTableReader(input_glob=app.predict_input_fp,
                                      input_schema=app.input_schema,
                                      batch_size=app.predict_batch_size)

        pred_writer = OdpsTableWriter(output_glob=app.predict_output_fp,
                                      output_schema=app.output_schema)

        app.run_predict(reader=pred_reader, writer=pred_writer, checkpoint_path=app.predict_checkpoint_path)

    def export():
        app = Application()
        app.export_model()

    if __name__ == "__main__":
        if FLAGS.mode == "train_and_evaluate_on_the_fly":
            train_and_evaluate_on_the_fly()
        elif FLAGS.mode == "predict_on_the_fly":
            predict_on_the_fly()
        elif FLAGS.mode == "export":
            export()
        else:
            raise RuntimeError("Run mode input")


4. 训练/评估
-----------------------------

脚本地址：`<http://gitlab.alibaba-inc.com/PAI-TL/ATP-tutorials/blob/master/tutorials-easytransfer/quick_start/run_finetune.sh>`_

.. code-block:: bash

    #!/usr/bin/env bash
    set -e

    odpscmd="/Users/jerry/Develops/odpscmd/bin/odpscmd"
    config="/Users/jerry/Develops/odpscmd/conf/odps_config_pai_exp_dev_tn_hz.ini"
    ini_path='/Users/jerry/Develops/odpscmd/conf/pai_exp_dev_zjk_tw.ini'
    role_arn_and_host=`cat ${ini_path}`
    oss_dir=tongruntwzjk


    cur_path=/Users/jerry/Develops/ATP-tutorials/tutorials-easytransfer
    cd ${cur_path}
    rm -f proj.tar.gz
    tar -zcf proj.tar.gz quick_start

    job_path='file://'${cur_path}'/proj.tar.gz'

    command="
    pai -name easytransfer_dev
    -project algo_platform_dev
    -Dmode=train_and_evaluate_on_the_fly
    -Dconfig=quick_start/config.json
    -Dtables='odps://pai_exp_dev/tables/clue_tnews_train,odps://pai_exp_dev/tables/clue_tnews_dev'
    -Dscript=${job_path}
    -DentryFile='quick_start/main_finetune.py'
    -Dbuckets=\"oss://tongruntwzjk/?${role_arn_and_host}\"
    -DworkerGPU=2
    -DworkerCount=1
    "

    echo "${command}"
    ${odpscmd} --config="${config}" -e "${command}"
    echo "finish..."


5. 预测
-----------------------------

预测脚本地址：`<http://gitlab.alibaba-inc.com/PAI-TL/ATP-tutorials/blob/master/tutorials-easytransfer/quick_start/run_predict.sh>`_

.. code-block:: bash

    #!/usr/bin/env bash
    set -e

    odpscmd="/Users/jerry/Develops/odpscmd/bin/odpscmd"
    config="/Users/jerry/Develops/odpscmd/conf/odps_config_pai_exp_dev_tn_hz.ini"
    ini_path='/Users/jerry/Develops/odpscmd/conf/pai_exp_dev_zjk_tw.ini'
    role_arn_and_host=`cat ${ini_path}`
    oss_dir=tongruntwzjk


    cur_path=/Users/jerry/Develops/ATP-tutorials/tutorials-easytransfer
    cd ${cur_path}
    rm -f proj.tar.gz
    tar -zcf proj.tar.gz quick_start

    job_path='file://'${cur_path}'/proj.tar.gz'

    command="
    pai -name easytransfer_dev
    -project algo_platform_dev
    -Dmode=predict_on_the_fly
    -Dconfig=quick_start/config.json
    -Dtables='odps://pai_exp_dev/tables/clue_tnews_dev'
    -Doutputs='odps://pai_exp_dev/tables/clue_tnews_pred_out'
    -Dscript=${job_path}
    -DentryFile='quick_start/main_finetune.py'
    -Dbuckets=\"oss://tongruntwzjk/?${role_arn_and_host}\"
    -DworkerGPU=1
    -DworkerCount=1
    "

    echo "${command}"
    ${odpscmd} --config="${config}" -e "${command}"
    echo "finish..."

6. 导出模型
-----------------------------

脚本地址：`<http://gitlab.alibaba-inc.com/PAI-TL/ATP-tutorials/blob/master/tutorials-easytransfer/quick_start/run_export.sh>`_

.. code-block:: bash

    #!/usr/bin/env bash
    set -e

    odpscmd="/Users/jerry/Develops/odpscmd/bin/odpscmd"
    config="/Users/jerry/Develops/odpscmd/conf/odps_config_pai_exp_dev_tn_hz.ini"
    ini_path='/Users/jerry/Develops/odpscmd/conf/pai_exp_dev_zjk_tw.ini'
    role_arn_and_host=`cat ${ini_path}`
    oss_dir=tongruntwzjk


    cur_path=/Users/jerry/Develops/ATP-tutorials/tutorials-easytransfer
    cd ${cur_path}
    rm -f proj.tar.gz
    tar -zcf proj.tar.gz quick_start

    job_path='file://'${cur_path}'/proj.tar.gz'

    command="
    pai -name easytransfer_dev
    -project algo_platform_dev
    -Dmode=export
    -Dconfig=quick_start/config.json
    -Dscript=${job_path}
    -DentryFile='quick_start/main_finetune.py'
    -Dbuckets=\"oss://tongruntwzjk/?${role_arn_and_host}\"
    -DworkerGPU=1
    -DworkerCount=1
    "

    echo "${command}"
    ${odpscmd} --config="${config}" -e "${command}"
    echo "finish..."

