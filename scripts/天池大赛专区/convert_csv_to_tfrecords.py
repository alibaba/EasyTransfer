import tensorflow as tf
from easytransfer import base_model
from easytransfer.datasets import CSVReader, TFRecordWriter
from easytransfer import preprocessors

class FinetuneSerialization(base_model):
    def __init__(self, **kwargs):
        super(FinetuneSerialization, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.config.tokenizer_name_or_path)
        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
        return input_ids, input_mask, segment_ids, label_ids

    def build_predictions(self, predict_output):
        input_ids, input_mask, segment_ids, label_ids = predict_output
        ret_dict = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "label_id": label_ids,
        }
        return ret_dict


def main(_):
    app = FinetuneSerialization()

    reader = CSVReader(input_glob=app.preprocess_input_fp,
                       is_training=False,
                       input_schema=app.input_schema,
                       batch_size=app.preprocess_batch_size)

    writer = TFRecordWriter(output_glob=app.preprocess_output_fp,
                            output_schema=app.output_schema)

    app.run_preprocess(reader=reader, writer=writer)

if __name__ == "__main__":
    tf.app.run()
