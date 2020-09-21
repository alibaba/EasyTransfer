import json
import six
import unicodedata
import os
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("task_name", None, "task name")
flags.DEFINE_string("task_data_dir", None, "task_data_dir")

def process_text(inputs, remove_space=True, lower=True):
    """preprocess data by removing extra space and normalize data."""
    outputs = inputs
    if remove_space:
        outputs = " ".join(inputs.strip().split())

    if six.PY2 and isinstance(outputs, str):
        try:
            outputs = six.ensure_text(outputs, "utf-8")
        except UnicodeDecodeError:
            outputs = six.ensure_text(outputs, "latin-1")

    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with tf.gfile.Open(input_file, "r") as f:
            lines = []
            for line in f:
                line = line.strip()
                json_line = json.loads(line)
                lines.append(json_line)
            return lines


class AFQMC_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        examples = []
        for (i, line) in enumerate(lines):
            text_a = process_text(line['sentence1'])
            text_b = process_text(line['sentence2'])
            label = process_text(line['label'])
            examples.append(
                InputExample(guid=str(i), text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "dev.json"))
        examples = []
        for (i, line) in enumerate(lines):
            text_a = process_text(line['sentence1'])
            text_b = process_text(line['sentence2'])
            label = process_text(line['label'])
            examples.append(
                InputExample(guid=str(i), text_a=text_a, text_b=text_b, label=label))
        return examples


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "test.json"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = str(line['id'])
            text_a = process_text(line['sentence1'])
            text_b = process_text(line['sentence2'])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples

    def get_labels(self, data_dir):
        return ["0", "1"]

class CMNLI_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        examples = []
        for (i, line) in enumerate(lines):
            text_a = process_text(line['sentence1'])
            text_b = process_text(line['sentence2'])
            label = process_text(line['label'])
            examples.append(
                InputExample(guid=str(i), text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "dev.json"))
        examples = []
        for (i, line) in enumerate(lines):
            text_a = process_text(line['sentence1'])
            text_b = process_text(line['sentence2'])
            label = process_text(line['label'])
            examples.append(
                InputExample(guid=str(i), text_a=text_a, text_b=text_b, label=label))
        return examples


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "test.json"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = str(line['id'])
            text_a = process_text(line['sentence1'])
            text_b = process_text(line['sentence2'])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples

    def get_labels(self, data_dir):
        return ["neutral", "entailment", "contradiction"]


class CSL_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        examples = []
        for (i, line) in enumerate(lines):
            text_a = process_text(" ".join(line['keyword']))
            text_b = process_text(line['abst'])
            label = process_text(line['label'])
            examples.append(
                InputExample(guid=str(i), text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "dev.json"))
        examples = []
        for (i, line) in enumerate(lines):
            text_a = process_text(" ".join(line['keyword']))
            text_b = process_text(line['abst'])
            label = process_text(line['label'])
            examples.append(
                InputExample(guid=str(i), text_a=text_a, text_b=text_b, label=label))
        return examples


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "test.json"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = str(line['id'])
            text_a = process_text(" ".join(line['keyword']))
            text_b = process_text(line['abst'])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples

    def get_labels(self, data_dir):
        return ["0", "1"]

class TNEWS_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        examples = []
        for (i, line) in enumerate(lines):
            text_a = process_text(line['sentence']) + " " + process_text(line['keywords'])
            label = process_text(line['label'])
            examples.append(
                InputExample(guid=str(i), text_a=text_a, text_b=None, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "dev.json"))
        examples = []
        for (i, line) in enumerate(lines):
            text_a = process_text(line['sentence']) + " " + process_text(line['keywords'])
            label = process_text(line['label'])
            examples.append(
                InputExample(guid=str(i), text_a=text_a, text_b=None, label=label))
        return examples


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "test.json"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = str(line['id'])
            text_a = process_text(line['sentence']) + " " + process_text(line['keywords'])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None))
        return examples

    def get_labels(self, data_dir):
        return ["100", "101", "102", "103", "104", "106", "107", "108", "109", "110", "112", "113", "114", "115", "116"]


class IFLYTEK_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        examples = []
        for (i, line) in enumerate(lines):
            text_a = process_text(line['sentence'])
            label = process_text(line['label'])
            examples.append(
                InputExample(guid=str(i), text_a=text_a, text_b=None, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "dev.json"))
        examples = []
        for (i, line) in enumerate(lines):
            text_a = process_text(line['sentence'])
            label = process_text(line['label'])
            examples.append(
                InputExample(guid=str(i), text_a=text_a, text_b=None, label=label))
        return examples


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "test.json"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = str(line['id'])
            text_a = process_text(line['sentence'])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None))
        return examples

    def get_labels(self, data_dir):
        return [str(idx) for idx in range(119)]

def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False

def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

class WSC_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "train-%d" % (i)
            span1_text = process_text(line['target']['span1_text'])
            span2_text = process_text(line['target']['span2_text'])
            span1_index = line['target']['span1_index']
            span2_index = line['target']['span2_index']
            text = process_text(line['text'])

            text_list = _tokenize_chinese_chars(text).split()
            if span2_index > span1_index:
                text_list.insert(span1_index, "_")
                text_list.insert(span1_index + len(span1_text) + 1, "_")
                text_list.insert(span2_index + 2, "[")
                text_list.insert(span2_index + len(span2_text) + 2 + 1, "]")
            else:
                text_list.insert(span2_index, "[")
                text_list.insert(span2_index + len(span2_text) + 1, "]")
                text_list.insert(span1_index + 2, "_")
                text_list.insert(span1_index + len(span1_text) + 2 + 1, "_")

            text_a = " ".join(text_list)

            label = line['label']
            if label == "true":
                label = "True"
            elif label == "false":
                label = "False"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "val.json"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "dev-%d" % (i)
            span1_text = process_text(line['target']['span1_text'])
            span2_text = process_text(line['target']['span2_text'])
            span1_index = line['target']['span1_index']
            span2_index = line['target']['span2_index']
            text = process_text(line['text'])

            text_list = _tokenize_chinese_chars(text).split()
            if span2_index > span1_index:
                text_list.insert(span1_index, "_")
                text_list.insert(span1_index + len(span1_text) + 1, "_")
                text_list.insert(span2_index + 2, "[")
                text_list.insert(span2_index + len(span2_text) + 2 + 1, "]")
            else:
                text_list.insert(span2_index, "[")
                text_list.insert(span2_index + len(span2_text) + 1, "]")
                text_list.insert(span1_index + 2, "_")
                text_list.insert(span1_index + len(span1_text) + 2 + 1, "_")

            text_a = " ".join(text_list)

            label = line['label']
            if label == "true":
                label = "True"
            elif label == "false":
                label = "False"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_json(os.path.join(data_dir, "test.json"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = str(line['id'])
            span1_text = process_text(line['target']['span1_text'])
            span2_text = process_text(line['target']['span2_text'])
            span1_index = line['target']['span1_index']
            span2_index = line['target']['span2_index']
            text = process_text(line['text'])

            text_list = _tokenize_chinese_chars(text).split()
            if span2_index > span1_index:
                text_list.insert(span1_index, "_")
                text_list.insert(span1_index + len(span1_text) + 1, "_")
                text_list.insert(span2_index + 2, "[")
                text_list.insert(span2_index + len(span2_text) + 2 + 1, "]")
            else:
                text_list.insert(span2_index, "[")
                text_list.insert(span2_index + len(span2_text) + 1, "]")
                text_list.insert(span1_index + 2, "_")
                text_list.insert(span1_index + len(span1_text) + 2 + 1, "_")

            text_a = " ".join(text_list)

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None))
        return examples

    def get_labels(self, data_dir):
        return ["True", "False"]


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    if not os.path.exists(FLAGS.task_data_dir):
        tf.gfile.MakeDirs(FLAGS.task_data_dir)

    processors = {
        "AFQMC": AFQMC_Processor,
        "TNEWS": TNEWS_Processor,
        "WSC": WSC_Processor,
        "IFLYTEK":IFLYTEK_Processor,
        "CMNLI": CMNLI_Processor,
        "CSL": CSL_Processor

    }

    processor = processors[FLAGS.task_name]()
    train_examples = processor.get_train_examples(FLAGS.task_data_dir)
    dev_examples = processor.get_dev_examples(FLAGS.task_data_dir)
    test_examples = processor.get_test_examples(FLAGS.task_data_dir)
    tf.logging.info("total train examples is {}".format(len(train_examples)))
    tf.logging.info("total dev examples is {}".format(len(dev_examples)))
    tf.logging.info("total test examples is {}".format(len(test_examples)))
    tf.logging.info("labels {}".format(processor.get_labels(FLAGS.task_data_dir)))
    with open(os.path.join(FLAGS.task_data_dir, "train.csv"), "w") as train_writer, \
            open(os.path.join(FLAGS.task_data_dir, "dev.csv"), "w") as dev_writer, \
        open(os.path.join(FLAGS.task_data_dir, "test.csv"), "w") as test_writer:

        for example in train_examples:
            if example.text_b is not None:
                train_writer.write(example.guid + '\t' +example.text_a + "\t" + example.text_b + "\t" + example.label + "\n")
            else:
                train_writer.write(example.guid + '\t' +example.text_a + "\t" + example.label + "\n")

        for example in dev_examples:
            if example.text_b is not None:
                dev_writer.write(example.guid + '\t' +example.text_a + "\t" + example.text_b + "\t" + example.label + "\n")
            else:
                dev_writer.write(example.guid + '\t' +example.text_a + "\t" + example.label + "\n")

        for example in test_examples:
            if example.text_b is not None:
                test_writer.write(example.guid + '\t' +example.text_a + "\t" + example.text_b  + "\n")
            else:
                test_writer.write(example.guid + '\t' +example.text_a + "\n")


if __name__ == "__main__":
    tf.app.run()
