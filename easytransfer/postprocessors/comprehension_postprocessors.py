# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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

import six
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl
import collections
import math
import numpy as np
import tensorflow as tf
from easytransfer.engines.distribution import Process
from easytransfer.preprocessors.tokenization import BasicTokenizer, convert_to_unicode


class ComprehensionPostprocessor(Process):
    """ Postprocessor for text comprehension, search and improve the answer span

    """
    def __init__(self,
                 output_schema,
                 n_best_size=20,
                 max_answer_length=30,
                 thread_num=None,
                 input_queue=None,
                 output_queue=None,
                 job_name='ComprehentionPostprocessor'):

        super(ComprehensionPostprocessor, self).__init__(
            job_name, thread_num, input_queue, output_queue, batch_size=1)
        self.output_schema = output_schema
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length

    @staticmethod
    def _get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    @staticmethod
    def _get_final_text(pred_text, orig_text, do_lower_case):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heruistic between
        # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            tf.logging.info("Couldn't map start position")
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            tf.logging.info("Couldn't map end position")
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

    @staticmethod
    def _compute_softmax(scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    def process(self, in_data):
        """ Post-process the model outputs

        Args:
            in_data (`dict`): a dict of model outputs
        Returns:
            ret (`dict`): a dict of post-processed model outputs
        """
        ret = dict()
        for output_col_name in self.output_schema.split(","):
            if output_col_name in in_data:
                ret[output_col_name] = in_data[output_col_name]
        if "predictions" not in self.output_schema.split(",") and \
                "probabilities" not in self.output_schema.split(","):
            return ret

        prediction_list = []
        probability_list = []
        for idx in range(len(in_data["start_logits"])):
            start_logits = in_data["start_logits"][idx]
            end_logits = in_data["end_logits"][idx]
            tokens = [convert_to_unicode(t) for t in in_data["tokens"][idx]]
            doc_tokens = [convert_to_unicode(t) for t in in_data["doc_tokens"][idx]]
            token_to_orig_map = {int(key): val for key, val
                                 in in_data["token_to_orig_map"][idx].items()}
            start_indexes = self._get_best_indexes(start_logits, self.n_best_size)
            end_indexes = self._get_best_indexes(end_logits, self.n_best_size)

            _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "PrelimPrediction",
                ["start_index", "end_index", "start_logit", "end_logit"])
            prelim_predictions = []
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(tokens):
                        continue
                    if end_index >= len(tokens):
                        continue
                    if start_index not in token_to_orig_map:
                        continue
                    if end_index not in token_to_orig_map:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > self.max_answer_length:
                        continue
                    prelim_predictions.append(
                      _PrelimPrediction(
                          start_index=start_index,
                          end_index=end_index,
                          start_logit=start_logits[start_index],
                          end_logit=end_logits[end_index]))

            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= self.n_best_size:
                    break
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = tokens[pred.start_index:(pred.end_index + 1)]
                    orig_doc_start = token_to_orig_map[pred.start_index]
                    orig_doc_end = token_to_orig_map[pred.end_index]
                    orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = self._get_final_text(tok_text, orig_text, True)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit))

            # if we didn't inlude the empty option in the n-best, inlcude it
            if "" not in seen_predictions:
                null_start_logit = start_logits[0]
                null_end_logit = end_logits[0]
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit,
                        end_logit=null_end_logit))
            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = self._compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = dict()
                output["text"] = entry.text
                output["probability"] = str(probs[i])
                output["start_logit"] = str(entry.start_logit)
                output["end_logit"] = str(entry.end_logit)
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            prediction_list.append(nbest_json[0]["text"].encode("utf8"))
            probability_list.append(nbest_json[0]["probability"])

        if "predictions" in self.output_schema.split(","):
            ret["predictions"] = np.array(prediction_list)

        if "probabilities" in self.output_schema.split(","):
            ret["probabilities"] = np.array(probability_list)

        return ret

if __name__ == "__main__":
    with open("tmp.in_data.pkl") as f:
        in_data = pkl.load(f)
    obj = ComprehensionPostprocessor(output_schema="predictions,example_id,answer")
    obj.process(in_data)
