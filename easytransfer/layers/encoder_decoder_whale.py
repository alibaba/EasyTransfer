import tensorflow as tf
import whale as wh
from tensorflow.python.layers.base import Layer
from .activations import gelu_new
from .attention import Attention
from .core import dense_dropoutput_layernorm, Dense
from .utils import get_initializer


class Block(Layer):
    def __init__(self, config, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.attention = Attention(config, name="attention")
        # Use gelu_new, then match results
        self.intermediate = Dense(
            units=config.intermediate_size,
            activation=gelu_new,
            kernel_initializer=get_initializer(config.initializer_range),
            name="intermediate/dense")

        self.bert_output = dense_dropoutput_layernorm(config, name="output")

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs
        attention_output = self.attention([hidden_states, attention_mask], training=training)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.bert_output([intermediate_output, attention_output], training=training)
        return layer_output, attention_output

class Encoder(Layer):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.layer = [Block(config, name="layer_{}".format(i)) for i in range(config.num_hidden_layers)]
        self.num_layers = config.num_hidden_layers

    def _stage_call(self, layer_index, all_hidden_states, all_att_outputs, hidden_states, attention_mask, training):
        layer_output, att_output = self.layer[layer_index]([hidden_states, attention_mask], training=training)
        hidden_states = layer_output
        all_hidden_states = all_hidden_states + (hidden_states,)
        all_att_outputs = all_att_outputs + (att_output, )
        return all_hidden_states, all_att_outputs, hidden_states

    def call(self, inputs, training=False):
        tf.logging.info("***************Inside stage to split model**********")
        hidden_states, attention_mask = inputs

        all_hidden_states = ()
        all_att_outputs = ()

        bert_base_layers_count = self.num_layers
        assert len(self.layer) == bert_base_layers_count
        # Use default scope.
        for i in range(0, 3):
            all_hidden_states, all_att_outputs, hidden_states = self._stage_call(i, all_hidden_states, all_att_outputs, hidden_states, attention_mask, training)

        with wh.stage():
            for i in range(3, 6):
                all_hidden_states, all_att_outputs, hidden_states = self._stage_call(i, all_hidden_states, all_att_outputs, hidden_states, attention_mask, training)
            wh.current_scope_as_default()

        with wh.stage():
            for i in range(6, 9):
                all_hidden_states, all_att_outputs, hidden_states = self._stage_call(i, all_hidden_states, all_att_outputs, hidden_states, attention_mask, training)
            wh.current_scope_as_default()

        with wh.stage():
            for i in range(9, 12):
                all_hidden_states, all_att_outputs, hidden_states = self._stage_call(i, all_hidden_states, all_att_outputs, hidden_states, attention_mask, training)
            wh.current_scope_as_default()

        final_outputs = []
        for hidden_states in all_hidden_states:
            final_outputs.append(hidden_states)

        return final_outputs, all_att_outputs
