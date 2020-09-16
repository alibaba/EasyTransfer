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


from tensorflow.python.layers.base import Layer
from .activations import gelu_new
from .attention import Attention, CrossAttention
from .core import dense_dropoutput_layernorm, Dense
from .utils import get_initializer

class EncoderBlock(Layer):
    def __init__(self, config, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
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


class DecoderBlock(Layer):
    def __init__(self, config, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.attention = Attention(config, name="decoder_attention")
        self.cross_attention = CrossAttention(config, name="decoder_cross_attention")
        # Use gelu_new, then match results
        self.intermediate = Dense(
            units=config.intermediate_size,
            activation=gelu_new,
            kernel_initializer=get_initializer(config.initializer_range),
            name="intermediate/dense")

        self.output_1 = dense_dropoutput_layernorm(config, name="output_1")

        self.output_2 = dense_dropoutput_layernorm(config, name="output_2")

    def call(self, inputs, training=False):
        hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask = inputs
        attention_output = self.attention([hidden_states, attention_mask], training=training)

        cross_attention_output = self.cross_attention([hidden_states, encoder_hidden_states,
                                                       encoder_attention_mask])

        attention_output = self.output_1([attention_output, cross_attention_output], training=training)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output_2([intermediate_output, attention_output], training=training)
        return layer_output, attention_output

class Encoder(Layer):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.layer = [EncoderBlock(config, name="layer_{}".format(i)) for i in range(config.num_hidden_layers)]

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs

        all_hidden_states = ()
        all_att_outputs = ()
        for i, layer_module in enumerate(self.layer):
            layer_output, att_output = layer_module([hidden_states, attention_mask], training=training)
            hidden_states = layer_output
            all_hidden_states = all_hidden_states + (hidden_states,)
            all_att_outputs = all_att_outputs + (att_output,)

        final_outputs = []
        for hidden_states in all_hidden_states:
            final_outputs.append(hidden_states)

        return final_outputs, all_att_outputs


class Decoder(Layer):
    def __init__(self, config, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.layer = [DecoderBlock(config, name="decoder_layer_{}".format(i)) for i in range(config.num_hidden_layers)]

    def call(self, inputs, training=False):
        hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask = inputs

        all_hidden_states = ()
        all_att_outputs = ()
        for i, layer_module in enumerate(self.layer):
            layer_output, att_output = layer_module([hidden_states,
                                                     encoder_hidden_states,
                                                     attention_mask,
                                                     encoder_attention_mask
                                                     ], training=training)
            hidden_states = layer_output
            all_hidden_states = all_hidden_states + (hidden_states,)
            all_att_outputs = all_att_outputs + (att_output,)

        final_outputs = []
        for hidden_states in all_hidden_states:
            final_outputs.append(hidden_states)

        return final_outputs, all_att_outputs
