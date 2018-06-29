# Copyright 2018 Deep Topology All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# noinspection PyUnresolvedReferences
import pathmagic
import tensorflow as tf
import modules


class LstmLastHiddenModule(modules.BaseModule):
    """ LSTM network that outputs the last hidden state. """
    def __init__(self, lstm_size, lstm_layers, num_frames, output_dim, scope_id=None):
        """ Initialize LSTM hidden module.
        :param lstm_size: int
        :param lstm_layers: int
        :param num_frames: num_frames x 1
        :param output_dim: int
        :param scope_id: Object
        """
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_dim = output_dim
        self.num_frames = num_frames
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for LstmLastHiddenModule.
        :param inputs: batch_size x max_frames x num_features
        :return: batch_size x output_dim
        """
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    self.lstm_size, forget_bias=1.0)
                for _ in range(self.lstm_layers)

            ])

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputs,
                                           sequence_length=self.num_frames,
                                           dtype=tf.float32)
        # Only output the hidden state at the end.
        return state[-1].h
