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

import collections
import json
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, to_be_excluded=None):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        if name in to_be_excluded:
            tf.logging.info("excluded {}".format(name))
            continue
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def load_npy(path):
    """load numpy array"""
    assert path.endswith(".npy"), "invalid filename {}".format(path)

    with gfile.GFile(path, 'rb') as f:
        restored = np.load(f)
    return restored


def load_arch(path):
    with gfile.GFile(path, 'r') as ips:
        given = json.load(ips)
        given_arch = dict()
        for k, v in given.items():
            sl = k.rfind('/')
            cm = k.rfind(':')
            keyword = k[sl + 1:cm]
            if keyword == 'alphaN':
                Kmax = np.argmax(v)
            else:
                tp = keyword[5:].split("to")
                given_arch[(int(tp[0]), int(tp[1]))] = softmax(np.asarray(v, dtype=np.float32)/0.2)
    print("====== {} cells ======".format(Kmax))
    print("searched op distributions ======>")
    print(given_arch)
    # reserve only two input edges for each node
    num_intermediates=3# hard-coded as this is not exposed to be configurable
    filtered_given_arch = dict()
    for i in range(num_intermediates):
        t = i + 2
        candidate_input_edges = list()
        for s in range(t):
            candidate_input_edges.append((s, np.max(given_arch[(s, t)])))
        sorted_cands = sorted(candidate_input_edges, key=lambda x: x[1])
        sorted_cands.reverse()
        pair = (sorted_cands[0][0], t)
        filtered_given_arch[pair] = np.argmax(given_arch[pair])
        pair = (sorted_cands[1][0], t)
        filtered_given_arch[pair] = np.argmax(given_arch[pair])
    print("derived arch ======>")
    print(filtered_given_arch)
    return Kmax, filtered_given_arch


class SearchResultsSaver(tf.train.SessionRunHook):

    def __init__(self, adabert_global_step, arch_params, ld_embs, output_dir, saved_steps, *args, **kwargs):
        super(SearchResultsSaver, self).__init__(*args, **kwargs)
        self._step = -1
        self._adabert_global_step = adabert_global_step
        self._arch_params = arch_params
        self._ld_embs = ld_embs
        self._output_dir = output_dir
        self.saved_steps = saved_steps

    def before_run(self, run_context):
        self._step += 1

    def after_run(self, run_context, run_values):
        if self._step % self.saved_steps == 0:
            tf.logging.info("Save arch for step %d..." % self._step)
            session = run_context.session
            arch_params_vals = session.run(self._arch_params)
            ld_embs_vals = session.run(self._ld_embs)
            with gfile.GFile(os.path.join(self._output_dir, "arch_%d.json" % self._step), 'w') as f:
                arch = dict()
                for var, var_val in zip(self._arch_params, arch_params_vals):
                    arch[var.name] = var_val.tolist()
                json.dump(arch, f)
            with gfile.GFile(os.path.join(self._output_dir, "wemb_%d.npy" % self._step), 'wb') as f:
                np.save(f, ld_embs_vals[0])
            with gfile.GFile(os.path.join(self._output_dir, "pemb_%d.npy" % self._step), 'wb') as f:
                np.save(f, ld_embs_vals[1])

    def end(self, session):
        #gs_val = session.run(self._global_step)
        arch_params_vals = session.run(self._arch_params)
        ld_embs_vals = session.run(self._ld_embs)

        with gfile.GFile(os.path.join(self._output_dir, "arch.json"), 'w') as f:
            arch = dict()
            for var, var_val in zip(self._arch_params, arch_params_vals):
                arch[var.name] = var_val.tolist()
            json.dump(arch, f)
        with gfile.GFile(os.path.join(self._output_dir, "wemb.npy"), 'wb') as f:
            np.save(f, ld_embs_vals[0])
        with gfile.GFile(os.path.join(self._output_dir, "pemb.npy"), 'wb') as f:
            np.save(f, ld_embs_vals[1])