# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for huggingface jax transformers.

tested with tf-2.6.0rc1, transformers-4.8.2

"""

# pylint: disable=missing-docstring,invalid-name,unused-argument
# pylint: disable=bad-classmethod-argument,wrong-import-position
# pylint: disable=import-outside-toplevel

import os
import time
import unittest
import zipfile
from unittest.case import skip

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as rt
import tensorflow as tf
import tf2onnx
from jax.experimental import jax2tf

compare_perf = True
time_to_run = 10
time_step = 10


class TestTransformers(unittest.TestCase):

    def setUp(self):
        tf.compat.v1.reset_default_graph()

    @classmethod
    def assertAllClose(cls, expected, actual, **kwargs):
        np.testing.assert_allclose(expected, actual, **kwargs)

    def run_onnxruntime(self, model_path, input_dict, output_names):
        """Run test against onnxruntime backend."""
        providers = ['CPUExecutionProvider']
        if rt.get_device() == "GPU":
            gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
            if gpus is None or len(gpus) > 1:
                providers = ['CUDAExecutionProvider']

        opt = rt.SessionOptions()
        m = rt.InferenceSession(
            model_path, sess_options=opt, providers=providers)
        results = m.run(output_names, input_dict)
        if compare_perf:
            n = 0
            time_start = time.time()
            time_stop = time_start + time_to_run
            while time.time() < time_stop:
                for _ in range(time_step):
                    _ = m.run(output_names, input_dict)
                n += time_step
            time_end = time.time()
            val = (time_end - time_start) / n
            print(f'= avg ort name={self.name}, time={val}, n={n}')
        return results

    def run_tf(self, func, input_dict):
        inputs = [tf.convert_to_tensor(v, name=k) for k, v in input_dict.items()]
        results = func(*inputs)
        if compare_perf:
            n = 0
            time_start = time.time()
            time_stop = time_start + time_to_run
            while time.time() < time_stop:
                for _ in range(time_step):
                    _ = func(*inputs)
                n += time_step
            time_end = time.time()
            val = (time_end - time_start) / n
            print(f'= avg tf name={self.name}, time={val}, n={n}')
        return results

    def run_jax(self, model, inputs):
        pred = model(**inputs)
        if compare_perf:
            n = 0
            time_start = time.time()
            time_stop = time_start + time_to_run
            while time.time() < time_stop:
                for _ in range(time_step):
                    _ = model(**inputs)
                n += time_step
            time_stop = time.time()
            val = (time_stop - time_start) / n
            print(f'= avg jax name={self.name}, time={val}, n={n}')
        return pred

    def run_test(self, model, input_dict, rtol=1e-2, atol=1e-4, outputs=None, large=True, extra_input=None):

        self.name = self._testMethodName.replace("test_", "")
        print(f"==== {self.name}")
        dst = os.path.join("/tmp", "test_transformers", self.name)
        os.makedirs(dst, exist_ok=True)

        spec, onnx_inputs = self.spec_and_pad(input_dict)

        # run jax model
        print("= running jax")
        jax_results = self.run_jax(model, input_dict)
        if not outputs:
            # no outputs given ... take all
            outputs = list(jax_results.keys())

        # filter outputs
        jax_results = [np.asarray(v)
                       for k, v in jax_results.items() if k in outputs]

        # input tensors to numpy
        input_dict = {k: np.asarray(v) for k, v in onnx_inputs.items()}

        func = tf.function(jax2tf.convert(model, enable_xla=False), autograph=False, jit_compile=True, input_signature=spec)

        print("= running tf")
        tf_results = self.run_tf(func, input_dict)
        tf_results = [v.numpy() for k, v in tf_results.items() if k in outputs]
        self.assertAllClose(jax_results, tf_results, rtol=rtol, atol=atol)

        model_path = os.path.join(dst, self.name)
        if not large:
            model_path = model_path + ".onnx"
        print("= convert")
        time_start = time.time()
        model = None
        _, _ = tf2onnx.convert.from_function(
            func, input_signature=spec, opset=13, large_model=large, output_path=model_path)
        time_stop = time.time()
        print(f"= convertsion took {time_stop - time_start}")

        if large:
            # need to unpack the zip for run_onnxruntime()
            with zipfile.ZipFile(model_path, 'r') as z:
                z.extractall(os.path.dirname(model_path))
            model_path = os.path.join(os.path.dirname(
                model_path), "__MODEL_PROTO.onnx")

        print("= running ort")
        if extra_input:
            onnx_inputs.update(extra_input)
        onnx_results = self.run_onnxruntime(model_path, input_dict, outputs)
        self.assertAllClose(jax_results, onnx_results, rtol=rtol, atol=atol)

    def spec_and_pad(self, input_dict, max_length=None, batchdim=None):
        spec = []
        new_dict = {}
        for k, v in input_dict.items():
            shape = v.shape
            # FIXME: how to handle dynamic inputs ?
            if False and len(shape) == 2:
                if not max_length:
                    shape = [batchdim, None]
                else:
                    shape = [batchdim, max_length]
            spec.append(tf.TensorSpec(shape, dtype=v.dtype, name=k))
            if max_length:
                l = len(v[0])
                v = tf.pad(v, [[0, 0], [0, max_length-l]])
            new_dict[k] = v
        return tuple(spec), new_dict

    # BERT

    def _test_JaxBertModel(self, size):
        from transformers import BertTokenizer, FlaxBertModel
        tokenizer = BertTokenizer.from_pretrained(size)
        model = FlaxBertModel.from_pretrained(size)
        inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')
        outputs = ["last_hidden_state"]
        self.run_test(model, inputs, outputs=outputs, large=True, rtol=1e-5)

    @unittest.skip("delivers wrong results for tf and onnx")
    def test_JaxBertModel(self):
        self._test_JaxBertModel('bert-base-uncased')

    def _test_JaxBertSquad(self, size):
        from transformers import BertTokenizer, FlaxBertForQuestionAnswering
        tokenizer = BertTokenizer.from_pretrained(size)
        model = FlaxBertForQuestionAnswering.from_pretrained(size)
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        inputs = tokenizer(question, text, return_tensors='jax')
        self.run_test(model, inputs, large=True, rtol=1e-5)

    def test_JaxBertSquad(self):
        self._test_JaxBertSquad('bert-base-uncased')

    # BART

    def _test_JaxBartModel(self, size):
        from transformers import BartTokenizer, FlaxBartModel
        tokenizer = BartTokenizer.from_pretrained(size)
        model = FlaxBartModel.from_pretrained(size)
        inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')
        outputs = ["last_hidden_state"]
        self.run_test(model, inputs, outputs=outputs, large=True, rtol=1e-5)

    @unittest.skip("crashes in grappler, see #1601 for stack trace")
    def test_JaxBartModel(self):
        self._test_JaxBartModel('facebook/bart-base')


if __name__ == "__main__":
    unittest.main()
