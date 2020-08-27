# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for Benchmarks."""
import os
import subprocess
from backend_test_base import Tf2OnnxBackendTestBase
from common import check_opset_after_tf_version, unittest_main

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,cell-var-from-loop
# pylint: disable=invalid-name
# pylint: enable=invalid-name

class BenchmarksTests(Tf2OnnxBackendTestBase):

    folder = os.path.join(os.path.dirname(__file__), '..', 'benchmarks')

    @check_opset_after_tf_version("2.0", 12, "might need Scan")
    def test_profile_conversion_time(self):
        filename = os.path.join(BenchmarksTests.folder, 'profile_conversion_time.py')
        proc = subprocess.Popen(
            ["python", filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            outs, errs = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            return
        assert b"Profile complete." in outs


if __name__ == '__main__':
    unittest_main()
