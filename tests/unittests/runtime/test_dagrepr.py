# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT_ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from towhee.runtime.dag_repr import DAGRepr, NodeRepr


towhee_dag = {
    '_input': {
        'inputs': ('a', 'b'),
        'outputs': ('a', 'b'),
        'fn_type': '_input',
        'iteration': 'map'
    },
    'e433a': {
        'function': 'towhee.decode',
        'init_args': ('a',),
        'init_kws': {'b': 'b'},
        'inputs': 'a',
        'outputs': 'c',
        'fn_type': 'hub',
        'iteration': 'map',
        'config': None,
        'tag': 'main',
        'param': None
    },
    'b1196': {
        'function': 'towhee.test',
        'init_args': ('a',),
        'init_kws': {'b': 'b'},
        'inputs': ('a', 'b'),
        'outputs': 'd',
        'fn_type': 'hub',
        'iteration': 'filter',
        'config': {'parallel': 3},
        'tag': '1.1',
        'param': {'filter_columns': 'a'}
    },
    '_output': {
        'inputs': 'd',
        'outputs': 'd',
        'fn_type': '_output',
        'iteration': 'map'
    },
}
dr = DAGRepr(towhee_dag)


class TestDAGRepr(unittest.TestCase):
    """
    DAGRepr test
    """
    def test_dag(self):
        self.assertEqual(dr.dag, towhee_dag)

    def test_nodes(self):
        nodes = dr.get_nodes()
        self.assertEqual(len(nodes), 4)
        self.assertTrue(isinstance(nodes[0], NodeRepr))
