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
import copy
import unittest

from towhee.runtime.dag_repr import DAGRepr, NodeRepr
from towhee.runtime.data_queue import ColumnType
from towhee.runtime.schema_repr import SchemaRepr


class TestDAGRepr(unittest.TestCase):
    """
    DAGRepr test
    """
    dag_dict = {
        '_input': {
            'inputs': ('a', 'b'),
            'outputs': ('a', 'b'),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': ['op1']
        },
        'op1': {
            'inputs': ('a',),
            'outputs': ('a', 'c'),
            'iter_info': {
                'type': 'flat_map',
                'param': None
            },
            'op_info': {
                'operator': 'towhee/decode',
                'type': 'hub',
                'init_args': ('x',),
                'init_kws': {'y': 'y'},
                'tag': 'main',
            },
            'config': None,
            'next_nodes': ['op2']
        },
        'op2': {
            'inputs': ('a', 'b'),
            'outputs': ('d',),
            'iter_info': {
                'type': 'filter',
                'param': {'filter_columns': 'a'}
            },
            'op_info': {
                'operator': 'towhee.test',
                'type': 'hub',
                'init_args': ('a',),
                'init_kws': {'b': 'b'},
                'tag': '1.1',
            },
            'config': {'parallel': 3},
            'next_nodes': ['_output']
        },
        '_output': {
            'inputs': ('d', 'c'),
            'outputs': ('d', 'c'),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': []
        },
    }

    def test_dag(self):
        dr = DAGRepr.from_dict(self.dag_dict)
        edges = dr.edges
        nodes = dr.nodes
        self.assertEqual(len(edges), 3)
        self.assertEqual(len(nodes), 4)
        for edge in edges.values():
            for schema in edge['schema']:
                self.assertTrue(isinstance(edge['schema'][schema], SchemaRepr))
        for node in nodes:
            self.assertTrue(isinstance(dr.nodes[node], NodeRepr))

    def test_check_input(self):
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test.pop('_input')
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)

    def test_check_output(self):
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test.pop('_output')
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)

    def test_check_schema(self):
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test['op1']['inputs'] = ('x', 'y')
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)

    def test_check_schema_equal(self):
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test['_input']['inputs'] = ('x',)
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)

    def test_check_schema_circle(self):
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test['op2']['next_nodes'] = ['op1', '_output']
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)

    def test_edges(self):
        """
        _input[(a,b)->(a,b)]->op1[(a,)->(c,)]->op2[(a,b)->(d,)]->_output[(d,e)->(d,e)]
                                |------->add_node[(c,)->(e,)]------^
        """
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test['op1']['next_nodes'] = ['add_node', 'op2']
        towhee_dag_test['add_node'] = {
            'inputs': ('c',),
            'outputs': ('e',),
            'iter_info': {
                'type': 'map',
                'param': {}
            },
            'op_info': {
                'operator': 'towhee.test',
                'type': 'hub',
                'init_args': ('x',),
                'init_kws': {'y': 'y'},
                'tag': 'main',
            },
            'config': {},
            'next_nodes': ['_output']
        }
        towhee_dag_test['_output']['inputs'] = ('d', 'e')
        towhee_dag_test['_output']['outputs'] = ('d', 'e')

        dr = DAGRepr.from_dict(towhee_dag_test)
        edges = dr.edges
        nodes = dr.nodes
        self.assertEqual(len(edges), 5)
        self.assertEqual(len(nodes), 5)

        self.assertEqual(edges[0]['data'], [('a', ColumnType.SCALAR), ('b', ColumnType.SCALAR)])
        self.assertEqual(edges[1]['data'], [('c', ColumnType.QUEUE)])
        self.assertEqual(edges[2]['data'], [('a', ColumnType.QUEUE), ('b', ColumnType.SCALAR)])
        self.assertEqual(edges[3]['data'], [('d', ColumnType.SCALAR)])
        self.assertEqual(edges[4]['data'], [('e', ColumnType.SCALAR)])

        self.assertEqual(nodes['_input'].in_edges, [0])
        self.assertEqual(nodes['_input'].out_edges, [0])
        self.assertEqual(nodes['op1'].in_edges, [0])
        self.assertEqual(nodes['op1'].out_edges, [1, 2])
        self.assertEqual(nodes['op2'].in_edges, [2])
        self.assertEqual(nodes['op2'].out_edges, [3])
        self.assertEqual(nodes['add_node'].in_edges, [1])
        self.assertEqual(nodes['add_node'].out_edges, [4])
        self.assertEqual(nodes['_output'].in_edges, [3, 4])
        self.assertEqual(nodes['_output'].out_edges, [3, 4])

    def test_edges_count(self):
        """
        _input[(a,b)-(a,b)]->op1[(a,)-(c,)]->op2[(a,b)-(d,)]->op4[(d,e)-(d,e)]->_output[(a, d,e)-(a, d,e)]
                                |------->op3[(c,)-(e,)]----------^
        """
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test['op1']['next_nodes'] = ['op2', 'op3']
        towhee_dag_test['op2']['next_nodes'] = ['op4']
        towhee_dag_test['op2']['outputs'] = ('a', 'd')
        towhee_dag_test['op3'] = {
            'inputs': ('c',),
            'outputs': ('c', 'e'),
            'iter_info': {
                'type': None,
                'param': {},
            },
            'op_info': {
                'operator': 'towhee.test3',
                'type': 'hub',
                'init_args': (),
                'init_kws': {'b': 'b'},
                'tag': 'main',
            },
            'config': None,
            'next_nodes': ['op4']
        }
        towhee_dag_test['op4'] = {
            'inputs': ('d', 'e'),
            'outputs': ('d', 'e'),
            'iter_info': {
                'type': 'map',
                'param': {},
            },
            'op_info': {
                'operator': 'towhee.test4',
                'type': 'hub',
                'init_args': ('a',),
                'init_kws': {'b': 'b'},
                'tag': 'main',
            },
            'config': None,
            'next_nodes': ['_output']
        }
        towhee_dag_test['_output']['inputs'] = ('a', 'd', 'e')
        towhee_dag_test['_output']['outputs'] = ('a', 'd', 'e')

        dr = DAGRepr.from_dict(towhee_dag_test)
        edges = dr.edges
        nodes = dr.nodes
        self.assertEqual(len(edges), 6)
        self.assertEqual(len(nodes), 6)

        self.assertEqual(edges[0]['schema']['a'].count, 1)
        self.assertEqual(edges[1]['schema']['a'].count, 2)
        self.assertEqual(edges[2]['schema']['a'].count, 2)
        self.assertEqual(edges[3]['schema']['a'].count, 3)
        self.assertEqual(edges[4]['schema']['a'].count, 2)
        self.assertEqual(edges[5]['schema']['a'].count, 3)
