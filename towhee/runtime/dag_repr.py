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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Any, Set, List, Tuple

from towhee.runtime.node_repr import NodeRepr
from towhee.runtime.schema_repr import SchemaRepr


def check_set(base: Set[str], parent: Set[str], equal: bool = False):
    """
    Check if the src is a valid input and output.

    Args:
        base (`Dict[str, Any]`): The base set will be check.
        parent (`Set[str]`): The parents set to check.
        equal (`bool`): Whether to check if two sets are equal

    Returns:
        (`bool | raise`)
            Return `True` if it is valid, else raise exception.
    """
    if equal and base != parent:
        raise ValueError(f'The DAG Nodes inputs {str(base)} is not equal to the output: {parent}')
    elif not base.issubset(parent):
        raise ValueError(f'The DAG Nodes inputs {str(base)} is not valid, which is not declared: {base - parent}.')


class DAGRepr:
    """
    A `DAGRepr` represents a complete DAG.

    Args:
        nodes (`Dict[str, NodeRepr]`): All nodes in the dag, which start with _input and end with _output node.
        edges (`Dict[str, List]`): The edges about data queue schema.
    """
    def __init__(self, nodes: Dict[str, NodeRepr], edges: Dict[int, Dict]):
        self._nodes = nodes
        self._edges = edges

    @property
    def nodes(self) -> Dict:
        return self._nodes

    @property
    def edges(self) -> Dict:
        return self._edges

    @staticmethod
    def check_nodes(nodes: Dict[str, NodeRepr]):
        top_sort = DAGRepr.get_top_sort(nodes)
        if len(top_sort) != len(nodes):
            raise ValueError('The DAG is not valid, it has a circle.')
        if top_sort[0] != '_input':
            raise ValueError('The DAG is not valid, it does not started with `_input`.')
        if top_sort[-1] != '_output':
            raise ValueError('The DAG is not valid, it does not ended with `_output`.')

        all_inputs = DAGRepr.get_all_inputs(nodes, top_sort)
        for name in nodes:
            check_set(set(nodes[name].inputs), set(all_inputs[name]))
        check_set(set(nodes['_input'].inputs), set(nodes['_input'].inputs), True)
        check_set(set(nodes['_output'].inputs), set(nodes['_output'].outputs), True)

    @staticmethod
    def get_top_sort(nodes: Dict[str, NodeRepr]):
        graph = dict((name, nodes[name].next_nodes) for name in nodes)
        result = []
        while True:
            temp_list = {j for i in graph.values() for j in i}
            node = [x for x in (list(graph.keys())) if x not in temp_list]
            if not node:
                break
            result.append(node[0])
            del graph[node[0]]
        return result

    @staticmethod
    def get_all_inputs(nodes: Dict[str, NodeRepr], top_sort: list):
        if '_input' not in nodes.keys():
            raise ValueError('The DAG Nodes is not valid, it does not have key `_input`.')
        all_inputs = dict((name, nodes['_input'].inputs) for name in nodes)
        for name in top_sort[1:]:
            for n in nodes[name].next_nodes:
                all_inputs[n] = all_inputs[n] + nodes[name].outputs + all_inputs[name]
        return all_inputs

    @staticmethod
    def dfs_used_schema(nodes: Dict[str, NodeRepr], name: str, ahead_edge: Set):
        ahead_schema = ahead_edge.copy()
        used_schema = set()
        stack = [name]
        visited = [name]
        while stack:
            n = stack.pop()
            common_schema = set(nodes[n].inputs) & ahead_schema
            for x in common_schema:
                ahead_schema.remove(x)
                used_schema.add(x)
            if len(ahead_schema) == 0:
                break
            next_nodes = nodes[n].next_nodes
            for i in next_nodes[::-1]:
                if i not in visited:
                    stack.append(i)
                    visited.append(i)
        return used_schema

    @staticmethod
    def get_edge_from_schema(schema: Tuple, outputs: Tuple, iter_type: str, ahead_edges: List) -> Dict:
        if ahead_edges is None:
            edge_schemas = dict((d, SchemaRepr.from_dag(d, iter_type)) for d in schema)
            edge = {'schema': edge_schemas, 'data': [(s, t.type) for s, t in edge_schemas.items()]}
            return edge
        ahead_schemas = ahead_edges[0]['schema']
        for ahead in ahead_edges[1:]:
            for a in ahead['schema']:
                if a not in ahead_schemas or ahead_schemas[a].count < ahead['schema'][a].count:
                    ahead_schemas[a] = ahead['schema'][a]

        edge_schemas = {}
        for d in schema:
            if d not in outputs:
                iter_type = 'map'
            if d not in ahead_schemas:
                edge_schemas[d] = SchemaRepr.from_dag(d, iter_type, None, 1)
            else:
                count = ahead_schemas[d].count
                if d in outputs:
                    count = count + 1
                edge_schemas[d] = SchemaRepr.from_dag(d, iter_type, ahead_schemas[d].type, count)
        edge = {'schema': edge_schemas, 'data': [(s, t.type) for s, t in edge_schemas.items()]}
        return edge

    @staticmethod
    def set_edges(nodes: Dict[str, NodeRepr]):
        out_id = 0
        edges = {out_id: DAGRepr.get_edge_from_schema(nodes['_input'].inputs, nodes['_input'].outputs, nodes['_input'].iter_info.type, None)}
        nodes['_input'].in_edges = [out_id]

        top_sort = DAGRepr.get_top_sort(nodes)
        input_next = nodes['_input'].next_nodes
        if len(nodes['_input'].next_nodes) == 1:
            nodes['_input'].out_edges = [out_id]
            nodes[input_next[0]].in_edges = [out_id]
            top_sort = top_sort[1:]
        for name in top_sort:
            ahead_schema = set(nodes[name].outputs)
            for i in nodes[name].in_edges:
                ahead_schema = ahead_schema | edges[i]['schema'].keys()
            for next_name in nodes[name].next_nodes:
                out_id += 1
                out_schema = DAGRepr.dfs_used_schema(nodes, next_name, ahead_schema)
                edges[out_id] = DAGRepr.get_edge_from_schema(tuple(out_schema), nodes[name].outputs, nodes[name].iter_info.type,
                                                             [edges[e] for e in nodes[name].in_edges])

                if nodes[next_name].in_edges is None:
                    nodes[next_name].in_edges = [out_id]
                else:
                    nodes[next_name].in_edges.append(out_id)

                if nodes[name].out_edges is None:
                    nodes[name].out_edges = [out_id]
                else:
                    nodes[name].out_edges.append(out_id)

        nodes['_output'].out_edges = nodes['_output'].in_edges
        return nodes, edges

    @staticmethod
    def from_dict(dag: Dict[str, Any]):
        nodes = dict((name, NodeRepr.from_dict(name, dag[name])) for name in dag)
        DAGRepr.check_nodes(nodes)
        dag_nodes, schema_edges = DAGRepr.set_edges(nodes)
        return DAGRepr(dag_nodes, schema_edges)
