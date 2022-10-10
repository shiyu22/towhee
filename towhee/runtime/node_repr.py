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

from typing import Dict, Any, Tuple, Union, Callable, Set


def check_keys(info: Dict[str, Any], essentials: Set[str]):
    """
    Check if the src is a valid node dictionary to describe.

    Args:
        info (`Dict[str, Any]`): The info dictionary.
        essentials (`Set[str]`): The essential keys that node dictionary should contain.
    """
    info_keys = set(info.keys())
    if not isinstance(info, dict) or not essentials.issubset(info_keys):
        raise ValueError(f'Node {str(info)} is not valid, lack attr {essentials - info_keys}')


class OperatorRepr:
    """
    OperatorRepr for operator representations.

    Args:
        operator (`Union[str, Callable]`): The operator, such as a callable (lambda, function) or the name of an op on the hub.
        type (`str`): The type of operator, such as 'hub', 'lambda' and 'callable'.
        init_args (`Tuple`): The args to initialize the operator.
        init_kws (`Dict[str, any]`): The kwargs to initialize the operator.
        tag (`str`): The tag for The function, defaults to 'main'.
    """
    def __init__(
        self,
        operator: Union[str, Callable],
        type: str,
        init_args: Tuple,
        init_kws: Dict[str, any],
        tag: str = 'main'
    ):
        self._operator = operator
        self._type = type
        self._init_args = init_args
        self._init_kws = init_kws
        self._tag = tag

    @property
    def operator(self):
        return self._operator

    @property
    def type(self) -> str:
        return self._type

    @property
    def init_args(self) -> Tuple:
        return self._init_args

    @property
    def init_kws(self) -> Dict:
        return self._init_kws

    @property
    def tag(self) -> str:
        return self._tag

    @staticmethod
    def from_dict(op_info: Dict[str, Any]) -> 'OperatorRepr':
        """Return a OperatorRepr from a description dict.

        Args:
            op_info (`Dict[str, Any]`): Dictionary about operator information.

        Returns:
            OperatorRepr object.
        """
        check_keys(op_info, {'operator', 'type', 'init_args', 'init_kws', 'tag'})
        return OperatorRepr(op_info['operator'], op_info['type'], op_info['init_args'], op_info['init_kws'], op_info['tag'])


class IterationRepr:
    """
    IterationRepr for iteration representations.

    Args:
        type (`str`): The type of the iteration, such as 'map', 'flat_map', 'filter', 'time_window'.
        param (`Dict[str, any]`): The parameter for the iteration, defaults to None.
    """
    def __init__(
        self,
        type: str,
        param: Dict[str, any] = None,
    ):
        self._type = type
        self._param = param

    @property
    def type(self) -> str:
        return self._type

    @property
    def param(self) -> Dict[str, any]:
        return self._param

    @staticmethod
    def from_dict(iter_info: Dict[str, Any]) -> 'IterationRepr':
        """Return a IterationRepr from a description dict.

        Args:
            iter_info (`Dict[str, Any]`): Dictionary about iteration information.

        Returns:
            IterationRepr object.
        """
        check_keys(iter_info, {'type', 'param'})
        return IterationRepr(iter_info['type'], iter_info['param'])


class NodeRepr:
    """
    NodeRepr for node representations.

    Args:
        name (`str`): Name of the node, such as '_input', '_output' and id for the node.
        inputs (`Union[str, Tuple]`): Input schema to this node.
        outputs (`Union[str, Tuple]`): Output schema to this node.
        iter_info (`IterationRepr`): The iteration to this node.
        op_info (`OperatorRepr`): The operator to this node.
        config (`Dict[str, any]`): The configuration to this node.
        next_node (`NodeRepr'): The next node, defaults to None.
    """
    def __init__(
        self,
        name: str,
        inputs: Union[str, Tuple],
        outputs: Union[str, Tuple],
        iter_info: IterationRepr,
        op_info: OperatorRepr,
        config: Dict[str, Any],
        next_node: 'NodeRepr' = None
    ):
        self._name = name
        self._inputs = inputs
        self._outputs = outputs
        self._iter_info = iter_info
        self._op_info = op_info
        self._config = config
        self._next_node = next_node

    @property
    def name(self):
        return self._name

    @property
    def inputs(self) -> Union[str, Tuple]:
        return self._inputs

    @property
    def outputs(self) -> Union[str, Tuple]:
        return self._outputs

    @property
    def iter_info(self) -> IterationRepr:
        return self._iter_info

    @property
    def op_info(self) -> OperatorRepr:
        return self._op_info

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def next_node(self) -> 'NodeRepr':
        return self._next_node

    @next_node.setter
    def next_node(self, next_node):
        self._next_node = next_node

    @staticmethod
    def from_dict(name: str, node: Dict[str, Any], next_node: 'NodeRepr' = None) -> 'NodeRepr':
        """Return a NodeRepr from a description dict.

        Args:
            name (`str`): Name of the node, such as 'input', 'output' or id for the node.
            node (`Dict[str, Any]`): Dictionary about node info from dag.
            next_node (`NodeRepr`): The next node to current node, defaults to None.

        Returns:
            NodeRepr object.
        """
        if name in ['_input', '_output']:
            check_keys(node, {'inputs', 'outputs', 'iter_info'})
            iter_repr = IterationRepr.from_dict(node['iter_info'])
            return NodeRepr(name, node['inputs'], node['outputs'], iter_repr, None, None, next_node)
        else:
            check_keys(node, {'inputs', 'outputs', 'iter_info', 'op_info', 'config'})
            iter_repr = IterationRepr.from_dict(node['iter_info'])
            op_repr = OperatorRepr.from_dict(node['op_info'])
            return NodeRepr(name, node['inputs'], node['outputs'], iter_repr, op_repr, node['config'], next_node)
