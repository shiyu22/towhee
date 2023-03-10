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

from towhee.runtime.operator_manager import OperatorAction
from towhee.runtime.factory import _OperatorWrapper
from towhee.runtime.constants import MapConst, ConcatConst, OPName


def get_nop_node_dict(input_schema, output_schema):
    fn_action = operator_to_action(OPName.NOP)
    nop_node = {
        'inputs': input_schema,
        'outputs': output_schema,
        'op_info': fn_action.serialize(),
        'iter_info': {
            'type': MapConst.name,
            'param': None,
        },
        'config': None,
        'next_nodes': [],
    }
    return nop_node


def operator_to_action(fn):
    if isinstance(fn, _OperatorWrapper):
        return OperatorAction.from_hub(fn.name, fn.init_args, fn.init_kws)
    if getattr(fn, '__name__', None) == '<lambda>':
        return OperatorAction.from_lambda(fn)
    if callable(fn):
        return OperatorAction.from_callable(fn)
    if fn in [OPName.NOP, ConcatConst.name]:
        return OperatorAction.from_builtin(fn)
    raise ValueError('Unknown operator, please make sure it is lambda, callable or operator with ops.')