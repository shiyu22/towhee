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

from towhee.runtime.data_queue import ColumnType
from towhee.utils.log import engine_log


# pylint: disable=redefined-builtin
class SchemaRepr:
    """
    A `SchemaRepr` represents the data queue schema.

    Args:
        name (`str`): The name column data.
        type (`ColumnType`): The type of the column data, such as ColumnType.SCALAR or ColumnType.QUEUE.
    """
    def __init__(self, name: str, type: 'ColumnType', count: int = 1):
        self._name = name
        self._type = type
        self._count = count

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> 'ColumnType':
        return self._type

    @property
    def count(self) -> int:
        return self._count

    @count.setter
    def count(self, count):
        self._count = count

    @staticmethod
    def from_dag(col_name: str, iter_type: str, input_type: 'ColumnType' = None, count: int = 1):
        if iter_type in ['flat_map', 'window']:
            col_type = ColumnType.QUEUE
        elif input_type is None:
            col_type = ColumnType.SCALAR
        elif iter_type in ['map', 'filter']:
            col_type = input_type
        else:
            engine_log.error('Unknown iteration type: %s', iter_type)
            raise ValueError(f'Unknown iteration type: {iter_type}')
        return SchemaRepr(col_name, col_type, count)
