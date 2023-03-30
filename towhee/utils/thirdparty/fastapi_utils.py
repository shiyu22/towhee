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

# pylint: disable=ungrouped-imports
# pylint: disable=unused-import

try:
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.testclient import TestClient
except ModuleNotFoundError as e:  # pragma: no cover
    from towhee.utils.dependency_control import prompt_install
    prompt_install('fastapi')
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.testclient import TestClient
