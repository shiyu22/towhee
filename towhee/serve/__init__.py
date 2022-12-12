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

from towhee.utils.log import engine_log
from .triton.pipeline_builder import PipelineBuilder as TritonModelBuilder
from .triton.docker_builder import DockerBuilder as TritonDockerBuilder


class Builder:
    """
    Build the server with towhee.

    Args:
        pipeline: RuntimePipeline
            Towhee pipeline ready to call.
        model_root: str
            The path to build the server.
        format_priority: list
            The format priority for model, such as ['onnx', 'tensorrt'].
        server: str
            The type of server, defaults to 'triton'.

    Examples:
        Build model for triton.
        >>> from towhee import Builder
        >>> from towhee.dc2 import pipe, ops
        >>> p = (
        ...        pipe.input('url')
        ...        .map('url', 'image', ops.image_decode.cv2())
        ...        .map('image', 'vec', ops.image_text_embedding.clip_vision())
        ...        .output('vec')
        ... )
        >>> Builder(
        ...        p,
        ...        './pipe',
        ...        server='triton',
        ...        format_priority=['onnx', 'trt']
        ...    ).build()
    """
    def __init__(self, pipeline: 'RuntimePipeline', model_root: str, format_priority: list, server: str = 'triton'):
        if server == 'triton':
            self._builder = TritonModelBuilder(pipeline, model_root, format_priority)
        else:
            engine_log.error('Unknown server type: %s.', server)
            self._builder = None

    def build(self):
        if self._builder is not None:
            return self._builder.build()
        return False


def build_docker_image(pipeline: 'RuntimePipeline', image_name: str, inference_server: list, server_config: dict, cuda: str = '11.7'):
    if inference_server == 'triton':
        return TritonDockerBuilder(pipeline, image_name, server_config, cuda).build()
    else:
        engine_log.error('Unknown server type: %s.', inference_server)
        return False
