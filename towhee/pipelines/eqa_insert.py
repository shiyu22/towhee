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
from pymilvus import FieldSchema, DataType, CollectionSchema

from towhee import ops, pipe, AutoPipes, AutoConfig
from towhee.pipelines.sentence_embedding import _get_embedding_op


@AutoConfig.register
class EnhancedQAInsertConfig:
    """
    Config of pipeline
    """
    def __init__(self):
        # config for text_loader
        self.chunk_size = 300
        self.source_type = 'file'
        # config for sentence_embedding
        self.model = 'all-MiniLM-L6-v2'
        self.dim = None
        self.openai_api_key = None
        self.customize_embedding_op = None
        self.normalize_vec = True
        self.device = -1
        # config for insert_milvus
        self.host = '127.0.0.1'
        self.port = '19530'
        self.collection_name = 'chatbot'
        self.user = None
        self.password = None
        self.collection_schema = None
        self.index_params = None


@AutoPipes.register
def enhanced_qa_insert_pipe(config):
    if not config:
        config = EnhancedQAInsertConfig()

    text_load_op = ops.text_loader()

    allow_triton, sentence_embedding_op = _get_embedding_op(config)
    sentence_embedding_config = {}
    if allow_triton:
        if config.device >= 0:
            sentence_embedding_config = AutoConfig.TritonGPUConfig(device_ids=[config.device], max_batch_size=128)
        else:
            sentence_embedding_config = AutoConfig.TritonCPUConfig()
    if not config.dim:
        config.dim = sentence_embedding_op.get_op().dimension

    assert config.dim
    config.collection_schema = CollectionSchema([
        FieldSchema(name='id', dtype=DataType.INT64, descrition='auto id', is_primary=True, auto_id=True),
        FieldSchema(name='text_id', dtype=DataType.VARCHAR, descrition='text id', max_length=500),
        FieldSchema(name='text', dtype=DataType.VARCHAR, descrition='text', max_length=500),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=config.dim),
    ])
    insert_milvus_op = ops.ann_insert.milvus_client(host=config.host,
                                                    port=config.port,
                                                    collection_name=config.collection_name,
                                                    user=config.user,
                                                    password=config.password,
                                                    collection_schema=config.collection_schema,
                                                    index_params=config.index_params,
                                                    )

    p = (
        pipe.input('doc')
            .flat_map('doc', 'sentence', text_load_op)
            .map('sentence', 'embedding', sentence_embedding_op, config=sentence_embedding_config)
    )

    if config.normalize_vec:
        p = p.map('vec', 'vec', ops.towhee.np_normalize())

    return (p.map(('doc', 'sentence', 'embedding'), 'mr', insert_milvus_op)
            .output('mr')
            )
