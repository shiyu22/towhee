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
import uuid
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    DataType,
    CollectionSchema,
    Collection,
    MilvusException,
)

from towhee.utils.log import engine_log
from towhee import ops, pipe, AutoPipes, AutoConfig
from towhee.pipelines.sentence_embedding import sentence_embedding
from towhee.pipelines.insert_milvus import milvus_insert_pipe


@AutoConfig.register
class EnhancedQAInsertConfig:
    """
    Config of pipeline
    """
    def __init__(self):
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
        # config for text_loader
        self.chunk_size = 300
        self.source_type = 'file'


def _create_collection(config):
    alias = uuid.uuid4().hex
    if config.user and config.password:
        connections.connect(alias=alias, host=config.host, port=config.port,
                            user=config.user, password=config.password, secure=True)
    else:
        connections.connect(alias=alias, host=config.host, port=config.port)

    if not utility.has_collection(config.collection_name, using=alias):
        schema = [
            FieldSchema(name='id', dtype=DataType.INT64, descrition='auto id', is_primary=True, auto_id=True),
            FieldSchema(name='text_id', dtype=DataType.VARCHAR, descrition='text id', max_length=500),
            FieldSchema(name='text', dtype=DataType.VARCHAR, descrition='text', max_length=500),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=config.dim)
        ]
        schema = CollectionSchema(schema)
        col = Collection(
            config.collection_name,
            schema=schema,
            consistency_level="Strong",
            using=alias,
        )
    else:
        engine_log.warning("The %s collection already exists, and it will be used directly.", config.collection_name)
        col = Collection(
            config.collection_name,
            consistency_level="Strong",
            using=alias,
        )

    if len(col.indexes) == 0:
        try:
            engine_log.info("Attempting creation of Milvus index.")
            index_params = {'metric_type': 'L2', 'index_type': 'HNSW', 'params': {'M': 8, 'efConstruction': 64}}
            col.create_index("embedding", index_params=index_params)
            engine_log.info("Creation of Milvus index successful.")
        except MilvusException as e:
            engine_log.warning("Error with building index: %s, and attempting creation of default index.", e)
            index_params = {"metric_type": "L2", "index_type": "AUTOINDEX", "params": {}}
            col.create_index("embedding", index_params=index_params)


@AutoPipes.register
def enhanced_qa_insert_pipe(config):
    if not config:
        config = EnhancedQAInsertConfig()

    text_load_op = ops.text_loader()
    sentence_embedding_pipe = sentence_embedding(config=config)

    assert config.dim
    _create_collection(config)
    insert_milvus_pipe = milvus_insert_pipe(config=config)

    return (
        pipe.input('doc')
            .flat_map('doc', 'sentence', text_load_op)
            .map('sentence', 'embedding', sentence_embedding_pipe)
            .map('doc', 'doc', lambda x: str(x))  # test
            .map(('doc', 'sentence', 'embedding'), 'mr', insert_milvus_pipe)
            .output('mr')
    )


