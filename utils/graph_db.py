import logging
import os
import sys
from abc import ABC
from typing import List, Dict, Any

import pandas as pd
from py2neo import Graph
# 配置日志
base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)


class GraphDB(ABC):
    def __init__(self):
        self.connection_params:dict

    def connect(self):
        raise NotImplementedError("Subclasses must implement this method")

    def clean_graphDB(self):
        raise NotImplementedError("Subclasses must implement this method")

    def create_nodes(self, df: pd.DataFrame, label: str):
        raise NotImplementedError("Subclasses must implement this method")

    def query(self, query, parameters=None):
        raise NotImplementedError("Subclasses must implement this method")


class Neo4JDB(GraphDB):
    def __init__(self):
        super().__init__()
        self.connection_params = {
            'host': 'localhost',
            'port': 7687,
            'user': 'neo4j',
            'password': '12345678',
            'name':'neo4j' # 可选，指定数据库名称
        }
        self.graph = None

    def connect(self):
        try:
            self.graph = Graph(**self.connection_params)
            logger.info(f"Neo4j连接成功 - 数据库: {self.connection_params.get('name', 'default')}")
            return True
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return False

    def clean_graphDB(self):
        if not self.graph:
            raise Exception("GraphDB is not connected.")
        try:
            self.graph.run("MATCH (n) DETACH DELETE n")
            logger.info("图数据库已清空")
        except Exception as e:
            logger.error(f"清空图数据库失败: {e}")
            raise

    def create_nodes(self, df: pd.DataFrame, label: str):
        """
        从 Pandas DataFrame 批量创建节点。
        为DataFrame中的每一行创建一个新的节点，并将所有列作为其属性。

        :param df: 包含节点数据的 DataFrame。
        :param label: 要为这些节点设置的标签 (Label)。
        """
        if not self.graph:
            logger.error("数据库未连接。")
            return

        # 将DataFrame转换为字典列表
        records = df.to_dict('records')

        # 构建批量创建节点的 Cypher 查询
        query = f"""
        UNWIND $props AS properties
        CREATE (n:{label})
        SET n = properties
        """

        try:
            self.graph.run(query, props=records)
            logger.info(f"成功创建了 {len(records)} 个 '{label}' 节点。")
        except Exception as e:
            logger.error(f"批量创建节点时出错: {e}")

    def create_node_batch(self, nodes: List[Any]) -> int:
        """批量创建节点"""
        try:
            tx = self.graph.begin()
            for node in nodes:
                tx.create(node)
            tx.commit()
            return len(nodes)
        except Exception as e:
            logger.error(f"批量创建节点失败: {e}")
            raise

    def query(self, query:str, **params) -> List[Dict]:
        if not self.graph:
            raise Exception("GraphDB is not connected.")
        try:
            result = self.graph.run(query, **params).data()
            return result
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            raise

    def create_indexes(self, index_queries: List[str]):
        """创建索引"""
        try:
            for query in index_queries:
                self.graph.run(query)
            logger.info(f"创建{len(index_queries)}个索引")
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            raise