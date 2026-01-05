"""
知识库管理器模块
支持动态配置和管理多个知识库
"""
import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from rag_agent.vector_store import VectorStore
from rag_agent.obsidian_connector import ObsidianConnector
from rag_agent.config import Config


@dataclass
class KnowledgeBaseConfig:
    """知识库配置数据类"""
    name: str
    type: str  # 'obsidian', 'folder', 'database', etc.
    path: str
    description: str = ""
    enabled: bool = True
    vector_store_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class KnowledgeBaseInfo:
    """知识库信息数据类"""
    name: str
    type: str
    path: str
    description: str
    enabled: bool
    indexed_documents_count: int = 0
    last_indexed: Optional[str] = None
    vector_store_path: Optional[str] = None


class KnowledgeBaseManager:
    """
    知识库管理器
    负责管理多个知识库的配置、索引和检索
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 知识库配置字典
        self.knowledge_bases: Dict[str, KnowledgeBaseConfig] = {}
        
        # 活跃的向量存储实例
        self.vector_stores: Dict[str, VectorStore] = {}
        
        # 知识库连接器实例
        self.connectors: Dict[str, Any] = {}
        
        # 加载配置
        self._load_knowledge_base_configs()
    
    def _load_knowledge_base_configs(self):
        """从配置中加载知识库配置"""
        # 从环境变量或配置文件加载预定义的知识库配置
        kb_configs_str = os.getenv("KNOWLEDGE_BASES_CONFIG", "")
        if kb_configs_str:
            try:
                kb_configs = json.loads(kb_configs_str)
                for kb_config_data in kb_configs:
                    kb_config = KnowledgeBaseConfig(**kb_config_data)
                    self.knowledge_bases[kb_config.name] = kb_config
            except json.JSONDecodeError as e:
                self.logger.error(f"解析知识库配置失败: {e}")
        
        # 如果没有配置，尝试从默认路径自动发现知识库
        if not self.knowledge_bases:
            self._discover_knowledge_bases_from_default_path()
    
    def _discover_knowledge_bases_from_default_path(self):
        """从默认路径自动发现知识库"""
        default_path = getattr(self.config, 'DEFAULT_KNOWLEDGE_BASE_PATH', self.config.OBSIDIAN_VAULT_PATH)
        
        if os.path.exists(default_path):
            # 扫描默认路径下的所有子目录作为知识库
            for item in os.listdir(default_path):
                item_path = os.path.join(default_path, item)
                if os.path.isdir(item_path):
                    # 使用目录名作为知识库名
                    kb_name = f"kb_{item.lower().replace(' ', '_').replace('-', '_')}"
                    
                    kb_config = KnowledgeBaseConfig(
                        name=kb_name,
                        type="obsidian",
                        path=item_path,
                        description=f"{item} 知识库",
                        enabled=True,
                        vector_store_path=f"./vector_store/{kb_name}"
                    )
                    
                    # 避免重复添加
                    if kb_name not in self.knowledge_bases:
                        self.knowledge_bases[kb_name] = kb_config
                        self.logger.info(f"自动发现知识库: {kb_name} at {item_path}")
        
        # 如果仍然没有找到知识库，添加传统默认知识库（保持向后兼容）
        if not self.knowledge_bases:
            default_kb = KnowledgeBaseConfig(
                name="default",
                type="obsidian",
                path=self.config.OBSIDIAN_VAULT_PATH,
                description="默认Obsidian知识库",
                enabled=True,
                vector_store_path=self.config.VECTOR_DB_PATH
            )
            self.knowledge_bases[default_kb.name] = default_kb
    
    def add_knowledge_base(self, kb_config: KnowledgeBaseConfig) -> bool:
        """
        添加新的知识库配置
        
        Args:
            kb_config: 知识库配置
            
        Returns:
            bool: 添加是否成功
        """
        try:
            if kb_config.name in self.knowledge_bases:
                self.logger.warning(f"知识库 {kb_config.name} 已存在，将被覆盖")
            
            # 验证路径是否存在
            if not os.path.exists(kb_config.path):
                self.logger.error(f"知识库路径不存在: {kb_config.path}")
                return False
            
            self.knowledge_bases[kb_config.name] = kb_config
            self.logger.info(f"成功添加知识库: {kb_config.name}")
            return True
        except Exception as e:
            self.logger.error(f"添加知识库失败: {e}")
            return False
    
    def remove_knowledge_base(self, kb_name: str) -> bool:
        """
        移除知识库配置
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            bool: 移除是否成功
        """
        if kb_name not in self.knowledge_bases:
            self.logger.warning(f"知识库 {kb_name} 不存在")
            return False
        
        try:
            # 移除对应的向量存储
            if kb_name in self.vector_stores:
                del self.vector_stores[kb_name]
            
            # 移除对应的连接器
            if kb_name in self.connectors:
                del self.connectors[kb_name]
            
            # 移除配置
            del self.knowledge_bases[kb_name]
            
            self.logger.info(f"成功移除知识库: {kb_name}")
            return True
        except Exception as e:
            self.logger.error(f"移除知识库失败: {e}")
            return False
    
    def get_knowledge_base(self, kb_name: str) -> Optional[KnowledgeBaseConfig]:
        """
        获取知识库配置
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            Optional[KnowledgeBaseConfig]: 知识库配置，不存在则返回None
        """
        return self.knowledge_bases.get(kb_name)
    
    def list_knowledge_bases(self) -> List[KnowledgeBaseInfo]:
        """
        列出所有知识库信息
        
        Returns:
            List[KnowledgeBaseInfo]: 知识库信息列表
        """
        kb_infos = []
        for name, config in self.knowledge_bases.items():
            # 获取向量存储的文档计数
            doc_count = 0
            if name in self.vector_stores:
                try:
                    doc_count = self.vector_stores[name].collection.count()
                except:
                    pass  # 如果向量存储未初始化，则计数为0
            
            kb_info = KnowledgeBaseInfo(
                name=name,
                type=config.type,
                path=config.path,
                description=config.description,
                enabled=config.enabled,
                indexed_documents_count=doc_count
            )
            kb_infos.append(kb_info)
        
        return kb_infos
    
    def get_vector_store(self, kb_name: str) -> Optional[VectorStore]:
        """
        获取指定知识库的向量存储实例
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            Optional[VectorStore]: 向量存储实例，不存在则返回None
        """
        if kb_name not in self.knowledge_bases:
            self.logger.error(f"知识库 {kb_name} 不存在")
            return None
        
        kb_config = self.knowledge_bases[kb_name]
        if not kb_config.enabled:
            self.logger.warning(f"知识库 {kb_name} 已禁用")
            return None
        
        # 如果向量存储实例不存在，则创建
        if kb_name not in self.vector_stores:
            vector_store_path = kb_config.vector_store_path or f"./vector_store/{kb_name}"
            self.vector_stores[kb_name] = VectorStore(persist_path=vector_store_path)
        
        return self.vector_stores[kb_name]
    
    def get_connector(self, kb_name: str) -> Optional[Any]:
        """
        获取指定知识库的连接器实例
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            Optional[Any]: 连接器实例，不存在则返回None
        """
        if kb_name not in self.knowledge_bases:
            self.logger.error(f"知识库 {kb_name} 不存在")
            return None
        
        kb_config = self.knowledge_bases[kb_name]
        if not kb_config.enabled:
            self.logger.warning(f"知识库 {kb_name} 已禁用")
            return None
        
        # 如果连接器实例不存在，则根据类型创建
        if kb_name not in self.connectors:
            if kb_config.type == "obsidian":
                self.connectors[kb_name] = ObsidianConnector(
                    vault_path=kb_config.path,
                    api_url=self.config.OBSIDIAN_API_URL,
                    api_key=self.config.OBSIDIAN_API_KEY,
                )
            # TODO: 添加其他类型知识库的支持
            else:
                self.logger.warning(f"不支持的知识库类型: {kb_config.type}")
                return None
        
        return self.connectors[kb_name]
    
    def index_knowledge_base(self, kb_name: str) -> bool:
        """
        索引指定知识库
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            bool: 索引是否成功
        """
        if kb_name not in self.knowledge_bases:
            self.logger.error(f"知识库 {kb_name} 不存在")
            return False
        
        kb_config = self.knowledge_bases[kb_name]
        if not kb_config.enabled:
            self.logger.warning(f"知识库 {kb_name} 已禁用")
            return False
        
        try:
            connector = self.get_connector(kb_name)
            vector_store = self.get_vector_store(kb_name)
            
            if not connector or not vector_store:
                self.logger.error(f"无法获取知识库 {kb_name} 的连接器或向量存储")
                return False
            
            self.logger.info(f"开始索引知识库: {kb_name}")
            
            # 根据知识库类型进行索引
            if kb_config.type == "obsidian":
                return self._index_obsidian_knowledge_base(kb_name, connector, vector_store)
            else:
                self.logger.error(f"不支持的知识库类型: {kb_config.type}")
                return False
                
        except Exception as e:
            self.logger.error(f"索引知识库 {kb_name} 失败: {e}")
            return False
    
    def _index_obsidian_knowledge_base(self, kb_name: str, connector: ObsidianConnector, vector_store: VectorStore) -> bool:
        """
        索引Obsidian知识库
        
        Args:
            kb_name: 知识库名称
            connector: Obsidian连接器
            vector_store: 向量存储
            
        Returns:
            bool: 索引是否成功
        """
        try:
            # 获取所有笔记列表
            notes = connector.list_notes()
            self.logger.info(f"知识库 {kb_name} 找到 {len(notes)} 个笔记")

            documents = []
            for note in notes:
                content = connector.get_note_content(note["id"])
                if content.strip():
                    # 为文档添加知识库标识
                    doc_id = f"{kb_name}_{note['id']}"
                    documents.append({
                        "id": doc_id,
                        "content": content,
                        "metadata": {
                            "title": note["title"],
                            "path": note["id"],
                            "knowledge_base": kb_name,  # 标识文档来自哪个知识库
                            "source_type": "obsidian"
                        },
                    })

            # 批量添加到向量库
            if documents:
                vector_store.add_documents(documents)
                self.logger.info(f"知识库 {kb_name} 索引完成，共添加 {len(documents)} 个文档块")
                return True
            else:
                self.logger.warning(f"知识库 {kb_name} 没有找到可索引的文档内容")
                return True  # 没有文档也算成功完成
                
        except Exception as e:
            self.logger.error(f"索引Obsidian知识库 {kb_name} 失败: {e}")
            return False
    
    def index_all_knowledge_bases(self) -> Dict[str, bool]:
        """
        索引所有启用的知识库
        
        Returns:
            Dict[str, bool]: 每个知识库的索引结果
        """
        results = {}
        for kb_name in self.knowledge_bases:
            kb_config = self.knowledge_bases[kb_name]
            if kb_config.enabled:
                results[kb_name] = self.index_knowledge_base(kb_name)
            else:
                results[kb_name] = True  # 禁用的知识库视为成功
                self.logger.info(f"跳过禁用的知识库: {kb_name}")
        
        return results
    
    def save_configs(self, config_file_path: str):
        """
        保存知识库配置到文件
        
        Args:
            config_file_path: 配置文件路径
        """
        try:
            configs_data = []
            for name, config in self.knowledge_bases.items():
                configs_data.append(asdict(config))
            
            with open(config_file_path, 'w', encoding='utf-8') as f:
                json.dump(configs_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"知识库配置已保存到: {config_file_path}")
        except Exception as e:
            self.logger.error(f"保存知识库配置失败: {e}")
    
    def load_configs(self, config_file_path: str):
        """
        从文件加载知识库配置
        
        Args:
            config_file_path: 配置文件路径
        """
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                configs_data = json.load(f)
            
            self.knowledge_bases = {}
            for kb_config_data in configs_data:
                kb_config = KnowledgeBaseConfig(**kb_config_data)
                self.knowledge_bases[kb_config.name] = kb_config
            
            self.logger.info(f"知识库配置已从 {config_file_path} 加载")
        except Exception as e:
            self.logger.error(f"加载知识库配置失败: {e}")