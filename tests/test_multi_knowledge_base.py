"""
多知识库功能测试
测试小魔仙RAG智能体的多知识库支持功能
"""
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from rag_agent.config import Config
from rag_agent.knowledge_base_manager import KnowledgeBaseManager, KnowledgeBaseConfig
from rag_agent.vector_store import VectorStore


class TestMultiKnowledgeBase(unittest.TestCase):
    """多知识库功能测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.config = Config()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """测试清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_knowledge_base_config(self):
        """测试知识库配置类"""
        kb_config = KnowledgeBaseConfig(
            name="test_kb",
            type="obsidian",
            path="/path/to/test",
            description="Test knowledge base",
            enabled=True,
            vector_store_path="./test_vector_store"
        )
        
        self.assertEqual(kb_config.name, "test_kb")
        self.assertEqual(kb_config.type, "obsidian")
        self.assertEqual(kb_config.path, "/path/to/test")
        self.assertTrue(kb_config.enabled)
    
    def test_knowledge_base_manager_initialization(self):
        """测试知识库管理器初始化"""
        kb_manager = KnowledgeBaseManager(self.config)
        
        # 检查默认知识库是否存在
        default_kb = kb_manager.get_knowledge_base("default")
        self.assertIsNotNone(default_kb)
        self.assertEqual(default_kb.name, "default")
    
    def test_add_and_remove_knowledge_base(self):
        """测试添加和移除知识库"""
        kb_manager = KnowledgeBaseManager(self.config)
        
        # 创建测试知识库配置
        test_kb_path = os.path.join(self.temp_dir, "test_kb")
        os.makedirs(test_kb_path, exist_ok=True)
        
        kb_config = KnowledgeBaseConfig(
            name="test_kb",
            type="obsidian",
            path=test_kb_path,
            description="Test knowledge base for testing",
            enabled=True
        )
        
        # 添加知识库
        result = kb_manager.add_knowledge_base(kb_config)
        self.assertTrue(result)
        
        # 检查知识库是否添加成功
        added_kb = kb_manager.get_knowledge_base("test_kb")
        self.assertIsNotNone(added_kb)
        self.assertEqual(added_kb.name, "test_kb")
        
        # 移除知识库
        result = kb_manager.remove_knowledge_base("test_kb")
        self.assertTrue(result)
        
        # 检查知识库是否移除成功
        removed_kb = kb_manager.get_knowledge_base("test_kb")
        self.assertIsNone(removed_kb)
    
    def test_list_knowledge_bases(self):
        """测试列出知识库"""
        kb_manager = KnowledgeBaseManager(self.config)
        
        # 获取初始知识库列表
        initial_kbs = kb_manager.list_knowledge_bases()
        initial_count = len(initial_kbs)
        
        # 添加一个测试知识库
        test_kb_path = os.path.join(self.temp_dir, "test_list_kb")
        os.makedirs(test_kb_path, exist_ok=True)
        
        kb_config = KnowledgeBaseConfig(
            name="test_list_kb",
            type="obsidian",
            path=test_kb_path,
            description="Test knowledge base for listing",
            enabled=True
        )
        
        kb_manager.add_knowledge_base(kb_config)
        
        # 获取更新后的知识库列表
        updated_kbs = kb_manager.list_knowledge_bases()
        self.assertEqual(len(updated_kbs), initial_count + 1)
        
        # 检查新增的知识库是否在列表中
        test_kb_info = None
        for kb_info in updated_kbs:
            if kb_info.name == "test_list_kb":
                test_kb_info = kb_info
                break
        
        self.assertIsNotNone(test_kb_info)
        self.assertEqual(test_kb_info.name, "test_list_kb")
        self.assertTrue(test_kb_info.enabled)
    
    def test_get_vector_store(self):
        """测试获取向量存储"""
        kb_manager = KnowledgeBaseManager(self.config)
        
        # 测试获取默认知识库的向量存储
        default_vs = kb_manager.get_vector_store("default")
        self.assertIsNotNone(default_vs)
        self.assertIsInstance(default_vs, VectorStore)
        
        # 添加一个新的知识库并测试获取其向量存储
        test_kb_path = os.path.join(self.temp_dir, "test_vs_kb")
        os.makedirs(test_kb_path, exist_ok=True)
        
        kb_config = KnowledgeBaseConfig(
            name="test_vs_kb",
            type="obsidian",
            path=test_kb_path,
            description="Test knowledge base for vector store",
            enabled=True,
            vector_store_path=os.path.join(self.temp_dir, "test_vs_store")
        )
        
        kb_manager.add_knowledge_base(kb_config)
        
        test_vs = kb_manager.get_vector_store("test_vs_kb")
        self.assertIsNotNone(test_vs)
        self.assertIsInstance(test_vs, VectorStore)
    
    def test_disabled_knowledge_base(self):
        """测试禁用的知识库"""
        kb_manager = KnowledgeBaseManager(self.config)
        
        # 添加一个禁用的知识库
        test_kb_path = os.path.join(self.temp_dir, "disabled_kb")
        os.makedirs(test_kb_path, exist_ok=True)
        
        kb_config = KnowledgeBaseConfig(
            name="disabled_kb",
            type="obsidian",
            path=test_kb_path,
            description="Disabled knowledge base",
            enabled=False
        )
        
        kb_manager.add_knowledge_base(kb_config)
        
        # 尝试获取禁用知识库的向量存储，应该返回None
        disabled_vs = kb_manager.get_vector_store("disabled_kb")
        self.assertIsNone(disabled_vs)
    
    @patch('rag_agent.obsidian_connector.ObsidianConnector')
    def test_get_connector(self, mock_connector):
        """测试获取连接器"""
        kb_manager = KnowledgeBaseManager(self.config)
        
        # 添加一个obsidian类型的知识库
        test_kb_path = os.path.join(self.temp_dir, "test_connector_kb")
        os.makedirs(test_kb_path, exist_ok=True)
        
        kb_config = KnowledgeBaseConfig(
            name="test_connector_kb",
            type="obsidian",
            path=test_kb_path,
            description="Test knowledge base for connector",
            enabled=True
        )
        
        kb_manager.add_knowledge_base(kb_config)
        
        # 获取连接器
        connector = kb_manager.get_connector("test_connector_kb")
        self.assertIsNotNone(connector)
    
    def test_save_and_load_configs(self):
        """测试保存和加载配置"""
        kb_manager = KnowledgeBaseManager(self.config)
        
        # 添加一些测试知识库
        test_kb_path1 = os.path.join(self.temp_dir, "test_save_kb1")
        os.makedirs(test_kb_path1, exist_ok=True)
        
        test_kb_path2 = os.path.join(self.temp_dir, "test_save_kb2")
        os.makedirs(test_kb_path2, exist_ok=True)
        
        kb_config1 = KnowledgeBaseConfig(
            name="test_save_kb1",
            type="obsidian",
            path=test_kb_path1,
            description="Test knowledge base 1 for saving",
            enabled=True
        )
        
        kb_config2 = KnowledgeBaseConfig(
            name="test_save_kb2",
            type="obsidian",
            path=test_kb_path2,
            description="Test knowledge base 2 for saving",
            enabled=False
        )
        
        kb_manager.add_knowledge_base(kb_config1)
        kb_manager.add_knowledge_base(kb_config2)
        
        # 保存配置
        config_file_path = os.path.join(self.temp_dir, "test_kb_configs.json")
        kb_manager.save_configs(config_file_path)
        
        # 验证配置文件存在
        self.assertTrue(os.path.exists(config_file_path))
        
        # 创建新的管理器实例并加载配置
        new_kb_manager = KnowledgeBaseManager(self.config)
        new_kb_manager.knowledge_bases = {}  # 清空默认配置
        new_kb_manager.load_configs(config_file_path)
        
        # 验证加载的配置
        loaded_kb1 = new_kb_manager.get_knowledge_base("test_save_kb1")
        self.assertIsNotNone(loaded_kb1)
        self.assertEqual(loaded_kb1.description, "Test knowledge base 1 for saving")
        self.assertTrue(loaded_kb1.enabled)
        
        loaded_kb2 = new_kb_manager.get_knowledge_base("test_save_kb2")
        self.assertIsNotNone(loaded_kb2)
        self.assertEqual(loaded_kb2.description, "Test knowledge base 2 for saving")
        self.assertFalse(loaded_kb2.enabled)


if __name__ == '__main__':
    unittest.main()