import logging
from pathlib import Path
from typing import Dict, List

import markdown
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ObsidianConnector:
    """
    Obsidian知识库连接器
    支持通过本地文件系统或Obsidian API读取笔记内容
    """

    def __init__(self, vault_path: str = "", api_url: str = "", api_key: str = ""):
        self.vault_path = Path(vault_path) if vault_path else None
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}" if api_key else "",
            "Content-Type": "application/json",
        }

    def list_notes(self) -> List[Dict[str, str]]:
        """
        列出所有笔记文件
        """
        if self.vault_path and self.vault_path.exists():
            # 通过文件系统读取
            notes = []
            # 使用rglob递归查找所有.md文件，处理包含空格的路径
            for file_path in self.vault_path.rglob("*.md"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.vault_path)
                    notes.append(
                        {
                            "id": str(relative_path),
                            "title": file_path.stem,
                            "path": str(file_path),
                            "modified_time": file_path.stat().st_mtime,
                        }
                    )
            return notes
        elif self.api_url:
            # 通过API读取（如果支持）
            try:
                response = requests.get(f"{self.api_url}/files", headers=self.headers)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"API获取笔记列表失败: {e}")
                # 回退到文件系统方式
                if self.vault_path and self.vault_path.exists():
                    return self._list_notes_from_fs()
                else:
                    return []
        else:
            return []

    def _list_notes_from_fs(self) -> List[Dict[str, str]]:
        """从文件系统获取笔记列表"""
        notes = []
        for file_path in self.vault_path.rglob("*.md"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.vault_path)
                notes.append(
                    {
                        "id": str(relative_path),
                        "title": file_path.stem,
                        "path": str(file_path),
                        "modified_time": file_path.stat().st_mtime,
                    }
                )
        return notes

    def get_note_content(self, note_id: str) -> str:
        """
        获取指定笔记的内容
        """
        if self.vault_path:
            # 从文件系统读取，处理包含空格的路径
            file_path = self.vault_path / note_id
            if file_path.exists():
                try:
                    # 使用with open读取文件，自动处理包含空格的路径
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # 提取纯文本内容
                        return self._extract_text_from_markdown(content)
                except FileNotFoundError:
                    logger.error(f"文件不存在: {file_path}")
                    return ""
                except Exception as e:
                    logger.error(f"读取笔记文件失败 {note_id}: {e}")
                    return ""
        elif self.api_url:
            # 通过API读取
            try:
                response = requests.get(
                    f"{self.api_url}/file",
                    params={"path": note_id},
                    headers=self.headers,
                )
                response.raise_for_status()
                content = response.json().get("content", "")
                return self._extract_text_from_markdown(content)
            except Exception as e:
                logger.error(f"API获取笔记内容失败 {note_id}: {e}")
                return ""

        return ""

    def _extract_text_from_markdown(self, markdown_content: str) -> str:
        """
        从Markdown内容中提取纯文本
        """
        try:
            # 使用markdown库转换为HTML，然后用BeautifulSoup提取文本
            html = markdown.markdown(markdown_content)
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text()
            # 清理多余的空白字符
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"提取Markdown文本失败: {e}")
            # 如果转换失败，直接返回原内容
            return markdown_content
