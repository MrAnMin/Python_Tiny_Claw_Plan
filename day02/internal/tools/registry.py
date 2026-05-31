from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from internal.schema.message import ToolCall, ToolDefinition, ToolResult


class Registry(ABC):
    """定义了工具的注册与分发执行接口"""

    @abstractmethod
    def GetAvailableTools(self) -> list[ToolDefinition]:
        """返回当前系统挂载的所有可用工具的 Schema"""
        raise NotImplementedError

    @abstractmethod
    def Execute(self, Ctx: Any, Call: ToolCall) -> ToolResult:
        """实际执行模型请求的工具，并返回结果

        Args:
            Ctx: 对应 Go 的 context.Context，用于传递取消信号与超时
            Call: 模型发起的工具调用请求

        Returns:
            工具执行完毕后的物理结果
        """
        raise NotImplementedError
