from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from internal.schema.message import Message, ToolDefinition


class LLMProvider(ABC):
    """定义了与大模型通信的统一契约"""

    @abstractmethod
    def Generate(
        self,
        Ctx: Any,
        Messages: list[Message],
        AvailableTools: list[ToolDefinition],
    ) -> Message:
        """接收当前的上下文历史、可用工具列表，并发起一次大模型推理

        Args:
            Ctx: 对应 Go 的 context.Context，用于传递取消信号与超时
            Messages: 当前上下文历史
            AvailableTools: 模型可调用的工具列表

        Returns:
            模型生成的下一条消息

        Raises:
            Exception: 推理失败时抛出异常（对应 Go 的 error 返回值）
        """
        raise NotImplementedError
