from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Role(str, Enum):
    """定义消息的角色，这是与大模型沟通的基石"""

    RoleSystem = "system"  # 系统提示词：确立 Agent 的性格与红线
    RoleUser = "user"  # 用户输入 / 工具执行的返回结果 (Observation)
    RoleAssistant = "assistant"  # 模型的输出：包含推理(Reasoning)或工具调用(ToolCall)


@dataclass
class ToolCall:
    """代表模型请求调用某个具体的工具"""

    ID: str  # 工具调用的唯一 ID
    Name: str  # 想要调用的工具名称 (例如 "bash")
    # 存放 JSON 参数，延迟解析，将解析责任交给具体的工具
    Arguments: Any


@dataclass
class Message:
    """代表上下文中传递的单条消息"""

    Role: Role
    Content: str  # 存放纯文本内容
    # 如果模型决定调用工具，此字段将被填充 (支持并行调用多个工具)
    ToolCalls: list[ToolCall] | None = None
    # 如果这是对某个工具调用的响应，此字段必须填写，以告知模型上下文的关联性
    ToolCallID: str | None = None


@dataclass
class ToolResult:
    """代表工具在本地执行完毕后返回的物理结果"""

    ToolCallID: str
    Output: str  # 工具执行的控制台输出或报错堆栈
    IsError: bool = False  # 标记是否失败，供后续的驾驭工程进行错误自愈


@dataclass
class ToolDefinition:
    """描述了一个大模型可以调用的工具元信息 (供模型理解工具有什么用)"""

    Name: str
    Description: str
    InputSchema: Any  # 对应 JSON Schema
