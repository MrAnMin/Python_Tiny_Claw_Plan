import logging
import os
import sys
from pathlib import Path
from typing import Any

# 将 day02 加入模块搜索路径，便于直接运行此入口
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from internal.engine.loop import NewAgentEngine
from internal.provider.interface import LLMProvider
from internal.schema.message import Message, Role, ToolCall, ToolDefinition, ToolResult
from internal.tools.registry import Registry

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ==========================================
# 1. 伪造的大模型 Provider
# ==========================================
class MockProvider(LLMProvider):
    def __init__(self) -> None:
        self.turn = 0

    # 模拟大模型的响应：第一轮请求执行 bash，第二轮输出最终结果
    def Generate(
        self,
        Ctx: Any,
        Messages: list[Message],
        AvailableTools: list[ToolDefinition],
    ) -> Message:
        self.turn += 1
        if self.turn == 1:
            return Message(
                Role=Role.RoleAssistant,
                Content="让我来看看当前目录下有什么文件。",
                ToolCalls=[
                    ToolCall(
                        ID="call_123",
                        Name="bash",
                        Arguments={"command": "ls -la"},
                    ),
                ],
            )

        return Message(
            Role=Role.RoleAssistant,
            Content="我看到了文件列表，里面包含 main.go，任务完成！",
        )


# ==========================================
# 2. 伪造的 Tool Registry
# ==========================================
class MockRegistry(Registry):
    def GetAvailableTools(self) -> list[ToolDefinition]:
        return []

    def Execute(self, Ctx: Any, Call: ToolCall) -> ToolResult:
        # 直接返回一段伪造的终端输出
        return ToolResult(
            ToolCallID=Call.ID,
            Output="-rw-r--r--  1 user group  234 Oct 24 10:00 main.go\n",
            IsError=False,
        )


# ==========================================
# 3. 组装运行
# ==========================================
def Main() -> None:
    # 获取当前执行目录作为 WorkDir 物理边界
    work_dir = os.getcwd()

    p = MockProvider()
    r = MockRegistry()

    # 实例化核心引擎
    eng = NewAgentEngine(p, r, work_dir)

    # 发起任务指令
    try:
        eng.Run(None, "帮我检查当前目录的文件")
    except Exception as e:
        logger.error("引擎崩溃: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    Main()
