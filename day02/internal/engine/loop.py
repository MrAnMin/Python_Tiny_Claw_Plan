from __future__ import annotations

import json
import logging
from typing import Any

from internal.provider.interface import LLMProvider
from internal.schema.message import Message, Role
from internal.tools.registry import Registry

logger = logging.getLogger(__name__)


def _FormatArguments(Arguments: Any) -> str:
    if isinstance(Arguments, str):
        return Arguments
    return json.dumps(Arguments, ensure_ascii=False)


class AgentEngine:
    """微型 OS 的核心驱动"""

    def __init__(self, Provider: LLMProvider, Registry: Registry, WorkDir: str) -> None:
        self.Provider = Provider
        self.Registry = Registry
        # WorkDir (工作区): 借鉴 OpenClaw 的理念，Agent 必须有一个明确的物理边界
        self.WorkDir = WorkDir

    def Run(self, Ctx: Any, UserPrompt: str) -> None:
        """启动 Agent 的生命周期"""
        logger.info("[Engine] 引擎启动，锁定工作区: %s", self.WorkDir)

        # 1. 初始化会话的 Context (上下文内存)
        # 在真实的场景中，这里会由动态 Prompt 组装器加载 AGENTS.md。目前我们先硬编码。
        context_history: list[Message] = [
            Message(
                Role=Role.RoleSystem,
                Content=(
                    "You are python-tiny-claw, an expert coding assistant. "
                    "You have full access to tools in the workspace."
                ),
            ),
            Message(
                Role=Role.RoleUser,
                Content=UserPrompt,
            ),
        ]

        turn_count = 0

        # 2. The Main Loop: 心跳开始 (标准的 ReAct 循环)
        while True:
            turn_count += 1
            logger.info("========== [Turn %d] 开始 ==========", turn_count)

            # 获取当前挂载的所有工具定义
            available_tools = self.Registry.GetAvailableTools()

            # 向大模型发起推理请求 (包含 Reasoning)
            logger.info("[Engine] 正在思考 (Reasoning)...")
            try:
                response_msg = self.Provider.Generate(Ctx, context_history, available_tools)
            except Exception as e:
                raise RuntimeError(f"模型生成失败: {e}") from e

            # 将模型的响应完整追加到上下文历史中
            context_history.append(response_msg)

            # 如果模型回复了纯文本，打印出来 (这通常是它的思考过程，或是最终结果)
            if response_msg.Content:
                print(f"🤖 模型: {response_msg.Content}")

            # 3. 退出条件判断
            # 如果模型没有请求任何工具调用，说明它认为任务已经完成，跳出循环。
            if not response_msg.ToolCalls:
                logger.info("[Engine] 任务完成，退出循环。")
                break

            # 4. 执行行动 (Action) 与 获取观察结果 (Observation)
            logger.info("[Engine] 模型请求调用 %d 个工具...", len(response_msg.ToolCalls))

            for tool_call in response_msg.ToolCalls:
                logger.info(
                    "  -> 🛠️ 执行工具: %s, 参数: %s",
                    tool_call.Name,
                    _FormatArguments(tool_call.Arguments),
                )

                # 通过 Registry 路由并执行底层工具
                result = self.Registry.Execute(Ctx, tool_call)

                if result.IsError:
                    logger.info("  -> ❌ 工具执行报错: %s", result.Output)
                else:
                    logger.info("  -> ✅ 工具执行成功 (返回 %d 字节)", len(result.Output))

                # 将工具执行的观察结果 (Observation) 封装为 User Message 追加到上下文中
                # 注意：ToolCallID 必须携带！这是维系大模型推理链条的关键
                observation_msg = Message(
                    Role=Role.RoleUser,
                    Content=result.Output,
                    ToolCallID=tool_call.ID,
                )
                context_history.append(observation_msg)

            # 循环回到开头，模型将带着新加入的 Observation 继续它的下一轮思考...


def NewAgentEngine(Provider: LLMProvider, Registry: Registry, WorkDir: str) -> AgentEngine:
    return AgentEngine(Provider, Registry, WorkDir)
