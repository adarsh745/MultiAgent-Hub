import json
import logging
from typing import Any, Dict

from pydantic import ValidationError

from agents.observability.trace import AgentTrace
from agents.base import BaseAgent
from agents.tools.decision import ToolDecision
from agents.tools.registry import ToolRegistry
from agents.tools.relevance import ToolRelevanceGuard
from prompts.system_prompt import Prompts

logger = logging.getLogger(__name__)


class AssistantAgent(BaseAgent):

    def __init__(
        self,
        name: str,
        tool_registry: ToolRegistry,
        llm_service
    ):
        super().__init__(name)

        self.tool_registry = tool_registry
        self.llm_service = llm_service
        self.relevance_guard = ToolRelevanceGuard()

        self.last_execution: Dict[str, Any] = {}

    def _clean_json_string(self, text: str) -> str:
        """Strip markdown syntax fences if present."""

        s = text.strip()

        if s.startswith("```json"):
            s = s[7:]

        elif s.startswith("```"):
            s = s[3:]

        if s.endswith("```"):
            s = s[:-3]

        return s.strip()

    async def run(self, message: str) -> Any:

        # ---------------------------------------------------------
        # 1. CREATE TRACE
        # ---------------------------------------------------------

        trace = AgentTrace(message)

        trace.log(
            "request_started",
            message=message
        )

        tool_definitions = self.tool_registry.get_tool_definitions()

        # ---------------------------------------------------------
        # 2. SYSTEM PROMPT
        # ---------------------------------------------------------
        system_prompt = Prompts.system_prompt
       

        max_retries = 3

        current_prompt = f"""
Available tools:

{json.dumps(tool_definitions, indent=2)}

User request:

{message}
"""

        # ---------------------------------------------------------
        # 3. EXECUTION STATE
        # ---------------------------------------------------------

        self.last_execution = {
            "user_request": message,
            "attempts": 0,
            "raw_llm_response": None,
            "decision": None,
            "tool_name": None,
            "operation": None,
            "arguments": None,
            "tool_result": None,
            "validation_errors": [],
            "relevance_errors": [],
            "error": None,
            "final_response": None,
        }

        # ---------------------------------------------------------
        # 4. AGENT LOOP
        # ---------------------------------------------------------

        for attempt in range(1, max_retries + 1):

            self.last_execution["attempts"] = attempt

            # Record attempt
            trace.log(
                "attempt_started",
                attempt=attempt,
                max_retries=max_retries
            )

            # -----------------------------------------------------
            # LLM CALL
            # -----------------------------------------------------

            llm_response = await self.llm_service.generate_response(
                prompt=current_prompt,
                system_prompt=system_prompt,
            )

            trace.log(
                "llm_response",
                attempt=attempt,
                response=llm_response
            )

            self.last_execution["raw_llm_response"] = llm_response

            print(
                f"\n[Attempt {attempt}/{max_retries}] "
                f"LLM decision output:\n{llm_response}"
            )

            # -----------------------------------------------------
            # 5. PARSE JSON
            # -----------------------------------------------------

            cleaned_json = self._clean_json_string(llm_response)

            try:

                raw_dict = json.loads(cleaned_json)

            except Exception as e:

                error_msg = f"Invalid JSON format. Error: {e}"

                print(f"[Validation Error] {error_msg}")

                self.last_execution["validation_errors"].append(
                    error_msg
                )

                trace.log(
                    "json_validation",
                    status="failed",
                    error=error_msg
                )

                if attempt < max_retries:

                    trace.log(
                        "retry_triggered",
                        reason="invalid_json",
                        next_attempt=attempt + 1
                    )

                current_prompt = self._build_correction_prompt(
                    message,
                    error_msg,
                    llm_response
                )

                continue

            trace.log(
                "json_validation",
                status="passed"
            )

            # -----------------------------------------------------
            # 6. TOOL DECISION VALIDATION
            # -----------------------------------------------------

            try:

                decision = ToolDecision.model_validate(raw_dict)

                trace.log(
                    "decision_validation",
                    status="passed"
                )

            except ValidationError as ve:

                error_msg = (
                    "Invalid tool decision format. "
                    f"Pydantic validation error: {ve}"
                )

                print(f"[Validation Error] {error_msg}")

                self.last_execution["validation_errors"].append(
                    error_msg
                )

                trace.log(
                    "decision_validation",
                    status="failed",
                    validation_error=str(ve)
                )

                if attempt < max_retries:

                    trace.log(
                        "retry_triggered",
                        reason="decision_validation_failed",
                        next_attempt=attempt + 1
                    )

                current_prompt = self._build_correction_prompt(
                    message,
                    error_msg,
                    llm_response
                )

                continue

            # -----------------------------------------------------
            # 7. FINAL RESPONSE WITHOUT TOOL
            # -----------------------------------------------------

            if decision.action == "final":

                self.last_execution["decision"] = "final"
                self.last_execution["tool_name"] = None
                self.last_execution["operation"] = None
                self.last_execution["final_response"] = (
                    decision.response
                )

                trace.log(
                    "final_response",
                    response=decision.response,
                    tool_used=False
                )

                trace.log(
                    "request_completed",
                    status="success"
                )

                trace.print_trace()

                return decision.response

            # -----------------------------------------------------
            # 8. TOOL ACTION
            # -----------------------------------------------------

            if decision.action == "tool":

                self.last_execution["decision"] = "tool"
                self.last_execution["tool_name"] = decision.tool

                arguments = decision.arguments or {}

                self.last_execution["arguments"] = arguments

                self.last_execution["operation"] = (
                    arguments.get("operation")
                    or arguments.get("conversion")
                )

                # -------------------------------------------------
                # 8A. TOOL EXISTENCE
                # -------------------------------------------------

                try:

                    tool = self.tool_registry.get(
                        decision.tool
                    )

                    trace.log(
                        "tool_validation",
                        tool_name=decision.tool,
                        status="exists"
                    )

                except ValueError as ve:

                    error_msg = (
                        f"Tool '{decision.tool}' not found "
                        f"in registry. Registered tools: "
                        f"{self.tool_registry.list_tools()}"
                    )

                    print(
                        f"[Validation Error] {error_msg}"
                    )

                    self.last_execution[
                        "validation_errors"
                    ].append(error_msg)

                    trace.log(
                        "tool_validation",
                        tool_name=decision.tool,
                        status="not_found",
                        error=str(ve)
                    )

                    if attempt < max_retries:

                        trace.log(
                            "retry_triggered",
                            reason="tool_not_found",
                            next_attempt=attempt + 1
                        )

                    current_prompt = (
                        self._build_correction_prompt(
                            message,
                            error_msg,
                            llm_response
                        )
                    )

                    continue

                # -------------------------------------------------
                # 8B. ARGUMENT VALIDATION
                # -------------------------------------------------

                try:

                    input_data = tool.input_schema(
                        **arguments
                    )

                    trace.log(
                        "argument_validation",
                        tool=decision.tool,
                        status="passed",
                        arguments=arguments
                    )

                except Exception as ve:

                    error_msg = (
                        f"Invalid arguments for tool "
                        f"'{decision.tool}'. Error: {ve}"
                    )

                    print(
                        f"[Validation Error] {error_msg}"
                    )

                    self.last_execution[
                        "validation_errors"
                    ].append(error_msg)

                    trace.log(
                        "argument_validation",
                        tool=decision.tool,
                        status="failed",
                        error=str(ve)
                    )

                    if attempt < max_retries:

                        trace.log(
                            "retry_triggered",
                            reason="invalid_arguments",
                            next_attempt=attempt + 1
                        )

                    current_prompt = (
                        self._build_correction_prompt(
                            message,
                            error_msg,
                            llm_response
                        )
                    )

                    continue

                # -------------------------------------------------
                # 8C. RELEVANCE GUARD
                # -------------------------------------------------

                is_relevant, relevance_reason = (
                    self.relevance_guard.validate_relevance(
                        message,
                        tool,
                        arguments
                    )
                )

                if not is_relevant:

                    logger.warning(
                        "[Relevance Error] "
                        "User request: %s | "
                        "Selected tool: %s | "
                        "Reason: %s",
                        message,
                        decision.tool,
                        relevance_reason
                    )

                    print(
                        f"[Relevance Error] "
                        f"User request: '{message}' | "
                        f"Selected tool: '{decision.tool}' | "
                        f"Reason: {relevance_reason}"
                    )

                    self.last_execution[
                        "relevance_errors"
                    ].append(relevance_reason)

                    trace.log(
                        "relevance_check",
                        tool=decision.tool,
                        status="failed",
                        reason=relevance_reason
                    )

                    if attempt < max_retries:

                        trace.log(
                            "retry_triggered",
                            reason="tool_not_relevant",
                            next_attempt=attempt + 1
                        )

                    current_prompt = (
                        self._build_relevance_correction_prompt(
                            message,
                            decision.tool,
                            tool.description,
                            relevance_reason
                        )
                    )

                    continue

                trace.log(
                    "relevance_check",
                    tool=decision.tool,
                    status="passed"
                )

                # -------------------------------------------------
                # 8D. TOOL EXECUTION
                # -------------------------------------------------

                # IMPORTANT:
                # Log START before executing the tool.

                trace.log(
                    "tool_execution_started",
                    tool=decision.tool,
                    arguments=arguments
                )

                try:

                    tool_result = tool.execute(
                        input_data
                    )

                    self.last_execution[
                        "tool_result"
                    ] = tool_result

                    print(
                        f"\n[Tool Execution Result] "
                        f"{tool_result}"
                    )

                    trace.log(
                        "tool_execution_completed",
                        tool=decision.tool,
                        result=tool_result
                    )

                except Exception as ex:

                    error_msg = (
                        f"Execution error in tool "
                        f"'{decision.tool}': {ex}"
                    )

                    print(
                        f"[Execution Error] {error_msg}"
                    )

                    self.last_execution[
                        "error"
                    ] = error_msg

                    trace.log(
                        "tool_execution_failed",
                        tool=decision.tool,
                        error=str(ex)
                    )

                    trace.log(
                        "request_completed",
                        status="failed"
                    )

                    trace.print_trace()

                    return f"ERROR: {ex}"

                # -------------------------------------------------
                # 9. FINAL LLM RESPONSE
                # -------------------------------------------------

                final_prompt = f"""
User request:

{message}

Tool used:

{decision.tool}

Tool result:

{tool_result}

Give the user the final answer based on the tool result.
"""

                final_response = (
                    await self.llm_service.generate_response(
                        prompt=final_prompt
                    )
                )

                trace.log(
                    "final_response",
                    response=final_response,
                    tool_used=True,
                    tool=decision.tool
                )

                self.last_execution[
                    "final_response"
                ] = final_response

                trace.log(
                    "request_completed",
                    status="success"
                )

                # Print the complete trace
                trace.print_trace()

                return final_response

        # ---------------------------------------------------------
        # 10. ALL RETRIES FAILED
        # ---------------------------------------------------------

        fallback_msg = (
            "Error: Failed to get a valid tool decision "
            "after 3 attempts."
        )

        self.last_execution["error"] = fallback_msg

        trace.log(
            "request_completed",
            status="failed",
            reason="max_retries_exceeded"
        )

        trace.print_trace()

        return fallback_msg

    # =============================================================
    # CORRECTION PROMPT
    # =============================================================

    def _build_correction_prompt(
        self,
        user_request: str,
        error_msg: str,
        previous_response: str
    ) -> str:

        return f"""
The previous response was invalid.

Error details:

{error_msg}

You MUST return ONLY valid JSON using exactly one
of these formats:

{{
    "action": "tool",
    "tool": "tool_name",
    "arguments": {{
        "key": "value"
    }}
}}

OR

{{
    "action": "final",
    "response": "your answer string"
}}

Previous invalid response:

{previous_response}

Original User request:

{user_request}
"""

    # =============================================================
    # RELEVANCE CORRECTION PROMPT
    # =============================================================

    def _build_relevance_correction_prompt(
        self,
        user_request: str,
        tool_name: str,
        tool_description: str,
        reason: str
    ) -> str:

        return f"""
The selected tool is not relevant to the user's request.

User request:

{user_request}

Selected tool:

{tool_name}

Tool description:

{tool_description}

Reason:

{reason}

Choose a relevant tool or return action='final'
if no tool is required.

Return ONLY valid JSON.
"""