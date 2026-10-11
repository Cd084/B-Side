"""LLM client abstraction for Anthropic, OpenAI, Google Gemini, Ollama, and custom providers."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import anthropic
from google import genai
from google.genai import types as genai_types
import httpx
from json_repair import repair_json
import openai

from backend.models import LLMConfig, OllamaModel, OllamaModelInfo, OllamaModelsResponse, OllamaStatus

logger = logging.getLogger(__name__)


# Cost per million tokens (updated Feb 2026)
MODEL_COSTS = {
    # Anthropic models (input/output per million tokens)
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # OpenAI models
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    # Google Gemini models
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}

# Context limits per model (in tokens) - used to calculate max tracks
MODEL_CONTEXT_LIMITS = {
    # Anthropic
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    # OpenAI
    "gpt-4.1": 128_000,
    "gpt-4.1-mini": 128_000,
    # Google Gemini
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
}

# Tokens per track (based on real-world testing, Feb 2026)
TOKENS_PER_TRACK = 40

# Tokens per album for recommendation flow (artist, album, year, genres)
TOKENS_PER_ALBUM = 25

# Default ceiling for the num_ctx we ask Ollama to allocate. Models often report
# a native context of 128k+, which would exhaust VRAM on typical home hardware.
# Batched selection works well with 8-16k, so cap unless the user overrides.
OLLAMA_DEFAULT_NUM_CTX_CAP = 16384

# Upper bound on the candidate pool for batched selection. The pool is no longer
# limited by context size because tracks are scanned in context-sized batches.
MAX_TRACK_POOL = 20000


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (~3 chars/token) for budgeting prompts."""
    return len(text) // 3 + 1


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    truncated: bool = False  # True if generation stopped on the output limit

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def estimated_cost(self) -> float:
        """Estimate cost in USD based on token usage."""
        return estimate_cost_for_model(self.model, self.input_tokens, self.output_tokens)


def estimate_cost_for_model(
    model: str, input_tokens: int, output_tokens: int, config: LLMConfig | None = None
) -> float:
    """Estimate cost in USD for a given model and token counts.

    Args:
        model: Model name (e.g., 'claude-haiku-4-5', 'gpt-4.1-mini')
        input_tokens: Estimated input token count
        output_tokens: Estimated output token count
        config: Optional LLMConfig to check for local providers

    Returns:
        Estimated cost in USD (0.0 for local providers)
    """
    costs = get_model_cost(model, config)
    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    return input_cost + output_cost


class LLMClient:
    """Unified LLM client for Anthropic, OpenAI, Gemini, Ollama, and custom providers."""

    def __init__(self, config: LLMConfig):
        """Initialize LLM client.

        Args:
            config: LLM configuration with provider and API key
        """
        self.config = config
        self.provider = config.provider
        self._client: Any = None
        self._ollama_thinking: dict[str, bool] = {}  # model -> supports thinking

        if config.provider == "anthropic":
            self._client = anthropic.Anthropic(api_key=config.api_key)
        elif config.provider == "openai":
            self._client = openai.OpenAI(api_key=config.api_key)
        elif config.provider == "gemini":
            self._client = genai.Client(api_key=config.api_key)
        elif config.provider == "custom":
            # Custom OpenAI-compatible endpoint
            self._client = openai.OpenAI(
                api_key=config.api_key or "not-needed",  # Use configured key or placeholder
                base_url=config.custom_url,
            )
        # Note: Ollama uses httpx directly, no persistent client needed

    @property
    def is_local(self) -> bool:
        """Whether the provider runs locally (Ollama or custom endpoint)."""
        return self.provider in ("ollama", "custom")

    @property
    def context_window(self) -> int:
        """Effective context window (tokens) available per request."""
        if self.provider == "ollama":
            return self.ollama_num_ctx
        if self.provider == "custom":
            return self.config.custom_context_window
        return MODEL_CONTEXT_LIMITS.get(self.config.model_generation, 128_000)

    @property
    def ollama_num_ctx(self) -> int:
        """num_ctx sent to Ollama on every request."""
        if self.config.ollama_num_ctx > 0:
            return self.config.ollama_num_ctx
        return min(self.config.ollama_context_window, OLLAMA_DEFAULT_NUM_CTX_CAP)

    @property
    def concurrency(self) -> int:
        """Number of LLM calls to run in parallel during batched selection."""
        if self.config.concurrency > 0:
            return self.config.concurrency
        return 1 if self.is_local else 4

    def _output_budget(self, prompt: str, system: str, max_output_tokens: int | None) -> int:
        """Pick a max output token count that fits in the remaining context."""
        requested = max_output_tokens or 8192
        if not self.is_local:
            return requested
        remaining = self.context_window - estimate_tokens(prompt) - estimate_tokens(system) - 64
        if max_output_tokens and remaining < requested:
            logger.warning(
                "Prompt leaves only ~%d tokens of a %d context for output (wanted %d)",
                remaining, self.context_window, requested,
            )
        return min(requested, max(256, remaining))

    def _complete_anthropic(
        self, prompt: str, system: str, model: str, max_output_tokens: int | None = None
    ) -> LLMResponse:
        """Make a completion request to Anthropic."""
        logger.info("Calling Anthropic API with %d char prompt", len(prompt))
        response = self._client.messages.create(
            model=model,
            max_tokens=max_output_tokens or 8192,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        logger.debug("Anthropic response received")

        content = response.content[0].text
        return LLMResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
            truncated=getattr(response, "stop_reason", None) == "max_tokens",
        )

    def _complete_openai(
        self, prompt: str, system: str, model: str, max_output_tokens: int | None = None
    ) -> LLMResponse:
        """Make a completion request to OpenAI or custom OpenAI-compatible endpoint."""
        logger.info("Calling OpenAI-compatible API with %d char prompt", len(prompt))
        response = self._client.chat.completions.create(
            model=model,
            # Bounded by the remaining context for local endpoints, which reject
            # (or silently truncate) requests where prompt + max_tokens > context.
            max_tokens=self._output_budget(prompt, system, max_output_tokens),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        logger.debug("OpenAI-compatible response received")

        choice = response.choices[0]
        content = choice.message.content or ""
        truncated = getattr(choice, "finish_reason", None) == "length"
        if truncated:
            logger.warning("OpenAI-compatible response hit the output token limit")
        # Use getattr for safe token access (custom providers may omit usage)
        input_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            truncated=truncated,
        )

    def _complete_gemini(
        self, prompt: str, system: str, model: str, max_retries: int = 3
    ) -> LLMResponse:
        """Make a completion request to Google Gemini with retry logic.

        Gemini 2.5 models have a known issue where responses can be truncated
        due to internal "thinking" consuming output tokens. We retry on
        truncation (MAX_TOKENS finish reason) or empty responses.
        """
        last_error = None

        for attempt in range(max_retries):
            logger.info("Calling Gemini API (attempt %d/%d) with %d char prompt",
                       attempt + 1, max_retries, len(prompt))

            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    # Don't set max_output_tokens - let the model use what it needs
                    # This avoids truncation from thinking token consumption
                ),
            )

            # Check finish reason
            finish_reason = None
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason

            # Extract content
            usage = response.usage_metadata
            content = response.text if response.text else ""

            logger.info("Gemini response: %d chars, finish_reason=%s, output_tokens=%d",
                       len(content), finish_reason, usage.candidates_token_count if usage else 0)

            # Check for truncation or empty response
            if finish_reason == genai_types.FinishReason.MAX_TOKENS:
                logger.warning("Gemini response truncated (MAX_TOKENS), attempt %d/%d",
                             attempt + 1, max_retries)
                last_error = "Response truncated due to MAX_TOKENS"
                continue

            if not content or len(content.strip()) < 10:
                logger.warning("Gemini returned empty/minimal response, attempt %d/%d",
                             attempt + 1, max_retries)
                last_error = "Empty or minimal response"
                continue

            # Success - return the response
            return LLMResponse(
                content=content,
                input_tokens=usage.prompt_token_count if usage else 0,
                output_tokens=usage.candidates_token_count if usage else 0,
                model=model,
            )

        # All retries exhausted
        raise RuntimeError(f"Gemini API failed after {max_retries} attempts: {last_error}")

    def _complete_ollama(
        self,
        prompt: str,
        system: str,
        model: str,
        timeout: float = 600.0,
        max_output_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Make a completion request to Ollama.

        Args:
            prompt: User prompt
            system: System prompt
            model: Model name (e.g., "llama3:8b")
            timeout: Request timeout in seconds (default 10 minutes for slow hardware)

        Returns:
            LLMResponse with content and token counts
        """
        num_ctx = self.ollama_num_ctx
        num_predict = self._output_budget(prompt, system, max_output_tokens)
        logger.info(
            "Calling Ollama API with %d char prompt (~%d tokens, num_ctx=%d, num_predict=%d)",
            len(prompt), estimate_tokens(prompt) + estimate_tokens(system), num_ctx, num_predict,
        )
        ollama_url = self.config.ollama_url.rstrip("/")

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            # Without num_ctx Ollama uses its small default context and silently
            # drops the start of long prompts (including the instructions).
            "options": {"num_ctx": num_ctx, "num_predict": num_predict},
            "keep_alive": "30m",
        }
        if json_mode:
            payload["format"] = "json"
        if self._ollama_supports_thinking(model):
            # Reasoning models would spend the small num_predict budget of a
            # batch call on hidden thinking and never emit the JSON answer.
            payload["think"] = False

        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{ollama_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

        logger.debug("Ollama response received")

        content = data.get("response", "")
        # Ollama returns token counts in the response
        # prompt_eval_count is input tokens, eval_count is output tokens
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        truncated = data.get("done_reason") == "length"
        if truncated:
            logger.warning("Ollama response hit num_predict=%d and was cut off", num_predict)
        if input_tokens and input_tokens >= num_ctx - 8:
            logger.warning(
                "Ollama prompt filled the %d token context; input was probably truncated",
                num_ctx,
            )

        # Check for empty response (common with small context windows)
        if not content or len(content.strip()) < 2:
            logger.warning("Ollama returned empty response. Input tokens: %d", input_tokens)
            raise RuntimeError(
                "Ollama returned an empty response. This may happen if the context "
                "window is too small for the request. Try reducing the number of "
                "tracks sent to AI or using a model with a larger context window."
            )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            truncated=truncated,
        )

    def _ollama_supports_thinking(self, model: str) -> bool:
        """Whether an Ollama model reports the 'thinking' capability (cached)."""
        if model not in self._ollama_thinking:
            supports = False
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        f"{self.config.ollama_url.rstrip('/')}/api/show", json={"name": model}
                    )
                    resp.raise_for_status()
                    capabilities = resp.json().get("capabilities") or []
                    supports = isinstance(capabilities, list) and "thinking" in capabilities
            except Exception as e:
                logger.debug("Could not read capabilities for %s: %s", model, e)
            self._ollama_thinking[model] = supports
            if supports:
                logger.info("Ollama model %s supports thinking; disabling it for requests", model)
        return self._ollama_thinking[model]

    def _complete(
        self,
        prompt: str,
        system: str,
        model: str,
        max_output_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Make a completion request to the configured provider.

        Args:
            max_output_tokens: Output cap (local providers are further bounded
                by the context left after the prompt)
            json_mode: Ask the provider to constrain output to valid JSON
                (Ollama only; others rely on prompting + repair)
        """
        if self.provider == "anthropic":
            return self._complete_anthropic(prompt, system, model, max_output_tokens)
        elif self.provider in ("openai", "custom"):
            return self._complete_openai(prompt, system, model, max_output_tokens)
        elif self.provider == "gemini":
            return self._complete_gemini(prompt, system, model)
        elif self.provider == "ollama":
            return self._complete_ollama(
                prompt, system, model,
                max_output_tokens=max_output_tokens, json_mode=json_mode,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def analyze(self, prompt: str, system: str, **kwargs: Any) -> LLMResponse:
        """Use the analysis model for understanding tasks.

        Args:
            prompt: User prompt to analyze
            system: System prompt with instructions

        Returns:
            LLMResponse with content and token counts
        """
        model = self.config.model_analysis
        return self._complete(prompt, system, model, **kwargs)

    def generate(self, prompt: str, system: str, **kwargs: Any) -> LLMResponse:
        """Use the generation model for track selection.

        If smart_generation is enabled, uses the analysis model instead.

        Args:
            prompt: User prompt for generation
            system: System prompt with instructions

        Returns:
            LLMResponse with content and token counts
        """
        if self.config.smart_generation:
            model = self.config.model_analysis
        else:
            model = self.config.model_generation
        return self._complete(prompt, system, model, **kwargs)

    def _extract_json_bounds(self, content: str) -> str | None:
        """Extract JSON array or object from content with extra text.

        Finds the first [ or { and its matching closing bracket,
        properly handling nested structures and strings.

        Args:
            content: Raw content that may contain JSON with extra text

        Returns:
            Extracted JSON string, or None if no valid JSON found
        """
        # Find start of JSON
        start_idx = -1
        open_char = None
        close_char = None

        for i, c in enumerate(content):
            if c == '[':
                start_idx = i
                open_char = '['
                close_char = ']'
                break
            elif c == '{':
                start_idx = i
                open_char = '{'
                close_char = '}'
                break

        if start_idx == -1:
            return None

        # Track bracket depth, handling strings
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(content)):
            c = content[i]

            if escape_next:
                escape_next = False
                continue

            if c == '\\' and in_string:
                escape_next = True
                continue

            if c == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    return content[start_idx:i + 1]

        return None

    def parse_json_response(self, response: LLMResponse) -> Any:
        """Parse JSON from LLM response, handling common issues.

        Args:
            response: LLM response to parse

        Returns:
            Parsed JSON data

        Raises:
            ValueError: If JSON cannot be parsed
        """
        content = response.content.strip()
        # Reasoning models (custom endpoints, or Ollama with thinking left on)
        # may prefix the answer with <think>...</think>; drop it.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if content.startswith("<think>"):
            content = content.split("</think>")[-1].strip() if "</think>" in content else ""

        # Check for empty response first
        if not content:
            raise ValueError(
                "LLM returned an empty response. This may happen if the context "
                "window is too small. Try reducing 'Max Tracks to AI' in filters."
            )

        # Try to extract JSON from markdown code blocks
        # Prefer ```json blocks first, then fall back to any code block
        json_match = re.search(r"```json\s*\n?(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if json_match:
            content = json_match.group(1).strip()
        else:
            # Fall back to first code block if no json block found
            match = re.search(r"```(?:\w+)?\s*\n?(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        # Replace curly/smart quotes with straight quotes (common LLM issue)
        content = content.replace("\u201c", '"').replace("\u201d", '"')
        content = content.replace("\u2018", "'").replace("\u2019", "'")

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            original_error = e

            # Strategy 1: If "Extra data" error, try to extract just the JSON portion
            if "Extra data" in str(e):
                extracted = self._extract_json_bounds(content)
                if extracted:
                    try:
                        return json.loads(extracted)
                    except json.JSONDecodeError:
                        pass  # Continue to next strategy

            # Strategy 2: Use json-repair library to fix common LLM JSON issues
            # (trailing commas, unescaped quotes, single quotes, etc.)
            try:
                repaired = repair_json(content, return_objects=True)
                logger.debug("JSON repair succeeded for malformed LLM response")
                return repaired
            except Exception:
                pass  # Fall through to error

            # All strategies failed - report original error
            preview = content[:200] + "..." if len(content) > 200 else content
            raise ValueError(
                f"Failed to parse LLM response as JSON: {original_error}\n"
                f"Response preview: {preview}"
            )


# Global client instance
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient | None:
    """Get the current LLM client instance."""
    return _llm_client


def init_llm_client(config: LLMConfig) -> LLMClient:
    """Initialize or reinitialize the LLM client."""
    global _llm_client
    _llm_client = LLMClient(config)
    return _llm_client


def get_max_tracks_for_model(
    model: str, buffer_percent: float = 0.10, config: LLMConfig | None = None
) -> int:
    """Calculate max tracks that can be sent to a model.

    Args:
        model: Model name
        buffer_percent: Buffer to leave (default 10%)
        config: Optional LLMConfig for local provider context window lookup

    Returns:
        Maximum number of tracks in the candidate pool. Selection is batched to
        fit the context window, so this is no longer bounded by context size.
    """
    return MAX_TRACK_POOL


def get_max_albums_for_model(
    model: str, buffer_percent: float = 0.10, config: LLMConfig | None = None
) -> int:
    """Calculate max albums that can be sent to a model.

    Args:
        model: Model name
        buffer_percent: Buffer to leave (default 10%)
        config: Optional LLMConfig for local provider context window lookup

    Returns:
        Maximum number of albums (minimum 100)
    """
    context_limit = get_model_context_limit(model, config)
    usable_tokens = int(context_limit * (1 - buffer_percent))

    # Reserve ~1000 tokens for system prompt and output
    available_for_albums = usable_tokens - 1000

    max_albums = available_for_albums // TOKENS_PER_ALBUM
    return max(100, max_albums)


def get_model_context_limit(model: str, config: LLMConfig | None = None) -> int:
    """Get the context limit for a model in tokens.

    Args:
        model: Model name
        config: Optional LLMConfig for local provider context window lookup

    Returns:
        Context limit in tokens
    """
    # Check standard model limits first
    if model in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model]

    # For local providers, use config-based context window
    if config:
        if config.provider == "custom":
            return config.custom_context_window
        if config.provider == "ollama":
            # What we actually request per call (num_ctx), not the model's
            # advertised maximum.
            if config.ollama_num_ctx > 0:
                return config.ollama_num_ctx
            return min(config.ollama_context_window, OLLAMA_DEFAULT_NUM_CTX_CAP)

    return 128_000  # Default fallback


def get_model_cost(model: str, config: LLMConfig | None = None) -> dict[str, float]:
    """Get cost per million tokens for a model.

    Args:
        model: Model name
        config: Optional LLMConfig to check for local providers

    Returns:
        Dict with "input" and "output" cost per million tokens
    """
    # Local providers have zero cost
    if config and config.provider in ("ollama", "custom"):
        return {"input": 0.0, "output": 0.0}

    return MODEL_COSTS.get(model, {"input": 1.0, "output": 2.0})


# =============================================================================
# Ollama API Functions
# =============================================================================


def list_ollama_models(ollama_url: str, timeout: float = 5.0) -> OllamaModelsResponse:
    """List available models from Ollama server.

    Args:
        ollama_url: Base URL of Ollama server (e.g., http://localhost:11434)
        timeout: Request timeout in seconds

    Returns:
        OllamaModelsResponse with models list or error
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{ollama_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            data = response.json()

            models = []
            for model_data in data.get("models", []):
                models.append(OllamaModel(
                    name=model_data.get("name", ""),
                    size=model_data.get("size", 0),
                    modified_at=model_data.get("modified_at", ""),
                ))

            return OllamaModelsResponse(models=models)

    except httpx.ConnectError:
        return OllamaModelsResponse(
            error=f"Cannot reach Ollama at {ollama_url}"
        )
    except httpx.TimeoutException:
        return OllamaModelsResponse(
            error=f"Timeout connecting to Ollama at {ollama_url}"
        )
    except Exception as e:
        logger.exception("Error listing Ollama models")
        return OllamaModelsResponse(error=str(e))


def get_ollama_model_info(ollama_url: str, model_name: str, timeout: float = 5.0) -> OllamaModelInfo | None:
    """Get detailed info about an Ollama model including context window.

    Args:
        ollama_url: Base URL of Ollama server
        model_name: Name of the model (e.g., "llama3:8b")
        timeout: Request timeout in seconds

    Returns:
        OllamaModelInfo with context window, or None if not found
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{ollama_url.rstrip('/')}/api/show",
                json={"name": model_name}
            )
            response.raise_for_status()
            data = response.json()

            # Extract context window - check multiple sources in priority order
            context_window = 32768  # Default fallback
            context_detected = False

            # Priority 1: Check model_info for native context_length (most reliable)
            # Keys are like "qwen3.context_length", "llama.context_length", etc.
            model_info = data.get("model_info", {})
            for key, value in model_info.items():
                if key.endswith(".context_length") and isinstance(value, int):
                    context_window = value
                    context_detected = True
                    break

            # Priority 2: Check for explicit num_ctx in parameters or modelfile
            # (This overrides native context if user configured it)
            parameters = data.get("parameters", "")
            modelfile = data.get("modelfile", "")

            for line in (parameters + "\n" + modelfile).split("\n"):
                line = line.strip().lower()
                if "num_ctx" in line:
                    # Extract number from line like "num_ctx 8192" or "PARAMETER num_ctx 8192"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "num_ctx" and i + 1 < len(parts):
                            try:
                                context_window = int(parts[i + 1])
                                context_detected = True
                                break
                            except ValueError:
                                pass

            # Get parameter size from details
            details = data.get("details", {})
            parameter_size = details.get("parameter_size")

            return OllamaModelInfo(
                name=model_name,
                context_window=context_window,
                context_detected=context_detected,
                parameter_size=parameter_size,
            )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        logger.exception("Error getting Ollama model info")
        return None
    except Exception:
        logger.exception("Error getting Ollama model info")
        return None


def get_ollama_status(ollama_url: str, timeout: float = 5.0) -> OllamaStatus:
    """Check Ollama connection status.

    Args:
        ollama_url: Base URL of Ollama server
        timeout: Request timeout in seconds

    Returns:
        OllamaStatus with connection status and model count
    """
    models_response = list_ollama_models(ollama_url, timeout)

    if models_response.error:
        return OllamaStatus(
            connected=False,
            model_count=0,
            error=models_response.error,
        )

    model_count = len(models_response.models)
    if model_count == 0:
        return OllamaStatus(
            connected=True,
            model_count=0,
            error="Connected but no models installed. Run `ollama pull llama3`",
        )

    return OllamaStatus(
        connected=True,
        model_count=model_count,
        error=None,
    )
