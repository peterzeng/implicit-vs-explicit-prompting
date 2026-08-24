"""
Multi-provider AI client infrastructure.

Supports:
- OpenAI (GPT-5.4, GPT-5.2, GPT-4o, etc.)
- Google Gemini (gemini-3-pro, gemini-3-flash, gemini-2.5-pro, etc.)

Each role (director/matcher) can use a different provider/model combination.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state import Player

# ---------------------------------------------------------------------------
# Provider Constants
# ---------------------------------------------------------------------------

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"

# Models that support the reasoning_effort parameter.
GPT_5_2_MODELS = frozenset({
    "gpt-5",
    "gpt-5.2",
    "gpt-5.2-mini",
    "gpt-5.2-chat-latest",
    "gpt-5.2-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
})

# Gemini models (for reference)
GEMINI_MODELS = frozenset({
    # Gemini 3 series (latest)
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3-pro-image-preview",
    # Gemini 2.5 series
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.0-flash",
    "gemini-2.0-flash-thinking-exp",
    # Gemini 1.5 series
    "gemini-1.5-pro",
    "gemini-1.5-flash",
})

# Default models per provider
DEFAULT_MODELS = {
    PROVIDER_OPENAI: "gpt-5.2",
    PROVIDER_GEMINI: "gemini-3-flash-preview",
}

# Track which Gemini SDK is available
_GEMINI_SDK_TYPE = None  # "new" for google-genai, "legacy" for google-generativeai

# ---------------------------------------------------------------------------
# Provider Detection
# ---------------------------------------------------------------------------


def detect_provider_from_model(model: str) -> str:
    """Detect which provider a model belongs to based on its name.
    
    Returns PROVIDER_OPENAI or PROVIDER_GEMINI.
    """
    if not model:
        return PROVIDER_OPENAI
    
    model_lower = model.lower()
    
    # Gemini models
    if model_lower.startswith("gemini"):
        return PROVIDER_GEMINI
    
    # Default to OpenAI
    return PROVIDER_OPENAI


def is_gpt_5_2_model(model: str) -> bool:
    """Check if a model supports GPT-5.x API features (reasoning_effort).
    
    Returns True for GPT-5, GPT-5.2, GPT-5.4, GPT-5.5, and GPT-5 snapshots that
    support the reasoning_effort parameter.
    """
    if not model:
        return False
    if model in GPT_5_2_MODELS:
        return True
    model_lower = model.lower()
    # Match GPT-5 snapshots plus GPT-5.x variants that support reasoning_effort.
    return (
        model_lower == "gpt-5"
        or model_lower.startswith("gpt-5-")
        or model_lower.startswith("gpt-5.2")
        or model_lower.startswith("gpt-5.4")
        or model_lower.startswith("gpt-5.5")
    )


def reasoning_effort_for_api(model: str, reasoning_effort: str | None) -> str | None:
    """Return a reasoning_effort value that is valid to send for a model."""
    if not is_gpt_5_2_model(model):
        return None
    effort = reasoning_effort or "none"
    model_lower = model.lower() if model else ""
    if effort == "none" and (model_lower == "gpt-5" or model_lower.startswith("gpt-5-")):
        return None
    return effort


def uses_max_completion_tokens(model: str) -> bool:
    """Check if a model requires max_completion_tokens instead of max_tokens."""
    if not model:
        return False
    model_lower = model.lower()
    if model_lower.startswith(("o1", "o3")):
        return True
    if model_lower.startswith("gpt-5"):
        return True
    return False


# ---------------------------------------------------------------------------
# Configuration Getters
# ---------------------------------------------------------------------------


def get_model_config_for_role(player: "Player" | None, role: str) -> dict[str, Any]:
    """Get model configuration for a specific role (director or matcher).
    
    Returns dict with keys:
        - provider: "openai" or "gemini"
        - model: model name string
        - reasoning_effort: for GPT-5.2+ models
        - thinking_level: for Gemini 3+ models ("low", "medium", "high")
        - thinking_budget: for Gemini 2.5 models (legacy, integer)
    
    Configuration priority:
    1. Session config role-specific: ai_director_model / ai_matcher_model
    2. Environment variable: OPENAI_MODEL / GEMINI_MODEL
    3. Defaults
    """
    config = {
        "provider": PROVIDER_OPENAI,
        "model": DEFAULT_MODELS[PROVIDER_OPENAI],
        "reasoning_effort": "none",
        "thinking_level": None,  # Gemini 3: "low", "medium", "high"
        "thinking_budget": None,  # Gemini 2.5 legacy: integer
    }
    
    session_cfg = {}
    if player is not None:
        try:
            if hasattr(player, "session") and player.session:
                session_cfg = player.session.config or {}
        except Exception:
            pass
    
    # Check role-specific model first
    role_model_key = f"ai_{role}_model"
    role_provider_key = f"ai_{role}_provider"
    role_reasoning_key = f"ai_{role}_reasoning_effort"
    role_thinking_level_key = f"ai_{role}_thinking_level"
    role_thinking_budget_key = f"ai_{role}_thinking_budget"
    
    model = None
    provider = None
    
    # Priority 1: Role-specific session config
    if session_cfg.get(role_model_key):
        model = session_cfg[role_model_key]
    if session_cfg.get(role_provider_key):
        provider = session_cfg[role_provider_key]
    
    # Priority 2: Global provider config
    if not provider and session_cfg.get("ai_provider"):
        provider = session_cfg["ai_provider"]
    
    # Priority 3: Environment variables
    if not model:
        model = os.environ.get("OPENAI_MODEL") or os.environ.get("GEMINI_MODEL")
    
    # Apply model
    if model:
        config["model"] = model
    
    # Detect provider from model if not explicitly set
    if not provider:
        provider = detect_provider_from_model(config["model"])
    config["provider"] = provider
    
    # Get reasoning effort (OpenAI GPT-5.2+)
    if session_cfg.get(role_reasoning_key):
        config["reasoning_effort"] = session_cfg[role_reasoning_key]
    elif session_cfg.get("ai_reasoning_effort"):
        config["reasoning_effort"] = session_cfg["ai_reasoning_effort"]
    
    # Get thinking level (Gemini 3+)
    if session_cfg.get(role_thinking_level_key):
        config["thinking_level"] = session_cfg[role_thinking_level_key]
    elif session_cfg.get("ai_thinking_level"):
        config["thinking_level"] = session_cfg["ai_thinking_level"]
    
    # Get thinking budget (Gemini 2.5 legacy)
    if session_cfg.get(role_thinking_budget_key):
        config["thinking_budget"] = session_cfg[role_thinking_budget_key]
    elif session_cfg.get("ai_thinking_budget"):
        config["thinking_budget"] = session_cfg["ai_thinking_budget"]
    
    return config


# ---------------------------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------------------------

_openai_client = None


def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    
    if _openai_client is not None:
        return _openai_client
    
    try:
        from openai import OpenAI
    except ImportError:
        logging.warning("[AI_PROVIDERS] OpenAI library not installed")
        return None
    
    base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if base_url:
        _openai_client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
        )
    elif api_key:
        _openai_client = OpenAI(api_key=api_key)
    else:
        logging.warning("[AI_PROVIDERS] No OpenAI API key configured")
        return None
    
    return _openai_client


# ---------------------------------------------------------------------------
# Gemini Client (supports new google-genai SDK and legacy google-generativeai)
# ---------------------------------------------------------------------------

_gemini_client = None


def _detect_gemini_sdk():
    """Detect which Gemini SDK is available.
    
    Returns "new" for google-genai, "legacy" for google-generativeai, None if neither.
    """
    global _GEMINI_SDK_TYPE
    
    if _GEMINI_SDK_TYPE is not None:
        return _GEMINI_SDK_TYPE
    
    # Try new SDK first (google-genai)
    try:
        from google import genai
        from google.genai import types
        _GEMINI_SDK_TYPE = "new"
        logging.info("[AI_PROVIDERS] Using new google-genai SDK")
        return "new"
    except ImportError:
        pass
    
    # Fall back to legacy SDK (google-generativeai)
    try:
        import google.generativeai as genai
        _GEMINI_SDK_TYPE = "legacy"
        logging.info("[AI_PROVIDERS] Using legacy google-generativeai SDK")
        return "legacy"
    except ImportError:
        pass
    
    _GEMINI_SDK_TYPE = None
    return None


def get_gemini_client():
    """Get or create Gemini client.
    
    Returns:
        - For new SDK: genai.Client instance
        - For legacy SDK: configured genai module
        - None if no SDK available or no API key
    """
    global _gemini_client
    
    if _gemini_client is not None:
        return _gemini_client
    
    sdk_type = _detect_gemini_sdk()
    if sdk_type is None:
        logging.warning("[AI_PROVIDERS] No Gemini SDK installed. Install google-genai (recommended) or google-generativeai")
        return None
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logging.warning("[AI_PROVIDERS] No Gemini API key configured (set GEMINI_API_KEY)")
        return None
    
    if sdk_type == "new":
        # New google-genai SDK
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
    else:
        # Legacy google-generativeai SDK
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_client = genai
    
    return _gemini_client


# ---------------------------------------------------------------------------
# Message Format Converters
# ---------------------------------------------------------------------------


def _parse_image_data_url(image_url: str) -> dict | None:
    """Parse a data URL into Gemini inline_data format."""
    if not image_url.startswith("data:"):
        return None
    try:
        # Format: data:image/png;base64,xxxxx
        header, b64_data = image_url.split(",", 1)
        mime_type = header.split(";")[0].split(":")[1]
        return {
            "mime_type": mime_type,
            "data": b64_data,
        }
    except Exception as e:
        logging.warning("[AI_PROVIDERS] Failed to parse image data URL: %s", e)
        return None


def convert_messages_to_gemini_format(messages: list[dict], sdk_type: str = "new") -> tuple[list | str, str | None]:
    """Convert OpenAI-style messages to Gemini format.
    
    Args:
        messages: List of OpenAI-format messages
        sdk_type: "new" for google-genai SDK, "legacy" for google-generativeai
    
    Returns:
        (contents, system_instruction)
        - contents: For new SDK, can be a simple string or list of Content objects
                   For legacy SDK, list of content dicts
        - system_instruction: combined system messages
    """
    system_parts = []
    contents = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role in ("developer", "system"):
            # Collect high-priority instruction messages for system_instruction.
            # OpenAI's developer role maps cleanly onto Gemini's system instruction.
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
        else:
            # Map assistant -> model, user -> user
            gemini_role = "model" if role == "assistant" else "user"
            
            # Build parts list
            parts = []
            
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append({"text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            # Handle image
                            image_url = part.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                inline_data = _parse_image_data_url(image_url)
                                if inline_data:
                                    parts.append({"inline_data": inline_data})
                            else:
                                logging.warning("[AI_PROVIDERS] Gemini requires inline images, not URLs")
            
            if parts:
                contents.append({
                    "role": gemini_role,
                    "parts": parts,
                })
    
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    
    # Gemini requires alternating roles - merge consecutive same-role messages
    merged_contents = []
    for content in contents:
        if merged_contents and merged_contents[-1]["role"] == content["role"]:
            # Merge parts
            merged_contents[-1]["parts"].extend(content["parts"])
        else:
            merged_contents.append(content)
    
    return merged_contents, system_instruction


# ---------------------------------------------------------------------------
# Unified API Call
# ---------------------------------------------------------------------------


def call_ai_api(
    messages: list[dict],
    model_config: dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> str | None:
    """Make an API call to the configured AI provider.
    
    Args:
        messages: OpenAI-format message list
        model_config: dict from get_model_config_for_role()
        temperature: optional temperature override
        max_tokens: optional max tokens
        response_format: optional response format (e.g., {"type": "json_object"})
    
    Returns:
        Response text string or None on failure
    """
    provider = model_config.get("provider", PROVIDER_OPENAI)
    model = model_config.get("model", DEFAULT_MODELS[provider])
    
    # logging.info(
        # "[AI_PROVIDERS] Calling %s with model %s",
        # provider.upper(), model
    # )
    
    if provider == PROVIDER_GEMINI:
        return _call_gemini_api(messages, model_config, temperature, max_tokens, response_format)
    else:
        return _call_openai_api(messages, model_config, temperature, max_tokens, response_format)


def _call_openai_api(
    messages: list[dict],
    model_config: dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> str | None:
    """Make OpenAI API call."""
    client = get_openai_client()
    if client is None:
        return None
    
    model = model_config.get("model", "gpt-5.2")
    reasoning_effort = model_config.get("reasoning_effort", "none")
    
    # Build kwargs
    kwargs = {
        "model": model,
        "messages": messages,
    }
    
    # Handle reasoning models (no custom temperature)
    model_lower = model.lower() if model else ""
    is_o_series = model_lower.startswith(("o1", "o3"))
    is_reasoning_mode = is_o_series or is_gpt_5_2_model(model) or (reasoning_effort != "none")
    
    if temperature is not None and not is_reasoning_mode:
        kwargs["temperature"] = temperature
    
    if max_tokens is not None:
        if uses_max_completion_tokens(model):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
    
    if response_format is not None:
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {})
            if hasattr(schema, "model_json_schema"):
                # Pass the raw Pydantic class to the beta parse method
                kwargs["response_format"] = schema
            else:
                kwargs["response_format"] = response_format
        else:
            kwargs["response_format"] = response_format
    
    api_reasoning_effort = reasoning_effort_for_api(model, reasoning_effort)
    if api_reasoning_effort is not None:
        kwargs["reasoning_effort"] = api_reasoning_effort
    
    try:
        if response_format and response_format.get("type") == "json_schema" and hasattr(response_format.get("json_schema", {}), "model_json_schema"):
            # Use beta parse for Pydantic schemas
            completion = client.beta.chat.completions.parse(**kwargs)
        else:
            completion = client.chat.completions.create(**kwargs)
        reply = completion.choices[0].message.content
        if isinstance(reply, list):
            reply = "".join(
                (part.get("text", "") if isinstance(part, dict) else str(getattr(part, 'text', '')))
                for part in reply
            )
        return (reply or "").strip()
    except Exception as e:
        logging.error("[AI_PROVIDERS] OpenAI API error: %s: %s", type(e).__name__, e)
        return None


def _call_gemini_api(
    messages: list[dict],
    model_config: dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> str | None:
    """Make Google Gemini API call.
    
    Supports both new google-genai SDK and legacy google-generativeai SDK.
    """
    client = get_gemini_client()
    if client is None:
        return None
    
    sdk_type = _detect_gemini_sdk()
    
    if sdk_type == "new":
        return _call_gemini_new_sdk(client, messages, model_config, temperature, max_tokens, response_format)
    else:
        return _call_gemini_legacy_sdk(client, messages, model_config, temperature, max_tokens, response_format)


def _call_gemini_new_sdk(
    client,
    messages: list[dict],
    model_config: dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> str | None:
    """Make Gemini API call using new google-genai SDK.
    
    Reference: https://ai.google.dev/gemini-api/docs/gemini-3
    """
    from google.genai import types
    
    model_name = model_config.get("model", "gemini-3-flash-preview")
    thinking_level = model_config.get("thinking_level")  # "low", "medium", "high"
    thinking_budget = model_config.get("thinking_budget")  # Legacy integer value
    
    logging.info("[AI_PROVIDERS] Gemini new SDK - model: %s, thinking_level: %s, thinking_budget: %s",
                 model_name, thinking_level, thinking_budget)
    
    try:
        # Convert messages to Gemini format
        logging.debug("[AI_PROVIDERS] Converting %d messages to Gemini format", len(messages))
        contents, system_instruction = convert_messages_to_gemini_format(messages, sdk_type="new")
        logging.debug("[AI_PROVIDERS] Converted to %d content items, system_instruction: %s",
                     len(contents) if isinstance(contents, list) else 1,
                     "yes" if system_instruction else "no")
        
        # Build generation config kwargs
        config_kwargs = {}
        
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        
        # Handle JSON response format
        if response_format:
            if response_format.get("type") == "json_object":
                config_kwargs["response_mime_type"] = "application/json"
            elif response_format.get("type") == "json_schema":
                config_kwargs["response_mime_type"] = "application/json"
                schema = response_format.get("json_schema", {})
                if hasattr(schema, "model_json_schema"):
                    config_kwargs["response_schema"] = schema
        
        # Handle system instruction
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        
        # Handle thinking configuration for Gemini 3+ and 2.5 models
        is_gemini_3 = "gemini-3" in model_name.lower()
        is_thinking_model = "2.5" in model_name or "3" in model_name or "thinking" in model_name.lower()
        
        logging.debug("[AI_PROVIDERS] is_gemini_3: %s, is_thinking_model: %s", is_gemini_3, is_thinking_model)
        
        if is_thinking_model:
            if thinking_level:
                # Use thinking_level for Gemini 3+ (preferred)
                logging.info("[AI_PROVIDERS] Using thinking_level: %s", thinking_level)
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level
                )
            elif thinking_budget is not None:
                # Use thinking_budget for backward compatibility
                logging.info("[AI_PROVIDERS] Using thinking_budget: %s", thinking_budget)
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )
            elif is_gemini_3:
                # Default to "low" for Gemini 3 to minimize latency
                logging.info("[AI_PROVIDERS] Using default thinking_level: low for Gemini 3")
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level="low"
                )
            else:
                # Default minimal thinking for 2.5 models
                logging.info("[AI_PROVIDERS] Using default thinking_budget: 0 for Gemini 2.5")
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=0
                )
        
        # Create config object
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        logging.info("[AI_PROVIDERS] Config created, making API call to %s...", model_name)
        
        # Make API call
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        
        logging.info("[AI_PROVIDERS] Gemini API call completed, processing response...")
        
        # Extract text from response
        if hasattr(response, 'text'):
            result = response.text.strip()
            logging.info("[AI_PROVIDERS] Got response via .text attribute (%d chars)", len(result))
            return result
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                parts = candidate.content.parts
                if parts:
                    text_parts = [p.text for p in parts if hasattr(p, 'text')]
                    result = "".join(text_parts).strip()
                    logging.info("[AI_PROVIDERS] Got response via candidates (%d chars)", len(result))
                    return result
        
        logging.warning("[AI_PROVIDERS] Gemini (new SDK) returned empty response. Response object: %s", type(response))
        return None
        
    except Exception as e:
        logging.error("[AI_PROVIDERS] Gemini (new SDK) API error: %s: %s", type(e).__name__, e)
        import traceback
        logging.error(traceback.format_exc())
        return None


def _call_gemini_legacy_sdk(
    genai,
    messages: list[dict],
    model_config: dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> str | None:
    """Make Gemini API call using legacy google-generativeai SDK."""
    model_name = model_config.get("model", "gemini-3-flash-preview")
    
    try:
        # Convert messages to Gemini format
        contents, system_instruction = convert_messages_to_gemini_format(messages, sdk_type="legacy")
        
        # Build generation config
        generation_config = {}
        
        if temperature is not None:
            generation_config["temperature"] = temperature
        
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        
        # Handle JSON response format. The legacy SDK does not reliably support
        # response schemas, but it can still be told to emit JSON for both
        # json_object and json_schema prompt contracts.
        if response_format and response_format.get("type") in ("json_object", "json_schema"):
            generation_config["response_mime_type"] = "application/json"
        
        # Create model instance
        model_kwargs = {}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config if generation_config else None,
            **model_kwargs,
        )
        
        # Generate response (legacy SDK doesn't reliably support thinking_config)
        response = model.generate_content(contents)
        
        # Extract text from response
        if hasattr(response, 'text'):
            return response.text.strip()
        elif hasattr(response, 'parts') and response.parts:
            return response.parts[0].text.strip()
        else:
            logging.warning("[AI_PROVIDERS] Gemini (legacy SDK) returned empty response")
            return None
            
    except Exception as e:
        logging.error("[AI_PROVIDERS] Gemini (legacy SDK) API error: %s: %s", type(e).__name__, e)
        import traceback
        logging.error(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def get_ai_client_for_role(player: "Player" | None, role: str):
    """Get the appropriate AI client for a role.
    
    Returns (client, model_config) tuple.
    """
    config = get_model_config_for_role(player, role)
    provider = config.get("provider", PROVIDER_OPENAI)
    
    if provider == PROVIDER_GEMINI:
        client = get_gemini_client()
    else:
        client = get_openai_client()
    
    return client, config


def has_ai_client_for_role(player: "Player" | None, role: str) -> bool:
    """Check if an AI client is available for a role."""
    client, _ = get_ai_client_for_role(player, role)
    return client is not None
