"""
llm/llm_client.py
==================
Pluggable LLM backend used by ReactAgent as the "Frozen Policy LLM" module
from the SIR architecture graph.

The paper's main experiments use PaLM-540B, which is not publicly accessible
(SIR implementation_assumptions[0], confidence 0.95). We therefore expose an
abstract `LLMClient` interface with two concrete backends:
  - `OpenAIClient`: any OpenAI-API-compatible chat/completions endpoint,
    reached through the `openai` Python SDK with a configurable `base_url`.
    Default substitute for the paper's PaLM-540B is DeepSeek's API
    (https://api.deepseek.com, models "deepseek-chat" / "deepseek-reasoner"),
    which is OpenAI-compatible -- no separate SDK is required. Any other
    OpenAI-compatible provider (including OpenAI itself) also works by
    setting `base_url`/`api_key_env_var` accordingly. The paper's own
    Appendix A.1 ablation similarly substitutes GPT-3 text-davinci-002 for
    PaLM-540B.
  - `HFLocalClient`: a local HuggingFace causal LM, used for the optional
    Appendix B.1 finetuning experiment (finetuned PaLM-8B/62B substitute
    checkpoints).

Paper section: Section 2 (policy pi(a_t | c_t)); Section 3.1 footnote 1;
Appendix A.1 (GPT-3 substitution); Appendix B.1 (finetuning).
"""

from __future__ import annotations

import abc
import os

from tenacity import retry, stop_after_attempt, wait_exponential


class LLMClient(abc.ABC):
    """Abstract interface decoupling the ReAct loop from any specific LLM backend.

    Implementations must be stateless with respect to the trajectory -- all
    conversational/episodic state lives in TrajectoryContext
    (agents/react_agent.py), not in the client.
    """

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        stop_sequences: list[str],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """Generate a single text continuation.

        Args:
            prompt: Full prompt string, i.e. few-shot prefix + serialized
                trajectory so far (TrajectoryContext.to_prompt_string()).
            stop_sequences: Strings at which generation should stop (e.g. a
                newline, so the model emits exactly one Thought/Action line
                at a time, matching the paper's step-by-step format).
            temperature: Sampling temperature. 0.0 for greedy decoding (main
                ReAct/Act/CoT results); 0.7 for CoT-SC (Section 3.2).
            max_tokens: Maximum number of generated tokens for this call.

        Returns:
            The raw generated text (not yet parsed into a Thought/DomainAction).
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}()"


#: Known OpenAI-compatible providers, mapped to (default base_url, default env var).
#: DeepSeek is the default remote substitute for the paper's PaLM-540B (see module
#: docstring); OpenAI itself remains supported for anyone who wants to switch back.
_PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}


class OpenAIClient(LLMClient):
    """LLMClient backed by any OpenAI-API-compatible chat/completions endpoint.

    Despite the class name (kept for backward compatibility / minimal diff),
    this is provider-agnostic: it talks to whatever `base_url` it's given
    using the standard `openai` Python SDK, since DeepSeek, OpenAI, and many
    other providers all expose the same request/response shape. Default
    substitute for the paper's PaLM-540B is DeepSeek (see module docstring).
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str = "deepseek",
        api_key_env_var: str | None = None,
    ) -> None:
        """
        Args:
            model_name: Model identifier, e.g. "deepseek-chat". ASSUMED: the
                paper does not specify a substitute model; this default is a
                config-level choice (configs/config.yaml::model.model_name).
            api_key: API key. If not provided, resolved from the environment
                -- see `api_key_env_var` / `provider` below.
            base_url: Custom API base URL. If not provided, defaults based on
                `provider` (DeepSeek: https://api.deepseek.com; OpenAI:
                https://api.openai.com/v1). Set explicitly for any other
                OpenAI-compatible provider.
            provider: One of "deepseek" (default) or "openai", used only to
                pick sensible defaults for `base_url` and the API-key
                environment variable when those aren't given explicitly.
            api_key_env_var: Explicit environment variable name to read the
                API key from. Overrides the `provider` default. Useful for
                a fully custom OpenAI-compatible provider.
        """
        try:
            import openai
        except ImportError as e:  # pragma: no cover - exercised only if openai is missing
            raise ImportError(
                "OpenAIClient requires the 'openai' package. Install it via "
                "`pip install -r requirements.txt`."
            ) from e

        if provider not in _PROVIDER_DEFAULTS and (base_url is None or api_key_env_var is None):
            raise ValueError(
                f"Unknown provider {provider!r}. Either use one of "
                f"{list(_PROVIDER_DEFAULTS)}, or pass both base_url and "
                f"api_key_env_var explicitly for a custom provider."
            )

        default_base_url, default_env_var = _PROVIDER_DEFAULTS.get(provider, (None, None))
        resolved_base_url = base_url or default_base_url
        resolved_env_var = api_key_env_var or default_env_var

        resolved_key = api_key or os.environ.get(resolved_env_var)
        if not resolved_key:
            raise ValueError(
                f"No API key provided for provider {provider!r}. Set "
                f"{resolved_env_var} in your .env file (see .env.example) "
                f"or pass api_key explicitly."
            )

        self.provider = provider
        self.model_name = model_name
        self._client = openai.OpenAI(api_key=resolved_key, base_url=resolved_base_url)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def generate(
        self,
        prompt: str,
        stop_sequences: list[str],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences or None,
        )
        return response.choices[0].message.content or ""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"OpenAIClient(provider={self.provider!r}, model_name={self.model_name!r})"


class HFLocalClient(LLMClient):
    """LLMClient backed by a local HuggingFace CausalLM.

    Used primarily for evaluating checkpoints produced by the optional
    Appendix B.1 finetuning experiment (training/finetune.py), where a
    substitute open-weight model is finetuned to imitate ReAct/Act/Standard/
    CoT trajectories.
    """

    def __init__(self, model_name_or_path: str, device: str = "cpu", precision: str = "bf16") -> None:
        """
        Args:
            model_name_or_path: HuggingFace hub id or local checkpoint path.
            device: "cpu" or "cuda".
            precision: One of "fp32", "fp16", "bf16". ASSUMED default "bf16"
                (SIR training_pipeline confidence 0.5, precision not stated
                in the paper text).
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "HFLocalClient requires 'torch' and 'transformers'. Install "
                "them via `pip install -r requirements.txt`."
            ) from e

        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        if precision not in dtype_map:
            raise ValueError(f"precision must be one of {list(dtype_map)}, got {precision!r}")

        self.model_name_or_path = model_name_or_path
        self.device = device
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype_map[precision]
        ).to(device)
        self._model.eval()

    def generate(
        self,
        prompt: str,
        stop_sequences: list[str],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature > 0.0
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        generated = self._tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for stop in stop_sequences:
            idx = generated.find(stop)
            if idx != -1:
                generated = generated[:idx]
        return generated

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"HFLocalClient(model_name_or_path={self.model_name_or_path!r}, device={self.device!r})"


def build_llm_client(config_model_section: dict) -> LLMClient:
    """Factory that builds the correct LLMClient from configs/config.yaml::model.

    `llm_backend: openai` is retained as the config value for ANY
    OpenAI-API-compatible remote provider (not just OpenAI itself) -- which
    concrete provider it hits is controlled by `model.provider` (default:
    "deepseek"), with `model.base_url` / `model.api_key_env_var` available
    as explicit overrides for a fully custom provider. This keeps the config
    schema stable while making DeepSeek the default remote backend.

    Args:
        config_model_section: The `model` sub-dict of AgentConfig.raw.

    Returns:
        A constructed LLMClient instance.

    Raises:
        ValueError: if `llm_backend` is not one of the supported values.
    """
    backend = config_model_section.get("llm_backend")
    if backend == "openai":
        return OpenAIClient(
            model_name=config_model_section["model_name"],
            provider=config_model_section.get("provider", "deepseek"),
            base_url=config_model_section.get("base_url"),
            api_key_env_var=config_model_section.get("api_key_env_var"),
        )
    if backend == "hf_local":
        return HFLocalClient(model_name_or_path=config_model_section["hf_model_name_or_path"])
    raise ValueError(f"Unknown model.llm_backend: {backend!r}. Expected 'openai' or 'hf_local'.")
