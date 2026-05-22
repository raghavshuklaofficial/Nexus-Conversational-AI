"""
Local HuggingFace text generation provider.

Uses a small model (e.g. Qwen1.5-0.5B-Chat, TinyLlama, or Phi) so the project
runs end-to-end without paid APIs.  Model inference runs in a bounded
thread-pool executor to keep the async event loop responsive.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

import structlog

from nexus.domain.ports import LLMProviderPort

logger = structlog.get_logger(__name__)

_LLM_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")


class LocalHuggingFaceProvider(LLMProviderPort):
    """Generate text with a local HuggingFace causal-LM pipeline."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen1.5-0.5B-Chat",
        device: str = "cpu",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        self._model_name = model_name
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._do_sample = do_sample
        self._repetition_penalty = 1.2
        self._pipeline = None
        self._semaphore = asyncio.Semaphore(1)  # single-request guard

    async def initialize(self) -> None:
        if self._pipeline is not None:
            return

        logger.info("loading_llm", model=self._model_name, device=self._device)

        loop = asyncio.get_event_loop()

        def _load():
            from transformers import pipeline
            return pipeline(
                "text-generation",
                model=self._model_name,
                device=0 if self._device == "cuda" else -1,
                torch_dtype="auto",
            )

        self._pipeline = await loop.run_in_executor(_LLM_EXECUTOR, _load)
        logger.info("llm_loaded", model=self._model_name)

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self._pipeline:
            raise RuntimeError("LLM not initialized")

        max_tokens = kwargs.get("max_new_tokens", self._max_new_tokens)
        temperature = kwargs.get("temperature", self._temperature)

        async with self._semaphore:
            loop = asyncio.get_event_loop()

            def _generate():
                # If model is an instruct model, format as ChatML internally
                formatted_prompt = prompt
                if "qwen" in self._model_name.lower():
                    # We pass the entire prompt minus the trailing "Assistant:" 
                    # into the user message so we don't lose RAG context.
                    clean_prompt = prompt.replace("Assistant:", "").strip()
                    messages = [
                        {"role": "system", "content": "You are Nexus, a helpful AI assistant."},
                        {"role": "user", "content": clean_prompt}
                    ]
                    
                    formatted_prompt = self._pipeline.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                results = self._pipeline(
                    formatted_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=self._do_sample,
                    repetition_penalty=self._repetition_penalty,
                    num_return_sequences=1,
                    return_full_text=False,
                    pad_token_id=self._pipeline.tokenizer.eos_token_id,
                )
                text = results[0]["generated_text"].strip()
                
                # Cleanup: stop generation if model starts talking to itself
                for stop_token in ["\nUser:", "User:", "\nAssistant:", "Assistant:", "<|im_end|>", "<|endoftext|>"]:
                    if stop_token in text:
                        text = text[:text.find(stop_token)].strip()
                return text

            return await loop.run_in_executor(_LLM_EXECUTOR, _generate)

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Simulate streaming by yielding sentence fragments."""
        full_text = await self.generate(prompt, **kwargs)
        # Yield in sentence-like chunks
        words = full_text.split()
        chunk_size = 5
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            yield chunk + " "
            await asyncio.sleep(0.05)

    async def health_check(self) -> bool:
        return self._pipeline is not None

    @property
    def model_name(self) -> str:
        return self._model_name
