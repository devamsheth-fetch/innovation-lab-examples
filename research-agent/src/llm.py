import os
import requests
from typing import Optional, List, Any
from pydantic import Field
from langchain_core.language_models.llms import LLM

class ASI1LLM(LLM):
    """
    Custom LangChain-compatible wrapper for the Fetch.ai ASI1 LLM.
    """
    api_key: str = Field(default_factory=lambda: os.getenv("ASI_LLM_KEY", ""))
    api_url: str = Field(default_factory=lambda: os.getenv("ASI_LLM_URL", "https://api.asi1.ai/v1/chat/completions"))
    model: str = Field(default_factory=lambda: os.getenv("ASI_LLM_MODEL", "asi1"))
    temperature: float = Field(default=0.7)
    fun_mode: bool = Field(default=False)
    web_search: bool = Field(default=False)
    enable_stream: bool = Field(default=False)
    max_tokens: int = Field(default=4096)

    @property
    def _llm_type(self) -> str:
        return "asi1_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        if not self.api_key:
            raise ValueError("ASI_LLM_KEY is not set.")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "fun_mode": self.fun_mode,
            "web_search": self.web_search,
            "stream": self.enable_stream,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        
        return (
            response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
