"""مزود OpenAI-Compatible — يعمل مع vLLM وLM Studio وأي نقطة نهاية متوافقة (ADR-0004).

الإعدادات من البيئة فقط — لا نقاط نهاية ولا مفاتيح في الملفات ملتزمة:
  RYKO_LLM_BASE_URL   مثل http://localhost:8000/v1
  RYKO_LLM_API_KEY    مفتاح النقطة (أي قيمة إن كان vLLM محلياً)
  RYKO_LLM_MODEL      مثل Qwen/Qwen2.5-7B-Instruct

عميل بتبعية صفرية: urllib من المكتبة القياسية.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List, Optional

from .base import Provider, ProviderError

_ENV_BASE = "RYKO_LLM_BASE_URL"
_ENV_KEY = "RYKO_LLM_API_KEY"
_ENV_MODEL = "RYKO_LLM_MODEL"


class OpenAICompatProvider(Provider):
    role = "generic"

    def __init__(self, name: str, family: str, base_url: str, api_key: str, model: str,
                 timeout: int = 120, retries: int = 2, role: str = "generic"):
        self.name = name
        self.family = family
        self.role = role
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key  # ryko:secret-ok (archived constructor, not a secret)
        self.model = model
        self.timeout = timeout
        self.retries = retries

    @classmethod
    def from_env(cls, name: str, family: str, role: str = "generic") -> "OpenAICompatProvider":
        base = os.environ.get(_ENV_BASE)
        key = os.environ.get(_ENV_KEY)
        model = os.environ.get(_ENV_MODEL)
        missing = [v for v, val in ((_ENV_BASE, base), (_ENV_KEY, key), (_ENV_MODEL, model)) if not val]
        if missing:
            raise ProviderError(f"متغيرات بيئة مفقودة للمزود {name}: {missing}")
        return cls(name=name, family=family, base_url=base or "", api_key=key or "",
                   model=model or "", role=role)

    def _request_body(self, system: str, prompt: str, temperature: float, max_tokens: int) -> bytes:
        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def generate(self, system: str, prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        url = f"{self.base_url}/chat/completions"
        last_error: Optional[Exception] = None
        for _ in range(self.retries + 1):
            req = urllib.request.Request(
                url,
                data=self._request_body(system, prompt, temperature, max_tokens),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError,
                    json.JSONDecodeError, TimeoutError) as exc:  # noqa: PERF203
                last_error = exc
        raise ProviderError(f"فشل الاستدعاء للمزود {self.name} بعد {self.retries + 1} محاولات: {last_error}")

    def fingerprint(self) -> dict:
        # لا نقطة نهاية ولا مفتاح في البصمة — فقط ما يكفي للتتبع
        return {"name": self.name, "family": self.family, "role": self.role, "model": self.model}
