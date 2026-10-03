from __future__ import annotations

import pytest

from data_synthesis.model_backends import OpenAICompatibleBackend


def test_api_key_is_environment_only(monkeypatch) -> None:
    monkeypatch.delenv("HARNESS_SQL_TEST_KEY", raising=False)
    backend = OpenAICompatibleBackend(
        model="test-model",
        base_url="http://127.0.0.1:1/v1",
        api_key_env="HARNESS_SQL_TEST_KEY",
    )
    with pytest.raises(RuntimeError, match="environment variable"):
        backend.complete([{"role": "user", "content": "hello"}])
