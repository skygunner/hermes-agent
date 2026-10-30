"""Test that call_llm vision path passes resolved provider args, not raw ones."""

from unittest.mock import patch, MagicMock


def test_vision_call_uses_resolved_provider_args():
    """Resolved provider/model/key/url from config must reach resolve_vision_provider_client."""
    from agent.auxiliary_client import call_llm

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="description"))],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
    )

    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("my-resolved-provider", "my-resolved-model", "http://resolved", "resolved-key", "chat_completions"),
    ), patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        return_value=("my-resolved-provider", fake_client, "my-resolved-model"),
    ) as mock_vision:
        call_llm(
            "vision",
            provider="raw-provider",
            model="raw-model",
            base_url="http://raw",
            api_key="raw-key",
            messages=[{"role": "user", "content": "describe this"}],
        )

    # The resolved values must be passed, not the raw call_llm arguments
    call_args = mock_vision.call_args
    assert call_args.kwargs["provider"] == "my-resolved-provider"
    assert call_args.kwargs["model"] == "my-resolved-model"
    assert call_args.kwargs["base_url"] == "http://resolved"
    assert call_args.kwargs["api_key"] == "resolved-key"


def test_vision_base_url_override_keeps_explicit_provider():
    """Explicit provider should still drive credential resolution with custom base_url."""
    from agent.auxiliary_client import resolve_vision_provider_client

    fake_client = MagicMock()
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=(
            "zai",
            "glm-4v",
            "https://open.bigmodel.cn/api/paas/v4",
            None,
            "chat_completions",
        ),
    ), patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fake_client, "glm-4v"),
    ) as mock_resolve:
        provider, client, model = resolve_vision_provider_client()

    assert provider == "zai"
    assert client is fake_client
    assert model == "glm-4v"
    assert mock_resolve.call_args.args[0] == "zai"
    assert mock_resolve.call_args.kwargs["explicit_base_url"] == "https://open.bigmodel.cn/api/paas/v4"


def test_vision_unavailable_provider_fallback_drops_pinned_model(tmp_path, monkeypatch):
    """Regression for issue #145182 (reporter's setup): `auxiliary.vision` pinned to an
    unresolvable provider WITH a pinned model; the explicit->auto fallback must not
    resurrect auxiliary.vision.model for the auto-detected backend (400 invalid model ID).
    Patches only `resolve_provider_client` (dead provider -> unavailable) so the fallback
    path itself runs against the real resolution chain and the reporter's config.yaml."""
    import agent.auxiliary_client as aux

    config = (
        "model:\n"
        "  provider: openrouter\n"
        "  default: deepseek/deepseek-v4.1-flash-1017\n"
        "auxiliary:\n"
        "  vision:\n"
        "    provider: custom:does-not-exist\n"
        "    model: Qwen3.5-9B-oQ4e-mtp\n"
    )
    (tmp_path / "config.yaml").write_text(config)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    aux._client_cache.clear()

    runtime = {"provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash-1017"}

    def fake_resolve_provider_client(*args, **kwargs):
        provider = args[0] if args else kwargs.get("provider")
        if str(provider).startswith("custom:does-not-exist"):
            return None, None
        return MagicMock(name="openrouter-client"), "deepseek/deepseek-v4.1-flash-1017"

    try:
        with patch.object(aux, "resolve_provider_client", side_effect=fake_resolve_provider_client), \
             patch.object(aux, "_main_model_supports_vision", return_value=True), \
             patch.object(aux, "_resolve_provider_vision_default", return_value=None):
            route = aux._resolve_call_client(
                task="vision", provider=None, model=None, base_url=None, api_key=None,
                resolved_provider="custom:does-not-exist", resolved_model="Qwen3.5-9B-oQ4e-mtp",
                resolved_base_url=None, resolved_api_key=None, resolved_api_mode=None,
                main_runtime=runtime, async_mode=False)
    finally:
        aux._client_cache.clear()

    assert route.effective_provider == "openrouter"
    # The pinned model belongs to the dead provider; the auto backend uses its own default.
    assert route.final_model != "Qwen3.5-9B-oQ4e-mtp"
