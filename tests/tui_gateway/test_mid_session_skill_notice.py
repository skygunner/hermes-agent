import base64

from tui_gateway import server


def test_mid_session_skill_notice_does_not_merge_into_user_bubble(tmp_path):
    """Regression for #145362: a must-deliver gateway note appended to a multimodal user
    message must not be concatenated onto the user's bubble body (display-only skip)."""
    image = tmp_path / "sample shot.png"
    image.write_bytes(b"image bytes")
    url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
    path = str(image)
    parts = [{"type": "image_url", "image_url": {"url": url}}]
    notice = "[System: The skills available in this conversation changed after its system prompt was built.\nNow available:\n    - some-skill: a description\n]"
    legacy = [{"type": "text", "text": f"Caption\n\n[Image attached at: {path}]"}, *parts,
              {"type": "text", "text": notice}]
    history = [{"role": "user", "content": legacy}]

    displayed = server._history_to_messages(history)
    assert displayed == [{"role": "user", "text": f"Caption\n\n@image:`{path}`"}]
    assert history[0]["content"][-1]["text"] == notice


def test_mid_session_skill_notice_dropped_from_text_only_user_bubble():
    """A text-only user turn carrying a ``[System: ...]`` note renders only the user text."""
    notice = "[System: Now available: some-skill]"
    history = [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": notice}]}]
    displayed = server._history_to_messages(history)
    assert displayed == [{"role": "user", "text": "hello"}]
