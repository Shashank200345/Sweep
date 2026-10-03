import pytest

from sweep.cli import main


def test_models_without_key_is_a_clear_message(monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        main(["models"])
    assert "TYPESAFE_API_KEY is not set" in str(e.value)
