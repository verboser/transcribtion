import pytest

from main import parse_args


def test_parse_args_rejects_zero_runs(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--runs", "0"])

    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_zero_anchor_group_size(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--anchor-group-size", "0"])

    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_negative_anchor_group_overlap(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--anchor-group-overlap", "-1"])

    with pytest.raises(SystemExit):
        parse_args()
