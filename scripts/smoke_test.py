"""Small smoke checks for portfolio/repo quality."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_python_files_parse() -> None:
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def test_required_project_files_exist() -> None:
    required = [
        "README.md",
        ".env.example",
        ".gitignore",
        "requirements.txt",
        "bot.py",
        "ai_client.py",
        "config.py",
        "knowledge_loader.py",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    assert not missing, f"Missing required files: {missing}"


def test_bot_does_not_import_old_client_name() -> None:
    bot_source = (ROOT / "bot.py").read_text(encoding="utf-8-sig")
    old_client_name = "gr" + "oq_client"
    assert old_client_name not in bot_source


if __name__ == "__main__":
    test_python_files_parse()
    test_required_project_files_exist()
    test_bot_does_not_import_old_client_name()
    print("Smoke checks passed.")
