from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
IGNORED_PARTS = {".git", ".pytest_cache", ".venv", "build", "dist"}


def _markdown_files() -> list[Path]:
    return sorted(
        path
        for path in REPOSITORY_ROOT.rglob("*.md")
        if not any(part in IGNORED_PARTS for part in path.relative_to(REPOSITORY_ROOT).parts)
    )


def _local_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    elif " " in target:
        target = target.split(" ", 1)[0]
    if not target or target.startswith("#"):
        return None
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None
    return unquote(parsed.path) or None


class MarkdownLinkTest(unittest.TestCase):
    def test_all_relative_markdown_links_resolve(self) -> None:
        missing: list[str] = []
        for document in _markdown_files():
            text = document.read_text(encoding="utf-8")
            for match in MARKDOWN_LINK.finditer(text):
                target = _local_target(match.group(1))
                if target is None:
                    continue
                resolved = (document.parent / target).resolve()
                try:
                    resolved.relative_to(REPOSITORY_ROOT.resolve())
                except ValueError:
                    missing.append(
                        f"{document.relative_to(REPOSITORY_ROOT)} -> outside repository: {target}"
                    )
                    continue
                if not resolved.exists():
                    missing.append(f"{document.relative_to(REPOSITORY_ROOT)} -> {target}")
        self.assertEqual([], missing, "broken local Markdown links:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
