from pathlib import Path
import re


class SFXLibrary:
    """Recursively index WAV SFX and rank matches using filename/path tags."""

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"SFX library not found: {self.root.resolve()}")
        self.files = self._scan()

    def _scan(self):
        files = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() != ".wav":
                continue
            if path.name.startswith("._"):
                continue
            if "__MACOSX" in path.parts:
                continue
            files.append(path)
        return sorted(files, key=lambda p: str(p).lower())

    @staticmethod
    def _normalize(text):
        text = str(text).lower()
        text = re.sub(r"[_\-.()]+", " ", text)
        return text

    def search_by_tags(self, tags, limit=30):
        normalized_tags = [self._normalize(tag).strip() for tag in tags if str(tag).strip()]
        results = []

        for path in self.files:
            haystack = self._normalize(path)
            score = 0
            for tag in normalized_tags:
                if tag and tag in haystack:
                    score += 1
            if score:
                results.append((score, path))

        results.sort(key=lambda item: (-item[0], str(item[1]).lower()))
        return [path for _, path in results[:limit]]
