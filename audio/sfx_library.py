from pathlib import Path


class SFXLibrary:
    def __init__(self, root_directory):
        self.root_directory = Path(root_directory)

        if not self.root_directory.exists():
            raise FileNotFoundError(
                f"SFX directory not found:\n"
                f"{self.root_directory.resolve()}"
            )

        self.refresh()

    def refresh(self):
        """
        Scan the entire SFX library.

        Folder structure does not matter.
        Every WAV file is treated as part of one library.
        """

        self.files = sorted(
            path
            for path in self.root_directory.rglob("*.wav")
            if not path.name.startswith("._")
            and "__MACOSX" not in path.parts
        )

    def search_by_tags(self, tags, limit=30):
        """
        Search every audio file using the supplied tags.

        A file matches when one or more tags appear
        in its filename OR its relative path.

        Results are ranked by number of matched tags.
        """

        if not tags:
            return []

        normalized_tags = [
            self.normalize(tag)
            for tag in tags
            if tag and tag.strip()
        ]

        results = []

        for file_path in self.files:

            relative_path = file_path.relative_to(
                self.root_directory
            )

            searchable_text = self.normalize(
                str(relative_path)
            )

            matched_tags = []

            for tag in normalized_tags:
                if tag in searchable_text:
                    matched_tags.append(tag)

            if not matched_tags:
                continue

            results.append({
                "path": file_path,
                "relative_path": relative_path.as_posix(),
                "matched_tags": matched_tags,
                "score": len(matched_tags),
            })

        # Highest number of matched tags first.
        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:limit]

    @staticmethod
    def normalize(text):
        """
        Normalize filenames so variations such as:

            church-bell
            church_bell
            Church Bell

        can be searched consistently.
        """

        text = text.lower()

        for character in [
            "_",
            "-",
            ".",
        ]:
            text = text.replace(
                character,
                " ",
            )

        return " ".join(
            text.split()
        )