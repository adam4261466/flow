from pathlib import Path

from audio.sfx_library import SFXLibrary
from ai.ollama_sfx import OllamaSFX


class SFXSelector:
    def __init__(
        self,
        sfx_directory,
        ollama_url=None,
        ollama_model=None,
    ):
        self.library = SFXLibrary(
            sfx_directory
        )

        if ollama_url is None and ollama_model is None:
            self.ollama = OllamaSFX()
        else:
            self.ollama = OllamaSFX(
                url=ollama_url,
                model=ollama_model,
            )

    def select_for_clip(self, clip):
        print(
            f"Selecting SFX for clip {clip.id}..."
        )

        # -------------------------------------------------
        # STEP 1: Generate searchable sound tags
        # -------------------------------------------------

        keywords = self.ollama.generate_keywords(
            clip
        )

        print("SFX keywords:")

        for keyword in keywords:
            print(f"  - {keyword}")

        # -------------------------------------------------
        # STEP 2: Search the entire SFX library
        # -------------------------------------------------

        candidates = self.library.search_by_tags(
            keywords,
            limit=30,
        )

        candidate_names = [
            item["relative_path"]
            for item in candidates
        ]

        print("SFX candidates:")

        for candidate in candidate_names:
            print(f"  - {candidate}")

        # -------------------------------------------------
        # No candidates
        # -------------------------------------------------

        if not candidate_names:
            print("No matching SFX found.")

            return {
                "keywords": keywords,
                "candidates": [],
                "selected": None,
            }

        # -------------------------------------------------
        # STEP 3: Ollama chooses from REAL candidates
        # -------------------------------------------------

        selected = self.ollama.choose_sound(
            clip=clip,
            candidate_files=candidate_names,
        )

        if selected is None:
            print(
                "Ollama decided that no candidate "
                "fits this clip."
            )

            return {
                "keywords": keywords,
                "candidates": candidate_names,
                "selected": None,
            }

        selected_path = (
            self.library.root_directory
            / selected
        )

        if not selected_path.exists():
            raise FileNotFoundError(
                f"Selected SFX does not exist:\n"
                f"{selected_path}"
            )

        print("Selected SFX:")
        print(selected_path)

        return {
            "keywords": keywords,
            "candidates": candidate_names,
            "selected": selected_path,
        }