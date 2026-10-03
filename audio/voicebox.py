import time
import shutil
from pathlib import Path

import requests


class Voicebox:
    """
    Interface to the local Voicebox server.

    Voicebox generates speech asynchronously:
        1. Start generation
        2. Wait for completion
        3. Find generated WAV
        4. Copy it into our project
    """

    def __init__(
        self,
        server="http://127.0.0.1:17493",
        profile="test1",
        data_directory=None,
    ):
        self.server = server.rstrip("/")
        self.profile = profile

        if data_directory is None:
            data_directory = (
                Path.home()
                / "AppData"
                / "Roaming"
                / "sh.voicebox.app"
            )

        self.data_directory = Path(data_directory)

        self.headers = {
            "Content-Type": "application/json",
            "X-Voicebox-Client-Id": "python-story-agent",
        }

    def generate_speech(self, text):
        """
        Start a Voicebox generation.

        Returns:
            generation_id
        """

        if not text or not text.strip():
            raise ValueError(
                "Cannot generate speech from empty text."
            )

        response = requests.post(
            f"{self.server}/speak",
            headers=self.headers,
            json={
                "text": text,
                "profile": self.profile,
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        generation_id = data.get("id")

        if not generation_id:
            raise RuntimeError(
                "Voicebox response did not contain a generation ID."
            )

        print("Voicebox generation started.")
        print("Generation ID:", generation_id)

        return generation_id

    def wait_for_generation(
        self,
        generation_id,
        poll_interval=1,
    ):
        """
        Wait until Voicebox finishes generating audio.
        """

        print("Waiting for Voicebox generation...")

        while True:

            response = requests.get(
                f"{self.server}/generate/{generation_id}/status",
                timeout=30,
            )

            response.raise_for_status()

            text = response.text

            if '"status": "completed"' in text:
                print("Voicebox generation completed.")
                return

            if '"status": "failed"' in text:
                raise RuntimeError(
                    f"Voicebox generation failed:\n{text}"
                )

            time.sleep(poll_interval)

    def find_audio_path(self, generation_id):
        """
        Find the generated WAV file in Voicebox history.
        """

        response = requests.get(
            f"{self.server}/history",
            timeout=30,
        )

        response.raise_for_status()

        history = response.json()

        for item in history.get("items", []):

            if item.get("id") != generation_id:
                continue

            audio_path = item.get("audio_path")

            if not audio_path:
                return None

            audio_path = self.data_directory / Path(audio_path)

            print("Voicebox audio path:")
            print(audio_path)

            return audio_path

        return None

    def generate_to_file(self, text, output_file):
        """
        Generate speech and copy the final WAV
        directly to output_file.
        """

        output_file = Path(output_file)

        generation_id = self.generate_speech(text)

        self.wait_for_generation(generation_id)

        audio_path = self.find_audio_path(generation_id)

        if audio_path is None:
            raise FileNotFoundError(
                "Voicebox completed the generation, "
                "but no audio path was found."
            )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Voicebox reported this file, "
                f"but it does not exist:\n{audio_path}"
            )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            audio_path,
            output_file,
        )

        print("Audio saved to:")
        print(output_file.resolve())

        return output_file