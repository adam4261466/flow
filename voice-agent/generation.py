import requests
import time
import shutil
from pathlib import Path

SERVER = "http://127.0.0.1:17493"
PROFILE = "test1"

# Voicebox data directory
VOICEBOX_DATA = Path(
    r"C:\Users\kabli\AppData\Roaming\sh.voicebox.app"
)

OUTPUT_FILE = Path("story.wav")

TEXT = """
The night was silent.

The streets were empty, and the cold wind moved slowly between the buildings.

Then, suddenly, someone knocked on the door.
"""

HEADERS = {
    "Content-Type": "application/json",
    "X-Voicebox-Client-Id": "python-story-agent"
}


def generate_speech():
    print("Starting speech generation...")

    response = requests.post(
        f"{SERVER}/speak",
        headers=HEADERS,
        json={
            "text": TEXT,
            "profile": PROFILE
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()
    generation_id = data["id"]

    print("Generation started.")
    print("ID:", generation_id)

    return generation_id


def wait_for_generation(generation_id):
    print("Waiting for generation...")

    # The Voicebox status endpoint returns SSE:
    # data: {"status": "generating", ...}
    # data: {"status": "completed", ...}

    while True:
        response = requests.get(
            f"{SERVER}/generate/{generation_id}/status",
            timeout=30
        )

        response.raise_for_status()

        text = response.text

        # Check for completion directly in the SSE text.
        if '"status": "completed"' in text:
            print("Generation completed.")
            return True

        if '"status": "failed"' in text:
            print("Generation failed.")
            print(text)
            return False

        time.sleep(1)


def find_audio_path(generation_id):
    print("Looking for generated audio...")

    response = requests.get(
        f"{SERVER}/history",
        timeout=30
    )

    response.raise_for_status()

    history = response.json()

    for item in history.get("items", []):
        if item.get("id") == generation_id:

            audio_path = item.get("audio_path")

            if not audio_path:
                return None

            print("Voicebox audio path:")
            print(audio_path)

            # Voicebox gives us a relative path such as:
            # generations\xxxxxxxx.wav
            return VOICEBOX_DATA / Path(audio_path)

    return None


def main():

    generation_id = generate_speech()

    if not wait_for_generation(generation_id):
        raise SystemExit(1)

    audio_path = find_audio_path(generation_id)

    if audio_path is None:
        print("ERROR: Could not find audio in Voicebox history.")
        raise SystemExit(1)

    print()
    print("Checking audio file:")
    print(audio_path)

    if not audio_path.exists():
        print()
        print("ERROR: Voicebox reported the file,")
        print("but the file does not exist at:")
        print(audio_path)
        raise SystemExit(1)

    # Copy Voicebox's WAV to our project folder
    shutil.copy2(audio_path, OUTPUT_FILE)

    print()
    print("========================================")
    print("Speech generation complete!")
    print("Audio saved to:")
    print(OUTPUT_FILE.resolve())
    print("========================================")


if __name__ == "__main__":
    main()
