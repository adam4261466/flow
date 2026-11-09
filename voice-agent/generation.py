import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.chatterbox_tts import DEFAULT_VOICE, ChatterboxTTS


OUTPUT_FILE = Path(__file__).resolve().parent / "story.wav"

TEXT = """
The night was silent.

The streets were empty, and the cold wind moved slowly between the buildings.

Then, suddenly, someone knocked on the door.
"""


def main():

    print("Generating speech with Chatterbox TTS...")

    tts = ChatterboxTTS(
        voice=DEFAULT_VOICE,
        language="en",
    )

    output_file = tts.generate_to_file(
        text=TEXT,
        output_file=OUTPUT_FILE,
    )

    print()
    print("========================================")
    print("Speech generation complete!")
    print("Audio saved to:")
    print(output_file.resolve())
    print("========================================")


if __name__ == "__main__":
    main()
