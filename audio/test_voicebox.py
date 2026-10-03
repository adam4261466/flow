from audio.voicebox import Voicebox


TEXT = """
The night was silent.

The streets were empty, and the cold wind moved slowly
between the buildings.

Then, suddenly, someone knocked on the door.
"""


def main():

    voicebox = Voicebox(
        profile="test1"
    )

    voicebox.generate_to_file(
        text=TEXT,
        output_file="output/test_story.wav"
    )


if __name__ == "__main__":
    main()