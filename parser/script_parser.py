from models.clip import Clip


FIELD_NAMES = [
    "Characters",
    "Moment",
    "Vibe",
    "Style",
    "Script",
]


def split_clips(text):
    """
    Split the complete script using $$ separators.
    """

    sections = text.split("$$")

    return [
        section.strip()
        for section in sections
        if section.strip()
    ]


def extract_field(section, field_name):
    """
    Extract one field from a CLIP section.

    Example:

    Characters:
    Adam

    Moment:
    Adam opens the door.

    Calling:
        extract_field(section, "Characters")

    returns:
        Adam
    """

    marker = field_name + ":"

    if marker not in section:
        return ""

    content = section.split(
        marker,
        1,
    )[1]

    positions = []

    for field in FIELD_NAMES:

        other_marker = field + ":"

        if other_marker == marker:
            continue

        position = content.find(
            other_marker
        )

        if position != -1:
            positions.append(
                position
            )

    if positions:

        content = content[
            :min(positions)
        ]

    return content.strip()

def extract_title(text):
    """
    Extract the title from the GLOBAL section.

    Supported:

    TITLE:
    The Last Door

    TITLE: The Last Door
    """

    sections = text.split("$$")

    for section in sections:

        lines = section.splitlines()

        # Find GLOBAL anywhere in the section,
        # allowing spaces before it.
        is_global = any(
            line.strip().upper() == "GLOBAL"
            for line in lines
        )

        if not is_global:
            continue

        for index, line in enumerate(lines):

            stripped = line.strip()

            if not stripped.upper().startswith("TITLE:"):
                continue

            # ---------------------------------------------
            # TITLE: The Last Door
            # ---------------------------------------------

            value = stripped[6:].strip()

            if value:
                return value

            # ---------------------------------------------
            # TITLE:
            # The Last Door
            # ---------------------------------------------

            for next_line in lines[index + 1:]:

                next_value = next_line.strip()

                if not next_value:
                    continue

                # Stop if another GLOBAL field starts.
                if next_value.lower().startswith(
                    "characters:"
                ):
                    break

                if next_value.lower().startswith(
                    "style:"
                ):
                    break

                return next_value

    return "Untitled"

def parse_clip(section, clip_id):
    """
    Convert one CLIP section into a Clip object.
    """

    characters = extract_field(
        section,
        "Characters",
    )

    moment = extract_field(
        section,
        "Moment",
    )

    vibe = extract_field(
        section,
        "Vibe",
    )

    style = extract_field(
        section,
        "Style",
    )

    script = extract_field(
        section,
        "Script",
    )

    return Clip(
        id=clip_id,
        characters=characters,
        moment=moment,
        vibe=vibe,
        style=style,
        script=script,
    )


def parse_script(text):
    """
    Parse the complete script and return
    a list of Clip objects.
    """

    sections = split_clips(
        text
    )

    clips = []

    clip_id = 1

    for section in sections:

        if section.startswith(
            "GLOBAL"
        ):
            continue

        if not section.startswith(
            "CLIP"
        ):
            continue

        clip = parse_clip(
            section,
            clip_id,
        )

        clips.append(
            clip
        )

        clip_id += 1

    return clips