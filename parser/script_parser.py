from models.clip import Clip

import re

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
    pattern = r"(?m)^[ \t]*" + re.escape(field_name) + r"[ \t]*:"
    match = re.search(pattern, section)

    # Fallback: old behaviour (flattened text)
    if not match:
        marker = field_name + ":"
        if marker not in section:
            return ""
        content = section.split(marker, 1)[1]
        positions = [
            content.find(f + ":")
            for f in FIELD_NAMES
            if f != field_name and content.find(f + ":") != -1
        ]
        return (content[:min(positions)] if positions else content).strip()

    content = section[match.end():]
    next_pos = [
        m.start()
        for f in FIELD_NAMES
        if f != field_name
        for m in [re.search(r"(?m)^[ \t]*" + re.escape(f) + r"[ \t]*:", content)]
        if m
    ]
    if next_pos:
        content = content[:min(next_pos)]
    return content.strip()

def extract_title(text):
    """
    Extract the title from the GLOBAL section.

    Supported:

    TITLE:
    The Last Door

    TITLE: The Last Door
    """

    text = text.lstrip("\ufeff")
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

                if next_value.lower().startswith(
                    "description:"
                ):
                    break

                return next_value

    return "Untitled"

GLOBAL_FIELD_NAMES = ["TITLE", "DESCRIPTION", "CHARACTERS", "STYLE"]


def extract_global_field(text, field_name):
    """Extract one field from the GLOBAL section, e.g. DESCRIPTION."""
    for section in text.lstrip("\ufeff").split("$$"):
        lines = section.splitlines()

        if not any(line.strip().upper() == "GLOBAL" for line in lines):
            continue

        match = re.search(
            r"(?im)^[ \t]*" + re.escape(field_name) + r"[ \t]*:[ \t]*(.*)$",
            section,
        )
        if not match:
            return ""

        parts = [match.group(1).strip()]

        for line in section[match.end():].splitlines():
            stripped = line.strip()
            upper = stripped.upper()

            if any(
                upper.startswith(name + ":")
                for name in GLOBAL_FIELD_NAMES
            ):
                break

            if stripped:
                parts.append(stripped)

        return "\n".join(part for part in parts if part).strip()

    return ""


def extract_description(text):
    """Return the optional DESCRIPTION written with the story script."""
    return extract_global_field(text, "Description")


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