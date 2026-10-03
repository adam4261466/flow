REQUIRED_FIELDS = [
    "characters",
    "moment",
    "vibe",
    "style",
    "script",
]


def validate_clip(clip):
    """
    Validate one Clip object.

    Returns:
        list[str]: validation errors.
    """

    errors = []

    clip_id = getattr(
        clip,
        "id",
        "unknown"
    )

    for field in REQUIRED_FIELDS:

        value = getattr(
            clip,
            field,
            None
        )

        if value is None:

            errors.append(
                f"Clip {clip_id}: missing {field}."
            )

            continue

        if not isinstance(value, str):

            errors.append(
                f"Clip {clip_id}: {field} must be text."
            )

            continue

        if not value.strip():

            errors.append(
                f"Clip {clip_id}: {field} is empty."
            )

    return errors


def validate_clips(clips):
    """
    Validate all Clip objects.

    Returns:
        list[str]: all validation errors.
    """

    errors = []

    if not clips:

        errors.append(
            "No clips were found."
        )

        return errors

    seen_ids = set()

    for clip in clips:

        # -----------------------------------------------
        # Validate fields
        # -----------------------------------------------

        clip_errors = validate_clip(
            clip
        )

        errors.extend(
            clip_errors
        )

        # -----------------------------------------------
        # Check duplicate IDs
        # -----------------------------------------------

        clip_id = getattr(
            clip,
            "id",
            None
        )

        if clip_id is None:
            continue

        if clip_id in seen_ids:

            errors.append(
                f"Duplicate clip ID: {clip_id}."
            )

        seen_ids.add(
            clip_id
        )

    return errors
