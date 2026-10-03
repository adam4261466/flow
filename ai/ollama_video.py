import json

from ai.ollama_sfx import OllamaSFX


class OllamaVideoPlanner(OllamaSFX):
    """
    Creates visual instructions for the entire story.

    One Ollama request produces:
        - camera
        - composition
        - motion
        - shake
        - transition
        - color grade

    This is intentionally done for all clips at once to
    reduce the number of Ollama executions.
    """

    ALLOWED_MOTIONS = {
        "static",
        "zoom_in",
        "zoom_out",
        "push_in",
        "pull_out",
        "pan_left",
        "pan_right",
        "pan_up",
        "pan_down",
    }

    ALLOWED_SHAKES = {
        "none",
        "subtle",
        "medium",
        "strong",
    }

    ALLOWED_TRANSITIONS = {
        "cut",
        "fade_black",
        "flash_white",
    }

    ALLOWED_GRADES = {
        "neutral",
        "cinematic",
        "moody",
        "cool",
        "warm",
    }
    def generate_title(self, clips):
        """
        Generate a short cinematic title for the entire story.
        """

        story_parts = []

        for clip in clips:

            story_parts.append(
                f"""
    CLIP {clip.id}

    Moment:
    {clip.moment}

    Vibe:
    {clip.vibe}

    Script:
    {clip.script}
    """.strip()
            )

        story_text = "\n\n".join(
            story_parts
        )

        system_prompt = """
    You are a professional film title writer.

    Generate ONE original title for the story.

    Rules:
    - The title must match the story.
    - Keep it short.
    - Prefer 2 to 6 words.
    - Make it memorable and cinematic.
    - Do not use quotation marks.
    - Do not add explanations.
    - Do not add "Title:".
    - Return JSON only.
    - Do not use Markdown.

    Required format:

    {
        "title": "Your Title"
    }
    """

        user_prompt = f"""
    Create a cinematic title for this complete story:

    {story_text}
    """

        content = self._chat(
            system_prompt,
            user_prompt,
        )

        data = self._parse_json_response(
            content
        )

        title = data.get(
            "title",
            ""
        )

        if not isinstance(
            title,
            str,
        ):
            raise RuntimeError(
                "Ollama generated an invalid title."
            )

        title = title.strip()

        if not title:
            raise RuntimeError(
                "Ollama returned an empty title."
            )

        return title
    def plan_clips(self, clips):

        clips_text = []

        for clip in clips:

            clips_text.append(
                f"""
    CLIP {clip.id}

    Characters:
    {clip.characters}

    Moment:
    {clip.moment}

    Vibe:
    {clip.vibe}

    Style:
    {clip.style}

    Script:
    {clip.script}
    """.strip()
                )

            story_text = "\n\n".join(
                clips_text
            )

            system_prompt = """
    You are the visual director for an AI cinematic video.

    Analyze the entire sequence and create visual instructions
    for every clip.

    For every clip choose:

    camera:
    A concrete cinematic camera shot.
    Examples:
    - wide establishing shot
    - medium shot
    - close-up
    - extreme close-up
    - over-the-shoulder
    - low-angle medium shot
    - high-angle shot
    - POV shot

    composition:
    Describe how the important subjects should be arranged
    inside the frame.

    motion:
    Choose exactly ONE:

    static
    zoom_in
    zoom_out
    push_in
    pull_out
    pan_left
    pan_right
    pan_up
    pan_down

    shake:
    Choose exactly ONE:

    none
    subtle
    medium
    strong

    Use shake only for physical movement, impact, running,
    explosions, crashes, sudden events, etc.

    transition_after:
    Choose exactly ONE:

    cut
    fade_black
    flash_white

    Choose the transition based on the relationship between
    this clip and the next clip.

    grade:
    Choose exactly ONE:

    neutral
    cinematic
    moody
    cool
    warm

    Keep the visual style consistent across the story.

    IMPORTANT:
    - Do not use camera shake for normal static scenes.
    - Do not use dramatic transitions everywhere.
    - Do not randomly change color grades.
    - Maintain visual continuity.
    - Return JSON only.
    - Do not use Markdown.
    - Do not wrap the response in ```.

    Required format:

    {
        "clips": [
            {
                "id": 1,
                "camera": "medium shot",
                "composition": "Adam on the left third...",
                "motion": "push_in",
                "shake": "none",
                "transition_after": "cut",
                "grade": "cinematic"
            }
        ]
    }
    """

            user_prompt = f"""
    Create the complete visual plan for this story.

    {story_text}
    """

            content = self._chat(
                system_prompt,
                user_prompt,
            )

            data = self._parse_json_response(
                content
            )

            raw_plans = data.get(
                "clips",
                []
            )

            if not isinstance(
                raw_plans,
                list,
            ):
                raise RuntimeError(
                    "Ollama visual plan 'clips' "
                    "must be a list."
                )

            plans = {}

            for item in raw_plans:

                if not isinstance(item, dict):
                    continue

                try:
                    clip_id = int(
                        item.get("id")
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                camera = str(
                    item.get(
                        "camera",
                        "medium cinematic shot",
                    )
                ).strip()

                composition = str(
                    item.get(
                        "composition",
                        "Balanced cinematic composition.",
                    )
                ).strip()

                motion = str(
                    item.get(
                        "motion",
                        "static",
                    )
                ).strip().lower()

                shake = str(
                    item.get(
                        "shake",
                        "none",
                    )
                ).strip().lower()

                transition = str(
                    item.get(
                        "transition_after",
                        "cut",
                    )
                ).strip().lower()

                grade = str(
                    item.get(
                        "grade",
                        "cinematic",
                    )
                ).strip().lower()

                if motion not in self.ALLOWED_MOTIONS:
                    motion = "static"

                if shake not in self.ALLOWED_SHAKES:
                    shake = "none"

                if transition not in self.ALLOWED_TRANSITIONS:
                    transition = "cut"

                if grade not in self.ALLOWED_GRADES:
                    grade = "cinematic"

                plans[clip_id] = {
                    "id": clip_id,
                    "camera": camera,
                    "composition": composition,
                    "motion": motion,
                    "shake": shake,
                    "transition_after": transition,
                    "grade": grade,
                }

            # -----------------------------------------------------
            # Make sure every clip received a plan.
            # -----------------------------------------------------

            for clip in clips:

                if clip.id not in plans:

                    plans[clip.id] = {
                        "id": clip.id,
                        "camera": "medium cinematic shot",
                        "composition": (
                            "Balanced cinematic composition."
                        ),
                        "motion": "static",
                        "shake": "none",
                        "transition_after": "cut",
                        "grade": "cinematic",
                    }

            # -----------------------------------------------------
            # Last clip should never transition into another clip.
            # -----------------------------------------------------

            if clips:

                last_id = clips[-1].id

                plans[last_id][
                    "transition_after"
                ] = "cut"

            return plans