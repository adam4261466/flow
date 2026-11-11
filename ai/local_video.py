"""Deterministic, fully offline planner.

This module is a drop-in replacement for :class:`ai.ollama_video.OllamaVideoPlanner`.
It produces a usable title, description, visual plan and thumbnail plan without
any model, network call or Ollama server.

The output is intentionally simple. It exists so the image / voice / clip /
final stages always stay runnable when no LLM is available (for example when an
Ollama cloud model returns ``402 Payment Required``). Use the Ollama planner
only when you explicitly want LLM-written titles and plans.
"""

import re

from parser.script_parser import extract_title

# Values accepted by the renderer. Keep these in sync with the validation in
# ai/ollama_video.py so a cached/local plan can never break a render.
_CAMERAS = [
    "wide cinematic shot",
    "medium cinematic shot",
    "close-up",
    "over-the-shoulder shot",
    "low angle",
    "high angle",
]

_COMPOSITIONS = [
    "centered",
    "rule of thirds",
    "foreground framing",
    "symmetrical",
    "leading lines",
    "deep composition",
]

_MOTIONS = [
    "static",
    "zoom_in",
    "zoom_out",
    "push_in",
    "pull_out",
    "pan_left",
    "pan_right",
]

_GRADES = ["cinematic", "warm", "cool", "neutral", "moody"]

# Camera directions that often start a "Moment" line and make poor titles.
_LEADING_SHOT_PATTERNS = (
    r"establishing shot of",
    r"wide shot of",
    r"medium shot of",
    r"close[ -]?up of",
    r"shot of",
    r"we see",
    r"the camera (?:pans|zooms|follows|tracks) (?:to|in on)",
)

# Words that end a noun phrase: prepositions, pronouns, conjunctions and the
# most common story verbs. This keeps titles like "Pip and the Shiny Red
# Button" instead of "Pip and the Shiny Red Button Was Found".
_PHRASE_BREAKS = {
    "of", "in", "at", "on", "from", "with", "for", "to", "behind",
    "under", "near", "beside", "into", "over", "through", "across",
    "around", "by", "as", "while", "and", "but", "that", "which",
    "who", "when", "then", "there", "here", "because", "if", "so",
    "he", "she", "it", "they", "we", "i", "you", "him", "them",
    "us", "me", "my", "his", "her", "their", "our", "your", "its",
    "lived", "woke", "had", "has", "have", "knew", "know", "knows",
    "was", "were", "is", "are", "went", "go", "goes", "saw", "see",
    "sees", "sat", "sit", "sits", "stood", "stands", "looked", "looks",
    "felt", "feels", "began", "begins", "started", "starts", "hopped",
    "hops", "tiptoed", "peeks", "peeked", "holding", "holds", "glowing",
    "glows", "stares", "lies", "kneels", "hangs", "heard", "hears",
    "noticed", "notices", "loved", "loves", "liked", "likes", "wanted",
    "wants", "found", "finds", "gave", "gives", "made", "makes",
    "said", "says", "asked", "asks", "whispered", "whispers", "smiled",
    "smiles", "laughed", "laughs", "cried", "cries", "ran", "runs",
    "flew", "flies", "swam", "swims", "jumped", "jumps", "climbed",
    "climbs", "opened", "opens", "closed", "closes", "put", "puts",
    "picked", "picks", "turned", "turns", "watched", "watches",
    "thought", "thinks", "remembered", "remembers", "decided",
    "decides", "tried", "tries", "helped", "helps", "followed",
    "follows", "carried", "carries", "wore", "wears", "ate", "eats",
    "drank", "drinks", "slept", "sleeps", "dreamed", "dreams",
    "wished", "wishes", "hoped", "hopes", "waited", "waits", "called",
    "calls", "named", "rose", "spread", "spreads", "stretched",
    "stretches", "floats", "drifts", "shines", "twinkles", "sparkles",
    "creaks", "sniffed", "sniffs", "wiggled", "wiggles", "took",
    "takes", "hung", "lay", "lays", "stopped", "stops", "like",
    "a", "an", "the",
}

_SMALL_WORDS = {
    "a", "an", "the", "and", "but", "or", "of", "in", "at", "on",
    "to", "for", "with", "from", "by", "as",
}

# Words that describe the camera rather than the story.
_CAMERA_WORDS = {
    "shot", "camera", "angle", "view", "frame", "closeup", "close-up",
    "establishing", "foreground", "background",
}

# Words that end a character's name and start a description.
_NAME_STOPS = {
    "the", "a", "an", "in", "with", "and", "who", "his", "her",
    "wearing", "about", "of", "on", "at", "holding", "standing",
    "beside", "near", "from", "is", "was",
}


def _clean_text(text):
    text = re.sub(r"[\r\n]+", " ", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _clean_line(text):
    return _clean_text(text).strip(" \t\"'`.,;:!?-")


def _first_sentence(text):
    text = _clean_line(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return parts[0] if parts else text


def _trim_words(text, limit=8):
    words = _clean_line(text).split()
    if len(words) > limit:
        words = words[:limit]
    return " ".join(words)


def _title_case(text):
    words = []
    for index, word in enumerate(_clean_text(text).split()):
        lowered = word.lower()
        if index > 0 and lowered in _SMALL_WORDS:
            words.append(lowered)
        else:
            words.append(lowered[:1].upper() + lowered[1:])
    return " ".join(words)


def _title_from_moment(moment):
    sentence = _first_sentence(moment)
    lowered = sentence.lower()

    for pattern in _LEADING_SHOT_PATTERNS:
        match = re.match(pattern + r"\s+", lowered)
        if match:
            sentence = sentence[match.end():]
            break

    title = _trim_words(sentence)
    return _title_case(title)


def _noun_phrase(text):
    """Pick the most descriptive short noun phrase from a few sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", _clean_text(text))
    block = " ".join(sentences[:5])
    best = ""
    best_score = 0

    breaks = "|".join(sorted(_PHRASE_BREAKS, key=len, reverse=True))

    for match in re.finditer(r"\b(?:a|an|the)\s+([^.;:!?]{1,100})", block, re.I):
        chunk = match.group(1).replace(",", " ")
        # Stop at the first preposition/verb so "the edge of a pond" becomes
        # "the edge" and "a sparkling pond" wins instead.
        head = re.split(r"\b(?:" + breaks + r")\b", chunk, maxsplit=1)[0]
        words = [
            word for word in _clean_line(head).split()
            if word.lower() not in _SMALL_WORDS
            and word.lower() not in _CAMERA_WORDS
        ]
        words = words[:4]
        if len(words) > best_score:
            best_score = len(words)
            best = " ".join(words)

    return _title_case(best)


def _title_from_clip(clip):
    name = _character_name(clip.characters)
    # Narration usually names the story's key object; the moment line is the
    # fallback because it often starts with a camera direction.
    phrase = _noun_phrase(clip.script) or _noun_phrase(clip.moment)

    if name and phrase:
        return f"{name} and the {phrase}"
    if phrase:
        return f"The {phrase}"

    return _title_from_moment(clip.moment)


def _character_name(characters):
    segment = re.split(r"[,:(]", _clean_text(characters), maxsplit=1)[0]
    name_words = []

    for word in segment.split():
        cleaned = word.strip(".,;:!?\"'")
        if not cleaned:
            continue
        if cleaned.lower() in _NAME_STOPS:
            # Allow a leading "The" in names such as "The Echo".
            if cleaned.lower() == "the" and not name_words:
                continue
            break
        if not cleaned[0].isupper():
            break
        name_words.append(cleaned)
        if len(name_words) >= 3:
            break

    return " ".join(name_words)


class AutoVideoPlanner:
    """Use Ollama when it works, otherwise the offline planner.

    Each method is attempted on ``primary`` first. Any failure is logged and
    the same method is retried on ``fallback``, so one broken Ollama call
    never stops the image/voice/clip pipeline.
    """

    def __init__(self, primary, fallback, log=print):
        self.primary = primary
        self.fallback = fallback
        self.log = log

    def _call(self, method, *args, **kwargs):
        try:
            return getattr(self.primary, method)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - fallback is the whole point
            self.log(
                f"[planner] Ollama {method} failed ({exc}); "
                "falling back to the offline planner."
            )
            return getattr(self.fallback, method)(*args, **kwargs)

    def generate_title(self, clips):
        return self._call("generate_title", clips)

    def generate_description(self, clips, title):
        return self._call("generate_description", clips, title)

    def plan_clips(self, clips):
        return self._call("plan_clips", clips)

    def generate_thumbnail_plan(self, clips, title):
        return self._call("generate_thumbnail_plan", clips, title)


class LocalVideoPlanner:
    """Offline replacement for the Ollama planner.

    ``script_text`` is optional; when given, a ``TITLE:`` header in the
    script's GLOBAL section is used for the video title. Otherwise the title is
    derived from the first clip's narration and moment.
    """

    def __init__(self, script_text="", kids=False, log=print):
        self.script_text = script_text or ""
        self.kids = bool(kids)
        self.log = log

    # ------------------------------------------------------------
    # TITLE
    # ------------------------------------------------------------

    def generate_title(self, clips):
        from_script = _clean_line(extract_title(self.script_text or ""))
        if from_script and from_script.lower() != "untitled":
            return from_script

        ordered = sorted(clips, key=lambda clip: clip.id)

        for clip in ordered:
            title = _title_from_clip(clip)
            if len(title.split()) >= 2:
                return title

        if ordered:
            name = _character_name(ordered[0].characters)
            if name:
                return f"{name}'s Story"

        return "A Little Story"

    # ------------------------------------------------------------
    # DESCRIPTION
    # ------------------------------------------------------------

    def generate_description(self, clips, title):
        ordered = sorted(clips, key=lambda clip: clip.id)
        beats = [
            _clean_text(clip.script)
            for clip in ordered
            if _clean_text(clip.script)
        ]

        if not beats:
            beats = [
                _clean_text(clip.moment)
                for clip in ordered
                if _clean_text(clip.moment)
            ]

        body = " ".join(beats[:3]).strip()
        if not body:
            body = f"{title} is a short cinematic story."

        closing = (
            "Watch to discover how the story unfolds."
            if not self.kids
            else "A gentle story for little ones to enjoy."
        )
        return f"{body}\n\n{closing}"

    # ------------------------------------------------------------
    # VISUAL PLAN
    # ------------------------------------------------------------

    def plan_clips(self, clips):
        plans = {}

        for index, clip in enumerate(sorted(clips, key=lambda item: item.id)):
            motion = _MOTIONS[index % len(_MOTIONS)]
            # Keep the first and last beats calm and clean.
            if index == 0:
                motion = "static"

            transition = "cut"
            if index == len(clips) - 1:
                transition = "fade_black"

            plans[clip.id] = {
                "camera": _CAMERAS[index % len(_CAMERAS)],
                "composition": _COMPOSITIONS[index % len(_COMPOSITIONS)],
                "motion": motion,
                # Camera shake stays hard-disabled, same as the Ollama planner.
                "shake": "none",
                "transition_after": transition,
                "grade": _GRADES[index % len(_GRADES)],
            }

        return plans

    # ------------------------------------------------------------
    # THUMBNAIL PLAN
    # ------------------------------------------------------------

    def generate_thumbnail_plan(self, clips, title):
        ordered = sorted(clips, key=lambda clip: clip.id)
        reference = ordered[0] if ordered else None

        subject = _character_name(reference.characters) if reference else ""
        subject = subject or "Main character"

        scene = _clean_line(reference.moment) if reference else ""
        scene = scene or "A memorable moment from the story"

        emotion = _clean_line(reference.vibe) if reference else ""
        emotion = emotion or ("wonder and warmth" if self.kids else "mystery")

        if self.kids:
            lighting = "soft warm golden storybook lighting"
            palette = "bright, friendly storybook colors"
            camera = "Wide, friendly storybook shot"
        else:
            lighting = "dramatic cinematic lighting"
            palette = "deep cinematic tones"
            camera = "Wide cinematic shot"

        return {
            "subject": subject,
            "scene": scene,
            "emotion": emotion,
            "camera": camera,
            "lighting": lighting,
            "color_palette": palette,
            "composition": "strong foreground/background separation",
            "title_position": "top_left",
            "reference_clip": reference.id if reference else 1,
        }
