"""Ask Claude for a new cinematic story and return parser-ready script text.

The browser automation uses the same real Chrome/CDP session as the Gemini
image flow (see :mod:`ai.chrome_cdp`).  That means the user signs in to Claude
once and later runs reuse the session.

Public helpers:
    generate_story_with_claude()  -> validated script text
    open_claude_login()           -> open the Claude tab for sign-in
"""

from __future__ import annotations

import re
import time

from playwright.sync_api import (
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

from ai.chrome_cdp import CLAUDE_URL, ChromeSession, open_login_tabs
from parser.script_parser import parse_script
from utils.validator import validate_clips


# ----------------------------------------------------------------------
# The user-provided story prompt, kept verbatim.
# ----------------------------------------------------------------------

STORY_PROMPT = """You are a professional cinematic storyteller writing scripts specifically for an automated AI video production system.

Your job is to generate a complete short cinematic story that will later be converted automatically into images, narration, sound effects, a thumbnail, and a final video.

IMPORTANT:
The video system will generate the title automatically, so DO NOT create a TITLE field.

Do not include subtitles, background music instructions, or editing instructions.

Return ONLY the script in the exact format specified below.
Do not add explanations before or after the script.

==================================================
STORY REQUIREMENTS
==================

Create an original, cinematic, emotionally engaging story.

The story should feel like a short film rather than a simple narration.

Choose a strong genre and atmosphere such as:

* mystery
* psychological thriller
* science fiction
* horror
* survival
* adventure
* crime
* dystopian
* supernatural
* emotional drama

The story should contain:

1. A strong opening hook.
2. A clear main character.
3. A recognizable setting.
4. A problem, mystery, danger, or objective.
5. Escalation and rising tension.
6. Several meaningful visual events.
7. A turning point.
8. A climax.
9. A satisfying ending.
10. A final image or moment that stays in the viewer's mind.

Do not make the story feel repetitive.

Avoid spending several consecutive clips describing the same action.

Every clip should introduce a useful visual change:

* new action
* new location
* new discovery
* new character reaction
* new object
* new danger
* new information
* change in emotional state
* escalation
* reveal

==================================================
VISUAL STORYTELLING
===================

This story will be converted into one generated image per clip.

Therefore, every clip must describe a moment that can be clearly represented by ONE cinematic image.

The "Moment" field must describe what should visibly exist in the frame.

Good:
"Adam stands in the abandoned laboratory as a red emergency light illuminates a strange metal door."

Bad:
"Adam becomes increasingly worried about what might happen."

Make actions visually concrete.

Prefer moments such as:

* opening a door
* finding an object
* looking at a photograph
* running through a corridor
* seeing something unexpected
* turning toward a sound
* standing before a machine
* discovering a hidden room
* watching a vehicle arrive
* holding an important object
* reacting to a revelation

==================================================
CHARACTER CONTINUITY
====================

Characters must remain visually consistent throughout the story.

In the GLOBAL section, define:

* name
* age
* gender when relevant
* hair
* clothing
* important physical characteristics
* important objects they carry

Do not completely redesign a character between clips.

When an important object appears, keep its appearance consistent.

When an important location appears repeatedly, keep its architecture and visual identity consistent.

==================================================
CLIP STRUCTURE
==============

Use approximately 12–20 clips for a short cinematic story.

Each clip should normally contain enough narration for roughly 15–30 seconds of natural speech.

Do not add a duration field.

The audio generated from the Script field will determine the final duration automatically.

Each clip MUST contain exactly these fields:

Characters:
Moment:
Vibe:
Style:
Script:

==================================================
FIELD RULES
===========

Characters:
Describe the character(s) visually present in this clip.

Only mention characters relevant to the current visual moment.

Moment:
Describe the exact visual moment represented by the image.

Vibe:
Describe the emotional atmosphere.

Examples:

* mysterious and tense
* calm but unsettling
* chaotic and frightening
* emotional and melancholic
* exciting and adventurous

Style:
Describe the visual style.

Examples:

* realistic cinematic science fiction
* dark photorealistic thriller
* cinematic horror realism
* dramatic dystopian realism

Keep the style consistent across the story.

Script:
Write the actual narration.

The narration should:

* sound natural when spoken aloud
* be easy for a TTS voice to pronounce
* advance the story
* avoid unnecessary repetition
* use natural sentence lengths
* create suspense where appropriate
* describe information the viewer needs
* remain understandable without subtitles

Write for spoken narration, not for reading silently.

==================================================
SOUND-DESIGN AWARENESS
======================

The automated system will analyze each clip and search a large sound-effect library using keywords derived from the story.

Therefore, when appropriate, include naturally audible events in the narration and visual moment.

Examples:

* a door slams
* footsteps echo
* glass breaks
* a phone rings
* an engine starts
* tires skid
* a gunshot
* rain hits the window
* metal scrapes
* a machine activates
* a key turns
* a bell rings
* a camera shutter clicks
* an object falls
* wind howls

Do NOT write a separate SFX field.

Do NOT force a sound into every clip.

Only include sound-worthy events when they naturally belong to the story.

==================================================
PACING
======

The first clip must immediately create curiosity.

Avoid long introductions.

Use a structure similar to:

Hook
→ Setup
→ Discovery
→ Escalation
→ Complication
→ Revelation
→ Climax
→ Resolution

Every few clips, introduce new information or a new visual development.

Do not resolve the main mystery too early.

==================================================
CAMERA-FRIENDLY WRITING
=======================

Write moments that can support cinematic camera choices such as:

* establishing shot
* wide shot
* medium shot
* close-up
* extreme close-up
* over-the-shoulder
* low-angle shot
* high-angle shot
* POV
* silhouette
* foreground/background composition

Do not explicitly add a "Camera:" field.

The later visual-planning system will choose the camera automatically.

==================================================
NO EDITING INSTRUCTIONS
=======================

Do NOT include:

* subtitles
* captions
* background music
* transition instructions
* zoom instructions
* pan instructions
* color grading instructions
* FPS
* resolution
* aspect ratio
* thumbnail instructions
* intro instructions
* outro instructions

Those are handled automatically by the video system.

==================================================
GLOBAL FORMAT
=============

The script MUST begin exactly like this:

GLOBAL

Characters:
[global character descriptions]

Style:
[global visual style]

Then separate every clip with:

$$

Each clip must use exactly:

CLIP

Characters:
...

Moment:
...

Vibe:
...

Style:
...

Script:
...

==================================================
IMPORTANT FORMATTING RULES
==================================================

- Do not use Markdown code fences.
- Do not number the clips.
- Do not add a title.
- Do not add explanations.
- Do not add commentary.
- Do not add a summary.
- Do not add "END".
- Do not add extra fields.
- Keep the field names exactly:
  Characters:
  Moment:
  Vibe:
  Style:
  Script:
- Separate clips using exactly:
$$

* Return only the finished script.

==================================================
FINAL QUALITY CHECK
===================

Before returning the script, verify that:

* The story has a strong beginning.
* The story has escalating tension.
* Every clip has a clear visual moment.
* Characters remain consistent.
* Locations remain consistent.
* The narration sounds natural.
* The story does not repeat the same visual idea.
* There are enough visually distinct moments.
* The ending is satisfying.
* The format exactly matches the required parser format.

Now generate the complete cinematic story script.
"""


# ----------------------------------------------------------------------
# Kids / family variant
# ----------------------------------------------------------------------
#
# The pipeline was originally written for dark cinematic stories.  The kids
# variant below keeps the exact same output format (GLOBAL / CLIP / $$) so the
# parser and the rest of the pipeline need no changes, but swaps the tone,
# genres, visual examples and sound examples for gentle bedtime content.
#
# Each pair is (text in STORY_PROMPT, replacement).  Replacements are applied
# in order and silently skipped if a future edit removes the source text, so
# importing this module can never fail because of prompt drift.  The test
# suite asserts the important kid-safe rules are still present.

_KIDS_PROMPT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (
        "You are a professional cinematic storyteller writing scripts specifically for an automated AI video production system.",
        "You are a professional children's storyteller writing gentle, joyful, age-appropriate scripts for an automated AI video production system aimed at children aged 3 to 8.",
    ),
    (
        "Your job is to generate a complete short cinematic story that will later be converted automatically into images, narration, sound effects, a thumbnail, and a final video.",
        "Your job is to generate a complete short bedtime story for young children that will later be converted automatically into images, narration, sound effects, a thumbnail, and a final video.",
    ),
    (
        "Create an original, cinematic, emotionally engaging story.",
        "Create an original, warm, playful, emotionally engaging story for young children.",
    ),
    (
        "The story should feel like a short film rather than a simple narration.",
        "The story should feel like an animated picture book rather than a simple narration.",
    ),
    (
        "Choose a strong genre and atmosphere such as:\n\n"
        "* mystery\n"
        "* psychological thriller\n"
        "* science fiction\n"
        "* horror\n"
        "* survival\n"
        "* adventure\n"
        "* crime\n"
        "* dystopian\n"
        "* supernatural\n"
        "* emotional drama",
        "Choose a gentle, child-friendly theme such as:\n\n"
        "* magical adventure\n"
        "* animal friendship\n"
        "* bedtime fantasy\n"
        "* kindness and sharing\n"
        "* curiosity and discovery\n"
        "* family and home\n"
        "* nature and the seasons\n"
        "* silly comedy\n"
        "* gentle mystery (never frightening)\n"
        "* learning a new skill",
    ),
    (
        "4. A problem, mystery, danger, or objective.",
        "4. A small, gentle problem or goal.",
    ),
    (
        "5. Escalation and rising tension.",
        "5. A playful complication that is easy for a child to follow.",
    ),
    (
        "* new danger",
        "* new friendly surprise",
    ),
    (
        'Adam stands in the abandoned laboratory as a red emergency light illuminates a strange metal door.',
        'Milo the rabbit stands in a sunny meadow as a giant friendly butterfly lands on a flower beside him.',
    ),
    (
        "Use approximately 12–20 clips for a short cinematic story.",
        "Use approximately 10–16 clips for a short children's story.",
    ),
    (
        "* mysterious and tense\n"
        "* calm but unsettling\n"
        "* chaotic and frightening\n"
        "* emotional and melancholic\n"
        "* exciting and adventurous",
        "* cozy and cheerful\n"
        "* curious and playful\n"
        "* silly and giggly\n"
        "* warm and reassuring\n"
        "* exciting and adventurous",
    ),
    (
        "* realistic cinematic science fiction\n"
        "* dark photorealistic thriller\n"
        "* cinematic horror realism\n"
        "* dramatic dystopian realism",
        "* bright 3D animated storybook\n"
        "* soft watercolor picture-book\n"
        "* colorful cartoon animation\n"
        "* gentle storybook illustration",
    ),
    (
        "* create suspense where appropriate",
        "* add gentle wonder and humor where appropriate",
    ),
    (
        "* a door slams\n"
        "* footsteps echo\n"
        "* glass breaks\n"
        "* a phone rings\n"
        "* an engine starts\n"
        "* tires skid\n"
        "* a gunshot\n"
        "* rain hits the window\n"
        "* metal scrapes\n"
        "* a machine activates\n"
        "* a key turns\n"
        "* a bell rings\n"
        "* a camera shutter clicks\n"
        "* an object falls\n"
        "* wind howls",
        "* a door creaks open\n"
        "* soft footsteps\n"
        "* leaves rustle\n"
        "* a bird chirps\n"
        "* a bicycle bell rings\n"
        "* water splashes\n"
        "* gentle rain\n"
        "* a kettle whistles\n"
        "* a toy squeaks\n"
        "* a page turns\n"
        "* a friendly bell rings\n"
        "* a ball bounces\n"
        "* wind whispers\n"
        "* a puppy barks happily",
    ),
    (
        "Hook\n"
        "→ Setup\n"
        "→ Discovery\n"
        "→ Escalation\n"
        "→ Complication\n"
        "→ Revelation\n"
        "→ Climax\n"
        "→ Resolution",
        "Hook\n"
        "→ Setup\n"
        "→ Discovery\n"
        "→ Playful Complication\n"
        "→ Helping Hand\n"
        "→ Happy Resolution",
    ),
    (
        "Do not resolve the main mystery too early.",
        "Do not resolve the small problem too early, and never introduce real danger.",
    ),
    (
        "==================================================\n"
        "GLOBAL FORMAT\n"
        "=============",
        "==================================================\n"
        "CHILD-SAFETY RULES (NON-NEGOTIABLE)\n"
        "==================================\n\n"
        "These stories are for children aged 3 to 8.\n\n"
        "- NO violence, weapons, blood, death, injury, or physical harm.\n"
        "- NO monsters, demons, ghosts, jump scares, or nightmares.\n"
        "- NO psychological fear, dread, abandonment, or peril.\n"
        "- NO adult themes: romance, crime, drugs, alcohol, smoking, or gambling.\n"
        "- NO sarcasm, insults, bullying, or mean-spirited humor.\n\n"
        "Characters solve problems through kindness, sharing, curiosity, teamwork,\n"
        "and asking a grown-up for help. Every story ends on a calm, happy, safe note\n"
        "that is comfortable at bedtime. If the premise conflicts with these rules,\n"
        "soften it into a gentle, friendly version instead of following it literally.\n\n"
        "==================================================\n"
        "GLOBAL FORMAT\n"
        "=============",
    ),
    (
        "* The story has escalating tension.",
        "* The story builds gentle, playful momentum.\n"
        "* The story is completely safe and comforting for young children.",
    ),
    (
        "Now generate the complete cinematic story script.",
        "Now generate the complete children's story script.",
    ),
)


def _build_kids_story_prompt(base: str) -> str:
    prompt = base
    for old, new in _KIDS_PROMPT_REPLACEMENTS:
        if old in prompt:
            prompt = prompt.replace(old, new, 1)
    return prompt


KIDS_STORY_PROMPT = _build_kids_story_prompt(STORY_PROMPT)


def story_prompt(kids: bool = False) -> str:
    """Return the story prompt for the requested audience."""
    return KIDS_STORY_PROMPT if kids else STORY_PROMPT


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------

class ClaudeError(RuntimeError):
    """Base error for Claude story generation."""


class ClaudeLoginRequired(ClaudeError):
    """Claude is not signed in in the automation Chrome profile."""


class ClaudeLimitError(ClaudeError):
    """Claude reported that the account/message limit was reached."""


class ClaudeScriptError(ClaudeError):
    """Claude replied, but the answer is not a valid parser script."""


# ----------------------------------------------------------------------
# Script cleanup / validation (pure, easy to unit-test)
# ----------------------------------------------------------------------

MIN_CLIPS = 3
MAX_CLIPS = 60


def clean_claude_script(raw: str) -> str:
    """Remove conversational noise and keep only the GLOBAL/CLIP script."""
    if not raw or not raw.strip():
        raise ClaudeScriptError("Claude returned an empty response.")

    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Strip a leading/trailing Markdown code fence, with or without a language.
    text = re.sub(r"^```[^\n]*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # Normalize common Markdown that Claude may add despite instructions.
    text = re.sub(
        r"\*\*\s*(GLOBAL|CLIP|Characters|Moment|Vibe|Style|Script)\s*:?\s*\*\*",
        r"\1:",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^\s*#{1,6}\s*(GLOBAL|CLIP|Characters|Moment|Vibe|Style|Script)\b\s*:?\s*$",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?m)^\s*GLOBAL\s*:\s*$", "GLOBAL", text)
    text = re.sub(r"(?m)^\s*CLIP\s*:?\s*\d*\s*$", "CLIP", text)

    # THE FINAL FIX:
    # Simply find the first occurrence of "GLOBAL" anywhere in the text.
    # This strips away "Claude responded: ..." or any other preamble.
    idx = text.find("GLOBAL")
    if idx == -1:
        preview = text[:300].replace("\n", " ")
        raise ClaudeScriptError(
            "Claude's response does not contain the required GLOBAL section. "
            f"Response started with: {preview!r}"
        )

    text = text[idx:].strip()

    # Some responses end the code fence after the last clip.
    text = re.sub(r"\n?```\s*$", "", text).strip()
    return text


def validate_claude_script(text: str, *, min_clips: int = MIN_CLIPS) -> list:
    """Parse and validate Claude's script; return the Clip objects."""
    clips = parse_script(text)

    errors = validate_clips(clips)
    if errors:
        raise ClaudeScriptError(
            "Claude returned a script the pipeline cannot use:\n- "
            + "\n- ".join(errors[:12])
        )

    if len(clips) < min_clips:
        raise ClaudeScriptError(
            f"Claude returned only {len(clips)} clip(s); expected at least "
            f"{min_clips}."
        )

    if len(clips) > MAX_CLIPS:
        raise ClaudeScriptError(
            f"Claude returned {len(clips)} clips; that is above the safety "
            f"limit of {MAX_CLIPS}."
        )

    return clips


# ----------------------------------------------------------------------
# Browser automation
# ----------------------------------------------------------------------

STABLE_SECONDS = 6
MIN_RESPONSE_SECONDS = 4
DEFAULT_TIMEOUT_SECONDS = 900


class ClaudeBrowser:
    """Drive claude.ai in the shared automation Chrome."""

    def __init__(
        self,
        cdp_url: str = "http://127.0.0.1:9222",
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.timeout = timeout
        self.session = ChromeSession(cdp_url)
        self.page: Page | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self.page is not None and not self.page.is_closed():
            return

        self.session.connect()
        self.page = self.session.page_for(
            CLAUDE_URL,
            match_substring="claude.ai",
            navigate=False,
        )

    def open_new_chat(self) -> None:
        if self.page is None:
            self.connect()
        assert self.page is not None

        print("Opening a new Claude conversation...")
        self.page.goto(CLAUDE_URL, wait_until="domcontentloaded", timeout=60000)
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=15000)
        except PlaywrightTimeoutError:
            pass
        self.page.wait_for_timeout(1500)

        self._raise_if_login_required()

        try:
            self.page.locator(
                'div[contenteditable="true"], textarea'
            ).first.wait_for(state="visible", timeout=30000)
        except PlaywrightTimeoutError as exc:
            raise ClaudeError(
                "Claude's message box did not appear. Make sure the Claude tab "
                "is signed in and fully loaded."
            ) from exc

        print(f"Claude conversation ready: {self.page.url}")

    def close(self) -> None:
        self.session.close()
        self.page = None

    # ------------------------------------------------------------------
    # Login / response-state detection
    # ------------------------------------------------------------------

    def _raise_if_login_required(self) -> None:
        assert self.page is not None
        url = (self.page.url or "").lower()
        if "claude.ai/login" in url or "claude.ai/onboarding" in url:
            raise ClaudeLoginRequired(
                "Claude is not signed in. Use the 'Open Claude + YouTube login' "
                "button, sign in, then run again."
            )

        sign_in_text = self._first_visible(
            (
                'button:has-text("Continue with Google")',
                'button:has-text("Sign in")',
                'a:has-text("Sign in")',
            )
        )
        composer = self._find_composer()
        if sign_in_text is not None and composer is None:
            raise ClaudeLoginRequired(
                "Claude is showing a sign-in screen. Use the login button, "
                "sign in to Claude, then run again."
            )

    def _first_visible(self, selectors, require_enabled: bool = False):
        assert self.page is not None
        for selector in selectors:
            try:
                matches = self.page.locator(selector)
                count = matches.count()
            except PlaywrightError:
                continue
            for index in range(count - 1, -1, -1):
                item = matches.nth(index)
                try:
                    if not item.is_visible():
                        continue
                    if require_enabled and not item.is_enabled():
                        continue
                    return item
                except PlaywrightError:
                    continue
        return None

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        return " ".join((text or "").split()).strip()

    def _find_composer(self):
        selectors = (
            'div.ProseMirror[contenteditable="true"]',
            'fieldset div[contenteditable="true"]',
            'div[contenteditable="true"][data-placeholder]',
            'div[contenteditable="true"][aria-label*="prompt" i]',
            'div[contenteditable="true"]',
            "textarea",
        )
        return self._first_visible(selectors, require_enabled=True)

    def _read_composer(self) -> str | None:
        composer = self._find_composer()
        if composer is None:
            return None
        try:
            return composer.evaluate(
                """el => typeof el.value === 'string'
                    ? el.value
                    : (el.innerText || el.textContent || '')"""
            )
        except PlaywrightError:
            return None

    def _is_streaming(self) -> bool:
        return self._first_visible(
            (
                'button[aria-label="Stop response"]',
                'button[aria-label*="Stop response" i]',
                'button[aria-label*="Stop" i]',
                '[data-is-streaming="true"]',
            )
        ) is not None

    def _send_button(self):
        return self._first_visible(
            (
                'button[aria-label="Send message"]',
                'button[aria-label*="Send message" i]',
                'button[aria-label*="Send" i]',
                'button[data-testid="send-button"]',
                'button[type="submit"]',
            ),
            require_enabled=True,
        )

    def _user_message_count(self) -> int:
        assert self.page is not None
        selectors = (
            '[data-testid="user-message"]',
            ".font-user-message",
            "[data-testid*='user-message' i]",
        )
        counts: list[int] = []
        for selector in selectors:
            try:
                counts.append(self.page.locator(selector).count())
            except PlaywrightError:
                continue
        return max(counts, default=0)

    # ------------------------------------------------------------------
    # Composer / submit
    # ------------------------------------------------------------------

    def _set_composer_text(self, prompt: str) -> bool:
        assert self.page is not None
        composer = self._find_composer()
        if composer is None:
            return False

        try:
            composer.click(timeout=8000)
        except PlaywrightError:
            try:
                composer.click(timeout=8000, force=True)
            except PlaywrightError:
                pass

        try:
            composer.press("Control+A", timeout=5000)
            composer.press("Delete", timeout=5000)
        except PlaywrightError:
            pass

        # Preferred: Playwright fill() updates contenteditable editors with a
        # proper input event.  Claude uses ProseMirror, which handles it well.
        try:
            composer.fill(prompt, timeout=30000)
            self.page.wait_for_timeout(500)
            if self._text_is_present(prompt):
                return True
        except PlaywrightError:
            pass

        # Fallback: insert text through the keyboard (fast, still fires input).
        try:
            composer.click(timeout=5000)
            composer.press("Control+A", timeout=5000)
            composer.press("Delete", timeout=5000)
            self.page.keyboard.insert_text(prompt)
            self.page.wait_for_timeout(500)
            if self._text_is_present(prompt):
                return True
        except PlaywrightError:
            pass

        # Last resort: set innerText directly.
        try:
            composer.evaluate(
                """(el, text) => {
                    el.innerText = text;
                    el.dispatchEvent(new InputEvent('input', {
                        bubbles: true,
                        cancelable: true,
                        inputType: 'insertText',
                        data: text,
                    }));
                }""",
                prompt,
            )
            self.page.wait_for_timeout(500)
            return self._text_is_present(prompt)
        except PlaywrightError:
            return False

    def _text_is_present(self, prompt: str) -> bool:
        """Loosely verify the composer holds the prompt.

        Contenteditable editors can normalize whitespace and line breaks, so an
        exact comparison would reject a perfectly good paste.  Check the length
        and the distinctive ending instead.
        """
        current = self._read_composer()
        if current is None:
            return False
        current_norm = self._normalize_text(current)
        prompt_norm = self._normalize_text(prompt)
        if not current_norm:
            return False
        if len(current_norm) < len(prompt_norm) * 0.9:
            return False
        return (
            prompt_norm[:80] in current_norm
            and prompt_norm[-80:] in current_norm
        )

    def _submit_prompt(self, prompt: str) -> None:
        assert self.page is not None

        if not self._set_composer_text(prompt):
            raise ClaudeError(
                "Claude's message box did not receive the complete prompt. "
                "The page may be re-rendering; try again."
            )

        messages_before = self._user_message_count()
        send = self._send_button()
        if send is None:
            # Some layouts only enable Send after a short delay.
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                send = self._send_button()
                if send is not None:
                    break
                self.page.wait_for_timeout(300)

        print("Submitting the story prompt to Claude...")
        if send is not None:
            try:
                send.click(timeout=8000)
            except PlaywrightError:
                pass
        else:
            composer = self._find_composer()
            if composer is not None:
                try:
                    composer.press("Enter", timeout=5000)
                except PlaywrightError:
                    pass

        def confirmed() -> bool:
            if self._user_message_count() > messages_before:
                return True
            if self._is_streaming():
                return True
            current = self._read_composer()
            # The composer provably held the prompt before Send was clicked, so
            # a verifiably empty composer means the message left the box.
            if current is not None and not self._normalize_text(current):
                return True
            return False

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if confirmed():
                print("Claude prompt submission confirmed.")
                return
            self.page.wait_for_timeout(300)

        # Retry with Enter if the prompt is still sitting in the composer.
        composer = self._find_composer()
        if composer is not None:
            try:
                composer.press("Enter", timeout=5000)
            except PlaywrightError:
                pass
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if confirmed():
                print("Claude prompt submission confirmed after Enter fallback.")
                return
            self.page.wait_for_timeout(300)

        raise ClaudeError(
            "Claude did not accept the prompt (it is still in the message box). "
            "Check the Claude tab for a pop-up or a sign-in prompt."
        )

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    def _assistant_texts(self) -> list[str]:
        """Snapshot the visible assistant response texts.
        
        User messages are explicitly excluded: after a prompt is submitted the
        user's own text would otherwise look like a "new assistant response"
        and trigger a false text-only error while the image is still rendering.
        """
        assert self.page is not None
        selectors = (
            'div[data-testid="assistant-message"]',
            "div.font-claude-response",
            "div.font-claude-message",
            '[data-message-author-role="assistant"]',
            "div[data-testid^='conversation-turn'] .font-claude-response",
            "div[data-testid^='conversation-turn'] .font-claude-message",
            'div[data-testid^="conversation-turn"]:not(:has([data-testid="user-message"]))',
        )

        texts: list[str] = []
        seen: set[str] = set()
        for selector in selectors:
            try:
                locator = self.page.locator(selector)
                count = locator.count()
            except PlaywrightError:
                continue
            for index in range(count):
                item = locator.nth(index)
                try:
                    # PRESERVE NEWLINES: use inner_text instead of _normalize_text
                    raw_value = item.inner_text(timeout=2000)
                except PlaywrightError:
                    continue
                
                # We use the normalized version only as a key for the 'seen' set
                key = self._normalize_text(raw_value)
                if not key or key in seen:
                    continue
                seen.add(key)
                texts.append(raw_value.replace("\r\n", "\n").strip())
        return texts

    def _wait_for_response(
        self,
        before_texts: list[str],
    ) -> str:
        assert self.page is not None
        before_set = set(before_texts)
        start = time.monotonic()
        deadline = start + self.timeout
        last_text: str | None = None
        stable_since: float | None = None
        last_log = 0.0

        print(f"Waiting up to {self.timeout} seconds for Claude's story...")

        while time.monotonic() < deadline:
            now = time.monotonic()
            texts = self._assistant_texts()
            new_texts = [text for text in texts if text not in before_set]

            if new_texts:
                current = new_texts[-1]
                if current != last_text:
                    last_text = current
                    stable_since = now
                    print(f"Claude is responding... ({len(current)} chars so far)")
                elif (
                    stable_since is not None
                    and not self._is_streaming()
                    and now - stable_since >= STABLE_SECONDS
                    and now - start >= MIN_RESPONSE_SECONDS
                ):
                    print(f"Claude response complete ({len(current)} chars).")
                    return current

            if now - last_log >= 20:
                print("Still waiting for Claude to finish writing...")
                last_log = now

            self.page.wait_for_timeout(1000)

        if last_text:
            # The stream never went completely quiet, but a long stable response
            # is more useful than throwing the whole run away.
            print("Claude response reached the timeout; using the latest text.")
            return last_text

        raise ClaudeError(
            "Claude did not produce a visible response within "
            f"{self.timeout} seconds. Check the Claude tab for a limit message."
        )

    # ------------------------------------------------------------------
    # Public operation
    # ------------------------------------------------------------------

    def generate_story(self, prompt: str = STORY_PROMPT) -> str:
        if self.page is None:
            self.connect()

        self.open_new_chat()
        before_texts = self._assistant_texts()
        self._submit_prompt(prompt)
        raw = self._wait_for_response(before_texts)
        self._raise_if_limit_message(raw)
        return raw

    @staticmethod
    def _raise_if_limit_message(raw: str) -> None:
        lowered = raw.lower()
        if (
            "message limit" in lowered
            or "usage limit" in lowered
            or ("limit" in lowered and "resets" in lowered)
        ) and "global" not in lowered:
            raise ClaudeLimitError(
                "Claude reported a message/usage limit. Wait for the limit to "
                f"reset and run again. Claude said: {raw[:300]!r}"
            )


# ----------------------------------------------------------------------
# High-level helper used by the GUI / CLI
# ----------------------------------------------------------------------

def generate_story_with_claude(
    prompt: str = STORY_PROMPT,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    attempts: int = 2,
) -> str:
    """Ask Claude for a story, clean it, validate it, and return the text.

    Invalid formatting is retried once in a fresh conversation.  Login and
    usage-limit errors are raised immediately because retrying cannot fix them.
    """
    browser = ClaudeBrowser(timeout=timeout)
    last_error: Exception | None = None

    try:
        for attempt in range(1, attempts + 1):
            print()
            print("=" * 40)
            print(f"CLAUDE STORY GENERATION (attempt {attempt}/{attempts})")
            print("=" * 40)

            raw = browser.generate_story(prompt)
            try:
                script = clean_claude_script(raw)
                clips = validate_claude_script(script)
            except ClaudeScriptError as exc:
                last_error = exc
                print(f"Claude response rejected: {exc}")
                if attempt >= attempts:
                    break
                print("Asking Claude again in a fresh conversation...")
                continue

            print(f"Claude returned a valid script with {len(clips)} clips.")
            return script
    finally:
        browser.close()

    if last_error is not None:
        raise last_error
    raise ClaudeError("Claude did not return a usable story.")


def open_claude_login() -> list[str]:
    """Open (or reuse) the Claude tab in the automation Chrome for sign-in."""
    return open_login_tabs((CLAUDE_URL,))
