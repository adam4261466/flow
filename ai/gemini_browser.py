"""Reliable Playwright automation for Gemini image generation.

Drop-in API used by main.py:
    gemini = GeminiBrowser()
    gemini.connect()
    gemini.open_gemini()
    path = gemini.generate_image(prompt, output_directory, filename)
    gemini.close()

The important fix is that generate_image() does not start waiting for an image
until the prompt has demonstrably left Gemini's composer / entered the chat.
"""

from __future__ import annotations

import base64
import json
import io
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image
from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from ai.chrome_cdp import (
    CDP_URL,
    CHROME_PROFILE,
    GEMINI_URL,
    cdp_is_ready,
    find_chrome,
    launch_chrome,
)
IMAGE_TIMEOUT_SECONDS = 300
SUBMIT_TIMEOUT_SECONDS = 18
TEXT_TIMEOUT_SECONDS = 180

# How long to wait after Gemini's text answer stops changing before deciding
# that it replied with text instead of generating an image.
TEXT_ONLY_GRACE_SECONDS = 10

# A critique answer is complete once its text has not changed for this long.
TEXT_STABLE_SECONDS = 4

# Gemini sometimes reports an exhausted image quota in normal chat text
# ("I can create more images as soon as your limit resets.") instead of a
# banner element.  Treat both forms as the same account limit so the factory
# pauses once instead of retrying (and repeating the message) for every clip.
LIMIT_TEXT_MARKERS = (
    "limit resets",
    "limit reached",
    "reached your limit",
    "usage limit",
    "out of images",
    "can create more images",
    "falling back",
    "quota",
)


def _looks_like_usage_limit(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in LIMIT_TEXT_MARKERS)


class GeminiLimitError(RuntimeError):
    """Raised when Gemini reports the account's usage limit is reached.

    This is a recoverable, account-level condition (not a code bug), so the
    pipeline can pause image generation, do other work, and resume later.
    """


class GeminiTextResponseError(RuntimeError):
    """Raised when Gemini replies with text instead of generating an image.

    Common causes are a refusal, a clarification question, or the model
    deciding to describe the scene. The caller should retry with a stricter
    image-only instruction rather than waiting out the full image timeout.
    """


class GeminiBrowser:
    """Connect to Chrome over CDP and generate/save images in Gemini."""

    def __init__(
        self,
        cdp_url: str = CDP_URL,
        image_timeout: int = IMAGE_TIMEOUT_SECONDS,
        conversation_file: str | Path | None = None,
        text_timeout: int = TEXT_TIMEOUT_SECONDS,
    ) -> None:
        self.cdp_url = cdp_url.rstrip("/")
        self.image_timeout = image_timeout
        self.text_timeout = text_timeout
        self.conversation_file = (
            Path(conversation_file) if conversation_file else None
        )
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.chrome_process: subprocess.Popen | None = None
        self._launched_chrome = False

    # ------------------------------------------------------------------
    # Chrome / Playwright lifecycle
    # ------------------------------------------------------------------

    def _cdp_is_ready(self) -> bool:
        return cdp_is_ready(self.cdp_url)

    @staticmethod
    def _find_chrome() -> str | None:
        return find_chrome()

    def _start_chrome(self) -> None:
        self.chrome_process = launch_chrome(
            initial_urls=[GEMINI_URL],
            cdp_url=self.cdp_url,
            profile=CHROME_PROFILE,
        )
        self._launched_chrome = True

    def connect(self) -> None:
        if self.page is not None and not self.page.is_closed():
            return

        if not self._cdp_is_ready():
            print("Chrome CDP is not available.")
            self._start_chrome()

        print("Connecting to Chrome through CDP...")
        self.playwright = sync_playwright().start()
        try:
            self.browser = self.playwright.chromium.connect_over_cdp(
                self.cdp_url,
                timeout=30000,
            )
            contexts = self.browser.contexts
            if contexts:
                self.context = contexts[0]
            else:
                self.context = self.browser.new_context(accept_downloads=True)

            # Reuse a Gemini tab whenever possible.
            for candidate in self.context.pages:
                if "gemini.google.com" in (candidate.url or ""):
                    self.page = candidate
                    break

            if self.page is None:
                self.page = self.context.new_page()

            self.page.set_default_timeout(15000)
            self.page.set_default_navigation_timeout(60000)
            print("Chrome connection established.")
        except Exception:
            try:
                self.playwright.stop()
            except Exception:
                pass
            self.playwright = None
            self.browser = None
            self.context = None
            self.page = None
            raise

    def open_gemini(self) -> None:
        if self.page is None:
            self.connect()

        assert self.page is not None
        print("Opening Gemini...")

        saved_conversation = self._read_conversation_url()
        current_url = self.page.url or ""

        if saved_conversation and current_url != saved_conversation:
            # Always return to the exact conversation used before so image
            # generation keeps the same context (character/style continuity).
            print(f"Returning to saved Gemini conversation: {saved_conversation}")
            try:
                self.page.goto(
                    saved_conversation,
                    wait_until="domcontentloaded",
                    timeout=60000,
                )
            except PlaywrightError as exc:
                print(f"Could not reopen saved conversation ({exc}); opening Gemini.")
                self.page.goto(GEMINI_URL, wait_until="domcontentloaded", timeout=60000)
        elif "gemini.google.com" not in current_url:
            self.page.goto(GEMINI_URL, wait_until="domcontentloaded", timeout=60000)
        else:
            self.page.bring_to_front()
            # Reload only when the tab is blank or the Gemini page failed to load.
            if current_url in ("about:blank", ""):
                self.page.goto(GEMINI_URL, wait_until="domcontentloaded", timeout=60000)

        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=15000)
        except PlaywrightTimeoutError:
            pass
        self.page.wait_for_timeout(1200)
        print("Gemini opened.")
        print(f"URL: {self.page.url}")
        self._remember_conversation()

    def start_new_conversation(self) -> None:
        """Open a fresh Gemini chat so each clip starts with clean context."""
        if self.page is None:
            self.connect()

        assert self.page is not None
        print("Opening a new Gemini conversation...")
        self.page.goto(
            GEMINI_URL,
            wait_until="domcontentloaded",
            timeout=60000,
        )
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=15000)
        except PlaywrightTimeoutError:
            pass
        self.page.wait_for_timeout(1200)
        self._dismiss_overlays()

        # Wait for the composer so an attached reference image is not lost
        # while the new chat is still rendering.
        try:
            self.page.locator(
                'rich-textarea [contenteditable="true"], '
                'div[role="textbox"][contenteditable="true"], '
                'textarea[aria-label*="prompt" i]'
            ).first.wait_for(state="visible", timeout=20000)
        except PlaywrightTimeoutError:
            print("Warning: Gemini's composer did not appear in time.")

        print(f"New conversation opened: {self.page.url}")

    # ------------------------------------------------------------------
    # Conversation continuity
    # ------------------------------------------------------------------

    def _read_conversation_url(self) -> str | None:
        """Return the previously saved conversation URL, if it has an id."""
        if self.conversation_file is None or not self.conversation_file.exists():
            return None
        try:
            value = self.conversation_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if value.startswith("https://gemini.google.com/app/"):
            return value
        return None

    def _remember_conversation(self) -> None:
        """Persist the conversation URL so later runs/retries keep the context.

        Only conversations that already have an id (``.../app/<id>``) are
        stored; the bare ``/app`` URL has no durable id yet.
        """
        if self.conversation_file is None or self.page is None:
            return
        url = self.page.url or ""
        if not url.startswith("https://gemini.google.com/app/"):
            return
        try:
            self.conversation_file.parent.mkdir(parents=True, exist_ok=True)
            self.conversation_file.write_text(url, encoding="utf-8")
        except OSError:
            pass

    def close(self) -> None:
        """Disconnect Playwright without intentionally killing an existing Chrome session."""
        print("Disconnecting from Chrome...")
        try:
            if self.playwright is not None:
                self.playwright.stop()
        except Exception as exc:
            print(f"Note while disconnecting from Chrome: {exc}")
        finally:
            self.playwright = None
            self.browser = None
            self.context = None
            self.page = None
        print("Chrome connection closed.")

    # ------------------------------------------------------------------
    # Composer detection and verified submission
    # ------------------------------------------------------------------

    def _first_visible(self, selectors: tuple[str, ...], require_enabled: bool = False):
        assert self.page is not None
        for selector in selectors:
            try:
                matches = self.page.locator(selector)
                for index in range(matches.count() - 1, -1, -1):
                    item = matches.nth(index)
                    try:
                        if not item.is_visible():
                            continue
                        if require_enabled and not item.is_enabled():
                            continue
                        return item
                    except PlaywrightError:
                        continue
            except PlaywrightError:
                continue
        return None

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        return " ".join((text or "").split()).strip()

    @staticmethod
    def _read_editor(editor):
        """Return the composer text, "" when empty, or None when unreadable.

        Returning None (instead of "") for a detached/rerendered composer is
        critical: an unreadable composer must never be mistaken for an empty
        one, otherwise the code reports a successful submission while the
        message is still sitting in the real text box.
        """
        try:
            result = editor.evaluate(
                """el => typeof el.value === 'string'
                    ? el.value
                    : (el.innerText || el.textContent || '')"""
            )
            return result or ""
        except PlaywrightError:
            return None

    def _find_editor(self):
        """Return the currently visible, enabled Gemini composer."""
        editor_selectors = (
            'rich-textarea [contenteditable="true"]',
            'div[role="textbox"][contenteditable="true"]',
            '[contenteditable="true"][aria-label*="prompt" i]',
            'textarea[aria-label*="prompt" i]',
            'div[contenteditable="true"]',
            "textarea",
        )
        return self._first_visible(editor_selectors, require_enabled=True)

    def _composer_text(self) -> str | None:
        """Current composer text: '' if empty, None if it cannot be read.

        Always re-acquires the editor because Gemini re-renders the composer,
        which invalidates any previously captured locator.
        """
        editor = self._find_editor()
        if editor is None:
            return None
        return self._read_editor(editor)


    def _user_message_count(self) -> int:
        assert self.page is not None
        selectors = (
            "user-query",
            '[data-message-author-role="user"]',
            '[data-testid*="user-query" i]',
            ".user-query",
        )
        counts: list[int] = []
        for selector in selectors:
            try:
                counts.append(self.page.locator(selector).count())
            except PlaywrightError:
                continue
        return max(counts, default=0)

    def _generation_is_running(self) -> bool:
        selectors = (
            'button[aria-label*="stop" i]',
            'button[title*="stop" i]',
            '[aria-label*="stop generating" i]',
            '[data-testid*="stop" i]',
        )
        return self._first_visible(selectors) is not None

    def _send_is_ready(self) -> bool:
        """True when Gemini's Send control is present and enabled."""
        send_selectors = (
            'button[aria-label="Send message"]',
            'button[aria-label="Send"]',
            'button[aria-label*="send" i]',
            '[role="button"][aria-label*="send" i]',
            'button[title*="send" i]',
            'button[data-tooltip*="send" i]',
        )
        return self._first_visible(send_selectors, require_enabled=True) is not None

    def _quota_limit_message(self) -> str | None:
        """Return Gemini's visible usage-limit banner text, if any.

        When the account's image quota is exhausted Gemini shows a banner
        ("Limit reached. Falling back to ...", "Your limit resets at ...")
        and simply never renders an image. Detecting it lets us stop
        immediately instead of burning the full 300-second timeout.
        """
        assert self.page is not None
        banners = (
            "[class*='banner']",
            "[class*='notice']",
            "[class*='upsell']",
            "[class*='error']",
            "[role='alert']",
        )
        for selector in banners:
            try:
                locator = self.page.locator(selector)
                count = locator.count()
            except PlaywrightError:
                continue
            for index in range(count):
                item = locator.nth(index)
                try:
                    if not item.is_visible():
                        continue
                    text = self._normalize_text(locator.nth(index).inner_text(timeout=1000))
                except PlaywrightError:
                    continue
                if not text:
                    continue
                if _looks_like_usage_limit(text):
                    return text
        return None

    def _focus_editor(self, editor) -> None:
        """Click the composer, working around overlay backdrops if present."""
        try:
            editor.click(timeout=8000)
            return
        except PlaywrightError:
            pass

        self._dismiss_overlays()

        try:
            editor.click(timeout=8000, force=True)
            return
        except PlaywrightError:
            pass

        # Do not raise: evaluate can still work.

    def _set_editor_text(self, prompt: str) -> bool:
        """Set the composer text directly via evaluate (bypasses all UI typing issues).

        Returns True if the text was set and the Send button is ready.
        """
        assert self.page is not None
        self._dismiss_overlays()

        editor = self._find_editor()
        if editor is None:
            return False

        try:
            self._focus_editor(editor)
        except PlaywrightError:
            pass

        # Directly set innerText and fire input event so Gemini's framework notices.
        # Use locator.evaluate which runs in the browser context with the element.
        try:
            js = """(el, text) => {
                el.innerText = text;
                el.dispatchEvent(new InputEvent('input', {bubbles: true, cancelable: true}));
                return el.innerText;
            }"""
            result = editor.evaluate(js, prompt)
            self.page.wait_for_timeout(300)
        except PlaywrightError as exc:
            print(f"evaluate set text failed: {exc}")
            return False

        # Verify it stuck.
        read = self._read_editor(editor) or ""
        if self._normalize_text(read) != self._normalize_text(prompt):
            print(f"evaluate text mismatch: got {len(read)} chars, expected {len(prompt)}")
            return False

        # Wait for Send to become ready.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self._send_is_ready():
                return True
            self.page.wait_for_timeout(200)
        return False

    def _submit_prompt_and_verify(self, prompt: str) -> None:
        """Set the prompt, submit it, and require positive confirmation.

        Uses evaluate-based text injection to avoid all UI typing flakiness.
        """
        assert self.page is not None

        normalized_prompt = self._normalize_text(prompt)

        # A viewer/dialog left open by a previous clip intercepts clicks on the
        # composer, so clear it before trying to interact.
        self._dismiss_overlays()

        editor = self._find_editor()
        if editor is None:
            raise RuntimeError(
                "Gemini prompt editor was not found. Make sure Gemini is open, "
                "loaded, and signed in."
            )

        # Set text via evaluate (most reliable). If it fails, fall back to the
        # old UI typing path once.
        if not self._set_editor_text(prompt):
            print("evaluate set failed; falling back to UI typing...")
            try:
                self._focus_editor(editor)
                editor.press("Control+A", timeout=5000)
                editor.press("Delete", timeout=5000)
                editor.type(prompt, delay=1, timeout=60000)
                self.page.wait_for_timeout(300)
            except PlaywrightError as exc:
                print(f"fallback typing failed: {exc}")

            # Verify fallback worked.
            read = self._read_editor(editor) or ""
            if self._normalize_text(read) != normalized_prompt:
                raise RuntimeError(
                    "Gemini did not receive the complete prompt in its composer. "
                    f"Expected {len(prompt)} characters, found {len(read)}."
                )

        # At this point the full prompt is in the composer and Send is ready.
        messages_before = self._user_message_count()

        def composer_text() -> str | None:
            return self._composer_text()

        def submission_confirmed() -> bool:
            # Positive evidence only. A composer that cannot be read (None) is
            # never treated as evidence that the message was sent.
            if self._user_message_count() > messages_before:
                return True

            text = composer_text()
            if text is None:
                return False

            # The composer is verifiably empty AND Gemini is generating.
            if not self._normalize_text(text) and self._generation_is_running():
                return True

            return False

        def wait_for_submission(seconds: float) -> bool:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if submission_confirmed():
                    return True
                self.page.wait_for_timeout(250)
            return submission_confirmed()

        def click_send() -> None:
            send_button = self._first_visible(
                (
                    'button[aria-label="Send message"]',
                    'button[aria-label="Send"]',
                    'button[aria-label*="send" i]',
                    '[role="button"][aria-label*="send" i]',
                    'button[title*="send" i]',
                    'button[data-tooltip*="send" i]',
                ),
                require_enabled=True,
            )
            if send_button is None:
                raise RuntimeError("Gemini Send button was not found or is disabled.")
            send_button.click(timeout=7000)

        print("Submitting prompt to Gemini...")

        # Click Send (Send is ready because _set_editor_text waited for it).
        try:
            click_send()
        except (PlaywrightError, RuntimeError) as exc:
            print(f"First submit attempt failed; will retry: {exc}")

        if wait_for_submission(6):
            print("Gemini prompt submission confirmed.")
            return

        # Retry with Enter only if the prompt is still sitting in the composer.
        if self._normalize_text(composer_text() or ""):
            try:
                editor = self._find_editor()
                if editor is not None:
                    editor.press("Enter", timeout=5000)
            except PlaywrightError as exc:
                print(f"Enter fallback failed: {exc}")

        if wait_for_submission(6):
            print("Gemini prompt submission confirmed after Enter fallback.")
            return

        # Final retry: click a freshly reacquired Send button.
        if self._normalize_text(composer_text() or ""):
            try:
                click_send()
            except (RuntimeError, PlaywrightError) as exc:
                print(f"Second Send attempt failed: {exc}")

        if wait_for_submission(6):
            print("Gemini prompt submission confirmed after retry.")
            return

        remaining = composer_text()
        raise TimeoutError(
            "Gemini did not submit the prompt: it is still in the composer and "
            "no new user message or active generation was detected. "
            f"The composer still has {len(remaining or '')} characters. "
            "Stopped before the 300-second image wait."
        )

    # ------------------------------------------------------------------
    # Reference image attachment
    # ------------------------------------------------------------------

    def _find_file_input(self):
        assert self.page is not None
        try:
            inputs = self.page.locator('input[type="file"]')
            count = inputs.count()
        except PlaywrightError:
            return None

        fallback = None
        for index in range(count):
            item = inputs.nth(index)
            try:
                accept = (item.get_attribute("accept") or "").lower()
                multiple = item.get_attribute("multiple")
            except PlaywrightError:
                continue
            # Prefer the dedicated image picker. The document picker and the
            # image picker can both be present in the current Gemini UI.
            if "image" in accept:
                return item
            if fallback is None and (
                not accept or "*/*" in accept or multiple is not None
            ):
                fallback = item
        return fallback

    def _wait_for_file_input(self, timeout_seconds: float = 8.0):
        """Poll for the file input, which Gemini creates lazily."""
        assert self.page is not None
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            found = self._find_file_input()
            if found is not None:
                return found
            self.page.wait_for_timeout(250)
        return None

    def _attachment_count(self) -> int:
        """Best-effort count of composer attachments (uploaded image chips)."""
        assert self.page is not None
        selectors = (
            '[class*="attachment" i] img',
            '[class*="upload" i] img',
            'img[src^="blob:"]',
            '[data-test-id*="attachment" i]',
            '[aria-label*="remove attachment" i]',
        )
        counts: list[int] = []
        for selector in selectors:
            try:
                counts.append(self.page.locator(selector).count())
            except PlaywrightError:
                continue
        return max(counts, default=0)

    def attach_reference_image(self, image_path: str | Path) -> None:
        """Attach a local image to the Gemini composer as visual reference."""
        assert self.page is not None

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Reference image does not exist: {path}"
            )

        self._dismiss_overlays()
        before_count = self._attachment_count()

        # The picker usually appears only after the "Upload & tools" menu is
        # opened, and Gemini sometimes creates it a beat later. Poll instead of
        # guessing a single fixed delay.
        file_input = self._wait_for_file_input(3)

        if file_input is None:
            attach_button = self._first_visible(
                (
                    'button[aria-label="Upload & tools"]',
                    'button[aria-label*="upload" i]',
                    'button[aria-label*="add" i]',
                    'button[aria-label*="attach" i]',
                    'button[aria-label*="plus" i]',
                    'button[aria-label*="insert" i]',
                ),
                require_enabled=True,
            )
            if attach_button is not None:
                try:
                    attach_button.click(timeout=5000)
                except PlaywrightError:
                    pass
                file_input = self._wait_for_file_input(8)

        if file_input is None:
            # Last resort: the menu item may open a native file chooser
            # instead of exposing an <input type="file">.
            try:
                menu_item = self.page.get_by_role(
                    "menuitem", name="Upload files", exact=False
                ).first
                menu_visible = (
                    menu_item.count() > 0 and menu_item.is_visible()
                )
            except PlaywrightError:
                menu_visible = False

            if menu_visible:
                try:
                    print(f"Attaching reference image: {path.name}")
                    with self.page.expect_file_chooser(timeout=8000) as info:
                        menu_item.click()
                    info.value.set_files(str(path))
                    self.page.wait_for_timeout(1500)
                    print("Reference image attached.")
                    return
                except (PlaywrightError, PlaywrightTimeoutError):
                    pass

        if file_input is None:
            raise RuntimeError(
                "Could not find Gemini's image upload control. "
                "The reference image was not attached."
            )

        print(f"Attaching reference image: {path.name}")
        try:
            file_input.set_input_files(str(path))
        except PlaywrightError as exc:
            raise RuntimeError(
                f"Could not attach reference image to Gemini: {exc}"
            ) from exc

        # Wait for the preview chip/thumbnail so the prompt is not submitted
        # while the upload is still in progress.
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            try:
                if self._attachment_count() > before_count:
                    break
                # An upload progress indicator is also positive evidence.
                if self._first_visible(
                    (
                        '[aria-label*="uploading" i]',
                        '[class*="upload-progress" i]',
                        '[class*="attachment-preview" i]',
                    )
                ) is not None:
                    break
            except PlaywrightError:
                pass
            self.page.wait_for_timeout(500)
        else:
            print("Warning: reference image preview was not detected; continuing.")

        self.page.wait_for_timeout(800)
        print("Reference image attached.")

    # ------------------------------------------------------------------
    # Detect text-only responses
    # ------------------------------------------------------------------

    def _assistant_texts(self) -> list[str]:
        """Snapshot the visible assistant response texts.

        User messages are explicitly excluded: after a prompt is submitted the
        user's own text would otherwise look like a "new assistant response"
        and trigger a false text-only error while the image is still rendering.
        """
        assert self.page is not None
        selectors = (
            "model-response",
            "message-content",
            '[data-message-author-role="model"]',
            '[data-testid*="model-response" i]',
        )
        texts: list[str] = []
        for selector in selectors:
            try:
                locator = self.page.locator(selector)
                count = locator.count()
            except PlaywrightError:
                continue
            for index in range(count):
                item = locator.nth(index)
                try:
                    in_user_message = item.evaluate(
                        """el => !!el.closest(
                            'user-query, .user-query, '
                            + '[data-message-author-role="user"]'
                        )"""
                    )
                except PlaywrightError:
                    in_user_message = False
                if in_user_message:
                    continue
                try:
                    value = self._normalize_text(
                        item.inner_text(timeout=1500)
                    )
                except PlaywrightError:
                    continue
                if value:
                    texts.append(value)
        return texts

    # ------------------------------------------------------------------
    # Detect the new generated image
    # ------------------------------------------------------------------

    def _snapshot_images(self) -> list[dict[str, Any]]:
        assert self.page is not None
        try:
            return self.page.locator("img").evaluate_all(
                """imgs => imgs.map((img, index) => {
                    const r = img.getBoundingClientRect();
                    const s = getComputedStyle(img);
                    return {
                        index,
                        src: img.currentSrc || img.src || '',
                        alt: img.alt || '',
                        naturalWidth: img.naturalWidth || 0,
                        naturalHeight: img.naturalHeight || 0,
                        rectWidth: Math.round(r.width),
                        rectHeight: Math.round(r.height),
                        visible: !!(r.width > 0 && r.height > 0) &&
                            s.display !== 'none' && s.visibility !== 'hidden' &&
                            Number(s.opacity || 1) > 0,
                        inUserMessage: !!img.closest(
                            'user-query, .user-query, [data-message-author-role="user"]'
                        )
                    };
                })"""
            )
        except PlaywrightError:
            return []

    @staticmethod
    def _image_signature(image: dict[str, Any]) -> tuple[str, int, int]:
        return (
            str(image.get("src", "")),
            int(image.get("naturalWidth", 0)),
            int(image.get("naturalHeight", 0)),
        )

    @staticmethod
    def _is_large_image(image: dict[str, Any]) -> bool:
        if not image.get("visible", False):
            return False
        # The reference image we attached is echoed back inside the user's
        # message. It is not a generated result, so never treat it as one.
        if image.get("inUserMessage", False):
            return False
        natural_w = int(image.get("naturalWidth", 0))
        natural_h = int(image.get("naturalHeight", 0))
        rect_w = int(image.get("rectWidth", 0))
        rect_h = int(image.get("rectHeight", 0))
        return (
            (natural_w >= 256 and natural_h >= 160)
            or (rect_w >= 280 and rect_h >= 160)
        )

    def _new_image_candidates(
        self,
        before_images: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        before_signatures = {self._image_signature(img) for img in before_images}
        before_by_index = {
            int(img["index"]): self._image_signature(img)
            for img in before_images
        }
        candidates: list[dict[str, Any]] = []
        for image in self._snapshot_images():
            if not self._is_large_image(image):
                continue
            signature = self._image_signature(image)
            index = int(image.get("index", -1))
            is_new = signature not in before_signatures
            changed_at_same_index = (
                index in before_by_index and before_by_index[index] != signature
            )
            if is_new or changed_at_same_index:
                candidates.append(image)
        return candidates

    def _wait_for_generated_image(
        self,
        before_images: list[dict[str, Any]],
        before_texts: list[str] | None = None,
        timeout_seconds: int | None = None,
    ):
        assert self.page is not None
        timeout_seconds = timeout_seconds or self.image_timeout
        before_signatures = {self._image_signature(img) for img in before_images}
        before_by_index = {int(img["index"]): self._image_signature(img) for img in before_images}
        before_text_set = set(before_texts or [])
        start = time.monotonic()
        deadline = start + timeout_seconds
        last_log = 0.0

        # Text-only detection state. Gemini sometimes answers with a written
        # description or a refusal instead of an image; waiting the full
        # 300 seconds for an image that will never come is what made the
        # pipeline feel stuck.
        text_signature: str | None = None
        text_stable_since: float | None = None
        latest_text = ""

        print(f"Waiting up to {timeout_seconds} seconds for a new Gemini image...")
        while time.monotonic() < deadline:
            images = self._snapshot_images()
            candidates: list[dict[str, Any]] = []
            for image in images:
                if not self._is_large_image(image):
                    continue
                signature = self._image_signature(image)
                index = int(image.get("index", -1))
                is_new = signature not in before_signatures
                changed_at_same_index = (
                    index in before_by_index and before_by_index[index] != signature
                )
                if is_new or changed_at_same_index:
                    candidates.append(image)

            if candidates:
                # Prefer the largest visible image among newly created/changed items.
                selected = max(
                    candidates,
                    key=lambda item: int(item.get("naturalWidth", 0))
                    * int(item.get("naturalHeight", 0))
                    or int(item.get("rectWidth", 0)) * int(item.get("rectHeight", 0)),
                )
                print("Detected new/updated Gemini image:")
                print(f"  Image index: {selected['index']}")
                print(
                    "  Size: "
                    f"{selected.get('rectWidth', 0)}x{selected.get('rectHeight', 0)} "
                    "(displayed), "
                    f"{selected.get('naturalWidth', 0)}x{selected.get('naturalHeight', 0)} "
                    "(source)"
                )
                return self.page.locator("img").nth(int(selected["index"]))

            now = time.monotonic()

            # ----------------------------------------------------------
            # Text-only response detection
            # ----------------------------------------------------------
            # Only consider assistant text that was not already on the page
            # before this prompt. Once that text stops changing and Gemini is
            # no longer generating, it replied with words instead of an image.
            try:
                new_texts = [
                    text
                    for text in self._assistant_texts()
                    if text not in before_text_set
                ]
            except PlaywrightError:
                new_texts = []

            if new_texts:
                current_signature = "\n".join(new_texts)
                if current_signature != text_signature:
                    text_signature = current_signature
                    text_stable_since = now
                    latest_text = new_texts[-1]
                elif (
                    text_stable_since is not None
                    and not self._generation_is_running()
                    and now - text_stable_since >= TEXT_ONLY_GRACE_SECONDS
                ):
                    # Final re-check: an image may have rendered while the
                    # text was settling. Only raise if there is still none.
                    if not self._new_image_candidates(before_images):
                        preview = self._normalize_text(latest_text)[:400]
                        if _looks_like_usage_limit(preview):
                            raise GeminiLimitError(
                                "Gemini image generation is blocked by the "
                                f"account limit: {preview}"
                            )
                        raise GeminiTextResponseError(
                            "Gemini replied with text instead of generating an "
                            f"image. Response: {preview!r}"
                        )

            if now - last_log >= 20:
                print("Still waiting for Gemini to render the new image...")
                last_log = now

            # If Gemini shows a usage-limit banner and no image is appearing,
            # stop early instead of waiting out the whole timeout.
            if now - start >= 15:
                limit_message = self._quota_limit_message()
                if limit_message:
                    raise GeminiLimitError(
                        "Gemini image generation is blocked by the account "
                        f"limit: {limit_message}"
                    )

            self.page.wait_for_timeout(1000)

        if text_signature:
            preview = self._normalize_text(latest_text)[:400]
            if _looks_like_usage_limit(preview):
                raise GeminiLimitError(
                    "Gemini image generation is blocked by the account "
                    f"limit: {preview}"
                )
            raise GeminiTextResponseError(
                "Gemini did not finish generating an image and left a text "
                f"response instead: {preview!r}"
            )

        raise TimeoutError(
            f"Gemini did not display a detectable generated image within "
            f"{timeout_seconds} seconds. The prompt was submitted successfully, "
            "so check Gemini for generation errors, quota limits, or a pending confirmation."
        )

    # ------------------------------------------------------------------
    # Download image without accidentally saving the whole page
    # ------------------------------------------------------------------

    def _try_download_buttons(self, output_path: Path) -> bool:
        assert self.page is not None
        # Prefer the exact full-size image control, then controls inside the
        # image viewer overlay. Generic "download" labels are tried last because
        # the page can contain unrelated download links (e.g. file attachments).
        selectors = (
            'button[aria-label="Download full size image"]',
            '[role="button"][aria-label="Download full size image"]',
            'button[aria-label*="full size image" i]',
            'div.cdk-overlay-container button[aria-label*="download" i]',
            'div.cdk-overlay-container [role="button"][aria-label*="download" i]',
            'button[aria-label*="download image" i]',
            'button[aria-label*="download" i]',
            '[role="button"][aria-label*="download" i]',
        )
        for selector in selectors:
            try:
                matches = self.page.locator(selector)
                for index in range(matches.count() - 1, -1, -1):
                    button = matches.nth(index)
                    try:
                        if not button.is_visible() or not button.is_enabled():
                            continue
                        print("Trying Gemini download control:")
                        label = ""
                        try:
                            label = (
                                button.get_attribute("aria-label")
                                or button.get_attribute("title")
                                or button.inner_text(timeout=1000)
                                or selector
                            )
                        except PlaywrightError:
                            label = selector
                        # Skip unrelated page downloads (attachments, files).
                        lowered_label = (label or "").lower()
                        if "more options" in lowered_label or (
                            "downloading" in lowered_label
                            and "image" not in lowered_label
                        ):
                            continue
                        print(f"  Label: {label}")
                        try:
                            with self.page.expect_download(timeout=10000) as download_info:
                                button.click(timeout=5000)
                            download = download_info.value
                            download.save_as(str(output_path))
                            if output_path.exists() and output_path.stat().st_size > 0:
                                self._normalize_to_png(output_path)
                                print("Gemini full-size download completed.")
                                return True
                        except PlaywrightTimeoutError:
                            # Some versions trigger a download after a delayed UI update.
                            self.page.wait_for_timeout(700)
                            if output_path.exists() and output_path.stat().st_size > 0:
                                return True
                        except PlaywrightError as exc:
                            print(f"Download control did not work: {exc}")
                    except PlaywrightError:
                        continue
            except PlaywrightError:
                continue
        return False

    def _try_fetch_image_source(self, image_locator, output_path: Path) -> bool:
        """Last-resort source download for same-session, directly fetchable image URLs."""
        assert self.page is not None
        try:
            data = image_locator.evaluate(
                """async img => {
                    const url = img.currentSrc || img.src;
                    if (!url) return null;
                    try {
                        const response = await fetch(url, {credentials: 'include'});
                        if (!response.ok) return null;
                        const blob = await response.blob();
                        return await new Promise(resolve => {
                            const reader = new FileReader();
                            reader.onload = () => resolve(reader.result);
                            reader.onerror = () => resolve(null);
                            reader.readAsDataURL(blob);
                        });
                    } catch (_) {
                        return null;
                    }
                }"""
            )
            if not data or "," not in data:
                return False
            header, encoded = data.split(",", 1)
            raw = base64.b64decode(encoded)
            if len(raw) < 1000:
                return False
            # Gemini may expose WebP/JPEG source bytes. Normalize them to PNG
            # because the pipeline requests .png filenames.
            with Image.open(io.BytesIO(raw)) as source_image:
                source_image.load()
                if source_image.mode not in ("RGB", "RGBA"):
                    source_image = source_image.convert("RGBA")
                source_image.save(output_path, format="PNG")
            return True
        except Exception:
            return False

    def _dismiss_overlays(self, attempts: int = 4) -> None:
        """Close Gemini's image viewer / dialogs so the page is clickable.

        Gemini shows a full-screen image viewer after a generated image is
        clicked. Its ``image-expansion-dialog-backdrop`` intercepts pointer
        events, so a leftover dialog makes the next prompt impossible to type
        and its download control impossible to reach. Press Escape and click
        any close control until no backdrop remains.
        """
        assert self.page is not None

        backdrop_selector = "div.cdk-overlay-backdrop-showing"

        for _ in range(attempts):
            try:
                if self.page.locator(backdrop_selector).count() == 0:
                    return
            except PlaywrightError:
                return

            # Prefer an explicit close button inside the overlay.
            close_button = self._first_visible(
                (
                    'div.cdk-overlay-container button[aria-label*="close" i]',
                    'div.cdk-overlay-container [role="button"][aria-label*="close" i]',
                    'button[aria-label*="close" i]',
                    'button[aria-label*="back" i]',
                ),
            )
            if close_button is not None:
                try:
                    close_button.click(timeout=4000)
                    self.page.wait_for_timeout(300)
                    continue
                except PlaywrightError:
                    pass

            # Escape closes the image viewer in every layout seen so far.
            try:
                self.page.keyboard.press("Escape")
                self.page.wait_for_timeout(400)
            except PlaywrightError:
                return

    @staticmethod
    def _normalize_to_png(output_path: Path) -> None:
        """Re-encode a downloaded image to real PNG.

        Gemini's "Download full size image" is actually a JPEG; keeping the
        .png filename with JPEG bytes is confusing and risks inconsistent
        handling downstream, so normalize it.
        """
        try:
            with Image.open(output_path) as source:
                if (source.format or "").upper() == "PNG":
                    return
                converted = source.convert("RGBA")
                converted.load()
            converted.save(output_path, format="PNG")
        except Exception:
            # Keep the original file if normalization fails; it is still usable.
            pass

    def _wait_for_download_control(self, timeout_ms: int = 10000) -> None:
        assert self.page is not None
        try:
            self.page.locator(
                'button[aria-label="Download full size image"], '
                'button[aria-label*="full size image" i], '
                'div.cdk-overlay-container button[aria-label*="download" i], '
                'button[aria-label*="download" i]'
            ).first.wait_for(state="visible", timeout=timeout_ms)
        except PlaywrightError:
            pass

    def _save_image(self, image_locator, output_path: Path) -> Path:
        assert self.page is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # A leftover viewer/backdrop blocks clicks, which is the usual reason
        # Gemini's download control seems to "disappear".
        self._dismiss_overlays()

        # Open the full-screen viewer first. Its download control is tied to
        # THIS image, whereas inline hover controls on the conversation can
        # belong to older images.
        try:
            image_locator.click(timeout=6000)
        except PlaywrightError:
            self._dismiss_overlays()
            try:
                image_locator.click(timeout=6000, force=True)
            except PlaywrightError:
                pass

        self._wait_for_download_control(timeout_ms=10000)

        if self._try_download_buttons(output_path):
            return output_path

        # Fall back to inline controls revealed by hovering the image.
        self._dismiss_overlays()
        try:
            image_locator.hover(timeout=5000)
            self.page.wait_for_timeout(400)
        except PlaywrightError:
            pass

        if self._try_download_buttons(output_path):
            return output_path

        # If the page's image source is directly fetchable, save those bytes.
        if self._try_fetch_image_source(image_locator, output_path):
            print("Image saved from Gemini's image source.")
            return output_path

        # Final fallback: crop to the image element, not the full browser page.
        # This should be rare; download buttons are preferred for full resolution.
        try:
            image_locator.screenshot(path=str(output_path), timeout=10000)
            if output_path.exists() and output_path.stat().st_size > 1000:
                print("Warning: saved the visible image element because Gemini's download control was unavailable.")
                return output_path
        except PlaywrightError as exc:
            raise RuntimeError(f"Could not save Gemini's generated image: {exc}") from exc

        raise RuntimeError("Gemini generated the image, but no usable download method succeeded.")

    # ------------------------------------------------------------------
    # Public operation
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt: str,
        output_directory: str | Path,
        filename: str,
        reference_image: str | Path | None = None,
        new_conversation: bool = False,
    ) -> Path:
        if self.page is None:
            self.connect()
        if self.page is None:
            raise RuntimeError("Playwright page is unavailable after connect().")
        if not prompt or not prompt.strip():
            raise ValueError("The Gemini image prompt cannot be empty.")

        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        # Do not mistake a previous run's file for a newly completed download.
        if output_path.exists():
            output_path.unlink()

        if new_conversation:
            # Each clip gets a clean conversation: the story and clip prompt
            # are pasted fresh, and the previous clip's image is attached as
            # the visual reference.
            self.start_new_conversation()
        elif "gemini.google.com" not in (self.page.url or ""):
            self.open_gemini()
        else:
            self.page.bring_to_front()
            # A viewer/dialog left open by a previous clip puts a backdrop over
            # the whole page that swallows clicks.
            self._dismiss_overlays()

        if reference_image is not None:
            try:
                self.attach_reference_image(reference_image)
            except (FileNotFoundError, RuntimeError, PlaywrightError) as exc:
                # Losing the reference weakens continuity but should not throw
                # away a clip we can still generate.
                print(f"Warning: could not attach reference image: {exc}")

        print("Recording current Gemini images and responses...")
        before_images = self._snapshot_images()
        before_texts = self._assistant_texts()

        # Critical ordering: verify message submission BEFORE waiting for output.
        print("Sending prompt to Gemini...")
        self._submit_prompt_and_verify(prompt)
        print("Prompt sent and verified.")

        # Gemini assigns the conversation id on the first message; remember it
        # so future runs/resumes continue in this exact conversation.
        self._remember_conversation()

        image_locator = self._wait_for_generated_image(
            before_images,
            before_texts=before_texts,
            timeout_seconds=self.image_timeout,
        )
        try:
            saved_path = self._save_image(image_locator, output_path)
        finally:
            # Always close Gemini's image viewer afterwards so the next clip's
            # composer and download control stay reachable.
            self._dismiss_overlays()
        print("Image saved to:")
        print(saved_path.resolve())
        return saved_path

    # ------------------------------------------------------------------
    # Public text operation (used by the critic layer)
    # ------------------------------------------------------------------

    def _wait_for_text_response(
        self,
        before_texts: list[str],
        timeout_seconds: int | None = None,
    ) -> str:
        """Wait until a new assistant answer stops changing, then return it.

        Unlike :meth:`_wait_for_generated_image` this expects words, so a
        stable text answer is the success condition.  The longest new text is
        returned because Gemini's DOM nests a short ``message-content`` inside
        the full ``model-response`` element.
        """

        assert self.page is not None
        timeout_seconds = timeout_seconds or self.text_timeout
        before_set = set(before_texts)
        start = time.monotonic()
        deadline = start + timeout_seconds
        signature: str | None = None
        stable_since: float | None = None
        latest = ""
        last_log = 0.0

        print(f"Waiting up to {timeout_seconds} seconds for Gemini's text answer...")
        while time.monotonic() < deadline:
            try:
                new_texts = [
                    text
                    for text in self._assistant_texts()
                    if text not in before_set
                ]
            except PlaywrightError:
                new_texts = []

            if new_texts:
                current = "\n".join(new_texts)
                if current != signature:
                    signature = current
                    stable_since = time.monotonic()
                    latest = max(new_texts, key=len)
                elif (
                    stable_since is not None
                    and not self._generation_is_running()
                    and time.monotonic() - stable_since >= TEXT_STABLE_SECONDS
                ):
                    return latest

            now = time.monotonic()
            if now - last_log >= 20:
                print("Still waiting for Gemini's text answer...")
                last_log = now

            if now - start >= 15:
                limit_message = self._quota_limit_message()
                if limit_message:
                    raise GeminiLimitError(
                        "Gemini text generation is blocked by the account "
                        f"limit: {limit_message}"
                    )

            self.page.wait_for_timeout(600)

        if latest:
            # The answer was visible but never fully settled; it is still more
            # useful to the critic than raising after a long wait.
            return latest

        raise TimeoutError(
            "Gemini did not produce a text answer within "
            f"{timeout_seconds} seconds."
        )

    def ask_text(
        self,
        prompt: str,
        *,
        new_conversation: bool = False,
        reference_image: str | Path | None = None,
        timeout_seconds: int | None = None,
    ) -> str:
        """Send a text-only prompt and return Gemini's answer.

        ``new_conversation=False`` keeps the current chat, which lets the
        image critic ask about the picture Gemini just generated without
        uploading it again.
        """

        if self.page is None:
            self.connect()
        if self.page is None:
            raise RuntimeError("Playwright page is unavailable after connect().")
        if not prompt or not prompt.strip():
            raise ValueError("The Gemini text prompt cannot be empty.")

        if new_conversation:
            self.start_new_conversation()
        elif "gemini.google.com" not in (self.page.url or ""):
            self.open_gemini()
        else:
            self.page.bring_to_front()
            self._dismiss_overlays()

        if reference_image is not None:
            try:
                self.attach_reference_image(reference_image)
            except (FileNotFoundError, RuntimeError, PlaywrightError) as exc:
                print(f"Warning: could not attach image for critique: {exc}")

        before_texts = self._assistant_texts()

        print("Sending text prompt to Gemini...")
        self._submit_prompt_and_verify(prompt)
        print("Text prompt sent and verified.")

        if not new_conversation:
            # Keep the story's saved image conversation intact when a critic
            # opens a throwaway chat to inspect an exported asset.
            self._remember_conversation()

        answer = self._wait_for_text_response(
            before_texts,
            timeout_seconds=timeout_seconds or self.text_timeout,
        )
        print(f"Gemini text answer received ({len(answer)} chars).")
        return answer

