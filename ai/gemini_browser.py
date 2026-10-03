import subprocess
import time
from pathlib import Path
from PIL import Image
import requests

from playwright.sync_api import (
    sync_playwright,
)

DEFAULT_CDP_URL = "http://127.0.0.1:9222"
GEMINI_URL = "https://gemini.google.com/app"

CHROME_PATHS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
]

CHROME_PROFILE = (
    Path.home()
    / "Desktop"
    / "flow"
    / "chrome-profile"
)


class GeminiBrowser:
    """
    Controls Gemini through a Chrome instance started by the agent.

    The agent automatically:
        1. Checks whether CDP is available.
        2. Starts Chrome if necessary.
        3. Enables remote debugging on port 9222.
        4. Uses a persistent Chrome profile.
        5. Connects Playwright through CDP.
        6. Keeps one Gemini conversation for the whole story.
    """

    def __init__(
        self,
        cdp_url=DEFAULT_CDP_URL,
        response_timeout=300,
    ):
        self.cdp_url = cdp_url
        self.response_timeout = response_timeout

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        self.chrome_process = None

    # ---------------------------------------------------------
    # CHROME MANAGEMENT
    # ---------------------------------------------------------
    def _validate_image_quality(self, image_path):
        with Image.open(image_path) as image:

            width, height = image.size

            print(
                f"Downloaded image resolution: "
                f"{width}x{height}"
            )

            if width < 900 or height < 500:
                raise RuntimeError(
                    "Gemini download appears to be a "
                    f"low-resolution preview: "
                    f"{width}x{height}"
                )
    def find_chrome(self):
        """
        Find Chrome installation.
        """

        for path in CHROME_PATHS:
            if path.exists():
                return path

        raise FileNotFoundError(
            "Could not find Google Chrome.\n\n"
            "Checked:\n"
            + "\n".join(str(path) for path in CHROME_PATHS)
        )

    def is_cdp_available(self):
        """
        Check whether Chrome's CDP server is already running.
        """

        try:
            response = requests.get(
                f"{self.cdp_url}/json/version",
                timeout=2,
            )

            return response.status_code == 200

        except requests.RequestException:
            return False

    def close_chrome(self):
        """
        Completely close Chrome.

        This is intentionally done so the agent can restart Chrome
        with the required remote debugging flag.
        """

        print("Closing existing Chrome...")

        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "chrome.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception as error:
            print("Could not close Chrome:", error)

        # Give Windows a moment to release Chrome processes.
        time.sleep(2)

    def start_chrome(self):
        """
        Start Chrome with remote debugging enabled.
        """

        chrome_path = self.find_chrome()

        CHROME_PROFILE.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("Starting Chrome with CDP...")

        command = [
            str(chrome_path),
            "--remote-debugging-port=9222",
            f"--user-data-dir={CHROME_PROFILE}",
        ]

        self.chrome_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print("Chrome process started.")

    def wait_for_cdp(self, timeout=30):
        """
        Wait until Chrome exposes the CDP endpoint.
        """

        print("Waiting for Chrome CDP...")

        deadline = time.time() + timeout

        while time.time() < deadline:

            if self.is_cdp_available():
                print("Chrome CDP is ready.")
                return True

            time.sleep(0.5)

        return False

    def ensure_chrome(self):
        """
        Make sure Chrome with CDP is running.

        Flow:

        Existing CDP
            ↓
        use it

        No CDP
            ↓
        close Chrome
            ↓
        start Chrome
            ↓
        wait for CDP
            ↓
        continue
        """

        if self.is_cdp_available():
            print("Chrome CDP is already running.")
            return

        print("Chrome CDP is not available.")

        self.close_chrome()

        # Check once more in case Chrome needed extra time to close.
        if self.is_cdp_available():
            print("Chrome CDP became available.")
            return

        self.start_chrome()

        if not self.wait_for_cdp(timeout=30):
            raise RuntimeError(
                "Chrome started, but CDP did not become available.\n\n"
                f"Expected endpoint:\n{self.cdp_url}/json/version"
            )

    # ---------------------------------------------------------
    # PLAYWRIGHT CONNECTION
    # ---------------------------------------------------------

    def connect(self):
        """
        Automatically prepare Chrome and connect Playwright.
        """

        if self.page is not None:
            return self.page

        # NEW:
        # The agent handles Chrome startup automatically.
        self.ensure_chrome()

        print("Connecting to Chrome through CDP...")

        self.playwright = sync_playwright().start()

        try:
            self.browser = (
                self.playwright
                .chromium
                .connect_over_cdp(self.cdp_url)
            )

        except Exception as error:

            self.playwright.stop()
            self.playwright = None

            raise RuntimeError(
                "Could not connect Playwright to Chrome.\n\n"
                f"CDP URL:\n{self.cdp_url}\n\n"
                f"Original error:\n{error}"
            ) from error

        contexts = self.browser.contexts

        if not contexts:
            self.close()

            raise RuntimeError(
                "Chrome is connected, but no browser context was found."
            )

        self.context = contexts[0]

        pages = self.context.pages

        if pages:
            # Prefer an existing Gemini page.
            gemini_pages = [
                page
                for page in pages
                if "gemini.google.com" in page.url
            ]

            if gemini_pages:
                self.page = gemini_pages[0]
            else:
                self.page = pages[0]

        else:
            self.page = self.context.new_page()

        print("Chrome connection established.")

        return self.page

    # ---------------------------------------------------------
    # GEMINI
    # ---------------------------------------------------------

    def open_gemini(self):
        """
        Open Gemini in the already-connected Chrome.
        """

        if self.page is None:
            raise RuntimeError(
                "Call connect() before open_gemini()."
            )

        print("Opening Gemini...")

        self.page.goto(
            GEMINI_URL,
            wait_until="domcontentloaded",
            timeout=60000,
        )

        self.page.wait_for_timeout(3000)

        print("Gemini opened.")
        print("URL:", self.page.url)

    def send_message(self, message):
        """
        Send a message to the current Gemini conversation.
        """

        if self.page is None:
            raise RuntimeError(
                "Gemini is not connected."
            )

        if not message or not message.strip():
            raise ValueError(
                "Cannot send an empty Gemini message."
            )

        textbox = self._find_message_box()

        print("Sending prompt to Gemini...")

        textbox.fill(message)
        textbox.press("Enter")

        print("Prompt sent.")

    def _find_message_box(self):
        """
        Find Gemini's message input.
        """

        selectors = [
            "textarea",
            "[contenteditable='true']",
            "div[role='textbox']",
            "[aria-label*='prompt' i]",
            "[placeholder*='prompt' i]",
        ]

        for selector in selectors:

            locator = self.page.locator(selector)

            try:
                count = locator.count()
            except Exception:
                continue

            for index in range(count):

                element = locator.nth(index)

                try:
                    if element.is_visible():
                        return element
                except Exception:
                    continue

        raise RuntimeError(
            "Could not find Gemini's message input box.\n"
            "Gemini's UI selectors may have changed."
        )

    # ---------------------------------------------------------
    # IMAGE GENERATION
    # ---------------------------------------------------------

    def _snapshot_images(self):
        """
        Record the currently visible large images.

        We store the image src so that we can later detect
        newly created OR updated images.
        """

        snapshot = {}

        images = self.page.locator("img")

        try:
            count = images.count()
        except Exception:
            return snapshot

        for index in range(count):

            image = images.nth(index)

            try:
                if not image.is_visible():
                    continue

                box = image.bounding_box()

                if not box:
                    continue

                width = box.get("width", 0)
                height = box.get("height", 0)

                # Ignore icons, avatars, small UI images, etc.
                if width < 300 or height < 300:
                    continue

                src = image.get_attribute("src")

                snapshot[index] = {
                    "src": src,
                    "width": width,
                    "height": height,
                }

            except Exception:
                continue

        return snapshot    

    def generate_image(
        self,
        prompt,
        output_directory,
        filename,
    ):
        if self.page is None:
            raise RuntimeError(
                "Gemini is not connected."
            )

        output_directory = Path(output_directory)
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = output_directory / filename

        print("Recording current Gemini images...")

        before_images = self._snapshot_images()

        self.send_message(prompt)

        print(
            "Waiting for Gemini to generate the image..."
        )

        image_locator = self._wait_for_generated_image(
            before_images
        )

        print("New Gemini image detected.")

        downloaded = self._download_image_for_element(
            image_locator,
            output_path,
        )

        if not downloaded:
            raise RuntimeError(
                "Gemini generated the image, but the "
                "full-size Download control could not be found."
            )

        print(
            "Image saved to:"
        )

        print(
            output_path.resolve()
        )

        return output_path

    def _wait_for_generated_image(self, before_images):
        """
        Detect the image generated by the latest Gemini prompt.

        We do not rely on only:
            - image count
            - src changes

        Gemini can reuse DOM elements or populate image properties
        progressively while generating.
        """

        deadline = time.time() + self.response_timeout

        # Highest image index that existed before generation.
        previous_max_index = (
            max(before_images.keys())
            if before_images
            else -1
        )

        print(
            f"Waiting up to "
            f"{self.response_timeout} seconds..."
        )

        while time.time() < deadline:

            try:
                images = self.page.locator("img")
                count = images.count()

                candidates = []

                for index in range(count):

                    image = images.nth(index)

                    try:
                        if not image.is_visible():
                            continue

                        box = image.bounding_box()

                        if not box:
                            continue

                        width = box.get("width", 0)
                        height = box.get("height", 0)

                        # Ignore avatars, icons, thumbnails, etc.
                        if width < 300 or height < 250:
                            continue

                        src = image.get_attribute("src")

                        # Read the actual browser image dimensions.
                        try:
                            image_info = image.evaluate(
                                """
                                img => ({
                                    complete: img.complete,
                                    naturalWidth: img.naturalWidth,
                                    naturalHeight: img.naturalHeight
                                })
                                """
                            )
                        except Exception:
                            image_info = {}

                        natural_width = image_info.get(
                            "naturalWidth",
                            0,
                        )

                        natural_height = image_info.get(
                            "naturalHeight",
                            0,
                        )

                        # Ignore images that have not actually loaded.
                        if (
                            natural_width < 300
                            or natural_height < 250
                        ):
                            continue

                        before = before_images.get(index)

                        is_new_element = (
                            index > previous_max_index
                        )

                        is_changed_element = False

                        if before is not None:

                            old_src = before.get(
                                "src"
                            )

                            old_width = before.get(
                                "width",
                                0,
                            )

                            old_height = before.get(
                                "height",
                                0,
                            )

                            if src != old_src:
                                is_changed_element = True

                            elif (
                                abs(width - old_width) > 20
                                or
                                abs(height - old_height) > 20
                            ):
                                is_changed_element = True

                        if not (
                            is_new_element
                            or is_changed_element
                        ):
                            continue

                        # New/changed image.
                        candidates.append(
                            {
                                "index": index,
                                "image": image,
                                "width": width,
                                "height": height,
                                "src": src,
                            }
                        )

                    except Exception:
                        continue

                if candidates:

                    # Prefer the newest DOM image.
                    candidates.sort(
                        key=lambda item: item["index"],
                        reverse=True,
                    )

                    candidate = candidates[0]

                    print(
                        "Detected new/updated Gemini image:"
                    )

                    print(
                        f"  Image index: "
                        f"{candidate['index']}"
                    )

                    print(
                        f"  Size: "
                        f"{int(candidate['width'])}x"
                        f"{int(candidate['height'])}"
                    )

                    return candidate["image"]

            except Exception:
                pass

            self.page.wait_for_timeout(1000)

        # ---------------------------------------------------------
        # Final fallback:
        #
        # Gemini may have finished generating but the image
        # metadata may not have changed in a way we detected.
        #
        # Search the page for the newest large loaded image.
        # ---------------------------------------------------------

        print(
            "Normal image detection timed out."
        )

        print(
            "Performing final Gemini image scan..."
        )

        try:

            images = self.page.locator("img")
            count = images.count()

            fallback_candidates = []

            for index in range(count - 1, -1, -1):

                image = images.nth(index)

                try:

                    if not image.is_visible():
                        continue

                    box = image.bounding_box()

                    if not box:
                        continue

                    width = box.get(
                        "width",
                        0,
                    )

                    height = box.get(
                        "height",
                        0,
                    )

                    if width < 300 or height < 250:
                        continue

                    image_info = image.evaluate(
                        """
                        img => ({
                            complete: img.complete,
                            naturalWidth: img.naturalWidth,
                            naturalHeight: img.naturalHeight
                        })
                        """
                    )

                    if (
                        image_info.get(
                            "naturalWidth",
                            0,
                        ) < 300
                    ):
                        continue

                    if (
                        image_info.get(
                            "naturalHeight",
                            0,
                        ) < 250
                    ):
                        continue

                    fallback_candidates.append(
                        (
                            index,
                            image,
                        )
                    )

                except Exception:
                    continue

            if fallback_candidates:

                fallback_candidates.sort(
                    key=lambda item: item[0],
                    reverse=True,
                )

                index, image = (
                    fallback_candidates[0]
                )

                print(
                    "Fallback detected image:"
                )

                print(
                    f"  Image index: {index}"
                )

                return image

        except Exception:
            pass

        raise TimeoutError(
            "Gemini did not display a detectable generated image "
            f"within {self.response_timeout} seconds."
            )
    def _download_image_for_element(
        self,
        image_locator,
        output_path,
    ):
        """
        Download the original/full-size Gemini image.

        Gemini's download control may be rendered in an overlay
        outside the image's immediate DOM hierarchy.

        We therefore:
            1. Hover the exact image.
            2. Search the whole visible page.
            3. Find download controls.
            4. Rank them by proximity to the image.
            5. Click the closest relevant control.
        """

        import re

        # ---------------------------------------------------------
        # Hover exact generated image
        # ---------------------------------------------------------

        try:
            image_locator.scroll_into_view_if_needed()

            image_locator.hover()

            self.page.wait_for_timeout(1200)

        except Exception:
            pass

        # ---------------------------------------------------------
        # Get image position
        # ---------------------------------------------------------

        image_box = None

        try:
            image_box = image_locator.bounding_box()
        except Exception:
            pass

        # ---------------------------------------------------------
        # Search the WHOLE page for download controls
        # ---------------------------------------------------------

        candidate_selectors = [
            "button",
            "[role='button']",
            "a[download]",
        ]

        candidates = []

        for selector in candidate_selectors:

            try:
                locator = self.page.locator(
                    selector
                )

                count = locator.count()

            except Exception:
                continue

            for index in range(count):

                element = locator.nth(index)

                try:

                    if not element.is_visible():
                        continue

                    box = element.bounding_box()

                    if not box:
                        continue

                    aria_label = (
                        element.get_attribute(
                            "aria-label"
                        )
                        or ""
                    )

                    title = (
                        element.get_attribute(
                            "title"
                        )
                        or ""
                    )

                    data_tooltip = (
                        element.get_attribute(
                            "data-tooltip"
                        )
                        or ""
                    )

                    try:
                        text = element.inner_text(
                            timeout=500
                        )
                    except Exception:
                        text = ""

                    searchable = " ".join(
                        [
                            aria_label,
                            title,
                            data_tooltip,
                            text,
                        ]
                    ).lower()

                    # -------------------------------------------------
                    # Is this likely a download control?
                    # -------------------------------------------------

                    if not any(
                        keyword in searchable
                        for keyword in [
                            "download full size",
                            "download",
                        ]
                    ):
                        continue

                    # -------------------------------------------------
                    # Calculate distance to exact image
                    # -------------------------------------------------

                    distance = 1_000_000

                    if image_box:

                        image_center_x = (
                            image_box["x"]
                            + image_box["width"] / 2
                        )

                        image_center_y = (
                            image_box["y"]
                            + image_box["height"] / 2
                        )

                        button_center_x = (
                            box["x"]
                            + box["width"] / 2
                        )

                        button_center_y = (
                            box["y"]
                            + box["height"] / 2
                        )

                        distance = (
                            (
                                button_center_x
                                - image_center_x
                            ) ** 2
                            +
                            (
                                button_center_y
                                - image_center_y
                            ) ** 2
                        ) ** 0.5

                    # Exact "Download full size" gets priority.
                    exact_match = (
                        "download full size"
                        in searchable
                    )

                    candidates.append(
                        {
                            "element": element,
                            "distance": distance,
                            "exact_match": exact_match,
                            "text": searchable,
                        }
                    )

                except Exception:
                    continue

        # ---------------------------------------------------------
        # Rank candidates
        # ---------------------------------------------------------

        candidates.sort(
            key=lambda item: (
                not item["exact_match"],
                item["distance"],
            )
        )

        # ---------------------------------------------------------
        # Try candidates
        # ---------------------------------------------------------

        for candidate in candidates:

            button = candidate["element"]

            try:

                print(
                    "Trying Gemini download control:"
                )

                print(
                    f"  Label: "
                    f"{candidate['text']}"
                )

                print(
                    f"  Distance: "
                    f"{candidate['distance']:.1f}"
                )

                with self.page.expect_download(
                    timeout=15000
                ) as download_info:

                    button.click()

                download = (
                    download_info.value
                )

                download.save_as(
                    str(output_path)
                )

                print(
                    "Gemini full-size download completed."
                )

                return True

            except Exception:
                continue

        # ---------------------------------------------------------
        # SECOND ATTEMPT:
        # Open Gemini's image viewer.
        # ---------------------------------------------------------

        try:

            print(
                "Full-size control not found."
            )

            print(
                "Opening Gemini image viewer..."
            )

            image_locator.click()

            self.page.wait_for_timeout(
                1500
            )

        except Exception:
            pass

        # ---------------------------------------------------------
        # Search page again after viewer opened.
        # ---------------------------------------------------------

        candidates = []

        for selector in candidate_selectors:

            try:

                locator = self.page.locator(
                    selector
                )

                count = locator.count()

            except Exception:
                continue

            for index in range(count):

                element = locator.nth(index)

                try:

                    if not element.is_visible():
                        continue

                    aria_label = (
                        element.get_attribute(
                            "aria-label"
                        )
                        or ""
                    )

                    title = (
                        element.get_attribute(
                            "title"
                        )
                        or ""
                    )

                    data_tooltip = (
                        element.get_attribute(
                            "data-tooltip"
                        )
                        or ""
                    )

                    try:
                        text = element.inner_text(
                            timeout=500
                        )
                    except Exception:
                        text = ""

                    searchable = " ".join(
                        [
                            aria_label,
                            title,
                            data_tooltip,
                            text,
                        ]
                    ).lower()

                    if not any(
                        keyword in searchable
                        for keyword in [
                            "download full size",
                            "download",
                        ]
                    ):
                        continue

                    exact_match = (
                        "download full size"
                        in searchable
                    )

                    candidates.append(
                        (
                            not exact_match,
                            element,
                            searchable,
                        )
                    )

                except Exception:
                    continue

        candidates.sort(
            key=lambda item: item[0]
        )

        for _, button, searchable in candidates:

            try:

                print(
                    "Trying viewer download control:"
                )

                print(
                    f"  Label: {searchable}"
                )

                with self.page.expect_download(
                    timeout=15000
                ) as download_info:

                    button.click()

                download = (
                    download_info.value
                )

                download.save_as(
                    str(output_path)
                )

                print(
                    "Gemini full-size download completed."
                )

                return True

            except Exception:
                continue

        return False
    def show_page_info(self):

        if self.page is None:
            raise RuntimeError(
                "No active page."
            )

        print("Gemini page:")
        print("URL:", self.page.url)

        try:
            print(
                "TITLE:",
                self.page.title(),
            )
        except Exception:
            pass

    # ---------------------------------------------------------
    # CLOSE CONNECTION
    # ---------------------------------------------------------

    def close(self):

        print("Disconnecting from Chrome...")

        try:

            if self.browser:
                self.browser.close()

        except Exception:
            pass

        try:

            if self.playwright:
                self.playwright.stop()

        except Exception:
            pass

        self.browser = None
        self.context = None
        self.page = None
        self.playwright = None

        print("Chrome connection closed.")