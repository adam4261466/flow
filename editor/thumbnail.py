from pathlib import Path

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageFilter,
    ImageEnhance,
)


class ThumbnailCreator:

    def __init__(
        self,
        output_directory,
        width=1280,
        height=720,
    ):
        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.width = width
        self.height = height

    # =========================================================
    # FONT
    # =========================================================

    def _find_font(self):
        """
        Find a bold Windows font.
        """

        candidates = [
            Path(
                r"C:\Windows\Fonts\arialbd.ttf"
            ),

            Path(
                r"C:\Windows\Fonts\segoeuib.ttf"
            ),

            Path(
                r"C:\Windows\Fonts\calibrib.ttf"
            ),
        ]

        for path in candidates:

            if path.exists():
                return path

        return None

    # =========================================================
    # FIT IMAGE
    # =========================================================

    def _fit_background(self, image):
        """
        Fill the thumbnail canvas while maintaining aspect ratio.
        """

        image = image.convert("RGB")

        source_ratio = (
            image.width
            / image.height
        )

        target_ratio = (
            self.width
            / self.height
        )

        if source_ratio > target_ratio:

            # Source is wider.
            new_height = self.height

            new_width = int(
                new_height
                * source_ratio
            )

        else:

            # Source is taller.
            new_width = self.width

            new_height = int(
                new_width
                / source_ratio
            )

        image = image.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.LANCZOS,
        )

        left = (
            new_width
            - self.width
        ) // 2

        top = (
            new_height
            - self.height
        ) // 2

        image = image.crop(
            (
                left,
                top,
                left + self.width,
                top + self.height,
            )
        )

        return image

    # =========================================================
    # TITLE WRAPPING
    # =========================================================

    def _wrap_title(
        self,
        draw,
        title,
        font,
        max_width,
    ):
        """
        Wrap title into a small number of lines.
        """

        words = title.split()

        lines = []
        current = ""

        for word in words:

            test = (
                word
                if not current
                else f"{current} {word}"
            )

            bbox = draw.textbbox(
                (0, 0),
                test,
                font=font,
            )

            width = (
                bbox[2]
                - bbox[0]
            )

            if width <= max_width:

                current = test

            else:

                if current:
                    lines.append(
                        current
                    )

                current = word

        if current:
            lines.append(
                current
            )

        return lines[:3]

    # =========================================================
    # CREATE
    # =========================================================

    def create(
        self,
        source_image,
        title,
        title_position="bottom_left",
    ):
        """
        Create the final professional thumbnail.
        """

        source_image = Path(
            source_image
        )

        if not source_image.exists():

            raise FileNotFoundError(
                f"Thumbnail source image not found:\n"
                f"{source_image}"
            )

        # -----------------------------------------------------
        # Background
        # -----------------------------------------------------

        image = Image.open(
            source_image
        )

        image = self._fit_background(
            image
        )

        # -----------------------------------------------------
        # Slight cinematic enhancement
        # -----------------------------------------------------

        image = ImageEnhance.Contrast(
            image
        ).enhance(1.08)

        image = ImageEnhance.Color(
            image
        ).enhance(1.04)

        # -----------------------------------------------------
        # Dark overlay
        # -----------------------------------------------------

        overlay = Image.new(
            "RGBA",
            image.size,
            (0, 0, 0, 0),
        )

        overlay_draw = ImageDraw.Draw(
            overlay
        )

        # Gradient-like dark area.
        for y in range(
            self.height
        ):

            progress = (
                y
                / self.height
            )

            alpha = int(
                150
                * progress
            )

            overlay_draw.line(
                (
                    0,
                    y,
                    self.width,
                    y,
                ),
                fill=(
                    0,
                    0,
                    0,
                    alpha,
                ),
            )

        image = Image.alpha_composite(
            image.convert("RGBA"),
            overlay,
        )

        # -----------------------------------------------------
        # Title
        # -----------------------------------------------------

        draw = ImageDraw.Draw(
            image
        )

        font_path = self._find_font()

        if font_path:

            font_size = 92

            font = ImageFont.truetype(
                str(font_path),
                font_size,
            )

        else:

            font = ImageFont.load_default()

        max_title_width = int(
            self.width * 0.72
        )

        lines = self._wrap_title(
            draw=draw,
            title=title,
            font=font,
            max_width=max_title_width,
        )

        # -----------------------------------------------------
        # Calculate title block size
        # -----------------------------------------------------

        spacing = 12

        line_sizes = []

        for line in lines:

            bbox = draw.textbbox(
                (0, 0),
                line,
                font=font,
                stroke_width=0,
            )

            line_width = (
                bbox[2]
                - bbox[0]
            )

            line_height = (
                bbox[3]
                - bbox[1]
            )

            line_sizes.append(
                (
                    line_width,
                    line_height,
                )
            )

        total_height = (
            sum(
                height
                for _, height
                in line_sizes
            )
            + spacing
            * (len(lines) - 1)
        )

        # -----------------------------------------------------
        # Position
        # -----------------------------------------------------

        padding = 70

        if title_position == "top_left":

            x = padding
            y = padding

        elif title_position == "top_right":

            y = padding

            max_width = max(
                width
                for width, _
                in line_sizes
            )

            x = (
                self.width
                - max_width
                - padding
            )

        elif title_position == "bottom_right":

            max_width = max(
                width
                for width, _
                in line_sizes
            )

            x = (
                self.width
                - max_width
                - padding
            )

            y = (
                self.height
                - total_height
                - padding
            )

        else:

            # Default: bottom-left.
            x = padding

            y = (
                self.height
                - total_height
                - padding
            )

        # -----------------------------------------------------
        # Add semi-transparent title plate
        # -----------------------------------------------------

        max_line_width = max(
            width
            for width, _
            in line_sizes
        )

        plate_padding_x = 28
        plate_padding_y = 22

        plate_left = (
            x
            - plate_padding_x
        )

        plate_top = (
            y
            - plate_padding_y
        )

        plate_right = (
            x
            + max_line_width
            + plate_padding_x
        )

        plate_bottom = (
            y
            + total_height
            + plate_padding_y
        )

        plate = Image.new(
            "RGBA",
            image.size,
            (0, 0, 0, 0),
        )

        plate_draw = ImageDraw.Draw(
            plate
        )

        plate_draw.rounded_rectangle(
            (
                plate_left,
                plate_top,
                plate_right,
                plate_bottom,
            ),
            radius=18,
            fill=(
                0,
                0,
                0,
                125,
            ),
        )

        # Slight blur behind title plate.
        plate = plate.filter(
            ImageFilter.GaussianBlur(0.4)
        )

        image = Image.alpha_composite(
            image,
            plate,
        )

        draw = ImageDraw.Draw(
            image
        )

        # -----------------------------------------------------
        # Draw title
        # -----------------------------------------------------

        current_y = y

        for index, line in enumerate(
            lines
        ):

            line_width, line_height = (
                line_sizes[index]
            )

            # Shadow.
            draw.text(
                (
                    x + 5,
                    current_y + 5,
                ),
                line,
                font=font,
                fill=(
                    0,
                    0,
                    0,
                    220,
                ),
                stroke_width=8,
                stroke_fill=(
                    0,
                    0,
                    0,
                    200,
                ),
            )

            # Main title.
            draw.text(
                (
                    x,
                    current_y,
                ),
                line,
                font=font,
                fill="white",
                stroke_width=3,
                stroke_fill=(
                    0,
                    0,
                    0,
                    255,
                ),
            )

            current_y += (
                line_height
                + spacing
            )

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        output_path = (
            self.output_directory
            / "thumbnail.png"
        )

        image.convert(
            "RGB"
        ).save(
            output_path,
            "PNG",
            optimize=True,
        )

        return output_path