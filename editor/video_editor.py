from pathlib import Path
import json
import subprocess

import imageio_ffmpeg

FONT_PATH = "C\\:/Windows/Fonts/arial.ttf"

class VideoEditor:

    def __init__(
        self,
        image_directory,
        audio_directory,
        output_directory,
        sfx_directory=None,
    ):
        self.image_directory = Path(
            image_directory
        )

        self.audio_directory = Path(
            audio_directory
        )

        self.output_directory = Path(
            output_directory
        )

        if sfx_directory is None:

            self.sfx_directory = (
                self.output_directory.parent
                / "sfx"
            )

        else:

            self.sfx_directory = Path(
                sfx_directory
            )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================
    # FFMPEG
    # =========================================================

    def _get_ffmpeg(self):

        try:
            return imageio_ffmpeg.get_ffmpeg_exe()

        except Exception as error:

            raise RuntimeError(
                "Could not locate FFmpeg."
            ) from error

    # =========================================================
    # DISCOVER CLIPS
    # =========================================================

    def discover_clip_ids(self):

        clip_ids = []

        for image_path in self.image_directory.glob(
            "image_*.png"
        ):

            try:

                clip_id = int(
                    image_path.stem.split(
                        "_"
                    )[1]
                )

            except (
                ValueError,
                IndexError,
            ):
                continue

            clip_ids.append(
                clip_id
            )

        return sorted(
            set(clip_ids)
        )

    # =========================================================
    # MOTION
    # =========================================================

    def _build_motion_filter(
        self,
        motion,
        shake,
        duration,
        fps,
        width,
        height,
    ):
        """
        Build FFmpeg zoompan motion.

        All motion is rendered inside FFmpeg.
        """

        frames = max(
            int(round(duration * fps)),
            1,
        )

        denominator = max(
            frames - 1,
            1,
        )

        progress = (
            f"(on/{denominator})"
        )

        # -----------------------------------------------------
        # Zoom
        # -----------------------------------------------------

        zoom_amount = 0.08

        if motion in {
            "zoom_in",
            "push_in",
            "pan_left",
            "pan_right",
            "pan_up",
            "pan_down",
        }:

            zoom = (
                f"(1+{zoom_amount}*"
                f"{progress})"
            )

        elif motion in {
            "zoom_out",
            "pull_out",
        }:

            zoom = (
                f"(1+{zoom_amount}*"
                f"(1-{progress}))"
            )

        else:

            zoom = "1"

        # -----------------------------------------------------
        # Base X
        # -----------------------------------------------------

        center_x = (
            f"((iw-iw/{zoom})/2)"
        )

        center_y = (
            f"((ih-ih/{zoom})/2)"
        )

        if motion == "pan_left":

            base_x = (
                f"((iw-iw/{zoom})*"
                f"{progress})"
            )

            base_y = center_y

        elif motion == "pan_right":

            base_x = (
                f"((iw-iw/{zoom})*"
                f"(1-{progress}))"
            )

            base_y = center_y

        elif motion == "pan_up":

            base_x = center_x

            base_y = (
                f"((ih-ih/{zoom})*"
                f"{progress})"
            )

        elif motion == "pan_down":

            base_x = center_x

            base_y = (
                f"((ih-ih/{zoom})*"
                f"(1-{progress}))"
            )

        else:

            base_x = center_x
            base_y = center_y

        # -----------------------------------------------------
        # Camera shake
        # -----------------------------------------------------

        shake_amounts = {
            "none": 0,
            "subtle": 4,
            "medium": 9,
            "strong": 16,
        }

        shake_amount = shake_amounts.get(
            shake,
            0,
        )

        if shake_amount > 0:

            shake_x = (
                f"+{shake_amount}"
                f"*sin(on*0.55)"
            )

            shake_y = (
                f"+{shake_amount * 0.7}"
                f"*cos(on*0.43)"
            )

        else:

            shake_x = ""
            shake_y = ""

        # -----------------------------------------------------
        # Clamp crop position.
        # -----------------------------------------------------

        x = (
            f"max(0,min(iw-iw/{zoom},"
            f"{base_x}{shake_x}))"
        )

        y = (
            f"max(0,min(ih-ih/{zoom},"
            f"{base_y}{shake_y}))"
        )

        # -----------------------------------------------------
        # Oversize source before zoompan.
        #
        # This gives the camera room to move.
        # -----------------------------------------------------

        oversize_width = int(
            width * 1.15
        )

        oversize_height = int(
            height * 1.15
        )

        filter_string = (
            f"scale={oversize_width}:"
            f"{oversize_height}:"
            f"force_original_aspect_ratio=increase,"
            f"zoompan="
            f"z='{zoom}':"
            f"x='{x}':"
            f"y='{y}':"
            f"d={frames}:"
            f"s={width}x{height}:"
            f"fps={fps}"
        )

        return filter_string

    # =========================================================
    # COLOR GRADING
    # =========================================================

    def _build_grade_filter(
        self,
        grade,
    ):

        grade = str(
            grade
        ).lower()

        grades = {

            "neutral": (
                "eq="
                "contrast=1.03:"
                "brightness=0:"
                "saturation=1.00"
            ),

            "cinematic": (
                "eq="
                "contrast=1.08:"
                "brightness=-0.01:"
                "saturation=0.96"
            ),

            "moody": (
                "eq="
                "contrast=1.14:"
                "brightness=-0.035:"
                "saturation=0.88"
            ),

            "cool": (
                "eq="
                "contrast=1.06:"
                "brightness=-0.01:"
                "saturation=0.94,"
                "colorchannelmixer="
                "rr=0.97:"
                "gg=1.00:"
                "bb=1.06"
            ),

            "warm": (
                "eq="
                "contrast=1.05:"
                "brightness=0.005:"
                "saturation=1.02,"
                "colorchannelmixer="
                "rr=1.05:"
                "gg=1.00:"
                "bb=0.94"
            ),
        }

        return grades.get(
            grade,
            grades["cinematic"],
        )

    # =========================================================
    # TRANSITIONS
    # =========================================================

    def _transition_color(
        self,
        transition,
    ):

        if transition == "flash_white":
            return "white"

        return "black"

    def _build_transition_filters(
        self,
        fade_in_transition,
        fade_out_transition,
        duration,
        fade_duration,
    ):
        """
        Fade transitions are applied to the individual clips.

        That makes:
            fade out → black → fade in

        without requiring MoviePy's timeline effects.
        """

        filters = []

        fade_duration = min(
            fade_duration,
            duration / 2,
        )

        if fade_in_transition in {
            "fade_black",
            "flash_white",
        }:

            color = self._transition_color(
                fade_in_transition
            )

            filters.append(
                f"fade="
                f"t=in:"
                f"st=0:"
                f"d={fade_duration}:"
                f"color={color}"
            )

        if fade_out_transition in {
            "fade_black",
            "flash_white",
        }:

            color = self._transition_color(
                fade_out_transition
            )

            fade_start = max(
                duration - fade_duration,
                0,
            )

            filters.append(
                f"fade="
                f"t=out:"
                f"st={fade_start}:"
                f"d={fade_duration}:"
                f"color={color}"
            )

        return filters

    # =========================================================
    # SFX
    # =========================================================

    def _build_audio_filter(
        self,
        duration,
        has_sfx,
        sfx_start,
        sfx_volume,
    ):

        if not has_sfx:

            return None

        delay_ms = int(
            sfx_start * 1000
        )

        return (
            f"[2:a]"
            f"volume={sfx_volume},"
            f"atrim=duration={duration},"
            f"adelay={delay_ms}:all=1"
            f"[sfx];"
            f"[1:a][sfx]"
            f"amix="
            f"inputs=2:"
            f"duration=first:"
            f"dropout_transition=0"
            f"[aout]"
        )

    # =========================================================
    # RENDER ONE CLIP
    # =========================================================

    def render_clip(
        self,
        clip_id,
        plan,
        fps,
        resolution,
        fade_duration,
        sfx_volume,
        sfx_start_ratio,
        first_clip,
        last_clip,
        previous_transition,
    ):

        width, height = resolution

        image_path = (
            self.image_directory
            / f"image_{clip_id:03d}.png"
        )

        narration_path = (
            self.audio_directory
            / f"audio_{clip_id:03d}.wav"
        )

        sfx_path = (
            self.sfx_directory
            / f"sfx_{clip_id:03d}.wav"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n"
                f"{image_path}"
            )

        if not narration_path.exists():
            raise FileNotFoundError(
                f"Audio not found:\n"
                f"{narration_path}"
            )

        output_path = (
            self.output_directory
            / f"clip_{clip_id:03d}.mp4"
        )

        # -----------------------------------------------------
        # Get narration duration.
        # -----------------------------------------------------

        duration = self._get_audio_duration(
            narration_path
        )

        # -----------------------------------------------------
        # Plan values.
        # -----------------------------------------------------

        motion = plan.get(
            "motion",
            "static",
        )

        shake = plan.get(
            "shake",
            "none",
        )

        grade = plan.get(
            "grade",
            "cinematic",
        )

        current_transition = plan.get(
            "transition_after",
            "cut",
        )

        # -----------------------------------------------------
        # Motion filter.
        # -----------------------------------------------------

        motion_filter = (
            self._build_motion_filter(
                motion=motion,
                shake=shake,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
            )
        )

        # -----------------------------------------------------
        # Color grading.
        # -----------------------------------------------------

        grade_filter = (
            self._build_grade_filter(
                grade
            )
        )

        # -----------------------------------------------------
        # Transitions.
        # -----------------------------------------------------

        fade_filters = (
            self._build_transition_filters(
                fade_in_transition=(
                    previous_transition
                    if not first_clip
                    else "cut"
                ),
                fade_out_transition=(
                    "cut"
                    if last_clip
                    else current_transition
                ),
                duration=duration,
                fade_duration=fade_duration,
            )
        )

        video_filters = [
            motion_filter,
            grade_filter,
        ]

        video_filters.extend(
            fade_filters
        )

        video_filter_string = ",".join(
            video_filters
        )

        # -----------------------------------------------------
        # SFX.
        # -----------------------------------------------------

        use_sfx = (
            sfx_path.exists()
        )

        sfx_start = (
            duration
            * sfx_start_ratio
        )

        # -----------------------------------------------------
        # Build FFmpeg command.
        # -----------------------------------------------------

        ffmpeg = self._get_ffmpeg()

        command = [
            ffmpeg,
            "-y",

            "-i",
            str(image_path),

            "-i",
            str(narration_path),
        ]

        if use_sfx:

            command.extend([
                "-i",
                str(sfx_path),
            ])

        if use_sfx:

            audio_filter = (
                self._build_audio_filter(
                    duration=duration,
                    has_sfx=True,
                    sfx_start=sfx_start,
                    sfx_volume=sfx_volume,
                )
            )

            filter_complex = (
                f"[0:v]"
                f"{video_filter_string}"
                f"[vout];"
                f"{audio_filter}"
            )

        else:

            filter_complex = (
                f"[0:v]"
                f"{video_filter_string}"
                f"[vout]"
            )

        command.extend([
            "-filter_complex",
            filter_complex,

            "-map",
            "[vout]",
        ])

        if use_sfx:
            command.extend([
                "-map",
                "[aout]",
            ])
        else:
            command.extend([
                "-map",
                "1:a:0",
            ])

        command.extend([
            "-t",
            f"{duration:.6f}",

            "-r",
            str(fps),

            "-c:v",
            "libx264",

            "-preset",
            "medium",

            "-crf",
            "18",

            "-pix_fmt",
            "yuv420p",

            "-c:a",
            "aac",

            "-b:a",
            "192k",

            "-ar",
            "48000",

            "-movflags",
            "+faststart",

            str(output_path),
        ])

        print(
            f"Rendering clip {clip_id}..."
        )

        print(
            f"  Motion: {motion}"
        )

        print(
            f"  Shake: {shake}"
        )

        print(
            f"  Grade: {grade}"
        )

        print(
            f"  Transition after: "
            f"{current_transition}"
        )

        if use_sfx:

            print(
                f"  SFX: "
                f"{sfx_path.name}"
            )

            print(
                f"  SFX start: "
                f"{sfx_start:.2f}s"
            )

        else:

            print(
                "  SFX: none"
            )

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "FFmpeg failed while rendering "
                f"clip {clip_id}.\n\n"
                f"{result.stderr}"
            )

        return output_path, duration
    def _create_title_file(self, title):
        """
        Create a temporary text file containing the title.

        Using a text file avoids FFmpeg escaping problems with
        apostrophes, colons, commas, etc.
        """

        title_file = (
            self.output_directory
            / "title.txt"
        )

        title_file.write_text(
            title,
            encoding="utf-8",
        )

        return title_file
    def _create_intro(
    self,
    title,
    fps,
    resolution,
    duration=3.0,
    ):
        """
        Create a cinematic black intro with the story title.
        """

        ffmpeg = self._get_ffmpeg()

        width, height = resolution

        title_file = self._create_title_file(
            title
        )

        output_path = (
            self.output_directory
            / "intro.mp4"
        )

        title_path = (
            title_file
            .resolve()
            .as_posix()
            .replace(
                ":",
                r"\:",
            )
        )

        filter_complex = (
            f"[0:v]drawtext="
            f"font='{FONT_PATH}':"
            f"textfile='{title_path}':"
            f"fontcolor=white:"
            f"fontsize={int(width * 0.055)}:"
            f"x=(w-text_w)/2:"
            f"y=(h-text_h)/2:"
            f"alpha='if(lt(t,1),t/1,"
            f"if(gt(t,{duration - 1}),"
            f"({duration}-t)/1,1))'"
            f"[vout]"
        )

        command = [
            ffmpeg,
            "-y",

            "-f",
            "lavfi",

            "-i",
            (
                f"color=c=black:"
                f"s={width}x{height}:"
                f"r={fps}"
            ),

            "-f",
            "lavfi",

            "-i",
            (
                "anullsrc="
                "channel_layout=stereo:"
                "sample_rate=48000"
            ),

            "-filter_complex",
            filter_complex,

            "-map",
            "[vout]",

            "-map",
            "1:a",

            "-t",
            str(duration),

            "-r",
            str(fps),

            "-c:v",
            "libx264",

            "-preset",
            "medium",

            "-crf",
            "18",

            "-pix_fmt",
            "yuv420p",

            "-c:a",
            "aac",

            "-b:a",
            "192k",

            "-ar",
            "48000",

            "-shortest",

            str(output_path),
        ]

        print()
        print("Creating intro...")

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "FFmpeg failed while creating intro:\n\n"
                f"{result.stderr}"
            )

        return output_path
    def _create_outro(
    self,
    title,
    fps,
    resolution,
    duration=4.0,
    ):
        """
        Create a cinematic outro.

        The final shot fades into black, followed by:
            THE END
            <title>
        """

        ffmpeg = self._get_ffmpeg()

        width, height = resolution

        title_file = self._create_title_file(
            title
        )

        output_path = (
            self.output_directory
            / "outro.mp4"
        )

        title_path = (
            title_file
            .resolve()
            .as_posix()
            .replace(
                ":",
                r"\:",
            )
        )

        main_title_y = (
            f"(h-text_h)/2-40"
        )

        filter_complex = (
            f"[0:v]drawtext="
            f"font='{FONT_PATH}':"
            f"text='THE END':"
            f"fontcolor=white:"
            f"fontsize={int(width * 0.055)}:"
            f"x=(w-text_w)/2:"
            f"y={main_title_y}:"
            f"alpha='if(lt(t,1),t/1,"
            f"if(gt(t,{duration - 1}),"
            f"({duration}-t)/1,1))',"

            f"drawtext="
            f"font='{FONT_PATH}':"
            f"textfile='{title_path}':"
            f"fontcolor=white:"
            f"fontsize={int(width * 0.025)}:"
            f"x=(w-text_w)/2:"
            f"y=(h-text_h)/2+50:"
            f"alpha='if(lt(t,1),t/1,"
            f"if(gt(t,{duration - 1}),"
            f"({duration}-t)/1,1))'"
            f"[vout]"
        )

        command = [
            ffmpeg,
            "-y",

            "-f",
            "lavfi",

            "-i",
            (
                f"color=c=black:"
                f"s={width}x{height}:"
                f"r={fps}"
            ),

            "-f",
            "lavfi",

            "-i",
            (
                "anullsrc="
                "channel_layout=stereo:"
                "sample_rate=48000"
            ),

            "-filter_complex",
            filter_complex,

            "-map",
            "[vout]",

            "-map",
            "1:a",

            "-t",
            str(duration),

            "-r",
            str(fps),

            "-c:v",
            "libx264",

            "-preset",
            "medium",

            "-crf",
            "18",

            "-pix_fmt",
            "yuv420p",

            "-c:a",
            "aac",

            "-b:a",
            "192k",

            "-ar",
            "48000",

            "-shortest",

            str(output_path),
        ]

        print()
        print("Creating outro...")

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "FFmpeg failed while creating outro:\n\n"
                f"{result.stderr}"
            )

        return output_path
    # =========================================================
    # AUDIO DURATION
    # =========================================================

    def _get_audio_duration(
        self,
        audio_path,
    ):

        ffmpeg = self._get_ffmpeg()

        ffprobe = Path(
            ffmpeg
        ).with_name(
            "ffprobe.exe"
        )

        if not ffprobe.exists():

            # FFmpeg itself can still provide enough
            # information for this fallback.
            command = [
                ffmpeg,
                "-i",
                str(audio_path),
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            text = (
                result.stdout
                + "\n"
                + result.stderr
            )

            marker = "Duration:"

            if marker not in text:

                raise RuntimeError(
                    "Could not determine audio duration:\n"
                    f"{audio_path}"
                )

            duration_text = (
                text.split(
                    marker,
                    1
                )[1]
                .split(",", 1)[0]
                .strip()
            )

            hours, minutes, seconds = (
                duration_text.split(":")
            )

            return (
                float(hours) * 3600
                + float(minutes) * 60
                + float(seconds)
            )

        command = [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:"
            "nokey=1",
            str(audio_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "Could not determine audio duration:\n"
                f"{result.stderr}"
            )

        try:

            return float(
                result.stdout.strip()
            )

        except ValueError as error:

            raise RuntimeError(
                "Invalid audio duration returned "
                f"for {audio_path}."
            ) from error

    # =========================================================
    # FINAL VIDEO
    # =========================================================
    def create_final_video(
        self,
        plans,
        title="Untitled",
        fps=30,
        resolution=(1920, 1080),
        bitrate="8M",
        fade_duration=0.4,
        sfx_volume=0.30,
        sfx_start_ratio=0.35,
        intro_duration=3.0,
        outro_duration=4.0,
    ):
        print()
        print("========================================")
        print("CREATING FINAL VIDEO")
        print("========================================")

        print(
            f"Title: {title}"
        )

        clip_ids = self.discover_clip_ids()

        if not clip_ids:
            raise RuntimeError(
                "No images were found."
            )

        print(
            f"Clips detected: {len(clip_ids)}"
        )

        print(
            f"Resolution: "
            f"{resolution[0]}x{resolution[1]}"
        )

        print(
            f"FPS: {fps}"
        )

        print(
            f"Bitrate: {bitrate}"
        )

        print(
            f"Intro duration: {intro_duration}s"
        )

        print(
            f"Outro duration: {outro_duration}s"
        )

        rendered_clips = []

        concat_file = (
            self.output_directory
            / "concat_list.txt"
        )

        try:

            # =====================================================
            # INTRO
            # =====================================================

            print()
            print("Creating intro...")

            intro_path = self._create_intro(
                title=title,
                fps=fps,
                resolution=resolution,
                duration=intro_duration,
            )

            print(
                f"Intro created:\n"
                f"{intro_path.resolve()}"
            )

            # =====================================================
            # STORY CLIPS
            # =====================================================

            previous_transition = "cut"

            for index, clip_id in enumerate(
                clip_ids
            ):

                print()
                print("----------------------------------------")
                print(
                    f"Rendering story clip {clip_id}"
                )
                print("----------------------------------------")

                first_clip = (
                    index == 0
                )

                last_clip = (
                    index == len(clip_ids) - 1
                )

                plan = plans.get(
                    clip_id,
                    {
                        "motion": "static",
                        "shake": "none",
                        "grade": "cinematic",
                        "transition_after": "cut",
                    },
                )

                output_path, duration = (
                    self.render_clip(
                        clip_id=clip_id,
                        plan=plan,
                        fps=fps,
                        resolution=resolution,
                        fade_duration=fade_duration,
                        sfx_volume=sfx_volume,
                        sfx_start_ratio=sfx_start_ratio,
                        first_clip=first_clip,
                        last_clip=last_clip,
                        previous_transition=(
                            previous_transition
                        ),
                    )
                )

                rendered_clips.append(
                    output_path
                )

                previous_transition = (
                    plan.get(
                        "transition_after",
                        "cut",
                    )
                )

                print(
                    f"Clip {clip_id} completed."
                )

                print(
                    f"Duration: {duration:.2f}s"
                )

            # =====================================================
            # OUTRO
            # =====================================================

            print()
            print("Creating outro...")

            outro_path = self._create_outro(
                title=title,
                fps=fps,
                resolution=resolution,
                duration=outro_duration,
            )

            print(
                f"Outro created:\n"
                f"{outro_path.resolve()}"
            )

            # =====================================================
            # COMPLETE TIMELINE
            # =====================================================

            all_video_parts = [
                intro_path,
                *rendered_clips,
                outro_path,
            ]

            print()
            print(
                "Complete timeline:"
            )

            for index, path in enumerate(
                all_video_parts,
                start=1,
            ):

                print(
                    f"{index:02d}. {path.name}"
                )

            # =====================================================
            # CONCAT FILE
            # =====================================================

            with open(
                concat_file,
                "w",
                encoding="utf-8",
            ) as file:

                for path in all_video_parts:

                    absolute_path = (
                        Path(path)
                        .resolve()
                    )

                    ffmpeg_path = (
                        absolute_path
                        .as_posix()
                    )

                    file.write(
                        f"file '{ffmpeg_path}'\n"
                    )

            # =====================================================
            # FINAL VIDEO
            # =====================================================

            final_path = (
                self.output_directory
                / "final.mp4"
            )

            ffmpeg = self._get_ffmpeg()

            command = [
                ffmpeg,
                "-y",

                "-f",
                "concat",

                "-safe",
                "0",

                "-i",
                str(concat_file),

                "-c:v",
                "libx264",

                "-preset",
                "medium",

                "-b:v",
                bitrate,

                "-pix_fmt",
                "yuv420p",

                "-c:a",
                "aac",

                "-b:a",
                "192k",

                "-ar",
                "48000",

                "-movflags",
                "+faststart",

                str(final_path),
            ]

            print()
            print(
                "Combining intro + story + outro..."
            )

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:

                raise RuntimeError(
                    "FFmpeg failed while creating "
                    "the final video.\n\n"
                    f"{result.stderr}"
                )

            if not final_path.exists():

                raise RuntimeError(
                    "FFmpeg reported success, but "
                    "final.mp4 was not created."
                )

            print()
            print("========================================")
            print("FINAL VIDEO CREATED")
            print("========================================")

            print()
            print(
                f"Title: {title}"
            )

            print()
            print(
                f"Output:\n"
                f"{final_path.resolve()}"
            )

            print()
            print(
                "Timeline:"
            )

            print(
                f"  Intro: "
                f"{intro_duration:.2f}s"
            )

            for clip_id in clip_ids:

                print(
                    f"  Clip {clip_id}"
                )

            print(
                f"  Outro: "
                f"{outro_duration:.2f}s"
            )

            return final_path

        finally:

            try:
                concat_file.unlink()

            except FileNotFoundError:
                pass