from pathlib import Path
import math
import shutil
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


class VideoEditor:
    """FFmpeg-based video renderer.

    Key guarantees:
    - No camera shake/vibration.
    - Camera motion is smooth and deterministic.
    - Narration is cleaned and limited before encoding.
    - SFX is attenuated and limited.
    - Intro/outro contain silent stereo audio, matching clip streams.
    - Final assembly uses FFmpeg's concat filter, NOT stream-copy concat.
      This avoids the intro being stretched over the whole final timeline.
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

    ALLOWED_GRADES = {"neutral", "cinematic", "moody", "cool", "warm"}
    ALLOWED_TRANSITIONS = {"cut", "fade_black", "flash_white"}

    def __init__(
        self,
        image_directory,
        audio_directory,
        sfx_directory,
        output_directory,
    ):
        self.image_directory = Path(image_directory)
        self.audio_directory = Path(audio_directory)
        self.sfx_directory = Path(sfx_directory)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self.ffprobe = shutil.which("ffprobe")

    # ------------------------------------------------------------
    # BASIC HELPERS
    # ------------------------------------------------------------

    def _run(self, cmd):
        printable = " ".join(
            f'"{str(x)}"' if " " in str(x) else str(x)
            for x in cmd
        )
        print(f"FFmpeg: {printable}")
        subprocess.run(cmd, check=True)

    def _duration(self, path):
        if self.ffprobe:
            cmd = [
                self.ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            value = result.stdout.strip()
            if value:
                return max(0.05, float(value))

        return 5.0

    # ------------------------------------------------------------
    # IMAGE / MOTION
    # ------------------------------------------------------------

    def _build_visual_filter(
        self,
        plan,
        fps,
        width,
        height,
        duration,
        fade_duration,
    ):
        motion = str(plan.get("motion", "static")).lower()
        if motion not in self.ALLOWED_MOTIONS:
            motion = "static"

        grade = str(plan.get("grade", "cinematic")).lower()
        if grade not in self.ALLOWED_GRADES:
            grade = "cinematic"

        # SHAKE IS HARD DISABLED. The value is intentionally ignored.
        shake = "none"
        _ = shake

        # Only monotonic/one-direction motion is used (no sin/cos),
        # so the camera never looks like it is vibrating.
        overscan = 1.12 if motion != "static" else 1.0
        scaled_w = int(math.ceil(width * overscan / 2) * 2)
        scaled_h = int(math.ceil(height * overscan / 2) * 2)

        if motion == "static":
            video_filter = [
                f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase",
                f"crop={width}:{height}:x='(iw-ow)/2':y='(ih-oh)/2'",
                "setsar=1",
            ]

        elif motion in {"zoom_in", "push_in"}:
            # Smooth monotonic zoom: scale grows from 1.00 to about 1.08.
            # eval=frame is REQUIRED because the expression uses t.
            video_filter = [
                f"scale=w='ceil({scaled_w}*(1+0.08*t/max(0.1,{duration:.6f}))/2)*2':"
                f"h='ceil({scaled_h}*(1+0.08*t/max(0.1,{duration:.6f}))/2)*2':"
                "eval=frame:"
                "force_original_aspect_ratio=increase",
                f"crop={width}:{height}:x='(iw-ow)/2':y='(ih-oh)/2'",
                "setsar=1",
            ]

        elif motion in {"zoom_out", "pull_out"}:
            # Smooth monotonic zoom-out.
            # eval=frame is REQUIRED because the expression uses t.
            video_filter = [
                f"scale=w='ceil({scaled_w}*(1.08-0.08*t/max(0.1,{duration:.6f}))/2)*2':"
                f"h='ceil({scaled_h}*(1.08-0.08*t/max(0.1,{duration:.6f}))/2)*2':"
                "eval=frame:"
                "force_original_aspect_ratio=increase",
                f"crop={width}:{height}:x='(iw-ow)/2':y='(ih-oh)/2'",
                "setsar=1",
            ]

        elif motion == "pan_left":
            video_filter = [
                f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase",
                f"crop={width}:{height}:x='(iw-ow)*(0.78-0.28*t/max(0.1,{duration:.6f}))':y='(ih-oh)/2'",
                "setsar=1",
            ]

        elif motion == "pan_right":
            video_filter = [
                f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase",
                f"crop={width}:{height}:x='(iw-ow)*(0.22+0.28*t/max(0.1,{duration:.6f}))':y='(ih-oh)/2'",
                "setsar=1",
            ]

        elif motion == "pan_up":
            video_filter = [
                f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase",
                f"crop={width}:{height}:x='(iw-ow)/2':y='(ih-oh)*(0.78-0.28*t/max(0.1,{duration:.6f}))'",
                "setsar=1",
            ]

        elif motion == "pan_down":
            video_filter = [
                f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase",
                f"crop={width}:{height}:x='(iw-ow)/2':y='(ih-oh)*(0.22+0.28*t/max(0.1,{duration:.6f}))'",
                "setsar=1",
            ]

        else:
            video_filter = [
                f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase",
                f"crop={width}:{height}:x='(iw-ow)/2':y='(ih-oh)/2'",
                "setsar=1",
            ]

        # Conservative cinematic grading.
        if grade == "cinematic":
            video_filter.append("eq=contrast=1.06:saturation=1.04:gamma=0.98")
        elif grade == "moody":
            video_filter.append("eq=contrast=1.10:saturation=0.92:gamma=0.93")
        elif grade == "cool":
            video_filter.append(
                "eq=contrast=1.05:saturation=1.00:gamma=0.98,colorbalance=bs=.06"
            )
        elif grade == "warm":
            video_filter.append(
                "eq=contrast=1.05:saturation=1.05:gamma=1.00,colorbalance=rs=.05"
            )

        # Optional end transition.
        transition = str(
            plan.get("transition_after", "cut")
        ).lower()

        if fade_duration > 0:
            end_fade_start = max(0.0, duration - fade_duration)

            if transition == "fade_black":
                video_filter.append(
                    f"fade=t=in:st=0:d={fade_duration}"
                )
                video_filter.append(
                    f"fade=t=out:st={end_fade_start:.3f}:d={fade_duration}:color=black"
                )

            elif transition == "flash_white":
                video_filter.append("fade=t=in:st=0:d=0.15")
                video_filter.append(
                    f"fade=t=out:st={end_fade_start:.3f}:d=0.12:color=white"
                )

        video_filter.append(f"fps={fps}")
        return ",".join(video_filter)

    # ------------------------------------------------------------
    # AUDIO
    # ------------------------------------------------------------

    def _build_audio_filter(
        self,
        duration,
        has_sfx,
        sfx_start,
        sfx_volume,
    ):
        # Clean narration.
        voice = (
            "[1:a]"
            "aresample=48000:async=1:first_pts=0,"
            "highpass=f=65,"
            "acompressor=threshold=-19dB:ratio=2.4:attack=15:release=140:makeup=1.0,"
            "alimiter=limit=0.88:level=disabled"
            "[voice]"
        )

        if not has_sfx:
            return (
                voice
                + ";"
                "[voice]"
                "aresample=48000,"
                "volume=0.95,"
                "alimiter=limit=0.90:level=disabled"
                "[aout]"
            )

        delay_ms = max(0, int(duration * sfx_start * 1000))
        safe_sfx_volume = max(0.0, min(float(sfx_volume), 0.20))

        sfx = (
            "[2:a]"
            "aresample=48000:async=1:first_pts=0,"
            f"volume={safe_sfx_volume:.4f},"
            "highpass=f=50,"
            "lowpass=f=14000,"
            "acompressor=threshold=-24dB:ratio=3:attack=10:release=100:makeup=0.5,"
            "alimiter=limit=0.55:level=disabled,"
            f"atrim=0:{duration:.3f},"
            f"adelay={delay_ms}:all=1"
            "[sfx]"
        )

        mix = (
            "[voice][sfx]"
            "amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            "volume=0.95,"
            "alimiter=limit=0.88:level=disabled"
            "[aout]"
        )

        return voice + ";" + sfx + ";" + mix

    # ------------------------------------------------------------
    # TITLE SCREEN
    # ------------------------------------------------------------

    @staticmethod
    def _font(size, bold=True):
        candidates = []

        if bold:
            candidates.extend([
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
            ])

        candidates.extend([
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ])

        for candidate in candidates:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)

        return ImageFont.load_default()

    def _create_title_image(
        self,
        title,
        path,
        width,
        height,
        kind,
    ):
        image = Image.new(
            "RGB",
            (width, height),
            (8, 8, 10),
        )
        draw = ImageDraw.Draw(image)

        font = self._font(
            max(48, int(width * 0.055)),
            bold=True,
        )
        subtitle_font = self._font(
            max(24, int(width * 0.020)),
            bold=False,
        )

        bbox = draw.multiline_textbbox(
            (0, 0),
            title,
            font=font,
            spacing=10,
            align="center",
        )

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (width - text_w) / 2
        y = (height - text_h) / 2 - 20

        draw.multiline_text(
            (x, y),
            title,
            font=font,
            fill=(245, 245, 245),
            spacing=10,
            align="center",
        )

        subtitle = (
            "AI CINEMATIC STORY"
            if kind == "intro"
            else "THE END"
        )

        sb = draw.textbbox(
            (0, 0),
            subtitle,
            font=subtitle_font,
        )
        sx = (width - (sb[2] - sb[0])) / 2
        sy = y + text_h + 45

        draw.text(
            (sx, sy),
            subtitle,
            font=subtitle_font,
            fill=(170, 170, 170),
        )

        image.save(path)

    # ------------------------------------------------------------
    # CLIP RENDER
    # ------------------------------------------------------------

    def _render_clip(
        self,
        clip_id,
        plan,
        fps,
        resolution,
        bitrate,
        fade_duration,
        sfx_volume,
        sfx_start_ratio,
    ):
        width, height = resolution

        image_path = (
            self.image_directory
            / f"image_{clip_id:03d}.png"
        )
        audio_path = (
            self.audio_directory
            / f"audio_{clip_id:03d}.wav"
        )
        sfx_path = (
            self.sfx_directory
            / f"sfx_{clip_id:03d}.wav"
        )
        output_path = (
            self.output_directory
            / f"clip_{clip_id:03d}.mp4"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Missing narration: {audio_path}"
            )

        duration = self._duration(audio_path)
        has_sfx = sfx_path.exists()

        visual_filter = self._build_visual_filter(
            plan=plan,
            fps=fps,
            width=width,
            height=height,
            duration=duration,
            fade_duration=fade_duration,
        )

        audio_filter = self._build_audio_filter(
            duration=duration,
            has_sfx=has_sfx,
            sfx_start=sfx_start_ratio,
            sfx_volume=sfx_volume,
        )

        cmd = [
            self.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "warning",
            "-loop", "1",
            "-i", str(image_path),
            "-i", str(audio_path),
        ]

        if has_sfx:
            cmd.extend([
                "-stream_loop", "-1",
                "-i", str(sfx_path),
            ])

        cmd.extend([
            "-filter_complex",
            f"[0:v]{visual_filter}[v];{audio_filter}",
            "-map", "[v]",
            "-map", "[aout]",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", bitrate,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart",
            str(output_path),
        ])

        self._run(cmd)
        return output_path

    # ------------------------------------------------------------
    # INTRO / OUTRO
    # ------------------------------------------------------------

    def _create_title_video(
        self,
        title,
        output_path,
        duration,
        fps,
        resolution,
        bitrate,
        kind,
    ):
        width, height = resolution
        title_image = output_path.with_suffix(".png")

        self._create_title_image(
            title,
            title_image,
            width,
            height,
            kind,
        )

        # IMPORTANT: add a silent stereo audio stream.
        # Every segment of the final timeline must have the same
        # video/audio stream structure.
        cmd = [
            self.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "warning",
            "-loop", "1",
            "-i", str(title_image),
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t", f"{duration:.3f}",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-vf", f"fps={fps},format=yuv420p",
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", bitrate,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]

        self._run(cmd)

        try:
            title_image.unlink()
        except OSError:
            pass

        return output_path

    # ------------------------------------------------------------
    # FINAL ASSEMBLY
    # ------------------------------------------------------------

    def _concat_with_filter(
        self,
        files,
        output_path,
        fps,
        bitrate,
    ):
        """Concatenate all segments safely using FFmpeg's concat filter.

        This intentionally does NOT use:
            -f concat ... -c copy

        The old stream-copy method was responsible for the malformed final
        timeline because intro/outro originally had no audio while clips did.
        """
        files = [Path(path).resolve() for path in files]

        if not files:
            raise RuntimeError("No video segments to concatenate.")

        filter_parts = []
        concat_inputs = []

        for index in range(len(files)):
            # Normalize timestamps for every input.
            filter_parts.append(
                f"[{index}:v:0]setpts=PTS-STARTPTS[v{index}]"
            )
            filter_parts.append(
                f"[{index}:a:0]aresample=48000,asetpts=N/SR/TB[a{index}]"
            )
            concat_inputs.append(
                f"[v{index}][a{index}]"
            )

        concat_chain = (
            "".join(concat_inputs)
            + f"concat=n={len(files)}:v=1:a=1[outv][outa]"
        )

        filter_complex = ";".join(
            filter_parts + [concat_chain]
        )

        temporary = output_path.with_name(
            output_path.stem + ".tmp.mp4"
        )

        if temporary.exists():
            temporary.unlink()

        cmd = [
            self.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "warning",
        ]

        for path in files:
            cmd.extend([
                "-i", str(path),
            ])

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-r", str(fps),
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", bitrate,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart",
            str(temporary),
        ])

        self._run(cmd)

        # Replace final output only after FFmpeg successfully completes.
        if output_path.exists():
            output_path.unlink()

        temporary.replace(output_path)
        return output_path

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------

    def create_final_video(
        self,
        plans,
        title,
        fps=30,
        resolution=(1920, 1080),
        bitrate="8M",
        fade_duration=0.4,
        sfx_volume=0.15,
        sfx_start_ratio=0.35,
        intro_duration=3.0,
        outro_duration=4.0,
    ):
        clip_ids = sorted(
            int(key)
            for key in plans.keys()
        )

        if not clip_ids:
            raise RuntimeError(
                "No visual plans were supplied to the video editor."
            )

        rendered_clips = []

        for clip_id in clip_ids:
            plan = dict(plans[clip_id])

            # HARD DISABLE SHAKE/VIBRATION at final render boundary.
            plan["shake"] = "none"

            print()
            print(f"Rendering clip {clip_id}...")

            rendered = self._render_clip(
                clip_id=clip_id,
                plan=plan,
                fps=fps,
                resolution=resolution,
                bitrate=bitrate,
                fade_duration=fade_duration,
                sfx_volume=sfx_volume,
                sfx_start_ratio=sfx_start_ratio,
            )

            rendered_clips.append(rendered)

        intro = self.output_directory / "intro.mp4"
        outro = self.output_directory / "outro.mp4"
        final = self.output_directory / "final_video.mp4"

        self._create_title_video(
            title=title,
            output_path=intro,
            duration=intro_duration,
            fps=fps,
            resolution=resolution,
            bitrate=bitrate,
            kind="intro",
        )

        self._create_title_video(
            title=title,
            output_path=outro,
            duration=outro_duration,
            fps=fps,
            resolution=resolution,
            bitrate=bitrate,
            kind="outro",
        )

        print()
        print("Assembling final video...")
        print("Using FFmpeg concat filter (safe audio/video synchronization).")

        return self._concat_with_filter(
            [intro, *rendered_clips, outro],
            final,
            fps=fps,
            bitrate=bitrate,
        )