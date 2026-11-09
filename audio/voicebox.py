"""Backward-compatible import path.

The Voicebox backend has been replaced by the Chatterbox TTS Server
client in :mod:`audio.chatterbox_tts`. Existing imports keep working.
"""

from audio.chatterbox_tts import ChatterboxTTS


Voicebox = ChatterboxTTS

__all__ = ["ChatterboxTTS", "Voicebox"]
