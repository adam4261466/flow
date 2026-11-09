import re
from ai.claude_browser import ClaudeBrowser

class PromptArchitect:
    """Agent that generates a list of cinematic story prompts for the pipeline."""
    
    SYSTEM_PROMPT = """You are a Viral Content Strategist and Cinematic Story Architect. 
Your goal is to generate high-concept, emotionally engaging, and visually stunning story prompts for an AI video pipeline.

A 'prompt' for this system is a high-level premise (1-2 sentences) that describes the world, the main character, and the core conflict.

RULES FOR VIRAL STORYTELLING:
1. High Contrast: Combine contrasting elements (e.g., 'Cyberpunk Victorian London', 'A silent monk in a screaming city').
2. Strong Hook: Start with a mystery or a forbidden object.
3. Visual Potential: Focus on premises that allow for epic cinematic imagery (huge scale, weird lighting, strange architecture).
4. Emotional Core: Ensure there is a clear feeling (melancholy, terror, wonder, betrayal).

OUTPUT FORMAT:
Return ONLY a list of prompts, one per line. 
No numbering, no bullets, no introductory text, no 'Here are your prompts'.
Just the raw prompts.
"""

    KIDS_SYSTEM_PROMPT = """You are a Children's Content Strategist and Bedtime Story Author.
Your goal is to generate warm, gentle, imaginative story prompts for an AI video pipeline that makes videos for children aged 3 to 8.

A 'prompt' for this system is a high-level premise (1-2 sentences) that describes the world, the main character, and a small, gentle problem or goal.

RULES FOR CHILDREN'S STORIES:
1. Safe and Cozy: No violence, weapons, monsters, ghosts, danger, death, or scary imagery.
2. Kind Characters: Animal friends, magical helpers, families, and curious children.
3. Gentle Conflict: A lost object, a shy feeling, a shared treat, a new skill, a small misunderstanding.
4. Warm Emotion: Wonder, friendship, kindness, laughter, and feeling proud or comforted.
5. Bedtime Friendly: Calm endings that help a child feel safe and sleepy.

Avoid anything frightening, stressful, or aimed at adults.

OUTPUT FORMAT:
Return ONLY a list of prompts, one per line. 
No numbering, no bullets, no introductory text, no 'Here are your prompts'.
Just the raw prompts.
"""

    def __init__(self, cdp_url="http://127.0.0.1:9222"):
        self.browser = ClaudeBrowser(cdp_url=cdp_url)

    def generate_batch(
        self,
        theme="dark cinematic mysteries",
        count=30,
        *,
        kids=False,
    ) -> list[str]:
        """Generates a batch of prompts based on a theme."""
        self.browser.connect()
        self.browser.open_new_chat()

        system = self.KIDS_SYSTEM_PROMPT if kids else self.SYSTEM_PROMPT
        audience = (
            "gentle children's bedtime story prompts"
            if kids
            else "high-concept cinematic story prompts"
        )
        user_prompt = (
            f"{system}\n\nGenerate {count} unique, {audience} "
            f"based on the theme: {theme}."
        )
        
        print(f"Architect is brainstorming {count} stories for theme: {theme}...")
        raw_response = self.browser.generate_story(user_prompt) # Reusing generate_story for a simple text prompt
        
        # Clean the response to get only the lines that look like prompts
        lines = raw_response.splitlines()
        prompts = []
        for line in lines:
            line = line.strip()
            # Remove numbering like "1. " or "- "
            line = re.sub(r'^(\d+[\.\)]|[-*•])\s*', '', line)
            if line and len(line) > 10: # Filter out short noise
                prompts.append(line)
        
        self.browser.close()
        return prompts
