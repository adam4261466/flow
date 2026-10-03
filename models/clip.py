class Clip:
    def __init__(self, id, characters, moment, vibe, style, script):
        self.id = id
        self.characters = characters
        self.moment = moment
        self.vibe = vibe
        self.style = style
        self.script = script

    def __repr__(self):
        return f"Clip(id={self.id}, characters={self.characters!r})"
