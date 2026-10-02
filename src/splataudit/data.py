from __future__ import annotations

from pathlib import Path

class Camera:
    pass
class Scene:
    pass

def load_scene(root: Path) -> Scene:
    raise NotImplementedError

def convert(dataset: str, src: Path, dst: Path):
    raise NotImplementedError
