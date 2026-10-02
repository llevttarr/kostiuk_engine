from __future__ import annotations

def run_cell(cfg) -> list[dict]:
    raise NotImplementedError
def sweep(cfg, vary: list[str]) -> None:
    raise NotImplementedError
def report(rows) -> None:
    raise NotImplementedError
