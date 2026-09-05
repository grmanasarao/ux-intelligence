from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_feedback(
    feedback: dict[str, Any],
    output_path: str = "reports/feedback.json",
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []

    feedback = {
        **feedback,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    existing.append(feedback)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
