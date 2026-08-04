"""Fixed benchmark prompts. Stable across runs so numbers are comparable.

Two workload shapes:
- code: structured output, high spec-decoding acceptance expected
- prose: open-ended, lower spec-decoding acceptance expected
"""

CODE_PROMPT = """Write a Python function `parse_log(line: str) -> dict` that takes a single
log line in the format:

  2024-04-26T10:15:32Z [INFO] component=auth user_id=4711 msg="login ok"

Return a dict with keys: timestamp (ISO string), level, component, user_id (int),
msg. Be tolerant of extra key=value pairs. Quote-aware splitting required.
Include a small if __name__ == "__main__" demo with three sample lines.
Output only the code, no commentary."""


PROSE_PROMPT = """In four short paragraphs, describe the working day of a long-haul truck driver
on a snowy highway in Idaho — the rhythm of the road, what they listen to, when
they stop, and the small rituals of the cab. Plain prose, no headings, no lists."""


# Map prompt name -> text. The harness picks one with --workload.
PROMPTS = {
    "code": CODE_PROMPT,
    "prose": PROSE_PROMPT,
}
