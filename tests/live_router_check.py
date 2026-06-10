"""LIVE quality check for the SOLO/TEAM router — hits a real model.

Not collected by pytest (no ``test_`` prefix). Run manually once an
API account has credits:

    python tests/live_router_check.py [model]

Default model: claude-haiku-4-5 (pass gpt-4.1-mini for OpenAI). Costs
fractions of a cent. Quality gate: ZERO hard fails (a TEAM-labelled
prompt routed SOLO loses verification); soft fails (SOLO routed to
team) only cost the status-quo delegation overhead — a few are fine.
"""

from __future__ import annotations

import sys

import anyio

from loom_code.credentials import load_credentials

# (prompt, expected). TEAM expectations are hard requirements.
CASES: list[tuple[str, str]] = [
    ("fix the typo in the README heading", "SOLO"),
    ("bump the version in pyproject.toml to 0.2.0", "SOLO"),
    ("add a unit test for the slugify helper", "SOLO"),
    ("rename get_usr to get_user in utils.py", "SOLO"),
    ("run the test suite and tell me if it passes", "SOLO"),
    ("add user authentication with sessions and password reset", "TEAM"),
    ("refactor the storage layer to support postgres and sqlite", "TEAM"),
    (
        "find out why the worker queue deadlocks under load and fix it",
        "TEAM",
    ),
    ("migrate all date handling to timezone-aware datetimes", "TEAM"),
    ("integrate the Stripe API for subscription billing", "TEAM"),
    ("clean up the codebase", "TEAM"),
    ("delete all deprecated endpoints and their tests", "TEAM"),
    ("fix the bug we discussed", "TEAM"),
]


async def main(model: str) -> int:
    from loomflow import Agent

    from loom_code.repl import _ROUTER_PROMPT

    router = Agent(_ROUTER_PROMPT, model=model, prompt_caching=True)
    hard_fails: list[str] = []
    soft_fails: list[str] = []
    cost = 0.0
    for prompt, expected in CASES:
        r = await router.run(prompt, user_id="qa")
        got = "SOLO" if "SOLO" in r.output.upper() else "TEAM"
        cost += r.cost_usd
        mark = "ok  " if got == expected else "MISS"
        print(f"  [{mark}] {expected}->{got}: {prompt[:60]}")
        if got != expected:
            (hard_fails if expected == "TEAM" else soft_fails).append(
                prompt
            )
    print(f"\nhard fails (TEAM misrouted to SOLO): {len(hard_fails)}")
    for p in hard_fails:
        print(f"  !! {p}")
    print(f"soft fails (SOLO routed to team):    {len(soft_fails)}")
    print(f"total cost: ${cost:.4f}")
    return 1 if hard_fails else 0


if __name__ == "__main__":
    load_credentials()
    chosen = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4-5"
    raise SystemExit(anyio.run(main, chosen))
