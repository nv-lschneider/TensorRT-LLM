#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re
import sys
import urllib.request


PROMPT = (
    "Calculate 17 + 25. Do not explain your work. On the final line, "
    "output only the decimal integer."
)
EXPECTED_PATTERN = re.compile(r"(?<!\d)42(?!\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and validate a plain-text inference canary.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def generated_text(response: dict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    choice = choices[0]
    text = choice.get("text")
    if text is None and isinstance(choice.get("message"), dict):
        text = choice["message"].get("content")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("first response choice has no generated text")
    return text


def write_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = {
        "model": args.model,
        "prompt": PROMPT,
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
    }
    artifact = {"request": payload, "url": None, "passed": False}
    url = f"http://{args.host}:{args.port}/v1/completions"
    artifact["url"] = url
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as reply:
            response = json.loads(reply.read().decode("utf-8"))
        text = generated_text(response)
        artifact["response"] = response
        artifact["generated_text"] = text
        artifact["passed"] = bool(EXPECTED_PATTERN.search(text.strip()))
        if not artifact["passed"]:
            artifact["error"] = "generated text does not contain a standalone answer of 42"
    except Exception as exc:
        artifact["error"] = f"{type(exc).__name__}: {exc}"

    write_artifact(args.output, artifact)
    print("Semantic canary generated text:")
    print(artifact.get("generated_text", "<no generated text>"))
    if not artifact["passed"]:
        print(f"Semantic canary failed: {artifact['error']}", file=sys.stderr)
        return 1
    print("Semantic canary passed: arithmetic answer is 42")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
