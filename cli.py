"""Command-line entry point with optional Persian RTL display compatibility.

The RTL transformation is presentation-only. User input and model answers stay
in their original logical Unicode order inside the application.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
ARABIC_SCRIPT = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")


def rtl_fix_enabled() -> bool:
    """Enable visual RTL processing by default on Windows consoles."""
    value = os.getenv("MATH_TUTOR_RTL_FIX", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return os.name == "nt"


def format_terminal_text(text: str, fix_rtl: bool) -> str:
    """Format *only displayed text* for terminals without RTL shaping.

    Reverse ordering and Arabic contextual shaping are done by libraries,
    without modifying messages passed to the language model.
    """
    if not fix_rtl or not ARABIC_SCRIPT.search(text):
        return text

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError as exc:
        raise RuntimeError(
            "RTL display requires: pip install arabic-reshaper python-bidi"
        ) from exc

    # Wrap before converting to visual order: terminal-side auto-wrap can
    # otherwise split a pre-reversed Persian line in the wrong place.
    terminal_columns = shutil.get_terminal_size(fallback=(92, 24)).columns
    width = max(24, min(84, terminal_columns - 6))
    displayed_lines: list[str] = []

    for line in text.split("\n"):
        if not ARABIC_SCRIPT.search(line):
            displayed_lines.append(line)
            continue
        chunks = textwrap.wrap(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        for chunk in chunks or [""]:
            if ARABIC_SCRIPT.search(chunk):
                shaped = arabic_reshaper.reshape(chunk)
                displayed_lines.append(get_display(shaped, base_dir="R"))
            else:
                displayed_lines.append(chunk)

    return "\n".join(displayed_lines)


def display(message: str = "", *, fix_rtl: bool) -> None:
    print(format_terminal_text(message, fix_rtl))


def read_question_from_file(filename: str) -> str:
    """Allow Persian questions via a UTF-8 file if console input is reversed."""
    path = Path(filename.strip().strip('"'))
    if not path.is_file():
        raise FileNotFoundError(f"Question file not found: {path}")
    question = path.read_text(encoding="utf-8-sig").strip()
    if not question:
        raise ValueError("Question file is empty.")
    return question


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    fix_rtl = rtl_fix_enabled()
    if "--display-test" in sys.argv[1:]:
        print(f"RTL display fix: {'ON' if fix_rtl else 'OFF'}")
        display("سلام دنیا! این یک تست نمایش فارسی است.", fix_rtl=fix_rtl)
        display("معادله x^2 - 5x + 6 = 0 را حل کن.", fix_rtl=fix_rtl)
        return

    print("=" * 52)
    print("MATH_TUTOR | Complex Mathematics Tutor Agent")
    display("سؤال یا مفهوم ریاضی را بنویسید.", fix_rtl=fix_rtl)
    print("/help: Help | /exit: Exit | /file <path>: UTF-8 input")
    print("/knowledge: Known concepts | /skip: Skip a diagnostic")
    print("=" * 52)

    # Delayed import means --display-test needs no AvalAI API key.
    from agent import MathTutorAgent

    try:
        agent = MathTutorAgent()
    except Exception as exc:
        print(f"Agent initialization error: {exc}")
        return

    while True:
        try:
            question = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"/exit", "/quit"}:
            print("Goodbye!")
            break
        if question.lower() == "/help":
            print("Enter a math problem, concept, or translation request.")
            print("Example: x^2 - 5x + 6 = 0")
            print("If Persian input appears reversed, write it in a UTF-8")
            print("text file and enter: /file question.txt")
            print("Use /knowledge to list previously confirmed concepts.")
            print("During a short diagnostic, type /skip if you do not know.")
            continue
        if question.lower() == "/knowledge":
            try:
                known = agent.known_concepts()
                print("Known concepts: " + (", ".join(known) if known else "(none yet)"))
            except Exception as exc:
                print(f"Knowledge error ({type(exc).__name__}): {exc}")
            continue
        if question.lower().startswith("/file "):
            try:
                question = read_question_from_file(question[6:])
            except (OSError, UnicodeError, ValueError) as exc:
                print(f"File input error: {exc}")
                continue

        exit_requested = False
        try:
            thread_id, result = agent.start_question(question)
            while result.get("__interrupt__"):
                payload = result["__interrupt__"][0].value
                print("\n[Diagnostic: " + payload["concept_name"] + "]")
                display(payload["question"], fix_rtl=fix_rtl)
                display("پاسخ کوتاه بنویس، یا /skip را وارد کن.", fix_rtl=fix_rtl)
                try:
                    diagnostic_answer = input("Your answer > ").strip()
                except (EOFError, KeyboardInterrupt):
                    exit_requested = True
                    break
                if diagnostic_answer.lower() in {"/exit", "/quit"}:
                    exit_requested = True
                    break
                result = agent.resume_question(thread_id, diagnostic_answer)
        except Exception as exc:
            print(f"Agent error ({type(exc).__name__}): {exc}")
            continue

        if exit_requested:
            print("\nGoodbye!")
            break

        print(
            f"\n[Route: {result['intent']} | "
            f"Topic: {result['math_topic']} | "
            f"Depth: {result['explanation_depth']} | "
            f"Diagnostics: {result['assessment_count']}]"
        )
        print("\nTutor >")
        display(result["final_answer"], fix_rtl=fix_rtl)
        print("\n[Phase 2: Independent mathematical verification is not enabled yet.]")


if __name__ == "__main__":
    main()
