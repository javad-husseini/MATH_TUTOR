"""Offline tests for CLI presentation and UTF-8 file input."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import ModuleType
import sys

from cli import format_terminal_text, read_question_from_file, rtl_fix_enabled


class CLIDisplayTests(unittest.TestCase):
    def test_no_fix_keeps_original_persian(self):
        text = "سلام دنیا! x^2 + 1 = 0"
        self.assertEqual(format_terminal_text(text, False), text)

    def test_ascii_only_is_unchanged(self):
        text = "x^2 - 5x + 6 = 0"
        self.assertEqual(format_terminal_text(text, True), text)

    def test_rtl_fix_uses_shaper_and_bidi(self):
        reshaper = ModuleType("arabic_reshaper")
        reshaper.reshape = lambda value: "shaped:" + value
        bidi = ModuleType("bidi")
        bidi_alg = ModuleType("bidi.algorithm")
        bidi_alg.get_display = lambda text, base_dir=None: f"visual({base_dir}):{text}"
        with patch.dict(sys.modules, {
            "arabic_reshaper": reshaper,
            "bidi": bidi,
            "bidi.algorithm": bidi_alg,
        }):
            result = format_terminal_text("سلام", True)
        self.assertEqual(result, "visual(R):shaped:سلام")

    def test_toggle_from_environment(self):
        with patch.dict("os.environ", {"MATH_TUTOR_RTL_FIX": "0"}):
            self.assertFalse(rtl_fix_enabled())
        with patch.dict("os.environ", {"MATH_TUTOR_RTL_FIX": "1"}):
            self.assertTrue(rtl_fix_enabled())

    def test_read_utf8_file_with_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "question.txt"
            path.write_text("\ufeffمشتق چیست؟", encoding="utf-8")
            self.assertEqual(read_question_from_file(str(path)), "مشتق چیست؟")


if __name__ == "__main__":
    unittest.main()
