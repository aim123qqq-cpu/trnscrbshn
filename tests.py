import tempfile
import unittest
from pathlib import Path

import app


class AppHelpersTest(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(app.format_timestamp(3661.8), "01:01:01")

    def test_markdown_to_paragraphs_strips_headings(self):
        self.assertEqual(app.markdown_to_paragraphs("# Title\n\n## Next")[0], "Title")

    def test_save_docx_creates_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.docx"
            app.save_docx("# Протокол\n\nТекст", path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
