import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile


spec = importlib.util.spec_from_file_location("release_builder", Path(__file__).resolve().parents[1] / "scripts/build_release.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class AnonymousReleaseTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        for name in builder.TOP_FILES:
            (root / name).write_text("Anonymous fixture\n")
        (root / "mora").mkdir()
        (root / "mora/__init__.py").write_text("VERSION = 1\n")
        return root

    def test_private_paths_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.fixture(Path(folder) / "repo")
            (root / "README.md").write_text("/".join(("", "home", "private-reviewer", "data.csv")))
            with self.assertRaises(ValueError):
                builder.audit(root, builder.release_files(root))

    def test_email_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.fixture(Path(folder) / "repo")
            (root / "README.md").write_text("contact" + "@" + "example.org")
            with self.assertRaises(ValueError):
                builder.audit(root, builder.release_files(root))

    def test_explicit_identity_token_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.fixture(Path(folder) / "repo")
            (root / "README.md").write_text("private-author-token")
            with self.assertRaises(ValueError):
                builder.audit(root, builder.release_files(root), ["private-author-token"])

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.fixture(Path(folder) / "repo")
            (root / "mora/linked.py").symlink_to(root / "README.md")
            with self.assertRaises(ValueError):
                builder.release_files(root)

    def test_weights_not_included(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.fixture(Path(folder) / "repo")
            (root / "mora/model.pth").write_bytes(b"not-a-model")
            with self.assertRaises(ValueError):
                builder.release_files(root)

    def test_zip_has_no_git_history_and_neutral_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.fixture(Path(folder) / "repo")
            (root / ".git").mkdir()
            (root / ".git/config").write_text("not-to-be-exported")
            output = Path(folder) / "release.zip"
            builder.build(root, output)
            with zipfile.ZipFile(output) as archive:
                self.assertTrue(all(".git/" not in n for n in archive.namelist()))
                self.assertTrue(all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in archive.infolist()))
            with self.assertRaises(FileExistsError):
                builder.build(root, output)


if __name__ == "__main__":
    unittest.main()
