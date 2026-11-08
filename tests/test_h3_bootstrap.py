"""CPU-only regression checks for the clean H3 bootstrap and phase driver."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_bootstrap", ROOT / "scripts/h3_bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def item(self, data=b"wheel-data"):
        return dict(filename="sample.whl", size=len(data),
                    sha256=hashlib.sha256(data).hexdigest(), url="https://example.invalid/sample.whl")

    def test_verified_rejects_same_size_corruption_and_truncation(self):
        p = self.root / "sample.whl"
        item = self.item()
        for data in (b"wheel-datX", b"short"):
            p.write_bytes(data)
            self.assertFalse(bootstrap.verified(p, item))
        p.write_bytes(b"wheel-data")
        self.assertTrue(bootstrap.verified(p, item))

    def test_fetch_reuses_verified_download_without_network(self):
        p = self.root / "sample.whl"
        p.write_bytes(b"wheel-data")
        with patch.object(bootstrap, "urlopen", side_effect=AssertionError("network used")):
            self.assertEqual(bootstrap.fetch(self.item(), self.root), p)

    def test_fetch_retries_bad_sha_before_atomic_promotion(self):
        with patch.object(bootstrap, "urlopen", side_effect=[
                io.BytesIO(b"wheel-datX"), io.BytesIO(b"wheel-data")]) as open_url:
            result = bootstrap.fetch(self.item(), self.root)
        self.assertEqual(open_url.call_count, 2)
        self.assertEqual(result.read_bytes(), b"wheel-data")
        self.assertFalse(result.with_suffix(".whl.partial").exists())

    def test_fetch_failure_does_not_promote_partial(self):
        with patch.object(bootstrap, "urlopen", side_effect=lambda *a, **k: io.BytesIO(b"too-long-data")):
            with self.assertRaisesRegex(RuntimeError, "Could not fetch"):
                bootstrap.fetch(self.item(), self.root)
        self.assertFalse((self.root / "sample.whl").exists())

    def test_fetch_rejects_filename_traversal(self):
        with self.assertRaisesRegex(RuntimeError, "filename"):
            bootstrap.fetch(dict(self.item(), filename="../sample.whl"), self.root)

    def test_work_rejects_afs_and_source_paths(self):
        for p in (self.root, self.root / "nested"):
            with self.assertRaisesRegex(RuntimeError, "AFS"):
                bootstrap.check_node_path(p, self.root)
        for p in (bootstrap.REPO, bootstrap.REPO.parent, Path(bootstrap.REPO.anchor),
                  bootstrap.REPO / "InfiniCore/build", bootstrap.REPO / "scripts/tmp"):
            with self.assertRaisesRegex(RuntimeError, "dedicated"):
                bootstrap.check_node_path(p)
        self.assertEqual(bootstrap.check_node_path(bootstrap.REPO / ".h3-reproduction"),
                         bootstrap.REPO / ".h3-reproduction")

    def test_work_resolves_symlink_before_afs_check(self):
        link = self.root / "link"
        target = self.root / "afs"
        target.mkdir()
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Host cannot create symlinks")
        with self.assertRaisesRegex(RuntimeError, "AFS"):
            bootstrap.check_node_path(link / "work", target)

    def group(self, key, child="kernel.hsaco", exists=True):
        directory = self.root / "jit" / key
        directory.mkdir(parents=True)
        if exists:
            (directory / child).write_bytes(b"kernel")
        group = directory / "__grp__kernel.json"
        group.write_text(json.dumps({"child_paths": {child: "/old-node/jit/" + child}}))
        return group

    def test_cache_relocation_preserves_kernel_bytes(self):
        group = self.group("a")
        bootstrap.rebase_cache(self.root)
        child = group.parent / "kernel.hsaco"
        self.assertEqual(json.loads(group.read_text())["child_paths"]["kernel.hsaco"], str(child))
        self.assertEqual(child.read_bytes(), b"kernel")

    def test_cache_missing_child_leaves_all_groups_unchanged(self):
        first = self.group("a")
        second = self.group("b", exists=False)
        before = {p: p.read_bytes() for p in (first, second)}
        with self.assertRaisesRegex(RuntimeError, "missing child"):
            bootstrap.rebase_cache(self.root)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_empty_cache_cannot_enter_eight_gpu_phase(self):
        with self.assertRaisesRegex(RuntimeError, "No JIT cache groups"):
            bootstrap.rebase_cache(self.root)

    def test_package_check_requires_bundled_flagtree_and_all_locked_versions(self):
        lock = {"wheels": [dict(name="Pillow", version="11.3.0")]}
        for versions in ({}, {"pillow": "11.2.0", "flagtree": "0.7.0+hcu3.6"},
                         {"pillow": "11.3.0"}):
            with patch.object(bootstrap, "installed_versions", return_value=versions):
                with self.assertRaisesRegex(RuntimeError, "lock mismatch"):
                    bootstrap.verify_packages(self.root, lock)
        with patch.object(bootstrap, "installed_versions",
                          return_value={"pillow": "11.3.0", "flagtree": "0.7.0+hcu3.6"}):
            bootstrap.verify_packages(self.root, lock)

    def test_config_key_changes_for_abi_toolchain_lua_but_not_cpp(self):
        repo = self.root / "repo"
        (repo / "InfiniCore/xmake").mkdir(parents=True)
        (repo / "scripts").mkdir()
        (repo / "scripts/build_hygon.sh").write_text("driver")
        lua = repo / "InfiniCore/xmake.lua"
        lua.write_text("config")
        cpp = repo / "InfiniCore/model.cpp"
        cpp.write_text("version1")
        xmake = self.root / "xmake"
        xmake.write_text("tool1")
        with patch.object(bootstrap, "REPO", repo), patch.object(bootstrap.shutil, "which", return_value=str(xmake)), \
                patch.dict(os.environ, {"H3_REPRO_WORK": str(self.root / "work"), "INFINILM_CXX11_ABI": "1"}):
            base = bootstrap.config_key("InfiniCore")
            cpp.write_text("version2")
            self.assertEqual(base, bootstrap.config_key("InfiniCore"))
            os.environ["INFINILM_CXX11_ABI"] = "0"
            self.assertNotEqual(base, bootstrap.config_key("InfiniCore"))
            os.environ["INFINILM_CXX11_ABI"] = "1"
            lua.write_text("new-config")
            self.assertNotEqual(base, bootstrap.config_key("InfiniCore"))
            lua.write_text("config")
            xmake.write_text("tool2")
            self.assertNotEqual(base, bootstrap.config_key("InfiniCore"))

    def test_phase_driver_rejects_unknown_phase_before_creating_work(self):
        if sys.platform != "win32":
            self.skipTest("Phase-driver checks require native Windows Git Bash")
        shell = os.environ.get("H3_TEST_GIT_BASH") or shutil.which("bash.exe")
        if not shell or "system32" in shell.lower():
            self.skipTest("Native Git Bash unavailable; never fall back to a Windows shell alias")
        work = self.root / "unused"
        env = dict(os.environ, H3_REPRO_WORK=work.as_posix(), H3_BASE_PYTHON=Path(sys.executable).as_posix())
        result = subprocess.run([shell, "--noprofile", "--norc",
                                 (ROOT / "scripts/reproduce_h3.sh").as_posix(), "invalid"],
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 64, result.stderr)
        self.assertFalse(work.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
