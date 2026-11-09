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
sys.path.insert(0, str(ROOT / "scripts"))
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

    def test_verified_uses_presence_and_size_without_reading_contents(self):
        p = self.root / "sample.whl"
        item = self.item()
        for data in (b"short", b""):
            p.write_bytes(data)
            self.assertFalse(bootstrap.verified(p, item))
        for data in (b"wheel-data", b"wheel-datX"):
            p.write_bytes(data)
            with patch.object(Path, "open", side_effect=AssertionError("content read")):
                self.assertTrue(bootstrap.verified(p, item))

    def test_fetch_reuses_verified_download_without_network(self):
        p = self.root / "sample.whl"
        p.write_bytes(b"wheel-data")
        with patch.object(bootstrap, "urlopen", side_effect=AssertionError("network used")):
            self.assertEqual(bootstrap.fetch(self.item(), self.root), p)

    def test_fetch_retries_truncation_before_atomic_promotion(self):
        with patch.object(bootstrap, "urlopen", side_effect=[
                io.BytesIO(b"short"), io.BytesIO(b"wheel-data")]) as open_url:
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

    def test_prepare_installs_without_hashes_and_migrates_old_receipt_in_place(self):
        import tarfile
        repo = self.root / "repo"
        tool = repo / "third_party/h3"
        tool.mkdir(parents=True)
        (repo / "requirements").mkdir()
        (repo / "scripts").mkdir()
        (repo / "scripts/h3_clang.sh").write_text("wrapper")
        wheel = tool / "flagtree.whl"
        wheel.write_bytes(b"wheel")
        archive = tool / "hcu-jit-clang18-minimal.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for name, data in (("clang-18", b"compiler"), ("lld", b"linker")):
                info = tarfile.TarInfo("bin/" + name)
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
        lock = repo / "requirements/h3_linux_cp311.lock.json"
        lock.write_text(json.dumps(dict(wheels=[])))
        assets = tool / "manifest.json"
        manifest = dict(files=[dict(path="third_party/h3/flagtree.whl", size=5)],
                        jit_toolchain=dict(binary_sizes={"clang-18": 8, "lld": 6}))
        assets.write_text(json.dumps(manifest))
        work = self.root / "work"
        calls = []
        def install(argv, **kwargs):
            calls.append(argv)
            if "--without-pip" in argv:
                python = Path(argv[-1]) / "bin/python3"
                python.parent.mkdir(parents=True)
                python.write_bytes(b"python")
        info = dict(python="3.11", torch="2.9.0", hip="6.3", abi=1)
        with patch.object(bootstrap, "REPO", repo), patch.object(bootstrap, "LOCK", lock), \
                patch.object(bootstrap, "ASSETS", assets), \
                patch.object(bootstrap, "platform_info", return_value=info), \
                patch.object(bootstrap, "verify_assets", return_value=manifest), \
                patch.object(bootstrap, "prepare_xmake", return_value=("xmake v3.0.3", "official_bundle")), \
                patch.object(bootstrap, "verify_packages"), patch.object(bootstrap, "run", side_effect=install), \
                patch.object(hashlib, "sha256", side_effect=AssertionError("SHA256 called")), \
                patch.dict(os.environ, {"IFAB_AFS_HOME": str(self.root / "afs")}):
            bootstrap.prepare(work)
            self.assertTrue((work / "python/.h3-lock").is_file())
            self.assertNotIn("--hash=", (work / "local-wheels.lock.txt").read_text())
            self.assertFalse(any("--require-hashes" in argv for argv in calls))
            receipt = json.loads((work / "prepared.json").read_text())
            receipt["key"] = "old-sha-key"
            receipt.pop("integrity")
            (work / "prepared.json").write_text(json.dumps(receipt))
            (work / "python/.h3-lock").write_text("old-sha-key")
            calls.clear()
            bootstrap.prepare(work)
            self.assertEqual(calls, [])
            migrated = json.loads((work / "prepared.json").read_text())
            self.assertEqual(migrated["integrity"], "presence_and_size")

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
            os.utime(xmake, ns=(1_000_000_000, 1_000_000_000))
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



class XmakeBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.bundle = self.repo / "tool"
        self.bundle.write_bytes(b"locked-tool")
        self.work = self.root / "work"
        self.item = dict(path="tool", size=11, sha256="historical-record-ignored")
        self.manifest = dict(xmake=dict(bundle_path="tool"), files=[self.item])

    def test_valid_bundle_replaces_source_build_and_is_reused(self):
        with patch.object(bootstrap, "REPO", self.repo), \
                patch.object(bootstrap.subprocess, "check_output", return_value="xmake v3.0.3"), \
                patch.object(bootstrap.subprocess, "run") as invoke:
            _, kind = bootstrap.prepare_xmake(self.work, self.manifest)
            self.assertEqual(kind, "official_bundle")
            self.assertEqual(invoke.call_count, 1)
            self.assertIn("show", invoke.call_args.args[0])
            before = (self.work / "xmake/bin/xmake").stat().st_mtime_ns
            bootstrap.prepare_xmake(self.work, self.manifest)
            self.assertEqual((self.work / "xmake/bin/xmake").stat().st_mtime_ns, before)

    def test_corrupt_bundle_never_runs(self):
        self.bundle.write_bytes(b"truncated")
        with patch.object(bootstrap, "REPO", self.repo), \
                patch.object(bootstrap.subprocess, "run") as invoke:
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                bootstrap.prepare_xmake(self.work, self.manifest)
            invoke.assert_not_called()

    def test_legacy_xmake_receipt_migrates_without_reinstall(self):
        executable = self.work / "xmake/bin/xmake"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"existing-tool")
        receipt = self.work / "xmake/tool.json"
        receipt.write_text(json.dumps(dict(kind="official_bundle", executable_sha256="old")))
        before = executable.stat().st_mtime_ns
        with patch.object(bootstrap.subprocess, "check_output", return_value="xmake v3.0.3"), \
                patch.object(bootstrap, "run", side_effect=AssertionError("reinstall")):
            bootstrap.prepare_xmake(self.work, self.manifest)
        self.assertEqual(executable.stat().st_mtime_ns, before)
        self.assertEqual(json.loads(receipt.read_text())["executable_bytes"], 13)

    def test_incompatible_bundle_falls_back_to_source_build(self):
        def source_command(argv, **kwargs):
            if argv[0] == "make" and "install" in argv:
                exe = self.work / "xmake/bin/xmake"
                exe.parent.mkdir(parents=True, exist_ok=True)
                exe.write_bytes(b"source-built")
        with patch.object(bootstrap, "REPO", self.repo), \
                patch.object(bootstrap.subprocess, "check_output",
                    side_effect=[OSError("incompatible loader"), "xmake v3.0.3"]), \
                patch.object(bootstrap, "run", side_effect=source_command) as invoke:
            _, kind = bootstrap.prepare_xmake(self.work, self.manifest)
            self.assertEqual(kind, "source")
            self.assertTrue(any(call.args[0][:2] == ["make", "-j120"] for call in invoke.call_args_list))


if __name__ == "__main__":
    unittest.main(verbosity=2)
