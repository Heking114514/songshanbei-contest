"""CPU checks for uploading source plus offline wheels, without a package."""
import contextlib
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
import h3_source_stage as stage
from h3_delivery import file_row
from h3_source_identity import digest, source_identity


class SourceStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3-source-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.afs = self.root / "afs"
        self.repo = self.afs / "songshanbei-contest"
        self.wheels = self.repo / "offline-wheels"
        self.node = self.root / "node"
        for name in stage.REQUIRED:
            self.put(name, ("source:" + name).encode())
        (self.repo / "ninetoothed/src").mkdir(parents=True)
        self.put("InfiniCore/operator.cpp", b"baseline")
        self.put("third_party/h3/tool.bin", b"tool")
        tool = file_row(self.repo / "third_party/h3/tool.bin", "third_party/h3/tool.bin")
        self.put("third_party/h3/manifest.json", json.dumps({"files": [tool]}).encode())
        self.wheels.mkdir()
        wheel = self.wheels / "a.whl"
        wheel.write_bytes(b"wheel")
        row = dict(filename="a.whl", size=5, sha256=digest(wheel))
        self.put("requirements/h3_linux_cp311.lock.json", json.dumps({"wheels": [row]}).encode())
        self.env = patch.dict(os.environ, {"IFAB_AFS_HOME": str(self.afs)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def put(self, name, data):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def run_stage(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return stage.stage(self.repo, self.wheels, self.node)

    def test_zip_source_without_git_or_package_is_staged_and_identified(self):
        self.run_stage()
        self.assertEqual((self.node / "source/InfiniCore/operator.cpp").read_bytes(), b"baseline")
        identity = source_identity(self.node / "source")
        self.assertEqual(identity["kind"], "source_directory_snapshot")
        self.assertIsNone(identity["git_head"])
        self.assertFalse((self.node / "source/offline-wheels").exists())
        self.assertFalse((self.repo / ".h3-source.json").exists())
        self.assertEqual(json.loads((self.node / "source-stage.json").read_text())["state"], "ready")

    def test_no_changes_preserve_source_and_object_mtimes(self):
        self.run_stage()
        cpp = self.node / "source/InfiniCore/operator.cpp"
        obj = self.node / "source/InfiniCore/build/.objs/keep.o"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b"already-built")
        before = {p: (digest(p), p.stat().st_mtime_ns) for p in (cpp, obj)}
        self.run_stage()
        self.assertEqual(before, {p: (digest(p), p.stat().st_mtime_ns) for p in before})
        self.assertEqual(json.loads((self.node / "source-stage.json").read_text())["changed_files"], 0)

    def test_changed_source_gets_new_mtime_even_when_uploaded_timestamp_is_old(self):
        self.run_stage()
        cpp = self.node / "source/InfiniCore/operator.cpp"
        untouched = self.node / "source/InfiniLM/xmake.lua"
        os.utime(cpp, (946684800, 946684800))
        before = untouched.stat().st_mtime_ns
        origin = self.put("InfiniCore/operator.cpp", b"optimized")
        os.utime(origin, (946684800, 946684800))
        self.run_stage()
        self.assertEqual(cpp.read_bytes(), b"optimized")
        self.assertGreater(cpp.stat().st_mtime, 946684800)
        self.assertEqual(untouched.stat().st_mtime_ns, before)

    def test_missing_or_corrupt_wheel_fails_before_node_creation(self):
        wheel = self.wheels / "a.whl"
        for data in (b"wrong", None):
            with self.subTest(data=data):
                if data is None:
                    wheel.unlink()
                else:
                    wheel.write_bytes(data)
                with self.assertRaisesRegex(RuntimeError, "Missing/corrupt"):
                    self.run_stage()
                self.assertFalse(self.node.exists())

    def test_build_outputs_uploaded_by_accident_are_excluded(self):
        self.put("InfiniCore/build/old.o", b"old")
        self.put("InfiniLM/python/infinilm/_infinilm.so", b"old-lib")
        self.put("scripts/__pycache__/old.pyc", b"cache")
        self.run_stage()
        for name in ("InfiniCore/build/old.o", "InfiniLM/python/infinilm/_infinilm.so",
                     "scripts/__pycache__/old.pyc"):
            self.assertFalse((self.node / "source" / name).exists(), name)

    def test_independent_node_edits_are_preserved_and_rejected(self):
        self.run_stage()
        cpp = self.node / "source/InfiniCore/operator.cpp"
        cpp.write_bytes(b"independent experiment")
        self.put("InfiniCore/operator.cpp", b"new upload")
        before = (self.node / "source-stage.json").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "independent edits"):
            self.run_stage()
        self.assertEqual(cpp.read_bytes(), b"independent experiment")
        self.assertEqual((self.node / "source-stage.json").read_bytes(), before)

    def test_removing_source_does_not_remove_build_outputs(self):
        self.run_stage()
        obj = self.node / "source/InfiniCore/build/keep.o"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b"keep")
        (self.repo / "InfiniCore/operator.cpp").unlink()
        self.run_stage()
        self.assertFalse((self.node / "source/InfiniCore/operator.cpp").exists())
        self.assertEqual(obj.read_bytes(), b"keep")

    def test_partial_source_copy_can_resume_without_replacing_unchanged_files(self):
        original = stage.shutil.copyfile
        def fail_once(origin, target):
            if str(origin).endswith("operator.cpp"):
                raise OSError("interrupted copy")
            return original(origin, target)
        with patch.object(stage.shutil, "copyfile", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.run_stage()
        self.run_stage()
        self.assertEqual((self.node / "source/InfiniCore/operator.cpp").read_bytes(), b"baseline")

    def test_afs_workspace_and_unowned_node_directories_are_rejected(self):
        for node in (self.afs / "node", self.root / "workspace/node", self.root):
            with self.subTest(node=node):
                with self.assertRaises(RuntimeError):
                    with contextlib.redirect_stderr(io.StringIO()):
                        stage.stage(self.repo, self.wheels, node)
        self.node.mkdir()
        (self.node / "unrelated").write_bytes(b"keep")
        with self.assertRaisesRegex(RuntimeError, "no source-stage receipt"):
            self.run_stage()
        self.assertEqual((self.node / "unrelated").read_bytes(), b"keep")

    def test_symlinked_source_is_rejected_before_copy(self):
        link = self.repo / "InfiniCore/escape.cpp"
        outside = self.root / "outside"
        outside.write_bytes(b"private")
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("Host cannot create symlinks")
        with self.assertRaisesRegex(RuntimeError, "Unsafe source"):
            self.run_stage()
        self.assertFalse(self.node.exists())

    def install_launcher(self):
        for name in ("run_submission.sh", "scripts/h3_source_stage.py", "scripts/h3_bootstrap.py",
                     "scripts/h3_delivery.py", "scripts/h3_source_identity.py"):
            shutil.copyfile(ROOT / name, self.repo / name)
        self.put("scripts/h3_submission_common.sh", b"""#!/usr/bin/env bash
set -eu
"$H3_BASE_PYTHON" - "$1" <<'WORKER_PY'
import json, os, sys
print("WORKER_ENV", json.dumps(dict(mode=sys.argv[1], node=os.environ["H3_NODE_ROOT"],
    output=os.environ["H3_AFS_OUTPUT_DIR"], wheels=os.environ["H3_WHEELHOUSE"],
    model=os.environ["MODEL_PATH"], jobs=os.environ["BUILD_JOBS"],
    snapshot=os.environ.get("H3_AFS_INFO_JSON"))))
WORKER_PY
""")

    def launch(self, cwd, args=(), extra=None):
        shell = os.environ.get("H3_TEST_GIT_BASH")
        if not shell or "system32" in shell.lower():
            self.skipTest("Explicit native Git Bash required")
        env = {k: v for k, v in os.environ.items() if not k.startswith("H3_")}
        for name in ("MODEL_PATH", "BUILD_JOBS", "OLDPWD"):
            env.pop(name, None)
        env.update(IFAB_AFS_HOME=self.afs.as_posix(),
                   H3_BASE_PYTHON=Path(sys.executable).as_posix())
        env.update(extra or {})
        return subprocess.run([shell, "--noprofile", "--norc",
            (self.repo / "run_submission.sh").as_posix(), *args],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=30)

    def worker_env(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(next(line.removeprefix("WORKER_ENV ") for line in result.stdout.splitlines()
                               if line.startswith("WORKER_ENV ")))

    def test_root_entry_runs_from_source_and_forwards_mode_and_wheelhouse(self):
        self.install_launcher()
        for mode, args in (("native", ()), ("check", ("check",))):
            with self.subTest(mode=mode):
                result = self.launch(self.repo, args, dict(H3_NODE_ROOT=self.node.as_posix()))
                data = self.worker_env(result)
                self.assertEqual(data["mode"], mode)
                self.assertEqual(Path(data["wheels"]), self.wheels)
                self.assertFalse((self.repo / "delivery-manifest.json").exists())
                self.assertIsNone(data["snapshot"])

    def test_copy_paste_launch_from_fresh_workspace_needs_no_manual_paths_or_json(self):
        self.install_launcher()
        workspace = self.root / "session/workspace"
        workspace.mkdir(parents=True)
        for args in ((), ("check",)):
            result = self.launch(workspace, args)
            data = self.worker_env(result)
            node = Path(data["node"])
            self.assertEqual(node.parent, workspace.parent)
            self.assertTrue(node.name.startswith("h3-native-"))
            self.assertTrue(Path(data["output"]).is_relative_to(self.afs / "h3-results"))
            self.assertIsNone(data["snapshot"])
            self.assertEqual(data["jobs"], "120")
            # Git Bash translates absolute Linux env paths for the Windows Python stub.
            self.assertTrue(data["model"].replace("\\\\", "/").endswith(
                "/public/home/aclsylqidf/shared/MiniMax-H3"))

    def test_repository_directory_launch_uses_previous_node_workspace(self):
        self.install_launcher()
        workspace = self.root / "session/workspace"
        workspace.mkdir(parents=True)
        data = self.worker_env(self.launch(self.repo, extra=dict(OLDPWD=workspace.as_posix())))
        self.assertEqual(Path(data["node"]).parent, workspace.parent)

    def test_automatic_node_directory_is_stable_and_preserves_cached_object(self):
        self.install_launcher()
        workspace = self.root / "session/workspace"
        workspace.mkdir(parents=True)
        first = self.worker_env(self.launch(workspace))
        node = Path(first["node"])
        obj = node / "source/InfiniCore/build/keep.o"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b"incremental object fixture")
        before = (digest(obj), obj.stat().st_mtime_ns)
        second = self.worker_env(self.launch(workspace))
        self.assertEqual(first["node"], second["node"])
        self.assertNotEqual(first["output"], second["output"])
        self.assertEqual(before, (digest(obj), obj.stat().st_mtime_ns))
        receipt = json.loads((node / "source-stage.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["changed_files"], 0)

    def test_missing_node_context_reports_the_initial_terminal_command(self):
        self.install_launcher()
        result = self.launch(self.repo)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NEW eight-GPU terminal", result.stderr)
        self.assertFalse((self.afs / "h3-native").exists())

    def test_auto_node_selection_does_not_use_workspace_inside_afs(self):
        workspace = self.afs / "workspace"
        workspace.mkdir()
        with patch.dict(os.environ, dict(OLDPWD=str(workspace))), \
                patch.object(stage.Path, "cwd", return_value=self.repo):
            with self.assertRaisesRegex(RuntimeError, "Cannot locate node workspace"):
                stage.node_directory(self.repo)


if __name__ == "__main__":
    unittest.main()
