import io
import os
import subprocess
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from warpdiff.__main__ import make_archive, rent, verify_saved_pair
from warpdiff.benchmark import git_commit, select
from warpdiff.report import combine_reports
from warpdiff.vast import APIError
from warpdiff.worker import revision_module


class BenchmarkTest(unittest.TestCase):
    def test_base_definition_and_revision_local_helpers(self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            original = Path.cwd()
            os.chdir(path)
            try:
                self.git("init", "-q")
                Path("kernels").mkdir()
                Path("kernels/operation.py").write_text("from .helper import value\n")
                helper = Path("kernels/helper.py")
                helper.write_text("value = 12\n")
                Path("bench.py").write_text("def prepare(case):\n    return case['size']\n")
                config = Path("warpdiff.toml")
                config.write_text('''
default = "row"
[benchmarks.row]
kernel = "kernels/operation.py"
files = ["kernels/helper.py"]
adapter = "bench.py"
gpus = ["RTX 3090", "RTX 4090"]
margin_pct = 4
cases = [{id = "scalar", label = "One scalar", size = 1}]
''')
                original_adapter = Path("bench.py").read_bytes()
                self.commit("base")
                base = git_commit("HEAD")
                config.write_text(config.read_text().replace("margin_pct = 4", "margin_pct = 90"))
                Path("bench.py").write_text("raise RuntimeError('candidate adapter')\n")
                helper.write_text("value = 20\n")
                self.commit("candidate")
                head = git_commit("HEAD")

                benchmark = select(base, "", None)
                self.assertEqual(benchmark.margin_pct, 4)
                self.assertEqual(benchmark.adapter, original_adapter)
                self.assertEqual(benchmark.job["cases"][0]["size"], 1)
                before, base_files = benchmark.sources(base)
                after, head_files = benchmark.sources(head)
                self.assertEqual(before["kernel_sha256"], after["kernel_sha256"])
                self.assertNotEqual(before["sources_sha256"], after["sources_sha256"])
                archive = make_archive(benchmark, base_files, head_files)
                destination = directory / "worker"
                with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
                    bundle.extractall(destination, filter="data")
                self.assertEqual(revision_module(destination, benchmark.kernel, "base").value, 12)
                self.assertEqual(revision_module(destination, benchmark.kernel, "head").value, 20)
            finally:
                os.chdir(original)

    @staticmethod
    def git(*args):
        subprocess.run(["git", *args], check=True, stdout=subprocess.DEVNULL)

    def commit(self, message):
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", message)

    def test_combine_respects_the_selected_benchmark(self):
        benchmark = {"id": "row", "gpus": ["RTX 3090", "RTX 4090"], "definition_sha256": "definition"}
        fixture = {
            "schema_version": 1, "provenance": "vast", "benchmark": benchmark,
            "kernel": "kernels/operation.py", "margin_pct": 4, "image": "image",
            "base": {"sha": "base", "sources_sha256": "original"},
            "head": {"sha": "head", "kernel_sha256": "kernel", "sources_sha256": "patched"},
            "started_at": "2026-11-27T17:26:48+00:00", "workflow_url": None,
        }
        measured = {"cleanup": "verified", "status": "improvement", "cases": [{"case_id": "scalar"}]}
        earlier = {**fixture, "run_id": "first", "cards": [{**measured, "requested_gpu": "RTX 4090"}]}
        current = {**fixture, "run_id": "retry", "cards": [{**measured, "requested_gpu": "RTX 3090"}]}
        verify_saved_pair(earlier, current)
        combined = combine_reports(earlier, current)
        self.assertEqual([card["requested_gpu"] for card in combined["cards"]], benchmark["gpus"])
        self.assertEqual([card["source_run_id"] for card in combined["cards"]], ["retry", "first"])
        changed = {**current, "benchmark": {**benchmark, "definition_sha256": "changed"}}
        with self.assertRaisesRegex(ValueError, "same benchmark"):
            verify_saved_pair(earlier, changed)

    def test_refresh_offers_after_the_provider_returns_a_wrapped_404(self):
        first = {"id": 17, "dph_total": 0.12}
        second = {"id": 18, "dph_total": 0.15}
        vast = Mock()
        vast.create.side_effect = [APIError(
            "PUT", "/api/v0/asks/17/", 400, "error 404/3603: no_such_ask Instance type by id 17 is not available.",
            "invalid_args",
        ), 20]
        vast.offers.return_value = [first, second]
        args = SimpleNamespace(startup_seconds=300, run_seconds=180, max_hourly=0.7,
                               max_transfer_rate=0.02, exclude_machine=[])
        rented = rent(vast, args, "RTX 3090", [first], "warpdiff-test", threading.Event())
        self.assertEqual(rented, (20, second))
        self.assertEqual([call.args[0] for call in vast.create.call_args_list], [17, 18])
