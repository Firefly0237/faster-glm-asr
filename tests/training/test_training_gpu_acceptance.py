from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    os.environ.get("RUN_4X3090_ACCEPTANCE") == "1",
    "requires an explicit RUN_4X3090_ACCEPTANCE=1 launch on the rented 4-GPU host",
)
class FourGpuAcceptanceTest(unittest.TestCase):
    def test_nccl_bf16_inventory_and_collective_preflight(self) -> None:
        output = (
            REPOSITORY_ROOT
            / "artifacts"
            / "private"
            / "training"
            / "acceptance"
            / f"preflight-{uuid.uuid4().hex}.json"
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "-m",
            "experiments.distributed_training.preflight",
            "--environment-candidate",
            str(REPOSITORY_ROOT / "configs" / "environments" / "rtx3090-ddp-v1.json"),
            "--output",
            str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["pass"])
        self.assertEqual(report["world_size"], 4)
        self.assertEqual(len(report["rank_inventory"]), 4)
        self.assertEqual(len({item["local_rank"] for item in report["rank_inventory"]}), 4)


if __name__ == "__main__":
    unittest.main()
