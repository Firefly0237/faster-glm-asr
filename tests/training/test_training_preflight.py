from __future__ import annotations

import unittest
from types import SimpleNamespace

from experiments.distributed_training.preflight import (
    NVIDIA_SMI_INVENTORY_COMMAND,
    _parse_nvidia_smi_inventory,
    _resolve_rank_gpu_binding,
)


def _inventory_capture() -> dict[str, object]:
    return {
        "command": NVIDIA_SMI_INVENTORY_COMMAND,
        "returncode": 0,
        "stdout": "".join(
            f"{index}, GPU-physical-{index}, 00000000:{index + 1:02X}:00.0, "
            "NVIDIA GeForce RTX 3090, 24576, 570.26.0\n"
            for index in range(4)
        ),
        "stderr": "",
    }


class DecoderTrainingPreflightIdentityTest(unittest.TestCase):
    def test_missing_torch_pci_properties_uses_uuid_bound_nvidia_bdf(self) -> None:
        rows = _parse_nvidia_smi_inventory(_inventory_capture())
        # These fixtures intentionally model the target Torch properties object:
        # UUID is available, but pci_domain_id/pci_bus_id/pci_device_id are absent.
        visible_properties = [
            SimpleNamespace(uuid=f"GPU-physical-{physical_index}")
            for physical_index in (2, 0, 3, 1)
        ]
        bindings = [
            _resolve_rank_gpu_binding(
                local_rank=local_rank,
                device_index=local_rank,
                selected_properties=visible_properties[local_rank],
                visible_properties=visible_properties,
                inventory_rows=rows,
                expected_world_size=4,
            )
            for local_rank in range(4)
        ]

        self.assertEqual(
            [(item["uuid"], item["pci_bus_id"]) for item in bindings],
            [
                ("GPU-physical-2", "00000000:03:00.0"),
                ("GPU-physical-0", "00000000:01:00.0"),
                ("GPU-physical-3", "00000000:04:00.0"),
                ("GPU-physical-1", "00000000:02:00.0"),
            ],
        )
        self.assertEqual([item["nvidia_smi_index"] for item in bindings], [2, 0, 3, 1])
        self.assertEqual(len({item["uuid"] for item in bindings}), 4)
        self.assertEqual(len({item["pci_bus_id"] for item in bindings}), 4)

    def test_selected_uuid_must_match_cuda_logical_device_index(self) -> None:
        rows = _parse_nvidia_smi_inventory(_inventory_capture())
        visible_properties = [
            SimpleNamespace(uuid=f"GPU-physical-{physical_index}")
            for physical_index in (2, 0, 3, 1)
        ]

        with self.assertRaisesRegex(ValueError, "does not match its logical device index"):
            _resolve_rank_gpu_binding(
                local_rank=0,
                device_index=0,
                selected_properties=SimpleNamespace(uuid="GPU-physical-0"),
                visible_properties=visible_properties,
                inventory_rows=rows,
                expected_world_size=4,
            )

    def test_cuda_and_nvidia_uuid_sets_must_match_exactly(self) -> None:
        rows = _parse_nvidia_smi_inventory(_inventory_capture())
        visible_properties = [
            SimpleNamespace(uuid=value)
            for value in ("GPU-physical-0", "GPU-physical-1", "GPU-physical-2", "GPU-other")
        ]

        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            _resolve_rank_gpu_binding(
                local_rank=0,
                device_index=0,
                selected_properties=visible_properties[0],
                visible_properties=visible_properties,
                inventory_rows=rows,
                expected_world_size=4,
            )

    def test_noncanonical_physical_index_set_is_rejected(self) -> None:
        rows = _parse_nvidia_smi_inventory(_inventory_capture())
        rows[3]["index"] = 7
        visible_properties = [SimpleNamespace(uuid=f"GPU-physical-{index}") for index in range(4)]

        with self.assertRaisesRegex(ValueError, "indexes are not the expected contiguous set"):
            _resolve_rank_gpu_binding(
                local_rank=0,
                device_index=0,
                selected_properties=visible_properties[0],
                visible_properties=visible_properties,
                inventory_rows=rows,
                expected_world_size=4,
            )

    def test_duplicate_physical_uuid_is_rejected(self) -> None:
        capture = _inventory_capture()
        capture["stdout"] = str(capture["stdout"]).replace("GPU-physical-1", "gpu-PHYSICAL-0", 1)

        with self.assertRaisesRegex(ValueError, "duplicate canonical_uuid"):
            _parse_nvidia_smi_inventory(capture)


if __name__ == "__main__":
    unittest.main()
