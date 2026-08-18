from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from experiments.distributed_training import preflight

GPU_UUIDS = [f"00000000-0000-4000-8000-{index:012x}" for index in range(4)]


def _inventory_capture() -> dict[str, object]:
    return {
        "command": preflight.NVIDIA_SMI_INVENTORY_COMMAND,
        "returncode": 0,
        "stdout": "".join(
            f"{index}, GPU-{GPU_UUIDS[index]}, 00000000:{index + 1:02X}:00.0, "
            "NVIDIA GeForce RTX 3090, 24576, 570.26.0\n"
            for index in range(4)
        ),
        "stderr": "",
    }


class DecoderTrainingPreflightIdentityTest(unittest.TestCase):
    def test_torch_cuuid_exact_type_is_accepted_but_strable_objects_are_not(self) -> None:
        class FakeCUuuid:
            def __init__(self, value: str) -> None:
                self.value = value

            def __str__(self) -> str:
                return self.value

        class FakeCUuuidSubclass(FakeCUuuid):
            pass

        class ArbitraryStrable:
            def __str__(self) -> str:
                return GPU_UUIDS[0]

        with mock.patch.object(preflight.torch._C, "_CUuuid", FakeCUuuid, create=True):
            self.assertEqual(preflight._canonical_gpu_uuid(FakeCUuuid(GPU_UUIDS[0])), GPU_UUIDS[0])
            self.assertIsNone(preflight._canonical_gpu_uuid(FakeCUuuidSubclass(GPU_UUIDS[0])))
            self.assertIsNone(preflight._canonical_gpu_uuid(ArbitraryStrable()))
            self.assertIsNone(preflight._canonical_gpu_uuid(FakeCUuuid("not-a-uuid")))
        self.assertIsNone(preflight._canonical_gpu_uuid("GPU-not-a-uuid"))
        self.assertIsNone(preflight._canonical_gpu_uuid(f" {GPU_UUIDS[0]}"))

    def test_missing_torch_pci_properties_uses_uuid_bound_nvidia_bdf(self) -> None:
        rows = preflight._parse_nvidia_smi_inventory(_inventory_capture())
        # These fixtures intentionally model the target Torch properties object:
        # UUID is available, but pci_domain_id/pci_bus_id/pci_device_id are absent.
        visible_properties = [
            SimpleNamespace(uuid=f"GPU-{GPU_UUIDS[physical_index]}")
            for physical_index in (2, 0, 3, 1)
        ]
        bindings = [
            preflight._resolve_rank_gpu_binding(
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
                (f"GPU-{GPU_UUIDS[2]}", "00000000:03:00.0"),
                (f"GPU-{GPU_UUIDS[0]}", "00000000:01:00.0"),
                (f"GPU-{GPU_UUIDS[3]}", "00000000:04:00.0"),
                (f"GPU-{GPU_UUIDS[1]}", "00000000:02:00.0"),
            ],
        )
        self.assertEqual([item["nvidia_smi_index"] for item in bindings], [2, 0, 3, 1])
        self.assertEqual(len({item["uuid"] for item in bindings}), 4)
        self.assertEqual(len({item["pci_bus_id"] for item in bindings}), 4)

    def test_selected_uuid_must_match_cuda_logical_device_index(self) -> None:
        rows = preflight._parse_nvidia_smi_inventory(_inventory_capture())
        visible_properties = [
            SimpleNamespace(uuid=f"GPU-{GPU_UUIDS[physical_index]}")
            for physical_index in (2, 0, 3, 1)
        ]

        with self.assertRaisesRegex(ValueError, "does not match its logical device index"):
            preflight._resolve_rank_gpu_binding(
                local_rank=0,
                device_index=0,
                selected_properties=SimpleNamespace(uuid=f"GPU-{GPU_UUIDS[0]}"),
                visible_properties=visible_properties,
                inventory_rows=rows,
                expected_world_size=4,
            )

    def test_cuda_and_nvidia_uuid_sets_must_match_exactly(self) -> None:
        rows = preflight._parse_nvidia_smi_inventory(_inventory_capture())
        other_uuid = "ffffffff-ffff-4fff-8fff-ffffffffffff"
        visible_properties = [
            SimpleNamespace(uuid=value)
            for value in (
                f"GPU-{GPU_UUIDS[0]}",
                f"GPU-{GPU_UUIDS[1]}",
                f"GPU-{GPU_UUIDS[2]}",
                f"GPU-{other_uuid}",
            )
        ]

        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            preflight._resolve_rank_gpu_binding(
                local_rank=0,
                device_index=0,
                selected_properties=visible_properties[0],
                visible_properties=visible_properties,
                inventory_rows=rows,
                expected_world_size=4,
            )

    def test_noncanonical_physical_index_set_is_rejected(self) -> None:
        rows = preflight._parse_nvidia_smi_inventory(_inventory_capture())
        rows[3]["index"] = 7
        visible_properties = [SimpleNamespace(uuid=f"GPU-{GPU_UUIDS[index]}") for index in range(4)]

        with self.assertRaisesRegex(ValueError, "indexes are not the expected contiguous set"):
            preflight._resolve_rank_gpu_binding(
                local_rank=0,
                device_index=0,
                selected_properties=visible_properties[0],
                visible_properties=visible_properties,
                inventory_rows=rows,
                expected_world_size=4,
            )

    def test_duplicate_physical_uuid_is_rejected(self) -> None:
        capture = _inventory_capture()
        capture["stdout"] = str(capture["stdout"]).replace(
            f"GPU-{GPU_UUIDS[1]}", f"gpu-{GPU_UUIDS[0].upper()}", 1
        )

        with self.assertRaisesRegex(ValueError, "duplicate canonical_uuid"):
            preflight._parse_nvidia_smi_inventory(capture)


if __name__ == "__main__":
    unittest.main()
