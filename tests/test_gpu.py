import os
import unittest
from unittest import mock

from vibiomir import gpu


class GpuMemoryFractionTest(unittest.TestCase):
    def test_unset_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(gpu.GPU_MEM_FRACTION_ENV, None)
            self.assertIsNone(gpu.gpu_memory_fraction())

    def test_parses_fraction(self):
        with mock.patch.dict(os.environ, {gpu.GPU_MEM_FRACTION_ENV: "0.4"}):
            self.assertAlmostEqual(gpu.gpu_memory_fraction(), 0.4)

    def test_rejects_out_of_range(self):
        for raw in ("0", "-0.2", "1.5", "abc"):
            with self.subTest(raw=raw), mock.patch.dict(
                os.environ, {gpu.GPU_MEM_FRACTION_ENV: raw}
            ):
                with self.assertRaises(ValueError):
                    gpu.gpu_memory_fraction()

    def test_apply_is_noop_without_cuda(self):
        # On CPU-only test machines this must return None and never raise.
        with mock.patch.dict(os.environ, {gpu.GPU_MEM_FRACTION_ENV: "0.4"}):
            try:
                import torch

                cuda_available = torch.cuda.is_available()
            except ImportError:
                cuda_available = False
            result = gpu.apply_gpu_memory_fraction()
            if not cuda_available:
                self.assertIsNone(result)

    def test_apply_calls_torch_when_cuda_present(self):
        fake_torch = mock.MagicMock()
        fake_torch.cuda.is_available.return_value = True
        gpu._applied_fraction = None
        with (
            mock.patch.dict(os.environ, {gpu.GPU_MEM_FRACTION_ENV: "0.4"}),
            mock.patch.dict("sys.modules", {"torch": fake_torch}),
        ):
            self.assertAlmostEqual(gpu.apply_gpu_memory_fraction(), 0.4)
            fake_torch.cuda.set_per_process_memory_fraction.assert_called_once_with(0.4)
            # A second call with the same fraction must not re-apply.
            gpu.apply_gpu_memory_fraction()
            fake_torch.cuda.set_per_process_memory_fraction.assert_called_once_with(0.4)
        gpu._applied_fraction = None


if __name__ == "__main__":
    unittest.main()
