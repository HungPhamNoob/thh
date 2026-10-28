import os
import unittest
from unittest import mock

from vibiomir import gpu


class GpuMemoryLimitTest(unittest.TestCase):
    def tearDown(self):
        gpu._applied = None

    def test_default_absolute_limit_is_below_operational_target(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(gpu.GPU_MAX_GIB_ENV, None)
            self.assertEqual(gpu.gpu_max_gib(), 3.0)

    def test_rejects_limit_above_four_gib(self):
        for raw in ("0", "-1", "4.1", "abc"):
            with self.subTest(raw=raw), mock.patch.dict(
                os.environ, {gpu.GPU_MAX_GIB_ENV: raw}
            ):
                with self.assertRaises(ValueError):
                    gpu.gpu_max_gib()

    def test_absolute_limit_is_derived_from_total_vram(self):
        fake_torch = mock.MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.get_device_properties.return_value.total_memory = 8 * 1024**3
        fake_torch.cuda.mem_get_info.return_value = (7 * 1024**3, 8 * 1024**3)
        fake_torch.cuda.get_device_name.return_value = "test-gpu"
        fake_torch.cuda.max_memory_allocated.return_value = 0
        fake_torch.cuda.max_memory_reserved.return_value = 0
        with (
            mock.patch.dict(os.environ, {gpu.GPU_MAX_GIB_ENV: "3.0"}),
            mock.patch.dict("sys.modules", {"torch": fake_torch}),
        ):
            report = gpu.apply_gpu_memory_limit()
        fake_torch.cuda.set_per_process_memory_fraction.assert_called_once_with(0.375, device=0)
        self.assertEqual(report["allocator_limit_gib"], 3.0)

    def test_cuda_oom_detection_is_specific(self):
        self.assertTrue(gpu.is_cuda_oom(RuntimeError("CUDA out of memory")))
        self.assertFalse(gpu.is_cuda_oom(RuntimeError("host out of memory")))


if __name__ == "__main__":
    unittest.main()
