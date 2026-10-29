from __future__ import annotations

import unittest

from vibiomir.models import repair_1d_position_id_buffers


class _FakeTensor:
    def __init__(self, values: list[int], ndim: int = 1):
        self.values = values
        self.ndim = ndim
        self.device = "cuda:0"
        self.dtype = "int64"

    def numel(self) -> int:
        return len(self.values)


class _FakeModule:
    def __init__(self, position_ids: _FakeTensor):
        self.position_ids = position_ids
        self.persistent: bool | None = None

    def register_buffer(self, name: str, value: _FakeTensor, persistent: bool) -> None:
        setattr(self, name, value)
        self.persistent = persistent


class _FakeModel:
    def __init__(self, children: list[_FakeModule]):
        self.children = children

    def modules(self):
        return iter(self.children)


class _FakeTorch:
    @staticmethod
    def arange(size: int, *, device: str, dtype: str) -> _FakeTensor:
        assert device == "cuda:0"
        assert dtype == "int64"
        return _FakeTensor(list(range(size)))


class ModelBufferTests(unittest.TestCase):
    def test_repair_recreates_only_one_dimensional_position_ids(self) -> None:
        broken = _FakeModule(_FakeTensor([99, -1, 42]))
        standard = _FakeModule(_FakeTensor([99, -1, 42], ndim=2))

        repaired = repair_1d_position_id_buffers(
            _FakeModel([broken, standard]), torch_module=_FakeTorch
        )

        self.assertEqual(repaired, 1)
        self.assertEqual(broken.position_ids.values, [0, 1, 2])
        self.assertFalse(broken.persistent)
        self.assertEqual(standard.position_ids.values, [99, -1, 42])
        self.assertIsNone(standard.persistent)


if __name__ == "__main__":
    unittest.main()
