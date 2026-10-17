from __future__ import annotations

import gzip
import hashlib
import pickle

import numpy as np
import pytest
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl

from polybot.training import replay_storage as storage


class Model:
    def __init__(self, replay=None):
        self.replay = replay

    def save_replay_buffer(self, stream):
        save_to_pkl(stream, self.replay)

    def load_replay_buffer(self, source):
        self.replay = load_from_pkl(source)


def replay():
    rng = np.random.default_rng(19)
    return {"observations": rng.normal(size=(1000, 1, 121)).astype(np.float32),
            "next_observations": rng.normal(size=(1000, 1, 121)).astype(np.float32),
            "actions": np.array([[-0., np.nan], [np.inf, -np.inf]], dtype=np.float32),
            "rewards": rng.normal(size=1000).astype(np.float64),
            "dones": rng.integers(0, 2, size=1000).astype(np.float32),
            "timeouts": rng.integers(0, 2, size=1000).astype(np.float32),
            "pos": 123, "full": True, "n_step": 16, "nested": {"reward_semantics": "unchanged"}}


def test_streaming_writer_bounds_compressor_output_without_copying_large_inputs():
    class Stream:
        closed = False

        def __init__(self):
            self.parts = []

        def write(self, part):
            assert isinstance(part, memoryview)
            assert part.obj is payload
            assert len(part) <= storage._STREAM_CHUNK_BYTES
            self.parts.append(part)

        def flush(self):
            pass

        def close(self):
            self.closed = True

    payload = b"x" * (3 * storage._STREAM_CHUNK_BYTES + 17)
    target = Stream()
    writer = storage._ChunkedWriter(target)
    assert writer.write(payload) == len(payload)
    assert len(target.parts) == 4
    assert b"".join(target.parts) == payload
    writer.close()
    writer.close()
    assert target.closed


@pytest.mark.parametrize("codec", ["plain", "gzip", "zstd"])
def test_all_replay_containers_preserve_exact_pickle_schema_arrays_and_special_values(tmp_path, codec):
    original = replay()
    serialized = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)
    path = tmp_path / "replay.pkl"
    if codec == "plain":
        path.write_bytes(serialized)
    elif codec == "gzip":
        path.write_bytes(gzip.compress(serialized, compresslevel=3))
    else:
        storage.save_replay_buffer(Model(original), path)
        assert path.read_bytes()[:4] == storage._ZSTD_MAGIC
        decoded = storage._zstd().decompress(path.read_bytes())
        assert hashlib.sha256(decoded).digest() == hashlib.sha256(serialized).digest()
    restored = Model()
    storage.load_replay_buffer(restored, path)
    for key, value in original.items():
        actual = restored.replay[key]
        if isinstance(value, np.ndarray):
            assert value.shape == actual.shape and value.dtype == actual.dtype
            assert value.tobytes() == actual.tobytes()
            assert value.flags.c_contiguous == actual.flags.c_contiguous
        else:
            assert value == actual


@pytest.mark.parametrize("failure", ["serialize", "fsync", "replace"])
def test_failed_save_preserves_existing_replay_and_removes_only_its_temporary_file(tmp_path, monkeypatch, failure):
    path = tmp_path / "replay.pkl"
    storage.save_replay_buffer(Model(replay()), path)
    before = path.read_bytes()

    class BrokenModel(Model):
        def save_replay_buffer(self, stream):
            stream.write(b"unfinished")
            raise OSError("injected serialization failure")

    def fail(*args):
        raise OSError("injected storage failure")

    if failure != "serialize":
        monkeypatch.setattr(storage.os, failure, fail)
    with pytest.raises(OSError, match="injected"):
        storage.save_replay_buffer(BrokenModel() if failure == "serialize" else Model(replay()), path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("damage", ["checksum", "truncate"])
def test_zstd_trailer_is_validated_even_when_pickle_stops_before_the_footer(tmp_path, damage):
    path = tmp_path / "replay.pkl"
    storage.save_replay_buffer(Model(replay()), path)
    data = bytearray(path.read_bytes())
    if damage == "checksum":
        data[-1] ^= 0xff
    else:
        del data[-2:]
    path.write_bytes(data)
    with pytest.raises((storage._zstd().ZstdError, EOFError)):
        storage.load_replay_buffer(Model(), path)
