"""NumPy skinning with the exact math of the OpenGL 4.3 compute shaders.

macOS stops at OpenGL 4.1 (no compute shaders), so mesh deformation runs here and
the result is uploaded as the same 8-float vertex records the shaders wrote.
"""
import numpy as np


class Topology:
    """Vertex-to-neighbour triangle pairs behind the smooth-normal pass."""
    def __init__(self, offsets, neighbors):
        offsets = np.asarray(offsets, np.int64)
        self.count = len(offsets) - 1
        self.owner = np.repeat(np.arange(self.count), np.diff(offsets))
        self.pairs = np.asarray(neighbors, np.int64).reshape(-1, 2)[:offsets[-1]]

    def normals(self, pos):
        c = pos[self.owner]
        cross = np.cross(pos[self.pairs[:, 0]] - c, pos[self.pairs[:, 1]] - c)
        n = np.stack([np.bincount(self.owner, weights=cross[:, k], minlength=self.count) for k in range(3)], 1)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


def records(pos, normal=None):
    out = np.zeros((len(pos), 8), np.float32)
    out[:, :3] = pos
    out[:, 3] = 1
    if normal is not None:
        out[:, 4:7] = normal
    return out


def _affine(frame, count):
    return np.asarray(frame, np.float32).reshape(-1, 16)[:count, :12].reshape(-1, 3, 4)


def _blend(weights, frame, count=55):
    """Per-vertex weighted 3x4 transform, as w * (A v) summed over joints."""
    return (weights @ _affine(frame, count).reshape(count, 12)).reshape(-1, 3, 4)


class SmplModel:
    def __init__(self, packed, vertex_count):
        n = 3 * vertex_count
        self.count = vertex_count
        self.template = packed[:n]
        self.shape = packed[n:21 * n].reshape(20, n)
        self.pose = packed[21 * n:507 * n].reshape(486, n)
        self.weights = packed[507 * n:507 * n + vertex_count * 55].reshape(vertex_count, 55)
        self._betas, self._shaped = None, None

    def skin(self, frame):
        frame = np.asarray(frame, np.float32)
        betas = frame[1366:1386].tobytes()
        if betas != self._betas:  # Betas are constant per person; skip the 20-way blend.
            self._betas, self._shaped = betas, self.template + frame[1366:1386] @ self.shape
        p = (self._shaped + frame[880:1366] @ self.pose).reshape(-1, 3)
        b = _blend(self.weights, frame[:880])
        return np.einsum('vij,vj->vi', b[:, :, :3], p) + b[:, :, 3] + frame[1386:1389]

    def bound_skin(self, frame, bind_vertices, bind_frames):
        frame = np.asarray(frame, np.float32)
        delta = (frame[880:1366] @ self.pose).reshape(-1, 3)
        m = (self.weights @ np.asarray(bind_frames, np.float32).reshape(55, 9)).reshape(-1, 3, 3)
        p = np.asarray(bind_vertices, np.float32).reshape(-1, 3) + np.einsum('vij,vj->vi', m, delta)
        b = _blend(self.weights, frame[:880])
        return np.einsum('vij,vj->vi', b[:, :, :3], p) + b[:, :, 3]


class RigModel:
    def __init__(self, packed, vertex_count):
        self.data = np.asarray(packed, np.float32).reshape(vertex_count, 7)
        self.bone = self.data[:, 6].astype(np.int64)

    def skin(self, frame):
        a = _affine(frame, 1 << 30)[self.bone]
        pos = np.einsum('vij,vj->vi', a[:, :, :3], self.data[:, :3]) + a[:, :, 3]
        n = np.einsum('vij,vj->vi', a[:, :, :3], self.data[:, 3:6])
        return pos, n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


class CharacterModel:
    def __init__(self, model, vertex_count):
        v = vertex_count
        self.vertices = model[:3 * v].reshape(v, 3)
        rows = model[3 * v:5 * v].reshape(v, 2).astype(np.int64)
        self.weights = model[5 * v:].reshape(-1, 2)
        counts = rows[:, 1]
        self.owner = np.repeat(np.arange(v), counts)
        start = np.cumsum(counts) - counts
        index = np.repeat(rows[:, 0] - start, counts) + np.arange(counts.sum())
        self.bone = self.weights[index, 0].astype(np.int64)
        self.weight = self.weights[index, 1]
        self.count = v

    def skin(self, palette):
        m = _affine(palette, 1 << 30)[self.bone]
        p = np.einsum('mij,mj->mi', m[:, :, :3], self.vertices[self.owner]) + m[:, :, 3]
        return np.stack([np.bincount(self.owner, weights=self.weight * p[:, k], minlength=self.count) for k in range(3)], 1)
