# SPDX-FileCopyrightText: Copyright (c) 2026 Steve Pederzani
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Quantum Circuit
#
# A state-vector simulator for quantum circuits. n qubits are stored as 2^n
# complex amplitudes on the GPU, and each gate is one kernel over the
# amplitudes it mixes. The example prepares a GHZ state, runs a quantum
# Fourier transform on a basis state, and runs a random circuit like the
# ones in Google's quantum supremacy experiment. The QFT is checked against
# its closed form, and every circuit is checked against NumPy when the state
# is small enough for NumPy.
#
# Qubit q is bit q of an amplitude's index. Amplitudes are complex64 (two
# float32 values), as in most GPU simulators, which allows up to 31 qubits
# (16 GiB).
#
###########################################################################

import math

import numpy as np

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def cmul(a: wp.vec2f, b: wp.vec2f) -> wp.vec2f:
    return wp.vec2f(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def with_zero_bit(k: int, q: int) -> int:
    # Insert a 0 into k at bit q, so every launch runs over the 2^(n-1) indices with one bit fixed.
    return ((k >> q) << (q + 1)) | (k & ((1 << q) - 1))


@wp.kernel
def gate_1q(state: wp.array[wp.vec2f], q: int, u00: wp.vec2f, u01: wp.vec2f, u10: wp.vec2f, u11: wp.vec2f):
    # Apply a 2x2 unitary on qubit q to each pair of amplitudes that differ only in bit q.
    i = with_zero_bit(wp.tid(), q)
    j = i | (1 << q)
    a = state[i]
    b = state[j]
    state[i] = cmul(u00, a) + cmul(u01, b)
    state[j] = cmul(u10, a) + cmul(u11, b)


@wp.kernel
def gate_cnot(state: wp.array[wp.vec2f], control: int, target: int):
    i = with_zero_bit(wp.tid(), target)
    if ((i >> control) & 1) == 1:
        j = i | (1 << target)
        a = state[i]
        state[i] = state[j]
        state[j] = a


@wp.kernel
def gate_cphase(state: wp.array[wp.vec2f], a: int, b: int, phase: wp.vec2f):
    # This gate is diagonal. The amplitudes with both bits set are multiplied by the phase, and a CZ is the phase -1.
    i = with_zero_bit(wp.tid(), a) | (1 << a)
    if ((i >> b) & 1) == 1:
        state[i] = cmul(phase, state[i])


@wp.kernel
def gate_swap(state: wp.array[wp.vec2f], a: int, b: int):
    # With a < b, the amplitude with bit a set and bit b clear swaps places with the one that has those bits the other way around.
    i = with_zero_bit(wp.tid(), b)
    if ((i >> a) & 1) == 1:
        j = (i ^ (1 << a)) | (1 << b)
        t = state[i]
        state[i] = state[j]
        state[j] = t


@wp.kernel
def pair_norms(state: wp.array[wp.vec2f], out: wp.array[float]):
    k = wp.tid()
    a = state[2 * k]
    b = state[2 * k + 1]
    out[k] = a[0] * a[0] + a[1] * a[1] + b[0] * b[0] + b[1] * b[1]


@wp.kernel
def set_basis(state: wp.array[wp.vec2f], x: int):
    i = wp.tid()
    if i == x:
        state[i] = wp.vec2f(1.0, 0.0)


@wp.kernel
def gather(state: wp.array[wp.vec2f], idx: wp.array[int], out: wp.array[wp.vec2f]):
    k = wp.tid()
    out[k] = state[idx[k]]


SQRT_X = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
SQRT_Y = np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2
SQRT_W = np.array([[1 + 1j, -1j * math.sqrt(2)], [math.sqrt(2), 1 + 1j]]) / 2  # The square root of (X + Y) / sqrt(2).
HADAMARD = np.array([[1, 1], [1, -1]]) / math.sqrt(2)


class Circuit:
    """A circuit's gates, which run on a state on the device, or in NumPy to check a small state."""

    def __init__(self, num_qubits: int):
        self.n = num_qubits
        self.ops = []

    def h(self, q):
        self.ops.append(("1q", q, HADAMARD))

    def u(self, q, matrix):
        self.ops.append(("1q", q, np.asarray(matrix, dtype=np.complex128)))

    def cnot(self, c, t):
        self.ops.append(("cnot", c, t))

    def cphase(self, a, b, theta):
        self.ops.append(("cphase", a, b, theta))

    def cz(self, a, b):
        self.ops.append(("cphase", a, b, math.pi))

    def swap(self, a, b):
        self.ops.append(("swap", min(a, b), max(a, b)))

    def run(self, state: wp.array):
        half = 1 << (self.n - 1)
        for op in self.ops:
            if op[0] == "1q":
                m = op[2]
                us = [wp.vec2f(float(z.real), float(z.imag)) for z in (m[0, 0], m[0, 1], m[1, 0], m[1, 1])]
                wp.launch(gate_1q, dim=half, inputs=[state, op[1], *us])
            elif op[0] == "cnot":
                wp.launch(gate_cnot, dim=half, inputs=[state, op[1], op[2]])
            elif op[0] == "cphase":
                phase = wp.vec2f(math.cos(op[3]), math.sin(op[3]))
                wp.launch(gate_cphase, dim=half, inputs=[state, op[1], op[2], phase])
            elif op[0] == "swap":
                wp.launch(gate_swap, dim=half, inputs=[state, op[1], op[2]])

    def run_numpy(self, psi: np.ndarray) -> np.ndarray:
        n = self.n
        idx = np.arange(1 << n)
        for op in self.ops:
            if op[0] == "1q":
                # Qubit q is axis n - 1 - q of the state viewed as an n-dimensional array in C order.
                axis = n - 1 - op[1]
                t = np.tensordot(op[2], psi.reshape([2] * n), axes=([1], [axis]))
                psi = np.moveaxis(t, 0, axis).reshape(-1)
            elif op[0] == "cnot":
                c, t = op[1], op[2]
                psi = psi[np.where((idx >> c) & 1, idx ^ (1 << t), idx)]
            elif op[0] == "cphase":
                both = ((idx >> op[1]) & 1) & ((idx >> op[2]) & 1)
                psi = psi * np.where(both == 1, np.exp(1j * op[3]), 1.0)
            elif op[0] == "swap":
                a, b = op[1], op[2]
                differ = ((idx >> a) & 1) ^ ((idx >> b) & 1)
                psi = psi[idx ^ (differ * ((1 << a) | (1 << b)))]
        return psi


def ghz(n):
    c = Circuit(n)
    c.h(0)
    for q in range(n - 1):
        c.cnot(q, q + 1)
    return c


def qft(n):
    # The textbook circuit, with qubit n - 1 as the most significant, gives QFT|x> = sum over k of exp(2 pi i x k / 2^n) |k> / sqrt(2^n).
    c = Circuit(n)
    for i in reversed(range(n)):
        c.h(i)
        for j in reversed(range(i)):
            c.cphase(j, i, math.pi / (1 << (i - j)))
    for q in range(n // 2):
        c.swap(q, n - 1 - q)
    return c


def random_circuit(n, depth, seed=1):
    # Each layer applies one of sqrt X, sqrt Y, or sqrt W to every qubit (never the same one twice in a row on a qubit), then CZ
    # on (q, q + 1) for even q in even layers and odd q in odd layers.
    rng = np.random.default_rng(seed)
    gates = [SQRT_X, SQRT_Y, SQRT_W]
    last = [-1] * n
    c = Circuit(n)
    for layer in range(depth):
        for q in range(n):
            g = int(rng.choice([k for k in range(3) if k != last[q]]))
            last[q] = g
            c.u(q, gates[g])
        for q in range(layer % 2, n - 1, 2):
            c.cz(q, q + 1)
    return c


class Example:
    def __init__(self, num_qubits=24, depth=20, check_max_qubits=20, verbose=True):
        self.n = num_qubits
        self.depth = depth
        self.check_max = check_max_qubits
        self.verbose = verbose
        self.state = wp.zeros(1 << num_qubits, dtype=wp.vec2f)
        self.pairs = wp.zeros(1 << (num_qubits - 1), dtype=float)
        self.results = {}

    def norm(self):
        wp.launch(pair_norms, dim=self.pairs.shape[0], inputs=[self.state, self.pairs])
        return float(wp.utils.array_sum(self.pairs))

    def amplitudes(self, idx=None):
        # Return every amplitude, or only those at idx, gathered on the device so a large state is never copied whole.
        if idx is None:
            a = self.state.numpy().astype(np.float64)
        else:
            out = wp.empty(len(idx), dtype=wp.vec2f)
            wp.launch(gather, dim=len(idx), inputs=[self.state, wp.array(idx, dtype=int), out])
            a = out.numpy().astype(np.float64)
        return a[:, 0] + 1j * a[:, 1]

    def run(self, name, circuit, x=0):
        self.state.zero_()
        wp.launch(set_basis, dim=self.state.shape[0], inputs=[self.state, x])
        wp.synchronize_device()
        with wp.ScopedTimer(name, print=False, synchronize=True) as timer:
            circuit.run(self.state)
        result = {"seconds": timer.elapsed / 1000.0, "gates": len(circuit.ops), "norm": self.norm(), "vs_numpy": None}
        if self.n <= self.check_max:
            psi = np.zeros(1 << self.n, dtype=np.complex128)
            psi[x] = 1.0
            result["vs_numpy"] = float(np.abs(self.amplitudes() - circuit.run_numpy(psi)).max())
        self.results[name] = result
        if self.verbose:
            vs = "" if result["vs_numpy"] is None else f"; largest difference from NumPy {result['vs_numpy']:.1e}"
            print(f"{name}: {self.n} qubits, {result['gates']} gates in {result['seconds']:.3f} s, norm {result['norm']:.6f}{vs}")
        return result

    def step(self):
        n = self.n
        self.run("GHZ state", ghz(n))
        ends = self.amplitudes(np.array([0, (1 << n) - 1], dtype=np.int32))
        if self.verbose:
            print(f"  |0...0> {ends[0].real:.6f}, |1...1> {ends[1].real:.6f} (each 1/sqrt(2) = 0.707107)")

        x = 5
        self.run(f"Quantum Fourier transform of |{x}>", qft(n), x=x)
        # Check against the QFT's own closed form, which doesn't depend on how the circuit is built. Up to 24 qubits every
        # amplitude is checked, and beyond that 2^20 randomly chosen amplitudes plus the first and last.
        if n <= 24:
            k = np.arange(1 << n, dtype=np.int64)
            got = self.amplitudes()
        else:
            k = np.unique(np.concatenate([[0, (1 << n) - 1], np.random.default_rng(2).integers(0, 1 << n, 1 << 20)]))
            got = self.amplitudes(k.astype(np.int32))
        want = np.exp(2j * np.pi * ((x * k) % (1 << n)) / (1 << n)) / math.sqrt(1 << n)
        err = float(np.abs(got - want).max())
        self.results["qft_closed_form"] = err
        if self.verbose:
            which = "every amplitude" if n <= 24 else f"{len(k):,} amplitudes"
            print(f"  against the closed form exp(2 pi i x k / 2^n) / sqrt(2^n), {which}: largest difference {err:.1e}")

        self.run(f"Random circuit, depth {self.depth}", random_circuit(n, self.depth))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num-qubits", type=int, default=24, help="Qubits in the state (2 to 31; 31 is 16 GiB).")
    parser.add_argument("--depth", type=int, default=20, help="Layers of the random circuit.")
    parser.add_argument("--check-max-qubits", type=int, default=20, help="Check every circuit against NumPy up to this many qubits.")

    args = parser.parse_known_args()[0]
    if not 2 <= args.num_qubits <= 31:
        raise SystemExit("--num-qubits must be between 2 and 31")

    with wp.ScopedDevice(args.device):
        example = Example(num_qubits=args.num_qubits, depth=args.depth, check_max_qubits=args.check_max_qubits)
        example.step()
