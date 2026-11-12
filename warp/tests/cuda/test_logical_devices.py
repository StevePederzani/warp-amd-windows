# SPDX-FileCopyrightText: Copyright (c) 2026 Steve Pederzani
# SPDX-License-Identifier: Apache-2.0

"""One AMD GPU shown as several Warp devices (WARP_HIP_DEVICES=n).

Each logical device holds an equal share of the card's memory, allocates from a memory pool of its own within its share, and has a
context of its own; its kernels run on the whole card, since HIP on Windows holds a stream to no part of it. Two shares of one card
reach each other's memory, so peer access between them is supported, and its state is the one Warp's API sets, as between two
GPUs. Each case runs in a subprocess, since the split is read when Warp starts.
"""

import json
import unittest

import warp as wp
from warp.tests.unittest_utils import run_python_subprocess

_CHILD = r"""
import json, sys
import numpy as np
import warp as wp
wp.init()
devices = wp.get_cuda_devices()
out = {"count": len(devices), "names": [d.name for d in devices], "sm": [d.sm_count for d in devices],
       "memory": [d.total_memory for d in devices]}

# (a kernel of Warp's own, a reduction, run on each device: a kernel defined in this source could not be compiled, since Warp reads a
#  kernel's source from its file and this runs from a string)
out["kernel"] = []
for d in devices:
    a = wp.full(1024, 1.5, dtype=float, device=d)
    out["kernel"].append(float(wp.utils.array_sum(a)))

if len(devices) >= 2:
    a = wp.full(4096, 3.0, dtype=float, device=devices[0])
    b = wp.zeros(4096, dtype=float, device=devices[1])
    # (a is filled on its own device's stream and the copy runs on b's: as with two GPUs, nothing orders two devices' streams but the
    #  caller, so the fill is waited for first)
    wp.synchronize_device(devices[0])
    wp.copy(b, a)
    out["copy"] = float(b.numpy().sum())
    out["peer_supported"] = bool(wp.is_peer_access_supported(devices[0], devices[1]))
    out["peer_before"] = bool(wp.is_peer_access_enabled(devices[0], devices[1]))
    wp.set_peer_access_enabled(devices[0], devices[1], True)
    out["peer_after"] = bool(wp.is_peer_access_enabled(devices[0], devices[1]))
    wp.set_peer_access_enabled(devices[0], devices[1], False)
    out["peer_off"] = bool(wp.is_peer_access_enabled(devices[0], devices[1]))
print("RESULT " + json.dumps(out), flush=True)
"""


def _hip_card():
    return any(d.is_hip for d in wp.get_cuda_devices()) if wp.is_cuda_available() else False


def _run(parts):
    result = run_python_subprocess(_CHILD, env_updates={"WARP_HIP_DEVICES": str(parts)}, timeout=300)
    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        raise AssertionError("no result from the child process\n" + result.stdout + "\n" + result.stderr)
    return json.loads(line[len("RESULT ") :])


@unittest.skipUnless(_hip_card(), "Logical devices are an AMD GPU feature of this port")
class TestLogicalDevices(unittest.TestCase):
    def test_one_card_as_two(self):
        whole = _run(1)
        split = _run(2)
        self.assertEqual(split["count"], 2 * whole["count"])
        # each share runs on all of the card's multiprocessors and holds half of its memory
        self.assertEqual(split["sm"], [whole["sm"][0]] * 2)
        self.assertLessEqual(abs(split["memory"][0] - whole["memory"][0] / 2), whole["memory"][0] * 0.01)
        self.assertIn("part 1 of 2", split["names"][0])
        self.assertIn("part 2 of 2", split["names"][1])
        # a kernel runs on each, and a copy moves between them
        self.assertEqual(split["kernel"], [1536.0, 1536.0])
        self.assertEqual(split["copy"], 4096 * 3.0)
        # one card's memory is reachable from both; the access state is the one Warp sets
        self.assertTrue(split["peer_supported"])
        self.assertFalse(split["peer_before"])
        self.assertTrue(split["peer_after"])
        self.assertFalse(split["peer_off"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
