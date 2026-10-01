################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S08 -- assignDerivedParameters EnableMatrixInstruction reject cluster (gfx942).

Reachable-invalid (category A): the DGEMM (double) TN config forks two valid
f64 MFMA MatrixInstruction shapes that pass the earlier validateMIParameters
gate and reach the type/MI reject cluster inside
Tensile/SolutionStructs/Solution.py:assignDerivedParameters, where each fork
trips a distinct reject branch and early-returns. No valid solution survives.

Solution.py lines that fire during the rejected derivation (probe-confirmed):
  Variant MI [16,16,4,...,3,1] -> waves=3 -> MIWaveGroup=[3,1].  This fork used
               to be stopped by a non-power-of-two MIWaveGroup guard in the
               LraTileAssignment vectorStaticRemainder path.  That guard was an
               alias bug in staticRemainder, not a real geometry constraint, and
               was lifted once the alias was fixed; the fork now runs on to
               DepthU selection, where NumThreads 192 tiles no DepthU, and
               rejects there instead.
  2015, 2016 : Variant MI [4,4,4,4] + ComputeDataType double + ISA (9,4,x)
               (!= IsaVersion(9,0,10)) + ScheduleIterAlg==3 -> "[4,4,4,4] is
               disabled" reject.

Both forks reject during derivation, so ``len(solutions_from_config(...)) == 0``
pins the reachable-invalid reject (category A). CPU-only; no GPU, no compile.
"""

import os

import pytest

from config_harness import assert_config_rejects

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "s08_assignderivedparameters_enablema.yaml",
)


def test_s08_assignderivedparameters_enablema_rejects_with_reason(monkeypatch, capsys):
    """Both matrix-instruction forks report their intended rejection."""
    assert_config_rejects(
        _CONFIG,
        _ARCH,
        monkeypatch,
        capsys,
        [
            # MIWaveGroup [3,1] fork: no longer stopped by the lifted
            # power-of-two guard, so it rejects one stage later instead.
            "reject: totalVectorsB 256 % NumThreads 192 != 0",
            "reject: No valid DepthU found",
            "reject: Currently Matrix instructions [4,4,4,4] is disabled.",
        ],
    )
