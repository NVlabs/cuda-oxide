# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit regressions for the cuda-gdb transcript harness."""

import subprocess

import pytest

from conftest import GdbSession, breakpoint_has_location, gdb_value_matches


def command_section(number, body=""):
    label = f"command_{number:03d}"
    return (
        f"@@CUDA_OXIDE_GDB_BEGIN:{label}@@\n"
        f"{body}\n"
        f"@@CUDA_OXIDE_GDB_END:{label}@@\n"
    )


def mocked_session(tmp_path, monkeypatch, output, returncode=0):
    completed = subprocess.CompletedProcess(
        args=["cuda-gdb"], returncode=returncode, stdout=output,
    )
    monkeypatch.setattr("conftest.subprocess.run", lambda *args, **kwargs: completed)
    return GdbSession("cuda-gdb", tmp_path / "binary", tmp_path / "session.gdb")


def test_nonzero_cuda_gdb_exit_fails(tmp_path, monkeypatch):
    session = mocked_session(
        tmp_path, monkeypatch, command_section(1), returncode=17,
    )
    session("show version")

    with pytest.raises(pytest.fail.Exception, match="exited with status 17"):
        session.execute_and_check()


def test_unchecked_command_must_reach_its_end_marker(tmp_path, monkeypatch):
    session = mocked_session(tmp_path, monkeypatch, command_section(1))
    session("set args values")
    session("run")

    with pytest.raises(pytest.fail.Exception, match="command 2.*markers are missing"):
        session.execute_and_check()


def test_final_quit_may_omit_its_end_marker(tmp_path, monkeypatch):
    quit_begin = "@@CUDA_OXIDE_GDB_BEGIN:command_002@@\n"
    session = mocked_session(
        tmp_path, monkeypatch, command_section(1) + quit_begin,
    )
    session("target cudacore fixture.core")
    session("quit")

    session.execute_and_check()


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("$1 = 1.59999", r"1\.5"),
        ("$2 = 200", r"20"),
        ("$3 = <optimized out>", r"\(0, 1\)"),
        ("$4 = <optimized out>", r"Direction::East"),
        ("$5 = <optimized out>", r"\[0, 1, 2, 3\]"),
        ("Cannot access memory at address 0x0", r"ConstantMemory .*2\.5"),
    ],
)
def test_exact_value_matching_rejects_incorrect_output(output, expected):
    assert not gdb_value_matches(output, expected)


def test_exact_value_matching_accepts_complete_expected_value():
    assert gdb_value_matches("$12 = (0, 1)\n", r"\(0, 1\)")


@pytest.mark.parametrize(
    "output",
    [
        "1 breakpoint keep y 0x1234 in debuginfo_values at src/main.rs:183",
        "\n".join([
            "1 breakpoint keep y <MULTIPLE>",
            "1.1 y 0x1234 in wrapping_add at src/main.rs:183",
            "1.2 y 0x5678 in debuginfo_values at src/main.rs:183",
        ]),
    ],
)
def test_breakpoint_location_accepts_single_and_multiple_rows(output):
    assert breakpoint_has_location(output, 1, "src/main.rs", 183)


def test_breakpoint_location_is_scoped_to_requested_breakpoint():
    output = "\n".join([
        "1 breakpoint keep y <MULTIPLE>",
        "1.1 y 0x1234 in wrapping_add at src/main.rs:184",
        "2 breakpoint keep y 0x5678 in debuginfo_values at src/main.rs:183",
    ])
    assert not breakpoint_has_location(output, 1, "src/main.rs", 183)
