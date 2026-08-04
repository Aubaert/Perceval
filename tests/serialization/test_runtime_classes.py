# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

from perceval.runtime.error_mitigation import (
    CompilationAveraging,
    DetectorBalancing,
    DistinguishablePhotonMitigation,
    PhotonRecycling,
)
from perceval.runtime.job_status import JobStatus, RunningStatus
from perceval.runtime.remote_computer import _RemoteGetter
from perceval.serialization import InputArchive, OutputArchive, Serialization


def _round_trip(obj):
    archive = OutputArchive()
    Serialization.serialize(obj, archive)
    return Serialization.deserialize(InputArchive.from_text(archive.to_text()))


def test_job_status_round_trip():
    status = JobStatus()
    status._status = RunningStatus.ERROR
    status._init_time_start = 10.
    status._running_time_start = 12.
    status._duration = 3.
    status._completed_time = 15.
    status._running_progress = .75
    status._running_phase = "processing"
    status._stop_message = "failed"

    restored = _round_trip(status)

    assert restored.__dict__ == status.__dict__


@pytest.mark.parametrize(
    "mitigation, expected_members",
    (
        (CompilationAveraging(3, 42), {"repetitions": 3, "starting_seed": 42}),
        (DetectorBalancing(), {}),
        (DistinguishablePhotonMitigation({2: 1}), {"_order": {2: 1}}),
        (PhotonRecycling(), {}),
    ),
)
def test_mitigation_round_trip(mitigation, expected_members):
    restored = _round_trip(mitigation)

    assert type(restored) is type(mitigation)
    assert restored.__dict__ == expected_members


def test_remote_getter_round_trip_omits_communication_layer():
    getter = _RemoteGetter(object(), "remote-job-id")
    getter._results = {"results": [1, 2]}
    getter._status.status = RunningStatus.SUCCESS

    restored = _round_trip(getter)

    assert restored._remote_id == getter._remote_id
    assert restored._results == getter._results
    assert restored._status.__dict__ == getter._status.__dict__
    assert restored._communication_layer is None
    assert restored._last_status_refresh == 0.
    assert restored._job_status_errors == 0
