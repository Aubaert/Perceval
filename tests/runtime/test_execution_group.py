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

import time
from unittest.mock import patch

import pytest
import tqdm

from perceval import RunningStatus, RemoteComputer, FockState, ContextManager
from perceval.components import Experiment
from perceval.runtime import Computation, Execution, ExecutionGroup, SimulatedComputer
from perceval.serialization import InputArchive, OutputArchive, Serialization
from perceval.providers.quandela.rpc_handler import RPCHandler
from perceval.runtime.communication_layer import RPCBasedCommunicationLayer

from ..providers.quandela._mock_rpc_handler import RPCHandlerResponsesBuilder

from .test_execution import execution

TOKEN = "test_token"
PLATFORM_NAME = "sim:test"
URL = "https://test"

RPC_HANDLER = RPCHandler(PLATFORM_NAME, URL, TOKEN)


def _execution(name="execution"):
    computer = SimulatedComputer("SLOS")
    exp = Experiment(2)
    exp.with_input(FockState([1, 0]))
    computation = Computation(computer.get_command("probs"), exp)
    computation.job_name = name
    return Execution(computation, computer)


def test_serialization(tmp_path, monkeypatch):
    monkeypatch.setattr(ExecutionGroup, "_DIR_PATH", str(tmp_path))

    group = ExecutionGroup("group")
    group.add(_execution())

    output = OutputArchive()
    Serialization.serialize(group, output)
    restored = Serialization.deserialize(InputArchive.from_text(output.to_text()))

    assert isinstance(restored, ExecutionGroup)
    assert restored.name == "group"
    assert len(restored) == 1
    assert isinstance(restored[0], Execution)
    assert restored.created_date == group.created_date


def test_persists_and_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(ExecutionGroup, "_DIR_PATH", str(tmp_path))

    group = ExecutionGroup("group")
    execution = _execution()
    group.add(execution)
    assert ExecutionGroup.list_locally_saved() == ["group"]

    restored = ExecutionGroup("group")

    assert len(restored) == 1
    assert restored[0].name == execution.name
    assert restored.progress() == {
        "Total": 1,
        "Finished": [0, {"successful": 0, "unsuccessful": 0}],
        "Unfinished": [1, {"sent": 0, "not sent": 1}],
    }


def test_add_wrong_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(ExecutionGroup, "_DIR_PATH", str(tmp_path))
    group = ExecutionGroup("group")

    with pytest.raises(TypeError, match="Only an Execution"):
        group.add(object())

    execution = _execution()
    group.add(execution)
    with pytest.raises(ValueError, match="Duplicate"):
        group.add(execution)


@patch.object(ExecutionGroup._PERSISTENT_DATA, 'write_file')
@patch.object(tqdm.tqdm, "display")
def test_classic_run(_, mock_write_file):
    initial_value = ExecutionGroup.STATUS_REFRESH_DELAY
    with ContextManager(lambda: setattr(ExecutionGroup, "STATUS_REFRESH_DELAY", 0),
                        lambda: setattr(ExecutionGroup, "STATUS_REFRESH_DELAY", initial_value)):
        rpc_handler_responses_builder = RPCHandlerResponsesBuilder(RPC_HANDLER)
        exec_nmb = 2

        eg = ExecutionGroup("group")

        expected_write_call_count = 1
        assert mock_write_file.call_count == expected_write_call_count

        for _ in range(exec_nmb):
            eg.add(_execution())
            expected_write_call_count += 1

        assert mock_write_file.call_count == expected_write_call_count
        assert len(eg) == exec_nmb

        group_progress = eg.progress()

        # no write since jobs have not been sent
        assert mock_write_file.call_count == expected_write_call_count

        assert group_progress == {'Total': exec_nmb,
                                  'Finished': [0, {'successful': 0, 'unsuccessful': 0}],
                                  'Unfinished': [exec_nmb, {'sent': 0, 'not sent': exec_nmb}]}

        # Running jobs
        rpc_handler_responses_builder.set_job_availability_count(2)

        eg.run_sequential(0)
        expected_write_call_count += 2 * exec_nmb

        assert mock_write_file.call_count == expected_write_call_count
        assert eg[0].job_group_name == eg.name

        group_progress = eg.progress()

        assert mock_write_file.call_count == expected_write_call_count
        assert group_progress == {'Total': exec_nmb,
                                  'Finished': [exec_nmb, {'successful': exec_nmb, 'unsuccessful': 0}],
                                  'Unfinished': [0, {'sent': 0, 'not sent': 0}]}

        for _ in range(exec_nmb):
            eg.add(_execution())
            expected_write_call_count += 1

        assert mock_write_file.call_count == expected_write_call_count

        group_progress = eg.progress()

        assert mock_write_file.call_count == expected_write_call_count

        current_group_progress = {'Total': exec_nmb*2,
                                  'Finished': [exec_nmb, {'successful': exec_nmb, 'unsuccessful': 0}],
                                  'Unfinished': [exec_nmb, {'sent': 0, 'not sent': exec_nmb}]}

        assert group_progress == current_group_progress

        assert mock_write_file.call_count == expected_write_call_count

        # Test complex load
        new_jg = ExecutionGroup("group")
        expected_write_call_count += 1
        assert mock_write_file.call_count == expected_write_call_count


@patch.object(tqdm.tqdm, "display")
def test_run_advance(_, tmp_path, monkeypatch):
    monkeypatch.setattr(ExecutionGroup, "_DIR_PATH", str(tmp_path))

    rpc_handler_responses_builder = RPCHandlerResponsesBuilder(RPC_HANDLER)
    rpc_handler_responses_builder.set_job_status_sequence(
        [RunningStatus.SUCCESS, RunningStatus.ERROR, RunningStatus.ERROR])

    eg = ExecutionGroup("group")

    computer = RemoteComputer(RPCBasedCommunicationLayer(RPC_HANDLER))
    experiment = Experiment(2)
    experiment.min_detected_photons_filter(1)
    experiment.with_input(FockState([1, 0]))
    computation = Computation(computer.get_command("probs"), experiment)
    execution = Execution(computation, computer)

    for i in range(3):
        eg.add(execution.clone(), max_shots = i)

    eg.run_parallel()

    assert rpc_handler_responses_builder.last_payload.get("job_group_name") == "group"
    assert rpc_handler_responses_builder.last_payload["payload"].get("max_shots") == 2

    eg.add(execution.clone(), max_shots = 1000)

    rpc_handler_responses_builder.set_job_status_sequence([])
    rpc_handler_responses_builder.set_default_job_status(RunningStatus.SUCCESS)

    assert eg.progress() == {'Total': 4,
                             'Finished': [3, {'successful': 1, 'unsuccessful': 2}],
                             'Unfinished': [1, {'sent': 0, 'not sent': 1}]}

    eg.run_parallel()

    assert eg.progress() == {'Total': 4,
                             'Finished': [4, {'successful': 2, 'unsuccessful': 2}],
                             'Unfinished': [0, {'sent': 0, 'not sent': 0}]}


@pytest.mark.long_test
@patch.object(ExecutionGroup._PERSISTENT_DATA, 'write_file')
def test_cancel_all(mock_write_file, execution):
    jg = ExecutionGroup("group")

    period = 1.

    for i in range(13):
        jg.add(execution.clone(), n = 5, period = 0. if i < 8 else period)

    jg.launch_async_executions()
    time.sleep(0.5)  # Give the time for the first threads to finish

    # Unfinished jobs are required in order to cancel_all() doing something
    assert jg.progress() == {'Total': 13,
                             'Finished': [8, {'successful': 8, 'unsuccessful': 0}],
                             'Unfinished': [5, {'sent': 5, 'not sent': 0}]}

    jg.cancel_all()

    time.sleep(period + 0.5)  # Give the time for all threads to reach the callback

    assert jg.progress() == {'Total': 13,
                             'Finished': [13, {'successful': 8, 'unsuccessful': 5}],
                             'Unfinished': [0, {'sent': 0, 'not sent': 0}]}
