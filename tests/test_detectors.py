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

from perceval.components import BS, PERM
from perceval.components.detector import get_detection_type, Detector, DetectionType, BSLayeredPPNR
from perceval.utils import BasicState


def test_detector():
    pnr_detector = Detector.pnr()
    assert pnr_detector.type == DetectionType.PNR

    for i in range(10):
        assert pnr_detector.detect(i) == BasicState([i])

    th_detector = Detector.threshold()
    assert th_detector.type == DetectionType.Threshold

    for i in range(10):
        assert th_detector.detect(i) == BasicState([i if i < 2 else 1])


def test_bs_layered_ppnr():
    detector = BSLayeredPPNR(1)
    assert detector.type == DetectionType.PPNR

    s0 = BasicState([0])
    s1 = BasicState([1])
    s2 = BasicState([2])
    result = detector.detect(0)
    assert result == s0
    result = detector.detect(1)
    assert result == s1
    result = detector.detect(2)
    assert s0 not in result
    assert result[s1] == pytest.approx(0.5)
    assert result[s2] == pytest.approx(0.5)

    result = detector.detect(3)
    assert s0 not in result
    assert result[s1] == pytest.approx(0.25)
    assert result[s2] == pytest.approx(0.75)


def test_bs_layered_ppnr_circuit():
    refl = 0.55
    detector = BSLayeredPPNR(1, refl)
    ppnr_circuit = detector.create_circuit()
    assert ppnr_circuit.ncomponents() == 1
    assert isinstance(ppnr_circuit[0, 0], BS)
    assert ppnr_circuit[0, 0].reflectivity == pytest.approx(refl)

    detector = BSLayeredPPNR(3, refl)
    ppnr_circuit = detector.create_circuit()
    assert ppnr_circuit.ncomponents() == 9  # 1 + 2 + 4 = 7 beam splitters + 2 permutations
    expected = ((0, BS), (0, PERM), (0, BS), (2, BS), (0, PERM), (0, BS), (2, BS), (4, BS), (6, BS))
    for component, expectation in zip(ppnr_circuit._components, expected):
        assert component[0][0] == expectation[0]
        assert isinstance(component[1], expectation[1])


def test_bs_layered_ppnr_bad_usage():
    with pytest.raises(AssertionError):
        BSLayeredPPNR(0)

    with pytest.raises(AssertionError):
        BSLayeredPPNR(1, -0.2)

    with pytest.raises(AssertionError):
        BSLayeredPPNR(1, 1.1)


def test_detection_type():
    pnr_detector_list = [Detector.pnr()] * 3  # Only PNR detectors
    thr_detector_list = [Detector.threshold()] * 3  # Only threshold detectors
    mixed_detector_list = [BSLayeredPPNR(1), Detector.pnr(), Detector.threshold()]

    assert get_detection_type(pnr_detector_list) == DetectionType.PNR
    assert get_detection_type(thr_detector_list) == DetectionType.Threshold
    # PPNR means mixed detectors in this context
    assert get_detection_type(pnr_detector_list + thr_detector_list) == DetectionType.Mixed
    assert get_detection_type(mixed_detector_list) == DetectionType.Mixed
