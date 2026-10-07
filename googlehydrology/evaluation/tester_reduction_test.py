# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `BaseTester._reduce_samples` point-estimate readouts.

For the `cmal_deterministic` head the trailing `samples` axis is not a set of
random draws. It is the fixed 10-point summary
`[mixture_mean, q0.1, q0.2, ..., q0.9]` emitted by
`utils.cmal_deterministic.generate_predictions`. These tests pin down that
`MIXTURE_MEAN` selects the mixture mean at index 0, and that neither `MEAN` nor
`MEDIAN` is a substitute for it.
"""

import types

from absl.testing import absltest
import numpy as np
import xarray

from googlehydrology.evaluation.tester import BaseTester
from googlehydrology.utils.config import TesterSamplesReduction


# A right-skewed CMAL summary, which is the normal shape for streamflow.
# Layout is [mixture_mean, q0.1, ..., q0.9]. The mean (5.0) sits above the
# median q0.5 (2.0), as it must for a right-skewed distribution.
_SKEWED_SUMMARY = [5.0, 0.1, 0.4, 0.9, 1.4, 2.0, 2.8, 3.9, 5.6, 9.0]
_MIXTURE_MEAN = _SKEWED_SUMMARY[0]
_Q50 = _SKEWED_SUMMARY[5]


def _make_sim(summary=None):
  """Builds a [date=2, samples=10] DataArray from a CMAL summary vector."""
  summary = _SKEWED_SUMMARY if summary is None else summary
  values = np.stack([np.asarray(summary), np.asarray(summary) * 2.0])
  return xarray.DataArray(values, dims=('date', 'samples'))


def _reduce(sim, reduction, head='cmal_deterministic'):
  """Invokes the unbound reducer against a minimal stand-in for the tester.

  `_reduce_samples` only touches `self.cfg`, so binding it to a namespace
  avoids constructing a full tester (which would require a model, a dataset
  and a run directory) for what is a pure function of its inputs.
  """
  fake_self = types.SimpleNamespace(
      cfg=types.SimpleNamespace(
          tester_sample_reduction=reduction,
          head=head,
      )
  )
  return BaseTester._reduce_samples(fake_self, sim)  # pylint: disable=protected-access


class ReduceSamplesTest(absltest.TestCase):

  def test_mixture_mean_selects_index_zero(self):
    out = _reduce(_make_sim(), TesterSamplesReduction.MIXTURE_MEAN)
    self.assertNotIn('samples', out.dims)
    np.testing.assert_allclose(
        out.values, [_MIXTURE_MEAN, _MIXTURE_MEAN * 2.0]
    )

  def test_mixture_mean_is_not_reproducible_by_mean_or_median(self):
    """Guards the trap that motivated adding a third readout.

    Setting `tester_sample_reduction: mean` does NOT yield the mixture mean.
    It averages the mixture mean together with nine quantiles, producing a
    third distinct statistic. This test fails if anyone "simplifies" the
    MIXTURE_MEAN branch away in favour of MEAN.
    """
    sim = _make_sim()
    mixture_mean = _reduce(sim, TesterSamplesReduction.MIXTURE_MEAN)
    plain_mean = _reduce(sim, TesterSamplesReduction.MEAN)
    plain_median = _reduce(sim, TesterSamplesReduction.MEDIAN)

    self.assertNotAlmostEqual(
        float(mixture_mean[0]), float(plain_mean[0]), places=3
    )
    self.assertNotAlmostEqual(
        float(mixture_mean[0]), float(plain_median[0]), places=3
    )

  def test_median_and_mixture_median_select_q50_quantile(self):
    """Verifies that MEDIAN and MIXTURE_MEDIAN select index 5 (q0.5) on cmal_deterministic."""
    for red in (TesterSamplesReduction.MEDIAN, TesterSamplesReduction.MIXTURE_MEDIAN):
      out = _reduce(_make_sim(), red)
      self.assertNotIn('samples', out.dims)
      np.testing.assert_allclose(out.values, [_Q50, _Q50 * 2.0])

  def test_mixture_mean_rejects_sampling_heads(self):
    """Index 0 is a real random draw for these heads, not a mean."""
    for head in ('cmal', 'regression'):
      with self.subTest(head=head):
        with self.assertRaisesRegex(ValueError, 'cmal_deterministic'):
          _reduce(
              _make_sim(), TesterSamplesReduction.MIXTURE_MEAN, head=head
          )

  def test_noop_without_samples_dimension(self):
    sim = xarray.DataArray(np.arange(3.0), dims=('date',))
    for reduction in TesterSamplesReduction:
      with self.subTest(reduction=reduction):
        out = _reduce(sim, reduction)
        np.testing.assert_allclose(out.values, sim.values)

  def test_unknown_reduction_raises(self):
    with self.assertRaises(KeyError):
      _reduce(_make_sim(), 'not_a_reduction')


if __name__ == '__main__':
  absltest.main()
