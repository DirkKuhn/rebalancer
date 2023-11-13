from collections.abc import Sequence

import numpy as np
import pytest
from pytest import approx

from rebalancer import Rebalancer


@pytest.mark.parametrize(
    "current_values,target_weights,number_investments,investment,expected_investment_distribution",
    [
        (
            [6_500, 1_100, 1_400, 1_000],
            [0.65, 0.11, 0.14, 0.10],
            1,
            10_000,
            [6_500, 1_100, 1_400, 1_000]
        ),
        (
            [6_500, 1_100, 1_200, 0],
            [0.65, 0.11, 0.14, 0.10],
            1,
            1_000,
            [0, 0, 96, 904]
        )
    ]
)
def test_distribution_fitting(
        current_values: Sequence[float],
        target_weights: Sequence[float],
        number_investments: int,
        investment: float,
        expected_investment_distribution: Sequence[float]
):
    rebalancer = Rebalancer()
    rebalancer._current_values = np.array(current_values)
    rebalancer._targeted_weights = np.array(target_weights)
    rebalancer._number_investments = number_investments
    rebalancer._investment = investment
    rebalancer._calculate_investment_distribution()

    expected_investment_distribution = np.array(expected_investment_distribution)

    assert rebalancer._investment_distribution == approx(expected_investment_distribution)
