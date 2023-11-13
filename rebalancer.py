from collections.abc import Collection
from json import load
from math import isclose

import numpy as np
import cvxpy as cp
from cvxpy import OPTIMAL
from tabulate import tabulate


DEFAULTS_LOCATION = "default_inputs.json"

ERROR_TOLERANCE = 1e-3

DECIMAL_PLACES_VALUES: int = 2
DECIMAL_PLACES_PERCENTAGES: int = 1


class Rebalancer:
    def __init__(self):
        # Input
        self._asset_names: Collection[str]
        self._current_values: np.ndarray
        self._targeted_weights: np.ndarray
        self._number_investments: int
        self._investment: float
        # Output
        self._targeted_weights_reachable: bool
        self._investment_distribution: np.ndarray
        self._current_percentages: np.ndarray
        self._new_values: np.ndarray
        self._new_percentages: np.ndarray
        self._targeted_percentages: np.ndarray

    def read_input(self) -> None:
        default_asset_names, default_target_weights = load_defaults()

        asset_names = input(
            "Bitte gebe die Namen der ETFs mit Kommas getrennt ein.\n"
            f"Leer lassen für Default: {default_asset_names}\n"
        )
        if asset_names == "":
            asset_names = default_asset_names
        self._asset_names = list(map(lambda s: s.strip(), asset_names.split(",")))
        print()

        current_values = input(
            "Bitte gebe die aktuellen Werte der ETFs in Euro mit Kommas getrennt ein.\n"
            "Kommazahlen bitte mit Punkt statt Komma eingeben.\n"
            "Bitte die gleiche Reihenfolge wie bei den Namen verwenden.\n"
            "Bsp.: 6500.00, 1100.00, 1400.00, 1000.00\n"
        )
        self._current_values = np.array(list(map(lambda v: float(v), current_values.split(","))))
        print()

        targeted_weights = input(
            "Bitte gebe die gewünschten Gewichte in Prozent der ETFs mit Kommas getrennt ein.\n"
            "Kommazahlen bitte mit Punkt statt Komma eingeben.\n"
            "Bitte die gleiche Reihenfolge wie bei den Namen verwenden.\n"
            f"Leer lassen für Default: {default_target_weights}\n"
        )
        if targeted_weights == "":
            targeted_weights = default_target_weights
        self._targeted_weights = np.array(list(map(lambda w: float(w), targeted_weights.split(",")))) / 100
        print()

        number_investments = input(
            "Bitte gebe bei einer Einmaleinzahlung eine 1 ein.\n"
            "Bei einem Sparplan bitte die Anzahl an Einzahlungen"
            "eingeben bei denen der Sparplan nicht verändert wird:\n"
        )
        self._number_investments = int(number_investments)
        print()

        investment = input(
            "Bitte gebe die Höhe der Einzahlung an."
            "Bei einem Sparplan nur der Betrag für EINE Ausführung:\n"
        )
        self._investment = float(investment)
        print()

        self._validate_input()

    def _validate_input(self) -> None:
        """
        Raise Exception if the input is invalid.
        """
        matching_number_of_inputs = len(self._asset_names) == len(self._current_values) \
            and len(self._current_values) == len(self._targeted_weights)
        if not matching_number_of_inputs:
            raise Exception("Es müssen gleich viele Namen, aktuelle Werte und Gewichte eingegeben werden!")

        at_least_one_asset = len(self._asset_names) >= 1
        if not at_least_one_asset:
            raise Exception("Es muss mindestens ein Asset (ETF, usw.) eingegeben werden!")

        weights_sum_up_to_1 = isclose(sum(self._targeted_weights), 1)
        if not weights_sum_up_to_1:
            raise Exception("Die gewünschten Gewichte müssen sich zu 100% aufsummieren!")

        number_payments_is_bigger_1 = self._number_investments > 0
        if not number_payments_is_bigger_1:
            raise Exception("Die Anzahl der Einzahlungen muss eine ganze Zahl größer gleich 1 sein!")

    def determine_output(self) -> None:
        """
        Raises an Exception if the problem could not be solved.
        """
        self._calculate_investment_distribution()

        self._current_percentages = self._current_values / self._current_values.sum() * 100
        self._targeted_percentages = self._targeted_weights * 100
        self._new_values = self._current_values + self._number_investments * self._investment_distribution
        self._new_percentages = self._new_values / self._new_values.sum() * 100

    def _calculate_investment_distribution(self) -> None:
        """
        Raise an Exception if the problem could not be solved. Solve:
              min ||desired - (current + new)||_2                                    s.t. x>=0, sum(x)==1
        <=>   min ||(sum(values)+n_pay*pay)*weights - (values+n_pay*pay*x)||_2       s.t. x>=0, sum(x)==1
        """
        total_amount_after_payment = self._current_values.sum() + self._number_investments * self._investment
        desired = total_amount_after_payment * self._targeted_weights

        num_assets = len(self._current_values)
        x = cp.Variable(num_assets, nonneg=True)
        new = self._number_investments * self._investment * x

        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(desired - (self._current_values + new))),
            [x.sum() == 1]
        )

        squared_error_or_error_msg = prob.solve()

        # Sugar-coat solution
        if squared_error_or_error_msg is str:
            raise Exception(
                f"Leider ist die Optimierung fehlgeschlagen. Fehlermeldung: {squared_error_or_error_msg}!"
            )
        if prob.status != OPTIMAL:
            raise Exception(
                f"Leider ist die Optimierung fehlgeschlagen. Status: {prob.status}!"
            )

        self._investment_distribution = x.value * self._investment

        # Check whether the target percentages can be achieved
        squared_error = squared_error_or_error_msg * (self._number_investments * self._investment) ** 2
        self._targeted_weights_reachable = squared_error < ERROR_TOLERANCE

    def print_statistics(self) -> None:
        num_assets = len(self._asset_names)

        print(f"Zahle die nächsten {self._number_investments} mal jeweils folgendes ein:")
        print(tabulate(
            {"Name": self._asset_names,
             "Einzahlung [€]": self._investment_distribution},
            headers="keys", tablefmt="plain", floatfmt=f".{DECIMAL_PLACES_VALUES}f"
        ))
        print()

        print("Damit verändert sich das Portfolio folgendermaßen:")
        print(tabulate(
            {"Name": self._asset_names,
             "Aktueller Wert [€]": self._current_values,
             "Aktuelles Gewicht [%]": self._current_percentages,
             "": num_assets*["=>"],
             "Neuer Wert [€]": self._new_values,
             "Neues Gewicht [%]": self._new_percentages,
             "Gewünschtes Gewicht [%]": self._targeted_percentages},
            headers="keys", tablefmt="plain",
            floatfmt=(
                "", f".{DECIMAL_PLACES_VALUES}f", f".{DECIMAL_PLACES_PERCENTAGES}f", "",
                f".{DECIMAL_PLACES_VALUES}f", f".{DECIMAL_PLACES_PERCENTAGES}f", f".{DECIMAL_PLACES_PERCENTAGES}f"
            )
        ))
        print()

        if self._targeted_weights_reachable:
            print("Die gewünschten Gewichte lassen sich erreichen!")
        else:
            print(
                "Die gewünschten Gewichte lassen sich nicht erreichen. Kein Grund zur Panik!\n"
                "Können von ETFs die ein zu großes Gewicht haben, Teile steuereffizient verkauft werden?"
            )


def load_defaults() -> tuple[str, str]:
    with open(f"./{DEFAULTS_LOCATION}", "r") as f:
        defaults = load(f)
    if "asset_names" not in defaults:
        raise Exception(f'Spezifizere "asset_names" in {DEFAULTS_LOCATION}!')
    if "target_weights" not in defaults:
        raise Exception(f'Spezifiziere "target_weights" in {DEFAULTS_LOCATION}!')
    default_asset_names = defaults["asset_names"]
    default_target_weights = defaults["target_weights"]
    if len(default_asset_names) != len(default_target_weights):
        raise Exception(
            f'Für "asset_names" und "target_weights" müssen die selbe Anzahl an Werten spezifiziert werde, '
            f'aber es wurde {default_asset_names} und {default_target_weights} erhalten!'
        )
    weights_sum_up_to_100 = isclose(sum(default_target_weights), 100)
    if not weights_sum_up_to_100:
        raise Exception(
            f'Für "target_weights" addieren sich die Gewichte nicht zu 100 auf: {default_target_weights}!'
        )
    default_asset_names = ", ".join(default_asset_names)
    default_target_weights = ", ".join(map(str, default_target_weights))
    return default_asset_names, default_target_weights


if __name__ == "__main__":
    rebalancer = Rebalancer()
    rebalancer.read_input()
    rebalancer.determine_output()
    rebalancer.print_statistics()
    input("\nDrücke Enter um dieses Fenster zu verlassen!")
