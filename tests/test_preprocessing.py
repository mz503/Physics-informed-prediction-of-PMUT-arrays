import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Helpers.data_loader import scale_target_for_training


class TargetScalingTests(unittest.TestCase):
    def test_target_scaler_is_fitted_on_training_pmuts_only(self):
        df = pd.DataFrame({
            "N_PMUT": [1, 1, 2, 2, 3, 3],
            "Amplitude_R_mean": [1.0, 3.0, 5.0, 7.0, 1000.0, 2000.0],
        })

        scaled_df, scaler = scale_target_for_training(df, [1, 2])

        self.assertAlmostEqual(scaler.mean_[0], 4.0)
        self.assertAlmostEqual(scaler.var_[0], 5.0)
        self.assertAlmostEqual(
            scaled_df.loc[scaled_df["N_PMUT"] == 3, "Amplitude_Scaled"].mean(),
            (1500.0 - 4.0) / (5.0 ** 0.5),
        )
        self.assertNotIn("Amplitude_Scaled", df.columns)

    def test_target_scaler_requires_present_training_pmuts(self):
        df = pd.DataFrame({
            "N_PMUT": [1],
            "Amplitude_R_mean": [1.0],
        })

        with self.assertRaisesRegex(ValueError, "missing from the data"):
            scale_target_for_training(df, [2])


if __name__ == "__main__":
    unittest.main()
