import csv
import unittest
from collections import defaultdict
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECTRA_CSV = PROJECT_ROOT / "data" / "processed" / "pmuts_all.csv"


class ProcessedSpectrumTests(unittest.TestCase):
    def test_spectra_have_aligned_frequency_coordinates_and_units(self):
        frequencies = defaultdict(list)
        required_columns = {
            "N_PMUT",
            "Frequency_MHz",
            "Amplitude_R_mean",
            "Amplitude_R_std",
            "Phase_deg_mean",
            "Phase_deg_std",
        }

        with SPECTRA_CSV.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertTrue(required_columns.issubset(reader.fieldnames or []))
            for row in reader:
                pmut_id = int(row["N_PMUT"])
                frequency_mhz = float(row["Frequency_MHz"])
                self.assertGreater(frequency_mhz, 0)
                self.assertLess(frequency_mhz, 100)
                frequencies[pmut_id].append(frequency_mhz)

        self.assertEqual(set(frequencies), set(range(1, 17)))
        self.assertTrue(all(len(values) == 1000 for values in frequencies.values()))

        reference = frequencies[1]
        self.assertEqual(reference, sorted(reference))
        for pmut_id, values in frequencies.items():
            with self.subTest(pmut_id=pmut_id):
                self.assertEqual(len(values), len(reference))
                self.assertTrue(
                    all(abs(actual - expected) <= 1e-9
                        for actual, expected in zip(values, reference)),
                    "frequency coordinates differ by more than 1e-9 MHz",
                )

    def test_frequency_grid_is_near_log_spaced(self):
        with SPECTRA_CSV.open(newline="", encoding="utf-8") as csv_file:
            frequencies = [
                float(row["Frequency_MHz"])
                for row in csv.DictReader(csv_file)
                if int(row["N_PMUT"]) == 1
            ]

        ratios = [
            right / left
            for left, right in zip(frequencies, frequencies[1:])
        ]
        intervals = [
            right - left
            for left, right in zip(frequencies, frequencies[1:])
        ]

        self.assertAlmostEqual(frequencies[0], 0.1, places=9)
        self.assertAlmostEqual(frequencies[-1], 2.5, places=9)
        self.assertAlmostEqual(median(ratios), 1.0032273, places=6)
        self.assertLess(max(ratios) - min(ratios), 2e-8)
        self.assertLess(intervals[0], median(intervals))
        self.assertGreater(intervals[-1], median(intervals))


if __name__ == "__main__":
    unittest.main()
