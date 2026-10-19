# Tests

`test_data_layout.py` verifies the processed spectra's schema, MHz-scale
frequency values, 1,000 points per PMUT, and shared frequency coordinates.

No experiment configuration lists were included. Checks for split/configuration
leakage, training-only scaling, ROI selection, and exact metric reproduction
remain blocked on authoritative run configurations and expected outputs.
