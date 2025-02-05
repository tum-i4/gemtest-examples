# Prerequisites

* As default only a small version of the GTSRB dataset is used.
* To download the full dataset run `python examples/traffic_sign_classifier/data/download_GTSRB_complete.py`.
* THe test suits will automatically detect if the full dataset is available and use it instead of the small version.

# Reproducing the results

* Execute the pytests located in the `examples/traffic_sign_classifier/metamorphic_tests` folder.
    * For the simple tests run `poetry run pytest examples/traffic_sign_classifier/metamorphic_tests/test_simple.py`
    * The test is parametrised to calculate the LSC metric for each dataset
    * The computation can be run in parallel per dataset. Add the `-n {number_workers}` argument to the `pytest` command
    * The results will be stored under the root `assets/` folder
