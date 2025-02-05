# GeMTest 💎: Examples

Metamorphic relation examples implemented in ``gemtest``.

To run all metamorphic relations from the 16 program domains 
install Python dependencies with ``poetry``:

``poetry install``

and run metamorphic test suites with:

``poetry run pytest  <options> <path to test file/folder>``

We also prepare poetry commands for running some examples such as:

```
poetry run example simple/test_add.py
poetry run example trigonometry/test_sin.py
poetry run example house_pricing/test_house_pricing.py
poetry run example shortest_path/test_path.py
poetry run example facial_keypoints/test_keypoints.py
poetry run example gtsrb/metamorphic_tests
poetry run example linalg/test_solve.py
```

## Requirements
Currently supported: Python Versions 3.8 - 3.12
- Tested under Ubuntu 20.04 with poetry 1.4.0 and python 3.10.12
- Tested under Windows 11 running WSL2 with poetry 1.5.1 and python 3.10.6 
- Tested under Windows 11 native with poetry 1.5.1 and python 3.10.6

## Citation
If you find the ``gemtest`` framework useful in your research or projects, please consider citing it:

```
@inproceedings{speth2025,
    author = {Speth, Simon and Pretschner, Alexander},
    title = {{GeMTest: A General Metamorphic Testing Framework}},
    booktitle = "Proceedings of the 47th International Conference on Software Engineering, (ICSE-Companion)",
    pages = {1--4},
    address = {Ottawa, ON, Canada},
    year = {2025},
}
```

## License
[MIT License](https://github.com/tum-i4/gemtest-examples/blob/main/LICENSE)

