Summary
=======

* Shuffle the training dataset with a randomized sampler.
* Keep validation and test datasets explicitly sequential.
* Seed worker initialization and sampler generation from each split's configured seed.
* Verify sampler selection and reproducible training order with unit tests.

Testing
=======

* ``pytest tests/utils/test_datasets.py``
