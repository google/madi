#     Copyright 2020 Google LLC
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# Lint as: python3
"""Open source setup for MADI."""
import io
import setuptools

with io.open("README.md", "r", encoding="utf8") as fh:
  long_description = fh.read()

setuptools.setup(
    name="madi",
    version="0.0.1",
    author="John Sipple",
    author_email="sipple@google.com",
    maintainer="John Sipple",
    maintainer_email="sipple@google.com",
    packages=setuptools.find_packages(),
    scripts=[],
    url="https://github.com/google/madi",
    license="Apache v.2.0",
    description="Multivariate Anomaly Detection with Interpretability (MADI)",
    long_description=long_description,
    python_requires=">=3.0.*",
    install_requires=[
        "setuptools >= 40.2.0",
        "numpy>=1.16.0",
        "tensorflow>=2.2.0"
        "scipy>=1.2.1",
        "scikit-learn>=0.21.3",
        "pandas>=0.24.2",
    ],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
    ],
)
