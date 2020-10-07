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
load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//research/colab:build_defs.bzl", "colab_binary")
load("//third_party/py/pytest:pytest_defs.bzl", "pytest_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files([
    "LICENSE",
])

#######  DETECTORS #######
pytype_strict_library(
    name = "base_detector",
    srcs = ["detectors/base_detector.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "base_interpreter",
    srcs = ["detectors/base_interpreter.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "isolation_forest_detector",
    srcs = ["detectors/isolation_forest_detector.py"],
    srcs_version = "PY3",
    deps = [
        ":base_detector",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/sklearn",
    ],
)

pytest_test(
    name = "isolation_forest_detector_test",
    python_version = "PY3",
    srcs_version = "PY3",
    tests = ["tests/isolation_forest_detector_test.py"],
    deps = [
        ":evaluation_utils",
        ":gaussian_mixture_dataset",
        ":isolation_forest_detector",
    ],
)

pytype_strict_library(
    name = "one_class_svm",
    srcs = ["detectors/one_class_svm.py"],
    srcs_version = "PY3",
    deps = [
        ":base_detector",
        ":sample_utils",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/sklearn",
    ],
)

pytest_test(
    name = "one_class_svm_test",
    python_version = "PY3",
    srcs_version = "PY3",
    tests = ["tests/one_class_svm_test.py"],
    deps = [
        ":evaluation_utils",
        ":gaussian_mixture_dataset",
        ":one_class_svm",
    ],
)

pytype_strict_library(
    name = "neg_sample_random_forest",
    srcs = ["detectors/neg_sample_random_forest.py"],
    srcs_version = "PY3",
    deps = [
        ":base_detector",
        ":sample_utils",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/sklearn",
    ],
)

pytest_test(
    name = "neg_sample_random_forest_test",
    # TODO(b/168828415): Fix flaky issues.
    flaky = True,
    python_version = "PY3",
    srcs_version = "PY3",
    tests = ["tests/neg_sample_random_forest_test.py"],
    deps = [
        ":evaluation_utils",
        ":gaussian_mixture_dataset",
        ":neg_sample_random_forest",
    ],
)

pytype_strict_library(
    name = "neg_sample_neural_net_detector",
    srcs = ["detectors/neg_sample_neural_net_detector.py"],
    srcs_version = "PY3",
    deps = [
        ":base_detector",
        ":base_interpreter",
        ":sample_utils",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

pytest_test(
    name = "neg_sample_neural_net_detector_test",
    # TODO(b/168828415): Fix flaky issues.
    flaky = True,
    python_version = "PY3",
    srcs_version = "PY3",
    tests = ["tests/neg_sample_neural_net_detector_test.py"],
    deps = [
        ":evaluation_utils",
        ":gaussian_mixture_dataset",
        ":neg_sample_neural_net_detector",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "integrated_gradients_interpreter",
    srcs = ["detectors/integrated_gradients_interpreter.py"],
    srcs_version = "PY3",
    deps = [
        ":base_interpreter",
        ":sample_utils",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/scipy",
        "//third_party/py/tensorflow",
    ],
)

pytest_test(
    name = "integrated_gradients_interpreter_test",
    data = [
        "tests/test_data/model-multivariate-ad/saved_model.pb",
        "tests/test_data/model-multivariate-ad/variables/variables.data-00000-of-00001",
        "tests/test_data/model-multivariate-ad/variables/variables.index",
        "tests/test_data/positive_sample.csv",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tests = ["tests/integrated_gradients_interpreter_test.py"],
    deps = [
        ":integrated_gradients_interpreter",
        ":sample_utils",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

####### DATASETS #######
pytype_strict_library(
    name = "base_dataset",
    srcs = ["datasets/base_dataset.py"],
    srcs_version = "PY3",
    deps = [
        ":file_utils",
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "gaussian_mixture_dataset",
    srcs = ["datasets/gaussian_mixture_dataset.py"],
    srcs_version = "PY3",
    deps = [
        ":base_dataset",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_strict_test(
    name = "gaussian_mixture_dataset_test",
    srcs = ["tests/gaussian_mixture_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gaussian_mixture_dataset",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "smart_buildings_dataset",
    srcs = ["datasets/smart_buildings_dataset.py"],
    srcs_version = "PY3",
    deps = [
        ":base_dataset",
        ":datasets_data",
        ":file_utils",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "datasets_data",
    srcs = ["datasets/data/__init__.py"],
    data = [
        "datasets/data/anomaly_detection_sample_1577622599.csv",
        "datasets/data/anomaly_detection_sample_1577622599_README.md",
        "datasets/data/forestcover_README.md",
    ],
    srcs_version = "PY3",
)

pytest_test(
    name = "smart_buildings_dataset_test",
    python_version = "PY3",
    srcs_version = "PY3",
    tests = ["tests/smart_buildings_dataset_test.py"],
    deps = [
        ":smart_buildings_dataset",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "forestcover_dataset",
    srcs = ["datasets/forestcover_dataset.py"],
    data = [
        "datasets/checksum/forest_cover.txt",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_dataset",
        ":datasets_data",
        ":file_utils",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_datasets:public_api",
    ],
)

pytest_test(
    name = "forestcover_dataset_test",
    data = [
        "tests/test_data/covtype.test.data",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tests = ["tests/forestcover_dataset_test.py"],
    deps = [
        ":forestcover_dataset",
        "//third_party/py/absl/logging",
        "//third_party/py/importlib_resources",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

####### UTILS #######
pytype_strict_library(
    name = "sample_utils",
    srcs = ["utils/sample_utils.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

pytest_test(
    name = "sample_utils_test",
    python_version = "PY3",
    srcs_version = "PY3",
    tests = ["tests/sample_utils_test.py"],
    deps = [
        ":sample_utils",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "evaluation_utils",
    srcs = ["utils/evaluation_utils.py"],
    srcs_version = "PY3",
    deps = ["//third_party/py/sklearn"],
)

pytest_test(
    name = "evaluation_utils_test",
    python_version = "PY3",
    srcs_version = "PY3",
    tests = ["tests/evaluation_utils_test.py"],
    deps = [
        ":evaluation_utils",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "file_utils",
    srcs = ["utils/file_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/attr",
        "//third_party/py/importlib_resources",
        "//third_party/py/tensorflow",
        "//third_party/py/typing_extensions",
    ],
)

pytest_test(
    name = "file_utils_test",
    data = [
        "tests/test_data/text_file.txt",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tests = [
        "tests/file_utils_test.py",
    ],
    deps = [
        ":file_utils",
        ":test_data_package",
    ],
)

# This target should be included when using package resources because
# importlib.resources requires that resources distributed as part of a package
# have an `__init__.py` in the directory and parent directories.
pytype_strict_library(
    name = "test_data_package",
    testonly = True,
    srcs = [
        "tests/__init__.py",
        "tests/test_data/__init__.py",
    ],
    srcs_version = "PY3",
)

pytype_strict_library(
    name = "madi",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":evaluation_utils",
        ":file_utils",
        ":forestcover_dataset",
        ":gaussian_mixture_dataset",
        ":integrated_gradients_interpreter",
        ":isolation_forest_detector",
        ":neg_sample_neural_net_detector",
        ":neg_sample_random_forest",
        ":one_class_svm",
        ":sample_utils",
        ":smart_buildings_dataset",
    ],
)

# BEGIN GOOGLE-INTERNAL
colab_binary(
    # The name of this binary. (Name; optional; defaults to "notebook").
    name = "notebook",

    # Data files to be baked into this binary. (List of labels; optional;
    # defaults to []).
    data = [
        "datasets/checksum/forest_cover.txt",
        "datasets/data/anomaly_detection_sample_1577622599.csv",
        "datasets/data/anomaly_detection_sample_1577622599_README.md",
    ],

    # List of source [Python] files that will be executed on kernel start
    # up, in the context of this notebook. (List of labels; optional;
    # defaults to []).
    kernel_init = [],

    # .par options passed directly to the py_binary. (List of strings;
    # optional; defaults to None).
    paropts = None,

    # Ensures the binary is Python 3.
    python_version = "PY3",
    srcs_version = "PY3",

    # Same as *.visibility. (List of labels; optional; defaults to None).
    visibility = ["//visibility:public"],

    # List of dependencies for this notebook. Should be python libraries.
    # (List of labels; optional; defaults to []).
    deps = [
        # Note that all the dm_notebook3 dependencies are already added by the
        # extends field above.
        # Additional custom dependencies can be added here.
        ":madi",
    ],
)

# END GOOGLE-INTERNAL
