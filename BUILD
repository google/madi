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

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files([
    "LICENSE",
])

#######  DETECTORS #######
py_library(
    name = "base_detector",
    srcs = ["detectors/base_detector.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/pandas",
        "//third_party/py/six",
    ],
)

py_library(
    name = "base_interpreter",
    srcs = ["detectors/base_interpreter.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/six",
    ],
)

py_library(
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

py_strict_test(
    name = "isolation_forest_detector_test",
    srcs = ["detectors/isolation_forest_detector_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":evaluation_utils",
        ":gaussian_mixture_dataset",
        ":isolation_forest_detector",
        "//third_party/py/absl/testing:absltest",
    ],
)

py_library(
    name = "neg_sample_neural_net_detector",
    srcs = ["detectors/neg_sample_neural_net_detector.py"],
    srcs_version = "PY3",
    deps = [
        ":base_detector",
        ":base_interpreter",
        ":sample_utils",
        "//third_party/py/absl/logging",
        "//third_party/py/keras",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "neg_sample_neural_net_detector_test",
    srcs = ["detectors/neg_sample_neural_net_detector_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":evaluation_utils",
        ":gaussian_mixture_dataset",
        ":neg_sample_neural_net_detector",
        "//third_party/py/absl/logging",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
    ],
)

####### DATASETS #######
py_library(
    name = "base_dataset",
    srcs = ["datasets/base_dataset.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/pandas",
        "//third_party/py/six",
        "//third_party/py/tensorflow",
    ],
)

py_library(
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
    srcs = ["datasets/gaussian_mixture_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gaussian_mixture_dataset",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "smart_buildings_dataset",
    srcs = ["datasets/smart_buildings_dataset.py"],
    data = [
        "datasets/data/anomaly_detection_sample_1577622599.csv",
        "datasets/data/anomaly_detection_sample_1577622599_README.md",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_dataset",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "smart_buildings_dataset_test",
    srcs = ["datasets/smart_buildings_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":smart_buildings_dataset",
        "//third_party/py/absl/logging",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_library(
    name = "forestcover_dataset",
    srcs = ["datasets/forestcover_dataset.py"],
    data = [
        "datasets/data/forestcover_README.md",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_dataset",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_datasets:public_api",
    ],
)

py_strict_test(
    name = "forestcover_dataset_test",
    srcs = ["datasets/forestcover_dataset_test.py"],
    data = [
        "datasets/data/covtype.test.data",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":forestcover_dataset",
        "//third_party/py/absl/logging",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

####### UTILS #######
py_library(
    name = "sample_utils",
    srcs = ["utils/sample_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_strict_test(
    name = "sample_utils_test",
    srcs = ["utils/sample_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sample_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_library(
    name = "evaluation_utils",
    srcs = ["utils/evaluation_utils.py"],
    srcs_version = "PY3",
    deps = ["//third_party/py/sklearn"],
)

py_strict_test(
    name = "evaluation_utils_test",
    srcs = ["utils/evaluation_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":evaluation_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/numpy",
    ],
)

# BEGIN GOOGLE-INTERNAL
# END GOOGLE-INTERNAL
