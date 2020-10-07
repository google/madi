# Lint as: python3
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
"""Common utilities for handling data sets."""

import functools
import os
import sys
import typing

import attr
import tensorflow as tf

# Imports with fallback to backports
# pylint: disable=g-import-not-at-top
if sys.version_info >= (3, 7):
  import importlib.resources as importlib_resources
else:
  import importlib_resources  # Backport for Python <3.7

try:
  from typing import Protocol
except ImportError:
  from typing_extensions import Protocol  # Backport for Python < 3.8
# pylint: enable=g-import-not-at-top

# As of version 2020.6.26, without this, pytype in Python 3.6 complains about
# importlib.resources not having an attribute "open_text"; presumably this is
# because it doesn't have any type hints for importlib.resources until Python
# 3.7.
assert hasattr(importlib_resources, "open_text")


@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackageResource:
  """Class for representing a resource distributed as part of a package."""
  package: importlib_resources.Package
  resource: importlib_resources.Resource


class TextIOContextManager(Protocol):
  """Protocol for context managers which, when entered, return TextIO.

  This is used to enforce that consumers of functions that return
  TextIOContextManager should enter the context rather than directly using the
  resource.
  """

  def __enter__(self) -> typing.TextIO:
    ...

  def __exit__(self, type_, value, traceback) -> None:
    ...


@functools.singledispatch
def open_text_resource(path: typing.Any) -> TextIOContextManager:
  """Accesses a text resource.

  This is roughly equivalent to `open(path, "rt")`, but when passed a path-like
  resource, it use `tensorflow.io.gfile.Gfile` to open the resource, and when
  passed a `PackageResource`, it uses `importlib.resources.open_text`.

  Args:
    path: A path to a text file or a `PackageResource` referring to a text
      resource distributed as part of a Python package.

  Returns:
    A context manager which manages an open file or resource.

  Raises:
    TypeError: If `path` is not one of the allowed types.
    IOError: If it is not possible to open the resource.
  """
  raise TypeError(
      f"Function takes a path-like or PackageResource, got: {type(path)!r}")


# In Python 3.7+, the explicit type registration decorators can be with
# @open_text_resource.register and the dispatch pattern will be inferred from
# the type signature.
@open_text_resource.register(str)
def _(path: str) -> TextIOContextManager:
  if not tf.io.gfile.exists(path):
    raise IOError(f"{path} does not exist.")

  # As of version 2.2.0, `tensorflow.io.gfile.Gfile`'s type stubs do not
  # include the __enter__ and __exit__ methods, but they do exist, so we will
  # typecast the result here.
  return typing.cast(TextIOContextManager, tf.io.gfile.GFile(path))


@open_text_resource.register(os.PathLike)
def _(path: os.PathLike) -> TextIOContextManager:
  return open_text_resource(os.fspath(path))


@open_text_resource.register(PackageResource)
def _(path: PackageResource) -> TextIOContextManager:
  return importlib_resources.open_text(path.package, path.resource)
