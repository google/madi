"""Tests for madi.utils.file_utils."""

import os
import pathlib
import textwrap

from . import test_data

from madi.utils import file_utils


class TestOpenTextResource:

  _FILE_CONTENTS = textwrap.dedent("""
  Lorem superposés valise pourparlers rêver chiots rendez-vous naissance Eiffel
  myrtille. Grèves Arc de Triomphe encore pourquoi sentiments baguette pédiluve
  une projet sentiments saperlipopette vachement le. Brume éphémère baguette
  Bordeaux en fait sommet avoir minitel.

  Nous avoir parole la nous moussant. Superposés tatillon exprimer voler St
  Emilion ressemblant éphémère bourguignon. Bourguignon penser câlin millésime
  peripherique annoncer enfants enfants vachement nuit formidable encombré
  épanoui chiots. Arc truc cacatoès lorem flâner.
  """)

  def _create_test_file(self, test_dir):
    temp_file = os.path.join(test_dir, "test_file.txt")
    with open(temp_file, "wt", encoding="utf-8") as f:
      f.write(self._FILE_CONTENTS)

    return temp_file

  def test_str_resource(self, tmpdir):
    test_file = self._create_test_file(tmpdir)
    with file_utils.open_text_resource(test_file) as f:
      test_contents = f.read()
      assert test_contents == self._FILE_CONTENTS

  def test_path_resource(self, tmpdir):
    test_file = pathlib.Path(self._create_test_file(tmpdir))
    with file_utils.open_text_resource(test_file) as f:
      test_contents = f.read()
      assert test_contents == self._FILE_CONTENTS

  def test_package_resource(self):
    package_resource = file_utils.PackageResource(test_data, "text_file.txt")

    expected = "Sphinx of black quartz — judge my vow. 紫の猿の皿洗い機."
    with file_utils.open_text_resource(package_resource) as f:
      test_contents = f.read()
      assert test_contents == expected
