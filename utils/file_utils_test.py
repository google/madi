"""Tests for madi.utils.file_utils."""

import pathlib
import textwrap

from absl.testing import absltest
from madi.utils import file_utils


class OpenTextResourceTest(absltest.TestCase):

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

  def _create_test_file(self):
    temp_file = self.create_tempfile().full_path
    with open(temp_file, "wt", encoding="utf-8") as f:
      f.write(self._FILE_CONTENTS)

    return temp_file

  def test_str_resource(self):
    test_file = self._create_test_file()
    with file_utils.open_text_resource(test_file) as f:
      test_contents = f.read()
      self.assertEqual(test_contents, self._FILE_CONTENTS)

  def test_path_resource(self):
    test_file = pathlib.Path(self._create_test_file())
    with file_utils.open_text_resource(test_file) as f:
      test_contents = f.read()
      self.assertEqual(test_contents, self._FILE_CONTENTS)

  def test_package_resource(self):
    package_resource = file_utils.PackageResource("madi.utils.test_data",
                                                  "text_file.txt")

    with file_utils.open_text_resource(package_resource) as f:
      test_contents = f.read()
      self.assertEqual(test_contents,
                       "Sphinx of black quartz — judge my vow. 紫の猿の皿洗い機.")


if __name__ == "__main__":
  absltest.main()
