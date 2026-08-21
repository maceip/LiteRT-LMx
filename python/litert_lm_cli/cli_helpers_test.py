# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from unittest import mock

from absl.testing import absltest
import click
from prompt_toolkit.keys import Keys
import questionary

from litert_lm_cli import cli_helpers
from litert_lm_cli import huggingface_download
from litert_lm_cli import model


class PromptSelectionTest(absltest.TestCase):

  @mock.patch.object(questionary, "select", autospec=True)
  def test_prompt_selection_success(self, mock_select):
    mock_select_instance = mock.MagicMock()
    mock_select_instance.ask.return_value = "choice1"
    mock_select.return_value = mock_select_instance

    result = cli_helpers.prompt_selection(["choice1", "choice2"])
    self.assertEqual(result, "choice1")
    mock_select.assert_called_once_with(
        "Please select an option",
        choices=["choice1", "choice2"],
        style=cli_helpers._QUESTIONARY_STYLE,
    )
    mock_select_instance.application.key_bindings.add.assert_has_calls(
        [
            mock.call(Keys.Escape, eager=True),
            mock.call(Keys.ControlD, eager=True),
        ],
        any_order=True,
    )

  @mock.patch.object(questionary, "select", autospec=True)
  def test_prompt_selection_abort(self, mock_select):
    mock_select_instance = mock.MagicMock()
    mock_select_instance.ask.return_value = None
    mock_select.return_value = mock_select_instance

    with self.assertRaises(click.Abort):
      cli_helpers.prompt_selection(["choice1", "choice2"])
    mock_select_instance.application.key_bindings.add.assert_has_calls(
        [
            mock.call(Keys.Escape, eager=True),
            mock.call(Keys.ControlD, eager=True),
        ],
        any_order=True,
    )


class ResolveModelFileTest(absltest.TestCase):

  @mock.patch.object(questionary, "select", autospec=True)
  @mock.patch.object(huggingface_download, "list_litertlm_files", autospec=True)
  def test_resolve_hf_interactive_select(self, mock_list, mock_select):
    mock_list.return_value = ["model1.litertlm", "model2.litertlm"]
    mock_select_instance = mock.MagicMock()
    mock_select_instance.ask.return_value = "model2.litertlm"
    mock_select.return_value = mock_select_instance

    result = cli_helpers.resolve_model_file(
        from_huggingface_repo="org/repo",
        huggingface_token="token",
    )

    self.assertEqual(result, "model2.litertlm")
    mock_list.assert_called_once_with(repo_id="org/repo", token="token")
    mock_select.assert_called_once_with(
        "Please select a model file",
        choices=["model1.litertlm", "model2.litertlm"],
        style=cli_helpers._QUESTIONARY_STYLE,
    )

  @mock.patch.object(huggingface_download, "list_litertlm_files", autospec=True)
  def test_resolve_hf_single_file_auto_select(self, mock_list):
    mock_list.return_value = ["only_one.litertlm"]

    result = cli_helpers.resolve_model_file(
        from_huggingface_repo="org/repo",
        huggingface_token="token",
    )

    self.assertEqual(result, "only_one.litertlm")
    mock_list.assert_called_once_with(repo_id="org/repo", token="token")

  @mock.patch.object(huggingface_download, "list_litertlm_files", autospec=True)
  def test_resolve_hf_no_files_error(self, mock_list):
    mock_list.return_value = []

    with self.assertRaises(click.ClickException) as context:
      cli_helpers.resolve_model_file(
          from_huggingface_repo="org/repo",
          huggingface_token="token",
      )
    self.assertIn("No .litertlm files found", str(context.exception))

  @mock.patch.object(questionary, "select", autospec=True)
  @mock.patch.object(huggingface_download, "list_litertlm_files", autospec=True)
  def test_resolve_hf_select_abort(self, mock_list, mock_select):
    mock_list.return_value = ["model1.litertlm", "model2.litertlm"]
    mock_select_instance = mock.MagicMock()
    mock_select_instance.ask.return_value = None
    mock_select.return_value = mock_select_instance

    with self.assertRaises(click.Abort):
      cli_helpers.resolve_model_file(
          from_huggingface_repo="org/repo",
          huggingface_token="token",
      )

  @mock.patch.object(questionary, "select", autospec=True)
  @mock.patch.object(model.Model, "get_all_models", autospec=True)
  def test_resolve_local_interactive_select(self, mock_get_all, mock_select):
    mock_model1 = mock.create_autospec(model.Model, instance=True)
    mock_model1.model_id = "local_model1"
    mock_model2 = mock.create_autospec(model.Model, instance=True)
    mock_model2.model_id = "local_model2"
    mock_get_all.return_value = [mock_model1, mock_model2]

    mock_select_instance = mock.MagicMock()
    mock_select_instance.ask.return_value = "local_model2"
    mock_select.return_value = mock_select_instance

    result = cli_helpers.resolve_model_file(
        from_huggingface_repo=None,
        huggingface_token=None,
    )

    self.assertEqual(result, "local_model2")
    mock_get_all.assert_called_once()
    mock_select.assert_called_once_with(
        "Please select a model",
        choices=["local_model1", "local_model2"],
        style=cli_helpers._QUESTIONARY_STYLE,
    )

  @mock.patch.object(questionary, "select", autospec=True)
  @mock.patch.object(model.Model, "get_all_models", autospec=True)
  def test_resolve_local_interactive_select_abort(
      self, mock_get_all, mock_select
  ):
    mock_model1 = mock.create_autospec(model.Model, instance=True)
    mock_model1.model_id = "local_model1"
    mock_model2 = mock.create_autospec(model.Model, instance=True)
    mock_model2.model_id = "local_model2"
    mock_get_all.return_value = [mock_model1, mock_model2]

    mock_select_instance = mock.MagicMock()
    mock_select_instance.ask.return_value = None
    mock_select.return_value = mock_select_instance

    with self.assertRaises(click.Abort):
      cli_helpers.resolve_model_file(
          from_huggingface_repo=None,
          huggingface_token=None,
      )

  @mock.patch.object(model.Model, "get_all_models", autospec=True)
  def test_resolve_local_no_models_error(self, mock_get_all):
    mock_get_all.return_value = []

    with self.assertRaises(click.ClickException) as context:
      cli_helpers.resolve_model_file(
          from_huggingface_repo=None,
          huggingface_token=None,
      )
    self.assertIn("No imported models found", str(context.exception))

  @mock.patch.object(model.Model, "get_all_models", autospec=True)
  def test_resolve_local_single_model_auto_select(self, mock_get_all):
    mock_model1 = mock.create_autospec(model.Model, instance=True)
    mock_model1.model_id = "only_one"
    mock_get_all.return_value = [mock_model1]

    result = cli_helpers.resolve_model_file(
        from_huggingface_repo=None,
        huggingface_token=None,
    )
    self.assertEqual(result, "only_one")
    mock_get_all.assert_called_once()


if __name__ == "__main__":
  absltest.main()
