from io import StringIO
from discordai_modelizer import openai as openai_wrapper
from . import expected_values
from .conftest import list_dict_comp


def test_model_list():
    models = openai_wrapper.list_models()
    for o in models:
        assert o in expected_values.list_module_expected


def test_model_list_full():
    models = openai_wrapper.list_models(full=True)
    for o in models:
        assert o in expected_values.list_module_expected_full


def test_job_list():
    jobs = openai_wrapper.list_jobs()
    for o in jobs:
        assert o in expected_values.list_job_expected


def test_job_list_full():
    jobs = openai_wrapper.list_jobs(full=True)
    for o in jobs:
        assert o in expected_values.list_job_expected_full


def test_job_info():
    info = openai_wrapper.get_job_info("ftjob-i2IyeV2xbLCSrYq45kTKSdwE")
    assert expected_values.job_info_expected == info


def test_job_events():
    events = openai_wrapper.get_job_events("ftjob-i2IyeV2xbLCSrYq45kTKSdwE")
    list_dict_comp(expected_values.job_events_expected, events)


def test_job_cancel():
    cancel = openai_wrapper.cancel_job("ftjob-i2IyeV2xbLCSrYq45kTKSdwE")
    assert expected_values.job_cancel_expected == cancel


def test_delete_model():
    delete = openai_wrapper.delete_model("whisper-1", force=True)
    assert expected_values.delete_model_expected == delete


def test_delete_model_cancel(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", StringIO("N\n"))
    openai_wrapper.delete_model("whisper-1")
    stdout = capsys.readouterr()
    assert "Cancelling model deletion..." in stdout.out
