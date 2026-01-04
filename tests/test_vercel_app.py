import json

import joblib
import numpy as np
import pytest

from src.config import METADATA_PATH, PIPELINE_PATH
from src.inference import predict as original_predict
from tests.test_app_inference import PROFILES
from vercel_app.app import app
from vercel_app.scoring import ROOT


@pytest.fixture
def client():
    return app.test_client()


def test_deployment_artifacts_match_canonical():
    assert (ROOT / "models/pipeline.joblib").read_bytes() == PIPELINE_PATH.read_bytes()
    assert (ROOT / "models/metadata.json").read_bytes() == METADATA_PATH.read_bytes()


@pytest.mark.parametrize("profile", PROFILES + [{**PROFILES[0], "ca": None, "thal": None}])
def test_api_matches_original_inference(client, profile):
    meta = json.loads(METADATA_PATH.read_text())
    bundle = {
        "meta": meta,
        "pipeline": joblib.load(PIPELINE_PATH),
        "threshold": meta["threshold"],
        "platt": meta["platt"],
    }
    expected = original_predict(bundle, profile)
    response = client.post("/api/predict", json=profile)
    assert response.status_code == 200
    actual = response.get_json()
    for key in ("probability", "raw_probability", "threshold", "raw_threshold"):
        assert actual[key] == pytest.approx(expected[key], abs=1e-12)
    assert actual["elevated"] == expected["elevated"]
    assert actual["label"] == expected["label"]
    assert actual["contributions"] == pytest.approx(expected["contributions"], abs=1e-12)
    assert [p[0] for p in actual["top"]] == [p[0] for p in expected["top"]]
    assert response.headers["Cache-Control"] == "no-store"


def _invalid(**changes):
    key = next(iter(changes))
    return pytest.param({"json": {**PROFILES[0], **changes}}, 400, key, id=f"{key}-{changes[key]}")


@pytest.mark.parametrize(
    "request_kwargs, status, error_key",
    [
        _invalid(age=28),
        _invalid(sex=3),
        _invalid(thal=0),
        _invalid(oldpeak=-1),
        _invalid(chol=None),
        _invalid(sex=True),
        _invalid(age=np.nan),
        _invalid(ca=2.5),
        pytest.param({"json": {}}, 400, None, id="empty-json"),
        pytest.param({"json": {**PROFILES[0], "extra": 1}}, 400, None, id="extra-field"),
        pytest.param({"data": "bad", "content_type": "application/json"}, 400, None, id="malformed-json"),
        pytest.param({"data": "x" * 9000, "content_type": "application/json"}, 413, None, id="payload-413"),
    ],
)
def test_rejects_invalid_missing_extra_and_malformed_inputs(client, request_kwargs, status, error_key):
    response = client.post("/api/predict", **request_kwargs)
    assert response.status_code == status
    if error_key:
        assert error_key in response.get_json()["errors"]


def test_metadata_has_ranges_defaults_and_no_private_runtime_configuration(client):
    response = client.get("/api/model")
    assert response.status_code == 200
    assert len(response.get_json()["input_ranges"]) == 13
    assert "DATABASE_URL" not in response.get_data(as_text=True)
