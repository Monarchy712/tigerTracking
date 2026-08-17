"""Export download endpoints — served from disk, no pipeline run needed."""

import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    from src.api.routes import router
    from src.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _write_bundle(tmp_path, run_id=1):
    bundle_dir = tmp_path / "exports" / f"run_{run_id}" / "mstripes"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "capture_records.csv").write_text("TrapID,PhotoID\nPEN01,10\n")
    (bundle_dir / "manifest.json").write_text('{"run_id": 1}')
    return bundle_dir


def test_mstripes_download_returns_zip_of_bundle(client, tmp_path):
    _write_bundle(tmp_path)

    response = client.get("/exports/1/mstripes")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "mstripes_run_1.zip" in response.headers["content-disposition"]

    archive = tmp_path / "downloaded.zip"
    archive.write_bytes(response.content)
    with zipfile.ZipFile(archive) as zf:
        assert set(zf.namelist()) == {"capture_records.csv", "manifest.json"}


def test_mstripes_download_404_when_bundle_missing(client):
    response = client.get("/exports/99/mstripes")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
