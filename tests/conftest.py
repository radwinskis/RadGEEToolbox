# tests/conftest.py
import os, json, tempfile, pytest, ee

def pytest_sessionstart(session):
    """Runs once before any tests. Initialize EE using Service Account creds if present."""
    key_json = os.getenv("GEE_SA_KEY")
    project  = os.getenv("GEE_PROJECT")

    # If secrets are missing (e.g., forked PRs where your secrets aren’t exposed), skip EE tests.
    if not key_json or not project:
        # Skip the *whole session* gracefully; non-EE unit tests would still run if you have any.
        pytest.skip("No Earth Engine credentials in env; skipping EE-dependent tests.")

    # Write the JSON to a temp file for ee.ServiceAccountCredentials
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(key_json)
        key_path = f.name

    email = json.loads(key_json)["client_email"]
    creds = ee.ServiceAccountCredentials(email, key_path)
    ee.Initialize(creds, project=project)

    # Quick sanity check (raises if auth failed)
    assert ee.Number(1).getInfo() == 1
