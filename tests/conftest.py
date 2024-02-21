"""
Conftest file for pytest
"""

from api.api import app

from fastapi.testclient import TestClient

client = TestClient(app)
