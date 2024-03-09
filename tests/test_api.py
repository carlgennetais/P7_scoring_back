import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from api.api import app

# app = FastAPI(default_response_class=ORJSONResponse)
client = TestClient(app)


def test_list_customers():
    response = client.get("/customers")
    status_code = response.status_code
    content = response.json()
    # test dtype and shape
    assert isinstance(content, list)
    assert len(content) > 50
    # test specific value
    assert 124332 in content


@pytest.mark.parametrize(
    "customer_id, expected_code", [(0, 404), (12, 404), (124332, 200), ("124332", 200)]
)
def test_read_single_customer(customer_id, expected_code):
    response = client.get(f"/customers/{customer_id}")
    status_code = response.status_code
    content = response.json()
    assert status_code == expected_code
    if expected_code == 200:
        assert isinstance(content, dict)
        assert len(content) > 10


def test_all_customers_stats():
    response = client.get("/customers_stats")
    status_code = response.status_code
    content = response.json()
    # test dtype and shape
    assert isinstance(content, dict)
    assert len(content) == 3  # 3 cols in describe : count, mean, std


@pytest.mark.parametrize(
    "customer_id, expected_code", [(0, 404), (12, 404), (124332, 200), ("124332", 200)]
)
def test_predict(customer_id, expected_code):
    response = client.get(f"/predict/{customer_id}")
    status_code = response.status_code
    content = response.json()
    assert status_code == expected_code
    if expected_code == 200:
        assert isinstance(content, int)
        # assert len(content) == 1
        assert (content == 0) | (content == 1)


@pytest.mark.parametrize(
    "customer_id, expected_code",
    [
        (0, 404),
        (12, 404),
        (124332, 200),
    ],
)
def test_shap_values(customer_id, expected_code):
    response = client.get(f"/shap/{customer_id}")
    status_code = response.status_code
    content = response.json()
    assert status_code == expected_code
    if expected_code == 200:
        assert isinstance(content, dict)
        assert (
            ("values" in content.keys())
            & ("data" in content.keys())
            & ("display_data" in content.keys())
            & ("base_values" in content.keys())
        )
