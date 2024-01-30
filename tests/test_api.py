import pytest
import requests

# from api.api import list_customers

API_URL = "http://127.0.0.1:8000"


def test_list_customers():
    response = requests.get(f"{API_URL}/customers")
    status_code = response.status_code
    content = response.json()
    # test dtype and shape
    assert isinstance(content, list)
    assert len(content) > 50
    # test specific value
    assert 100006 in content


@pytest.mark.parametrize(
    "customer_id, expected_code", [(0, 404), (12, 404), (100006, 200)]
)
def test_read_single_customer(customer_id, expected_code):
    response = requests.get(f"{API_URL}/customers/{customer_id}")
    status_code = response.status_code
    content = response.json()
    assert status_code == expected_code
    if expected_code == 200:
        assert isinstance(content, dict)
        assert len(content) > 10


def test_all_customers_stats():
    response = requests.get(f"{API_URL}/customers_stats")
    status_code = response.status_code
    content = response.json()
    # test dtype and shape
    assert isinstance(content, dict)
    assert len(content) == 3  # 3 cols in describe : count, mean, std


@pytest.mark.parametrize(
    "customer_id, expected_code", [(0, 404), (12, 404), (100006, 200)]
)
def test_predict(customer_id, expected_code):
    response = requests.get(f"{API_URL}/predict/{customer_id}")
    status_code = response.status_code
    content = response.json()
    assert status_code == expected_code
    if expected_code == 200:
        assert isinstance(content, dict)
        assert len(content) == 2


@pytest.mark.parametrize(
    "customer_id, expected_code", [(0, 404), (12, 404), (100006, 200)]
)
def test_shap_values(customer_id, expected_code):
    response = requests.get(f"{API_URL}/shap/{customer_id}")
    status_code = response.status_code
    content = response.json()
    assert status_code == expected_code
    if expected_code == 200:
        assert isinstance(content, dict)
        assert len(content) == 2
