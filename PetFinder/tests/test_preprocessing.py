import pytest
from src.preprocessing import load_csv

url = "http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip"

@pytest.fixture
def csv():
    return load_csv(url)
    
def test_csv_contents(csv):
    print(csv)
    assert 0
    