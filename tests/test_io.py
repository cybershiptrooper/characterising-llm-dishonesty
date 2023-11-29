from utils.io import *
import os

def test_get_labels_from_response():
    response_file = os.path.join(os.path.dirname(__file__), "dummy_response.txt")
    with open(response_file) as f:
        response = f.read()
    
    labels = get_labels_from_response(response)
    assert labels == [True, True, True, False]
   