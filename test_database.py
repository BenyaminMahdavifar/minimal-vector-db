import numpy as np
import pytest
from vectordb.database import MinimalVectorDB

def test_insert_and_get():
    db = MinimalVectorDB()
    vec = np.array([1.0, 0.0, 0.0])
    metadata = {"text": "apple"}
    
    vector_id = db.insert(vec, metadata)
    
    result = db.get(vector_id)
    assert result is not None
    assert np.array_equal(result["vector"], vec)
    assert result["metadata"]["text"] == "apple"
    
    assert db.get(999) is None

def test_update():
    db = MinimalVectorDB()
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.0, 1.0])
    vector_id = db.insert(vec1, {"text": "old"})
    
    success = db.update(vector_id, new_vector=vec2, new_metadata={"text": "new"})
    assert success is True
    
    result = db.get(vector_id)
    assert np.array_equal(result["vector"], vec2)
    assert result["metadata"]["text"] == "new"
    
    assert db.update(999, new_vector=vec1) is False

def test_delete():
    db = MinimalVectorDB()
    vec = np.array([1.0, 0.0])
    vector_id = db.insert(vec, {"text": "to_delete"})
    
    success = db.delete(vector_id)
    assert success is True
    
    assert db.get(vector_id) is None
    
    assert db.delete(vector_id) is False

def test_search():
    db = MinimalVectorDB()
    
    vec_apple = np.array([1.0, 0.0, 0.0])
    vec_orange = np.array([0.0, 1.0, 0.0])
    vec_fruit = np.array([1.0, 1.0, 0.0])
    
    db.insert(vec_apple, {"text": "apple"})
    db.insert(vec_orange, {"text": "orange"})
    db.insert(vec_fruit, {"text": "fruit"})
    
    query = np.array([0.9, 0.1, 0.0])
    
    results = db.search(query, top_k=2)
    
    assert len(results) == 2
    
    top_id, top_score = results[0]
    assert db.metadata[top_id]["text"] == "apple"
    
    second_id, second_score = results[1]
    assert db.metadata[second_id]["text"] == "fruit"

def test_search_after_delete():
    db = MinimalVectorDB()
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.0, 1.0])
    
    id1 = db.insert(vec1, {"text": "keep"})
    id2 = db.insert(vec2, {"text": "delete"})
    
    db.delete(id2)
    
    results = db.search(np.array([0.0, 1.0]), top_k=5)
    
    assert len(results) == 1  
    assert results[0][0] == id1