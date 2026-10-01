# Minimal Vector DB

A tiny, educational vector database implemented from scratch with Python and NumPy.

The project focuses on the core mechanics behind a vector database without relying on a dedicated vector-search engine. It stores vectors and metadata in memory, provides basic CRUD operations, and performs vectorized cosine-similarity search using NumPy.

> **Status:** Educational / experimental project  
> **Storage:** In-memory only  
> **Search:** Exact brute-force cosine similarity  
> **Core dependency:** NumPy

---

## Overview

Vector databases are commonly used in semantic search, recommendation systems, Retrieval-Augmented Generation (RAG), image similarity, and other machine-learning applications.

At their core, many vector-search systems need to solve a simple problem:

> Given a query vector, find the stored vectors that are most similar to it.

This repository implements that core idea in a deliberately small codebase.

The project provides:

- Vector insertion
- Integer vector IDs
- Metadata storage
- Vector retrieval
- Vector/metadata updates
- Deletion
- Exact cosine-similarity search
- Top-`k` nearest-neighbor retrieval
- NumPy-vectorized similarity computation
- Automated tests
- A Jupyter notebook demonstrating the system with 1,000 synthetic 128-dimensional vectors

The goal is not to replace production systems such as FAISS, Qdrant, Milvus, Weaviate, or Pinecone. Instead, it provides a compact implementation that makes the fundamental data structures and search operation easy to inspect and understand.

---

## Why this project?

A production vector database contains many layers of engineering:

- Persistent storage
- Index structures
- Approximate nearest-neighbor algorithms
- Filtering
- Concurrency
- Distributed storage
- Replication
- Durability
- APIs
- Monitoring
- Memory management

That complexity can hide the basic mechanism of vector search.

This project intentionally removes those layers.

The implementation makes the following pipeline explicit:

```text
Embedding / Vector
       │
       ▼
   Insert
       │
       ├── Vector
       ├── ID
       └── Metadata
       │
       ▼
 In-memory storage
       │
       ▼
 Query Vector
       │
       ▼
 Cosine Similarity
       │
       ▼
 Sort scores
       │
       ▼
   Top-k IDs
```

---

## Features

### 1. Insert

Store a vector together with arbitrary metadata.

Each inserted vector receives an automatically generated integer ID.

### 2. Get

Retrieve a stored vector and its metadata using its ID.

### 3. Update

Update either:

- the vector
- the metadata
- both

### 4. Delete

Remove a vector from the active database.

Deleted entries are excluded from future searches.

### 5. Similarity Search

Search uses cosine similarity:

[
mathrm{cos}(x,y)
=
\frac{x \cdot y}
{\|x\|\|y\|}
]

Higher scores indicate greater directional similarity.

### 6. NumPy Vectorization

The search operation converts the active vectors into a NumPy array and computes similarities using vectorized matrix operations rather than calculating every vector independently in Python.

### 7. Tests

The repository includes tests for:

- insertion and retrieval
- updates
- deletion
- similarity search
- searching after deletion

---

## Repository Structure

```text
minimal-vector-db/
│
├── vectordb/
│   ├── __init__.py
│   └── database.py
│
├── demo.ipynb
├── test_database.py
├── .gitignore
└── README.md
```

### `vectordb/database.py`

Contains the `MinimalVectorDB` implementation.

### `test_database.py`

Contains the automated test suite covering the main database operations.

### `demo.ipynb`

A runnable demonstration that:

1. Generates 1,000 synthetic 128-dimensional vectors.
2. Inserts them into the database.
3. Performs cosine-similarity search.
4. Retrieves the top 5 results.
5. Deletes a result.
6. Searches again to demonstrate that deleted vectors are no longer returned.

---

# Installation

## Requirements

- Python 3.9+
- NumPy
- pytest for running the test suite
- Jupyter Notebook or JupyterLab for the demo notebook

The core implementation only requires NumPy.

## Clone the repository

```bash
git clone https://github.com/BenyaminMahdavifar/minimal-vector-db.git
cd minimal-vector-db
```

## Create a virtual environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install numpy pytest jupyter
```

---

# Quick Start

Create a database and insert a vector:

```python
import numpy as np

from vectordb.database import MinimalVectorDB

db = MinimalVectorDB()

vector = np.array([1.0, 0.0, 0.0])
metadata = {
    "text": "apple",
    "category": "fruit"
}

vector_id = db.insert(vector, metadata)

print(vector_id)
```

The first inserted vector receives ID `0`.

---

# Retrieving a Vector

Use `get()` with the vector ID:

```python
result = db.get(vector_id)

print(result)
```

The returned object has the following structure:

```python
{
    "vector": ...,
    "metadata": ...
}
```

For example:

```python
{
    "vector": array([1., 0., 0.]),
    "metadata": {
        "text": "apple",
        "category": "fruit"
    }
}
```

If the ID does not exist, `get()` returns `None`.

---

# Updating Data

## Update the vector

```python
new_vector = np.array([0.9, 0.1, 0.0])

db.update(
    vector_id,
    new_vector=new_vector
)
```

## Update metadata

```python
db.update(
    vector_id,
    new_metadata={
        "text": "green apple",
        "category": "fruit"
    }
)
```

## Update both

```python
db.update(
    vector_id,
    new_vector=np.array([0.8, 0.2, 0.0]),
    new_metadata={
        "text": "green apple",
        "category": "fruit"
    }
)
```

The method returns:

- `True` if the ID exists and the update is applied.
- `False` if the ID does not exist.

---

# Deleting Data

Delete an existing vector:

```python
success = db.delete(vector_id)

print(success)
```

The method returns `True` when the vector exists and is deleted.

Deleting an already deleted or unknown ID returns `False`.

Deleted vectors are not included in future similarity searches.

---

# Similarity Search

The main feature is nearest-neighbor search using cosine similarity.

Insert several vectors:

```python
db.insert(
    np.array([1.0, 0.0, 0.0]),
    {"text": "apple"}
)

db.insert(
    np.array([0.0, 1.0, 0.0]),
    {"text": "orange"}
)

db.insert(
    np.array([1.0, 1.0, 0.0]),
    {"text": "fruit"}
)
```

Then search:

```python
query = np.array([0.9, 0.1, 0.0])

results = db.search(query, top_k=2)

print(results)
```

The result format is:

```python
[
    (vector_id, similarity_score),
    ...
]
```

For example:

```text
[
    (0, 0.9939),
    (2, 0.7809)
]
```

The exact numerical scores can vary depending on the input vectors.

To retrieve the metadata:

```python
for vector_id, score in results:
    item = db.get(vector_id)

    print(
        vector_id,
        score,
        item["metadata"]
    )
```

---

# How Search Works

The search implementation follows five main steps.

## 1. Select active IDs

The database maintains an ID-to-storage-index mapping:

```python
self.id_map
```

Only active IDs are considered during search.

---

## 2. Build the active vector matrix

The stored vectors corresponding to active IDs are collected into a NumPy array.

Conceptually:

```text
[v₁]
[v₂]
[v₃]
...
[vₙ]
```

This creates an (n \times d) matrix where:

- (n) = number of active vectors
- (d) = vector dimension

---

## 3. Normalize the vectors

Each vector is normalized using its L2 norm:

[
\hat{x} = \frac{x}{\|x\|}
]

The query vector is normalized in the same way.

Zero-norm vectors are handled safely so that division by zero does not occur.

---

## 4. Compute similarities

After normalization, cosine similarity becomes a dot product:

[
\mathrm{cos}(x,q)
=
\hat{x}\cdot\hat{q}
]

The implementation computes all active-vector similarities using a NumPy matrix multiplication.

Conceptually:

```text
[n × d] · [d × 1]
        │
        ▼
      [n × 1]
```

---

## 5. Select top-k results

The similarity scores are sorted in descending order and the first `top_k` entries are returned.

```python
np.argsort(similarities)[::-1][:top_k]
```

The result is therefore an exact top-(k) search over all active vectors.

---

# Internal Data Model

The database uses three primary structures.

## Vector storage

```python
self.vectors = []
```

Vectors are stored in a Python list.

The list position acts as an internal storage index.

---

## ID map

```python
self.id_map = {}
```

Maps public vector IDs to their positions in `self.vectors`.

Conceptually:

```text
vector ID → storage index
```

Example:

```python
{
    0: 0,
    1: 1,
    4: 3
}
```

This allows IDs to remain stable even when other entries are deleted.

---

## Metadata

```python
self.metadata = {}
```

Stores metadata separately using the vector ID as the key.

Conceptually:

```text
vector ID → metadata
```

Example:

```python
{
    0: {"text": "apple"},
    1: {"text": "orange"}
}
```

---

# Deletion Design

Deletion does not physically remove an item from the `vectors` list.

Instead, the implementation:

1. Finds the internal storage index.
2. Replaces the vector with `None`.
3. Removes the ID from `id_map`.
4. Removes its metadata.

This preserves the positions of other stored vectors.

For example:

```text
Before deletion:

vectors
│
├── index 0 → vector A
├── index 1 → vector B
└── index 2 → vector C

After deleting B:

vectors
│
├── index 0 → vector A
├── index 1 → None
└── index 2 → vector C
```

Because `id_map` no longer contains B's ID, search ignores the deleted entry.

This is simple and keeps the implementation easy to understand, but it also means deleted slots remain allocated in the underlying list.

---

# API Reference

## `MinimalVectorDB()`

Creates an empty in-memory vector database.

```python
db = MinimalVectorDB()
```

### Internal state

| Attribute | Purpose |
|---|---|
| `vectors` | Stores vector objects |
| `metadata` | Maps IDs to metadata |
| `id_map` | Maps IDs to vector-list indexes |
| `next_id` | Generates new IDs |

---

## `insert(vector, metadata)`

Adds a vector and its metadata.

### Parameters

| Parameter | Description |
|---|---|
| `vector` | Vector-like object accepted by NumPy operations |
| `metadata` | Metadata associated with the vector |

### Returns

`int` — newly assigned vector ID.

### Example

```python
vector_id = db.insert(
    np.array([1.0, 0.0]),
    {"text": "example"}
)
```

---

## `get(vector_id)`

Retrieves an active vector.

### Returns

A dictionary containing:

```python
{
    "vector": vector,
    "metadata": metadata
}
```

Returns `None` when the ID does not exist.

---

## `update(vector_id, new_vector=None, new_metadata=None)`

Updates an existing vector.

### Parameters

- `vector_id`: ID of the vector.
- `new_vector`: Optional replacement vector.
- `new_metadata`: Optional replacement metadata.

### Returns

`True` if successful, otherwise `False`.

---

## `delete(vector_id)`

Deletes an active vector.

### Returns

`True` if deleted, otherwise `False`.

---

## `search(query_vector, top_k=5)`

Returns the most similar active vectors.

### Parameters

| Parameter | Description |
|---|---|
| `query_vector` | Query vector |
| `top_k` | Maximum number of results |

### Returns

A list of:

```python
(vector_id, similarity_score)
```

sorted by similarity in descending order.

If the database is empty or the query vector has zero magnitude, the method returns an empty list.

---

# Complexity

Let:

- (N) = number of active vectors
- (D) = vector dimension
- (K) = requested number of results

## Insert

Appending the vector and updating the dictionaries is approximately:

[
O(1)
]

excluding any cost associated with external vector construction.

## Get

Dictionary lookup plus list indexing:

[
O(1)
]

## Update

Approximately:

[
O(1)
]

for the database operations themselves.

## Delete

Approximately:

[
O(1)
]

for the dictionary/list operations.

## Search

The similarity computation requires processing all active vectors:

[
O(ND)
]

The sorting step is:

[
O(N\log N)
]

Therefore the current search strategy is fundamentally a **linear scan** over the database.

This is exact nearest-neighbor search, not approximate nearest-neighbor (ANN) search.

---

# Performance Characteristics

The implementation uses NumPy to move the expensive similarity computation into vectorized numerical operations.

For example, instead of conceptually doing:

```python
for vector in vectors:
    calculate_similarity(vector, query)
```

the implementation constructs a matrix and performs a vectorized dot product.

This can be considerably faster than a pure Python loop for moderate in-memory datasets.

However, vectorization does **not** change the asymptotic search complexity.

The system still compares the query against every active vector.

---

# Demonstration

The repository includes `demo.ipynb`.

The notebook creates:

- **1,000 vectors**
- **128 dimensions per vector**
- synthetic floating-point values

It then:

1. Inserts all vectors.
2. Creates a query similar to vector 42.
3. Searches for the top 5 results.
4. Displays IDs, metadata, and similarity scores.
5. Deletes the best result.
6. Runs the search again.

This provides a small end-to-end example of how an embedding store can be used.

Run it with:

```bash
jupyter notebook demo.ipynb
```

or:

```bash
jupyter lab
```

---

# Running Tests

Install pytest:

```bash
pip install pytest
```

Run the test suite:

```bash
pytest -q
```

The current tests cover the main behavior of the database:

```text
✓ insert + get
✓ update
✓ delete
✓ similarity search
✓ search after deletion
```

---

# Example: Mini Semantic Search

A real embedding model can produce vectors that are stored in this database.

For example:

```python
documents = [
    "Python is a programming language.",
    "Neural networks are machine learning models.",
    "Vector databases support semantic search."
]
```

An embedding model could convert each document into:

```text
Document → Embedding Vector
```

Those vectors could then be inserted:

```python
db.insert(
    embedding,
    {
        "text": document,
        "source": "example"
    }
)
```

A query can be embedded using the same model and searched:

```python
results = db.search(
    query_embedding,
    top_k=5
)
```

The database itself does not generate embeddings. It only stores vectors and performs similarity search.

---

# Design Philosophy

The project intentionally follows a minimal design.

### No external vector database

The implementation does not depend on a database server or vector-search service.

### No ANN index

There is no HNSW, IVF, PQ, tree index, or other approximate nearest-neighbor structure.

### No persistence

All data exists only in the Python process.

### No framework

The core database is implemented directly with Python and NumPy.

### Readable over feature-heavy

The implementation is small enough to inspect from beginning to end.

This makes the project useful for learning how vector databases work internally before moving to production systems.

---

# Limitations

This implementation is intentionally minimal and should not be treated as a production database.

## In-memory only

All data is lost when the Python process exits.

There is currently no:

- disk persistence
- WAL
- snapshotting
- recovery

## Brute-force search

Every search compares the query against all active vectors.

This becomes increasingly expensive as (N) grows.

## No filtering

Search cannot currently filter results by metadata.

For example, there is no built-in operation such as:

```text
search(query, category="science")
```

## No batch API

The current API inserts vectors one at a time.

## No concurrency model

The implementation is not designed as a concurrent or multi-process database.

## No dimensionality validation

The current implementation does not explicitly enforce one fixed vector dimension at insertion time.

Applications should ensure that stored vectors are compatible with the query vectors.

## No persistence or durability guarantees

There are no transactional or durability guarantees.

## Deleted slots are retained

Deletion leaves `None` entries in the underlying vector list rather than compacting the storage.

---

# What This Project Demonstrates

Despite its small size, the repository demonstrates several important concepts used by larger vector systems:

- Vector storage
- ID management
- Metadata association
- Cosine similarity
- L2 normalization
- Matrix-vector multiplication
- Exact nearest-neighbor search
- Top-(k) retrieval
- Active/deleted record management
- Separation of vector data and metadata

It can therefore serve as a compact reference implementation for understanding the basic mechanics behind vector search.

---

# Possible Future Extensions

The current design can be extended in several directions.

## 1. Persistence

Add a storage layer using:

- SQLite
- memory-mapped files
- custom binary storage
- Parquet or another structured format

## 2. Batch insertion

Add an API such as:

```python
db.insert_many(vectors, metadata)
```

This could reduce Python-level overhead.

## 3. Metadata filtering

Support queries such as:

```python
db.search(
    query_vector,
    top_k=5,
    filter={"category": "science"}
)
```

## 4. ANN indexing

Introduce approximate nearest-neighbor algorithms such as:

- HNSW
- IVF
- Product Quantization

This would make large-scale search substantially more efficient.

## 5. Vector normalization at insertion

Vectors could optionally be normalized when inserted, allowing cosine search to use simpler dot products.

## 6. Batch search

Support multiple query vectors in one operation:

```text
Q × Vᵀ
```

where (Q) contains multiple normalized query vectors.

## 7. Better top-k selection

For large datasets, a full sort of all (N) scores is unnecessary when only a small (K) is required.

A partial-selection approach could reduce sorting overhead.

## 8. Memory optimization

Instead of storing individual Python/NumPy objects in a list, active vectors could be maintained in a contiguous matrix.

## 9. Type and dimension validation

The API could explicitly validate:

- vector dimensionality
- numeric dtype
- metadata types
- empty vectors
- invalid `top_k`

---

# Learning Path

If you are using this repository to learn vector databases, a useful progression is:

```text
1. Store vectors
      ↓
2. Calculate cosine similarity
      ↓
3. Vectorize the calculation with NumPy
      ↓
4. Implement top-k search
      ↓
5. Add metadata filtering
      ↓
6. Add persistence
      ↓
7. Add an ANN index
      ↓
8. Benchmark against production vector databases
```

This progression makes it easier to understand why production vector databases require specialized indexes and storage engines.

---

# Production Alternatives

For production workloads, specialized systems provide functionality beyond this educational implementation, including persistence, indexing, filtering, and scalable deployment.

Examples include:

- FAISS
- Qdrant
- Milvus
- Weaviate
- Pinecone

The purpose of this project is to understand the underlying mechanics rather than compete with these systems.

---

# Project Goals

The primary goals are:

1. Keep the implementation small.
2. Make vector-search mechanics easy to inspect.
3. Demonstrate cosine similarity with NumPy.
4. Provide a clean CRUD interface.
5. Provide tests for the core behavior.
6. Provide a reproducible demonstration.
7. Create a foundation for experimenting with vector-database concepts.

---

# License

No license file is currently included in the repository.

If you intend others to reuse, modify, or distribute the project, consider adding an explicit open-source license.

---

# Author

**Benyamin Mahdavifar**

GitHub: https://github.com/BenyaminMahdavifar

Repository: https://github.com/BenyaminMahdavifar/minimal-vector-db

---

## Summary

`minimal-vector-db` is a deliberately small implementation of a vector database built with Python and NumPy.

It provides the essential idea behind vector retrieval:

```text
Store vectors
    +
Associate metadata
    +
Normalize vectors
    +
Compute cosine similarity
    +
Return top-k results
```

The implementation is intentionally simple, exact, and transparent, making it useful as a learning project and a starting point for experimenting with more advanced vector-search techniques.
