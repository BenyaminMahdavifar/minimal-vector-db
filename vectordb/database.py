import numpy as np

class MinimalVectorDB:
    def __init__(self):
        self.vectors = []       
        self.metadata = {}      
        self.id_map = {}        
        self.next_id = 0        

    def insert(self, vector, metadata):
        vector_id = self.next_id
        self.next_id += 1
        
        index = len(self.vectors)
        self.vectors.append(vector)
        
        self.id_map[vector_id] = index
        self.metadata[vector_id] = metadata
        return vector_id

    def get(self, vector_id):
        if vector_id in self.id_map:
            index = self.id_map[vector_id]
            return {
                "vector": self.vectors[index],
                "metadata": self.metadata[vector_id]
            }
        return None

    def update(self, vector_id, new_vector=None, new_metadata=None):
        if vector_id not in self.id_map:
            return False
        
        index = self.id_map[vector_id]
        
        if new_vector is not None:
            self.vectors[index] = new_vector
        if new_metadata is not None:
            self.metadata[vector_id] = new_metadata
            
        return True

    def delete(self, vector_id):
        if vector_id in self.id_map:
            index = self.id_map[vector_id]
            self.vectors[index] = None  
            
            del self.id_map[vector_id]
            del self.metadata[vector_id]
            return True
        return False

    def _cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    def search(self, query_vector, top_k=5):
        if not self.id_map:
            return []

        valid_ids = list(self.id_map.keys())
        
        valid_vectors = np.array([self.vectors[self.id_map[i]] for i in valid_ids])
        
        query = np.array(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query_normalized = query / query_norm

        norms = np.linalg.norm(valid_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  
        vectors_normalized = valid_vectors / norms

        similarities = np.dot(vectors_normalized, query_normalized)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            v_id = valid_ids[idx]
            score = float(similarities[idx])
            results.append((v_id, score))
        return results