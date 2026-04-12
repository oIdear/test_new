import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.data = []
        self.dimension = None

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        if self.dimension is None:
            self.dimension = embeddings.shape[1]
        return embeddings

    def build_index(self, data_path):
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

            texts = [item['content'] for item in self.data]
            embeddings = self.generate_embeddings(texts)

            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)

            print('Built index with', len(self.data), 'items')
            print('Index dimension:', self.dimension)

        except Exception as e:
            print('Error building index:', e)

    def save_index(self, index_path, data_path):
        try:
            if self.index is not None:
                faiss.write_index(self.index, index_path)
                print('Saved index to', index_path)

            if self.data:
                with open(data_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
                print('Saved data to', data_path)
        except Exception as e:
            print('Error saving index:', e)

    def load_index(self, index_path, data_path):
        try:
            self.index = faiss.read_index(index_path)
            print('Loaded index from', index_path)

            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print('Loaded', len(self.data), 'items from', data_path)

            self.dimension = self.index.d
            print('Index dimension:', self.dimension)
        except Exception as e:
            print('Error loading index:', e)

    def search(self, query, k=5):
        try:
            if self.index is None:
                print('Index not built. Please build the index first.')
                return []

            query_embedding = self.model.encode([query])[0]
            query_embedding = np.array([query_embedding])

            distances, indices = self.index.search(query_embedding, k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.data):
                    results.append((self.data[idx], distances[0][i]))

            return results
        except Exception as e:
            print('Error searching:', e)
            return []

if __name__ == '__main__':
    store = EmbeddingStore()

    data_path = '/home/zzx-king/zzx/routing-anomaly-detection-master/rag_system/parsed_data.json'
    store.build_index(data_path)

    index_path = '/home/zzx-king/zzx/routing-anomaly-detection-master/rag_system/faiss_index.bin'
    save_data_path = '/home/zzx-king/zzx/routing-anomaly-detection-master/rag_system/index_data.json'
    store.save_index(index_path, save_data_path)

    test_query = 'BGP route leak'
    results = store.search(test_query, k=3)
    print('Search results for', test_query, ':')
    for i, (result, distance) in enumerate(results):
        print('Result', i+1, '(distance:', distance, '):')
        print('File:', result['file'])
        print('Content:', result['content'][:200], '...')
        print()
