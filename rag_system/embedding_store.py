import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# bge 系列模型检索时需要加此前缀以激活检索优化
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingStore:
    def __init__(self, model_name='BAAI/bge-base-en-v1.5'):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.index = None
        self.data = []
        self.dimension = None
        # BM25 相关
        self.bm25 = None
        self.tokenized_corpus = None

    def _add_query_prefix(self, text):
        if 'bge' in self.model_name.lower():
            return BGE_QUERY_PREFIX + text
        return text

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        if self.dimension is None:
            self.dimension = embeddings.shape[1]
        return embeddings

    def build_index(self, data_path):
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

            texts = [item['content'] for item in self.data]

            # 向量索引
            embeddings = self.generate_embeddings(texts)
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)

            # BM25 索引
            self.tokenized_corpus = [text.lower().split() for text in texts]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

            print(f'向量索引构建完成，共 {len(self.data)} 条，维度 {self.dimension}')
            print('BM25 索引构建完成')

        except Exception as e:
            print('Error building index:', e)
            raise

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
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.dimension = self.index.d

            # 重建 BM25 索引（不持久化，启动时快速重建）
            texts = [item['content'] for item in self.data]
            self.tokenized_corpus = [text.lower().split() for text in texts]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

            print(f'加载索引：{len(self.data)} 条文档，向量维度 {self.dimension}')
            print('BM25 索引重建完成')
        except Exception as e:
            print('Error loading index:', e)

    # ------------------------------------------------------------------ #
    #  核心检索接口                                                         #
    # ------------------------------------------------------------------ #

    def search(self, query, k=5, hybrid=True):
        """
        hybrid=True：BM25 + 向量检索，RRF 融合后 MMR 去重
        hybrid=False：退化为纯向量检索（兼容旧调用）
        """
        if not hybrid or self.bm25 is None:
            return self._vector_search(query, k)

        vec_results  = self._vector_search(query, k=min(20, len(self.data)))
        bm25_results = self._bm25_search(query,  k=min(20, len(self.data)))
        merged = self._rrf_merge(vec_results, bm25_results, k=min(k * 3, len(self.data)))
        return self._mmr_select(merged, k=k)

    # ------------------------------------------------------------------ #
    #  内部方法                                                             #
    # ------------------------------------------------------------------ #

    def _vector_search(self, query, k=5):
        try:
            if self.index is None:
                return []
            q = self._add_query_prefix(query)
            emb = self.model.encode([q])[0].astype(np.float32)
            emb = np.array([emb])
            faiss.normalize_L2(emb)
            distances, indices = self.index.search(emb, k)
            return [
                (self.data[idx], float(distances[0][i]))
                for i, idx in enumerate(indices[0])
                if idx < len(self.data)
            ]
        except Exception as e:
            print('Vector search error:', e)
            return []

    def _bm25_search(self, query, k=20):
        try:
            scores = self.bm25.get_scores(query.lower().split())
            top_idx = np.argsort(scores)[::-1][:k]
            return [
                (self.data[i], float(scores[i]))
                for i in top_idx
                if scores[i] > 0
            ]
        except Exception as e:
            print('BM25 search error:', e)
            return []

    def _rrf_merge(self, vec_results, bm25_results, k=15, c=60):
        """倒数排名融合（Reciprocal Rank Fusion）"""
        scores = {}
        for rank, (item, _) in enumerate(vec_results):
            key = id(item)
            if key not in scores:
                scores[key] = {'item': item, 'score': 0.0}
            scores[key]['score'] += 1.0 / (c + rank + 1)

        for rank, (item, _) in enumerate(bm25_results):
            # bm25 结果按 content 匹配到 data 中的对象
            matched_key = None
            for key, val in scores.items():
                if val['item']['content'] == item['content']:
                    matched_key = key
                    break
            if matched_key is None:
                matched_key = id(item)
                scores[matched_key] = {'item': item, 'score': 0.0}
            scores[matched_key]['score'] += 1.0 / (c + rank + 1)

        ranked = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        return [(r['item'], r['score']) for r in ranked[:k]]

    def _mmr_select(self, results, k=5, lambda_=0.6):
        """
        最大边际相关（MMR）去重：在相关性和多样性之间取平衡。
        lambda_ 越大越偏向相关性，越小越偏向多样性。
        """
        if len(results) <= k:
            return results

        selected = []
        candidates = list(results)

        while len(selected) < k and candidates:
            if not selected:
                selected.append(candidates.pop(0))
                continue

            best_idx = max(
                range(len(candidates)),
                key=lambda i: (
                    lambda_ * candidates[i][1]
                    - (1 - lambda_) * max(
                        self._jaccard(candidates[i][0]['content'], s[0]['content'])
                        for s in selected
                    )
                )
            )
            selected.append(candidates.pop(best_idx))

        return selected

    @staticmethod
    def _jaccard(text_a, text_b):
        a = set(text_a.lower().split())
        b = set(text_b.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


if __name__ == '__main__':
    import sys
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent

    print("=" * 60)
    print("开始构建 RAG 向量索引（bge-base-en-v1.5 + BM25）")
    print("=" * 60)

    store = EmbeddingStore()

    data_path = base_dir / 'parsed_data.json'
    if not data_path.exists():
        print(f'错误: 找不到 {data_path}，请先运行 data_parser.py')
        sys.exit(1)

    store.build_index(str(data_path))

    index_path     = base_dir / 'faiss_index.bin'
    save_data_path = base_dir / 'index_data.json'
    store.save_index(str(index_path), str(save_data_path))

    print(f'\n索引构建完成')
    print(f'  文档数: {len(store.data)}')
    print(f'  向量维度: {store.dimension}')
