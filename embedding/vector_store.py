import faiss
import numpy as np

class VectorMemory:
    def __init__(self):
        self.index = faiss.IndexFlatL2(512)
        self.events = []

    def store(self, event):
        vec = self.embed(event)
        self.index.add(np.array([vec]).astype('float32'))
        self.events.append(event)

    def embed(self, text):
        return np.random.rand(512)  # Replace with real embedding later

    def search(self, query, top_k=3):
        qvec = self.embed(query).astype('float32').reshape(1, -1)
        D, I = self.index.search(qvec, top_k)
        return [self.events[i] for i in I[0] if i < len(self.events)]