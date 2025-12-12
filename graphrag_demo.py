#!/usr/bin/env python3
"""
GraphRAG Full Demo (offline, no LLM)
Drop PDFs into `docs/` and run: python graphrag_full_demo.py
"""

import os
import sys
import pickle
from typing import List, Tuple, Dict, Any
from pypdf import PdfReader
import re
import networkx as nx
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Optional OCR imports
USE_OCR = False
try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    USE_OCR = True
except Exception:
    # OCR not available; scanned PDFs will fail text extraction
    USE_OCR = False

# -----------------------
# Config
# -----------------------
DOCS_DIR = "docs"
INDEX_PATH = "faiss.index"
META_PATH = "meta.pkl"
GRAPH_PATH = "graph.pkl"
CHUNK_SIZE_SENT = 6      # sentences per chunk
CHUNK_OVERLAP = 2       # overlap in sentences
EMB_MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast

# -----------------------
# Models
# -----------------------
nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer(EMB_MODEL_NAME)

# -----------------------
# Helpers: text extraction (with optional OCR fallback)
# -----------------------
def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text_pages = []
    for page in reader.pages:
        try:
            t = page.extract_text()
        except Exception:
            t = None
        if t:
            text_pages.append(t)
    text = "\n".join(text_pages).strip()
    if text:
        return text
    # fallback to OCR if available
    if USE_OCR:
        print(f"[INFO] No selectable text in {path}; attempting OCR (may be slow)...")
        images = convert_from_path(path)
        ocr_text = []
        for img in images:
            ocr_text.append(pytesseract.image_to_string(img))
        return "\n".join(ocr_text)
    else:
        return ""

# -----------------------
# Helpers: chunking by sentences (with overlap)
# -----------------------
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def sentence_split(text: str) -> List[str]:
    if not text:
        return []
    sents = SENT_SPLIT_RE.split(text.replace("\r\n", " ").replace("\n", " "))
    # remove empty
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def chunk_sentences(text: str, max_sentences=CHUNK_SIZE_SENT, overlap=CHUNK_OVERLAP) -> List[str]:
    sents = sentence_split(text)
    chunks = []
    if not sents:
        return chunks
    step = max_sentences - overlap if max_sentences > overlap else max_sentences
    for i in range(0, len(sents), step):
        chunk = " ".join(sents[i:i+max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# -----------------------
# Loading docs -> chunks + meta
# -----------------------
def load_docs_to_chunks(doc_dir: str) -> Tuple[List[str], List[Dict[str,Any]]]:
    all_chunks = []
    meta = []
    files = sorted([f for f in os.listdir(doc_dir) if f.lower().endswith(".pdf")])
    if not files:
        print("[WARN] No PDFs found in docs/ folder.")
    for fname in files:
        path = os.path.join(doc_dir, fname)
        print(f"[INFO] Processing {fname} ...")
        text = extract_text_from_pdf(path)
        if not text:
            print(f"[WARN] No text extracted from {fname}. (scanned PDF and OCR not available?)")
            continue
        pages = text.split("\f") if "\f" in text else [text]
        # chunk whole text (not per-page here)
        chunks = chunk_sentences(text)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            meta.append({"doc": fname, "chunk_id": i})
    return all_chunks, meta

# -----------------------
# Build graph: entity nodes + chunk nodes
# -----------------------
def build_graph(chunks: List[str], meta: List[Dict[str,Any]]) -> nx.Graph:
    G = nx.Graph()
    for idx, chunk in enumerate(chunks):
        chunk_node = f"chunk:{idx}"
        G.add_node(chunk_node, type="chunk", text=chunk, meta=meta[idx])
        doc = nlp(chunk)
        entities = []
        for ent in doc.ents:
            ent_text = ent.text.strip()
            if not ent_text:
                continue
            ent_node = f"ent:{ent_text}"
            if not G.has_node(ent_node):
                G.add_node(ent_node, type="entity", label=ent_text, ent_type=ent.label_)
            # connect entity -> chunk
            if not G.has_edge(ent_node, chunk_node):
                G.add_edge(ent_node, chunk_node, relation="mentions")
            entities.append(ent_node)
        # co-occurrence edges between entities in same chunk
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                a, b = entities[i], entities[j]
                if G.has_edge(a, b):
                    G[a][b]['weight'] = G[a][b].get('weight', 1) + 1
                else:
                    G.add_edge(a, b, relation="cooccur", weight=1)
    return G

# -----------------------
# Embeddings and FAISS index
# -----------------------
def build_embeddings_index(chunks: List[str]) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    if not chunks:
        raise ValueError("No chunks for embedding.")
    print("[INFO] Computing embeddings (may take a moment)...")
    emb = embed_model.encode(chunks, show_progress_bar=True)
    emb = np.array(emb).astype("float32")
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return index, emb

# -----------------------
# Save / Load helpers
# -----------------------
def save_state(index: faiss.IndexFlatL2, meta: List[Dict[str,Any]], graph: nx.Graph):
    print("[INFO] Saving FAISS index, metadata, and graph...")
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(graph, f)
    print("[INFO] Saved.")

def load_state():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH) or not os.path.exists(GRAPH_PATH):
        return None, None, None
    print("[INFO] Loading saved index + metadata + graph...")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)
    print("[INFO] Loaded.")
    return index, meta, graph

# -----------------------
# Query: entity extraction, graph neighborhood, vector retrieval, combine
# -----------------------
def extract_entities(text: str) -> List[str]:
    doc = nlp(text)
    return [ent.text.strip() for ent in doc.ents if ent.text.strip()]

def get_graph_chunk_candidates(G: nx.Graph, entities: List[str], hops: int = 1, limit: int = 200) -> List[int]:
    candidates = set()
    for ent in entities:
        ent_node = f"ent:{ent}"
        if not G.has_node(ent_node):
            # fuzzy fallback: match entity-containing names
            for node in G.nodes:
                if node.startswith("ent:") and ent.lower() in node[4:].lower():
                    ent_node = node
                    break
            if not G.has_node(ent_node):
                continue
        reachable = nx.single_source_shortest_path_length(G, ent_node, cutoff=hops).keys()
        for n in reachable:
            if n.startswith("chunk:"):
                candidates.add(int(n.split(":")[1]))
    return list(candidates)[:limit]

def vector_retrieve(index: faiss.IndexFlatL2, query: str, k: int = 5):
    q_emb = embed_model.encode([query]).astype("float32")
    dists, idxs = index.search(q_emb, k)
    return dists[0].tolist(), [int(i) for i in idxs[0]]

def retrieve_hybrid(query: str, chunks: List[str], index: faiss.IndexFlatL2, G: nx.Graph, meta: List[Dict[str,Any]], k_vector=5, hops=1):
    ents = extract_entities(query)
    graph_idxs = get_graph_chunk_candidates(G, ents, hops=hops, limit=200) if ents else []
    dists, vec_idxs = vector_retrieve(index, query, k=k_vector)
    # combine prioritizing graph hits
    combined_order = list(dict.fromkeys(graph_idxs + vec_idxs))
    results = []
    for idx in combined_order:
        r = {
            "idx": idx,
            "chunk": chunks[idx],
            "meta": meta[idx],
            "distance": None
        }
        if idx in vec_idxs:
            pos = vec_idxs.index(idx)
            r["distance"] = float(dists[pos])
        results.append(r)
    return results

# -----------------------
# Main
# -----------------------
def main():
    # Optionally load saved state
    index, meta, G = load_state()
    chunks = []
    emb_matrix = None

    if index is None:
        # build from scratch
        chunks, meta = load_docs_to_chunks(DOCS_DIR)
        if not chunks:
            print("[ERROR] no chunks created. Add PDFs in docs/ or enable OCR.")
            sys.exit(1)
        print(f"[INFO] Total chunks: {len(chunks)}")
        G = build_graph(chunks, meta)
        print(f"[INFO] Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
        index, emb_matrix = build_embeddings_index(chunks)
        # save for future runs
        save_state(index, meta, G)
    else:
        # We need chunks loaded from meta for runtime; the saved meta doesn't contain text,
        # so we must reconstruct 'chunks' by re-chunking the source docs.
        # Simpler: if index exists, rebuild chunks here (cheap for small corpora).
        chunks, meta = load_docs_to_chunks(DOCS_DIR)
        if not chunks:
            print("[ERROR] Unable to rebuild chunks from docs/ while loading saved index.")
            sys.exit(1)
        print(f"[INFO] Rebuilt {len(chunks)} chunks from docs/")

    print("[READY] You can ask questions (type 'exit' or Ctrl+C to quit).")

    while True:
        try:
            q = input("\nQuestion: ").strip()
        except KeyboardInterrupt:
            print("\n[EXIT]")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        results = retrieve_hybrid(q, chunks, index, G, meta, k_vector=5, hops=1)
        if not results:
            print("[NO RESULTS] Try a different question or check your docs.")
            continue
        # print top candidates
        print(f"\nTop {min(10, len(results))} candidate chunks (graph-prioritized):\n")
        for i, r in enumerate(results[:10]):
            print(f"#{i+1} IDX:{r['idx']} DOC:{r['meta']['doc']} CHUNK_ID:{r['meta']['chunk_id']} DIST:{r['distance']}")
            print(r['chunk'][:600].replace("\n", " "))
            print("-" * 60)

if __name__ == "__main__":
    main()
