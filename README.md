# LexiFind — Advanced RAG Synonym Finder

A Streamlit vocabulary application using NLTK WordNet as its evidence corpus. It includes lexical retrieval, optional local semantic retrieval, reciprocal-rank fusion (RRF), evidence-aware reranking, synonym deduplication, definitions, examples, saved words, history, and CSV export.

## Important accuracy note

No retrieval system can guarantee that every suggested word is interchangeable in every sentence. WordNet groups lemmas by lexical sense; users should inspect the displayed definition and examples. The current version is evidence-grounded RAG-style retrieval, not an LLM that invents facts. It does not claim perfect sentence-level word-sense disambiguation.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
streamlit run app.py
```

Open the local URL shown by Streamlit.

## Optional semantic retrieval

For embedding-based semantic ranking, install the optional packages:

```bash
pip install sentence-transformers numpy
```

Enable **Local semantic retrieval** in the sidebar. The first run downloads `all-MiniLM-L6-v2`; this is optional and may require extra disk space and internet access. If it cannot load, the application continues with lexical retrieval.

## Retrieval architecture

1. **Normalize query** — lowercases and cleans whitespace.
2. **Retrieve evidence** — looks up WordNet synsets for the query phrase and its tokens.
3. **Lexical ranking** — ranks exact term, lemma, and definition overlap.
4. **Optional semantic retrieval** — embeds query and evidence documents.
5. **RRF fusion** — merges lexical and semantic rankings with reciprocal-rank fusion.
6. **Evidence-aware reranking** — modestly boosts exact lexical matches and overlap.
7. **Deduplicate** — keeps one best-supported entry per candidate word.
8. **Grounded display** — shows WordNet definitions, examples, and synset IDs.

## What this version does not claim

- It does not guarantee perfect synonym precision.
- The semantic retriever measures similarity, not synonymy.
- It does not use an LLM to rewrite whole sentences in this starter release.
- Its lexical ranking is a transparent heuristic, not a trained cross-encoder reranker.
- A full advanced RAG production system should add word-sense disambiguation, a curated evaluation dataset, calibrated confidence thresholds, and tests for precision@k.

## Suggested next improvements

- Add a curated benchmark of word/context pairs and measure precision@5.
- Add contextual word-sense disambiguation before retrieval.
- Add a cross-encoder reranker behind an optional dependency.
- Add a FastAPI API layer and React frontend if a full-stack portfolio version is desired.
- Add unit tests and CI.
