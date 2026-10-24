import re
import json
import csv
import io
from collections import defaultdict
from pathlib import Path

import streamlit as st
import nltk
from nltk.corpus import wordnet as wn

APP_TITLE = "LexiFind"
APP_SUBTITLE = "Evidence-grounded vocabulary intelligence"

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 10% 0%, #e9f5ff 0, #f7f9fc 36%, #f8f7ff 100%); }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
.hero { padding: 1.6rem 1.8rem; border-radius: 24px; background: linear-gradient(120deg,#111827,#172554 60%,#312e81); color:white; margin-bottom:1.2rem; box-shadow: 0 16px 40px rgba(30,41,59,.16); }
.hero h1 { font-family:Manrope,sans-serif; font-size:2.5rem; font-weight:800; margin:0; letter-spacing:-1.5px; }
.hero p { color:#dbeafe; font-size:1rem; margin:.5rem 0 0; }
.pill { display:inline-block; background:#dbeafe; color:#1d4ed8; padding:4px 10px; border-radius:999px; font-size:.75rem; font-weight:700; margin-bottom:.65rem; }
.result { background:rgba(255,255,255,.9); border:1px solid #e5eaf2; border-radius:18px; padding:1rem 1.1rem; margin:.55rem 0; box-shadow:0 4px 14px rgba(15,23,42,.035); }
.word { font-family:Manrope,sans-serif; font-size:1.25rem; font-weight:800; color:#111827; }
.meta { color:#64748b; font-size:.82rem; }
.small { color:#475569; font-size:.92rem; }
.stButton>button { border-radius:12px; font-weight:700; }
section[data-testid="stSidebar"] { background:#f1f5f9; }
</style>
""", unsafe_allow_html=True)

# ---------- NLTK resources ----------
@st.cache_resource
def prepare_wordnet():
    for resource, path in [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
    try:
        wn.synsets("test")
        return True
    except LookupError:
        return False

WORDNET_READY = prepare_wordnet()

POS_MAP = {
    "Any part of speech": None,
    "Noun": wn.NOUN,
    "Verb": wn.VERB,
    "Adjective": wn.ADJ,
    "Adverb": wn.ADV,
}
POS_LABEL = {wn.NOUN: "noun", wn.VERB: "verb", wn.ADJ: "adjective", wn.ADV: "adverb"}

def normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def tokens(text):
    return re.findall(r"[a-zA-Z][a-zA-Z'-]*", normalize(text))

def lexical_docs(query, pos=None, max_synsets=80):
    """Create evidence documents directly from WordNet; no model-generated facts."""
    query_tokens = tokens(query)
    if not query_tokens:
        return []
    synsets = []
    seen_ids = set()
    # Search each token; multiword expressions are also tried as underscore phrases.
    candidates = []
    phrase = "_".join(query_tokens)
    candidates.extend(wn.synsets(phrase, pos=pos))
    for token in query_tokens:
        candidates.extend(wn.synsets(token, pos=pos))
    for syn in candidates:
        if syn.name() not in seen_ids:
            seen_ids.add(syn.name())
            synsets.append(syn)
    docs = []
    for syn in synsets[:max_synsets]:
        lemmas = []
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() not in {x.lower() for x in lemmas}:
                lemmas.append(name)
        docs.append({
            "id": syn.name(),
            "synset": syn,
            "term": syn.name().split(".")[0].replace("_", " "),
            "pos": POS_LABEL.get(syn.pos(), "adjective satellite" if syn.pos() == "s" else syn.pos()),
            "definition": syn.definition(),
            "examples": list(syn.examples()),
            "lemmas": lemmas,
            "lemma_names": [x.lower() for x in lemmas],
            "source": "WordNet",
        })
    return docs

def overlap_score(query, doc):
    q = set(tokens(query))
    term_tokens = set(tokens(doc["term"]))
    lemma_tokens = set(tokens(" ".join(doc["lemmas"])))
    definition_tokens = set(tokens(doc["definition"]))
    # Simple transparent lexical relevance score.
    return 4 * len(q & term_tokens) + 2 * len(q & lemma_tokens) + 0.4 * len(q & definition_tokens)

@st.cache_resource
def embedding_model():
    """Optional local semantic retriever. If unavailable, lexical retrieval still works."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None

def retrieve(query, pos=None, use_semantic=False, top_k=10):
    docs = lexical_docs(query, pos=pos)
    if not docs:
        return [], {"lexical": 0, "semantic": 0, "fused": 0, "reranked": 0, "semantic_enabled": False}
    # Lexical ranking is always available and works offline once WordNet is downloaded.
    lexical_ranked = sorted(docs, key=lambda d: overlap_score(query, d), reverse=True)
    lexical_rank = {d["id"]: i + 1 for i, d in enumerate(lexical_ranked)}
    semantic_rank = {}
    semantic_used = False
    if use_semantic:
        model = embedding_model()
        if model is not None:
            try:
                import numpy as np
                query_vec = model.encode([query], normalize_embeddings=True)[0]
                texts = [d["term"] + ". " + d["definition"] + ". " + " ".join(d["lemmas"]) for d in docs]
                vectors = model.encode(texts, normalize_embeddings=True)
                sims = np.dot(vectors, query_vec)
                semantic_rank = {docs[i]["id"]: rank + 1 for rank, i in enumerate(np.argsort(-sims))}
                semantic_used = True
            except Exception:
                semantic_rank = {}
    # Reciprocal Rank Fusion (RRF) combines lexical and semantic ranked lists.
    k = 60
    for d in docs:
        d["rrf_score"] = 1 / (k + lexical_rank.get(d["id"], 10_000))
        if semantic_rank:
            d["rrf_score"] += 1 / (k + semantic_rank.get(d["id"], 10_000))
        d["lexical_score"] = overlap_score(query, d)
    fused = sorted(docs, key=lambda d: (d["rrf_score"], d["lexical_score"]), reverse=True)
    # Lightweight evidence-aware reranking: exact query sense / term and query overlap.
    q_tokens = set(tokens(query))
    for d in fused:
        exact = 1.0 if normalize(d["term"]) == normalize(query) else 0.0
        lemma_hit = 1.0 if any(normalize(x) == normalize(query) for x in d["lemmas"]) else 0.0
        definition_hit = len(q_tokens & set(tokens(d["definition"])))
        d["final_score"] = d["rrf_score"] + exact * .04 + lemma_hit * .02 + definition_hit * .001
    reranked = sorted(fused, key=lambda d: d["final_score"], reverse=True)
    return reranked[:top_k], {
        "lexical": len(lexical_ranked),
        "semantic": len(semantic_rank),
        "fused": len(fused),
        "reranked": min(top_k, len(reranked)),
        "semantic_enabled": semantic_used,
    }

def collect_synonyms(query, pos=None, use_semantic=False, top_k=10, include_query=False):
    docs, stats = retrieve(query, pos, use_semantic, top_k=80)
    q_norm = normalize(query).replace("_", " ")
    found = {}
    # Prefer terms supported by more relevant senses; retain the best supporting evidence.
    for doc in docs:
        for lemma in doc["lemmas"]:
            key = normalize(lemma)
            if not key or (not include_query and key == q_norm):
                continue
            if key in {"."}:
                continue
            # Do not call every lemma a perfect synonym: preserve sense and evidence.
            item = {
                "word": lemma,
                "pos": doc["pos"],
                "definition": doc["definition"],
                "examples": doc["examples"],
                "source": doc["source"],
                "synset": doc["id"],
                "score": doc["final_score"],
            }
            if key not in found or item["score"] > found[key]["score"]:
                found[key] = item
    ranked = sorted(found.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k], docs, stats

# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "favorites" not in st.session_state:
    st.session_state.favorites = []

# ---------- Header ----------
st.markdown(f"""
<div class="hero">
  <div class="pill">ADVANCED RAG • WORDNET EVIDENCE</div>
  <h1>📚 {APP_TITLE}</h1>
  <p>{APP_SUBTITLE}. Find alternatives with definitions, sense-aware evidence, and transparent retrieval.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Search settings")
    pos_choice = st.selectbox("Part of speech", list(POS_MAP.keys()), index=0)
    result_count = st.slider("Number of suggestions", min_value=5, max_value=25, value=10, step=5)
    use_semantic = st.toggle("Enable local semantic retrieval", value=False, help="Requires sentence-transformers and a model download on first use. The app falls back to lexical retrieval if unavailable.")
    st.caption("Semantic retrieval is optional. WordNet remains the evidence source.")
    st.markdown("---")
    st.markdown("### Pipeline")
    st.caption("Query normalization → WordNet retrieval → lexical/optional semantic ranking → RRF fusion → evidence-aware reranking → deduplication")
    st.markdown("---")
    st.markdown("### Saved words")
    if st.session_state.favorites:
        for word in st.session_state.favorites[-12:]:
            st.write("⭐ " + word)
    else:
        st.caption("Your saved words will appear here.")

if not WORDNET_READY:
    st.error("WordNet could not be loaded. Check your internet connection and run `python -m nltk.downloader wordnet omw-1.4`, then restart the app.")
    st.stop()

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("Enter a word or short phrase", placeholder="e.g. happy, run, make a decision", label_visibility="collapsed")
with col2:
    search_clicked = st.button("Find synonyms", type="primary", use_container_width=True)

st.caption("Tip: A single word usually gives the clearest lexical results. A sentence can help identify intent, but this version does not claim full sentence-level word-sense disambiguation.")

if search_clicked and query.strip():
    pos = POS_MAP[pos_choice]
    with st.spinner("Retrieving and ranking lexical evidence…"):
        results, evidence, stats = collect_synonyms(query, pos, use_semantic, result_count)
    st.session_state.history.insert(0, {"query": query.strip(), "count": len(results)})
    st.session_state.history = st.session_state.history[:20]
    st.session_state.last_query = query.strip()
    st.session_state.last_results = results
    st.session_state.last_evidence = evidence
    st.session_state.last_stats = stats

if search_clicked and not query.strip():
    st.warning("Enter a word or phrase first.")

if "last_results" in st.session_state:
    active_query = st.session_state.get("last_query", "")
    results = st.session_state.last_results
    evidence = st.session_state.last_evidence
    stats = st.session_state.last_stats

    st.markdown(f"## Results for “{active_query}”")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Suggestions", len(results))
    m2.metric("Evidence senses", len(evidence))
    m3.metric("Lexical candidates", stats["lexical"])
    m4.metric("Semantic retrieval", "On" if stats["semantic_enabled"] else "Off")

    if not results:
        st.info("No synonym candidates were found. Try a common English word or select a different part of speech.")
    else:
        st.markdown("### Ranked alternatives")
        st.caption("Ranking is a relevance estimate, not a guarantee that every alternative fits every context. Read the definition and examples before choosing.")
        for idx, item in enumerate(results, 1):
            with st.container():
                st.markdown(f"""
                <div class="result">
                  <div class="word">{idx}. {item['word']}</div>
                  <div class="meta">{item['pos'].title()} · Source: {item['source']} · Evidence: {item['synset']}</div>
                  <p class="small">{item['definition']}</p>
                </div>
                """, unsafe_allow_html=True)
                if item["examples"]:
                    st.caption("Example: “" + item["examples"][0] + "”")
                c1, c2 = st.columns([1, 5])
                with c1:
                    if st.button("⭐ Save", key=f"save_{active_query}_{idx}_{item['word']}"):
                        if item["word"] not in st.session_state.favorites:
                            st.session_state.favorites.append(item["word"])
                        st.toast(f"Saved {item['word']}")
                with c2:
                    st.caption("Use only when its definition matches your intended meaning.")

        export_data = [{
            "query": active_query,
            "word": x["word"],
            "part_of_speech": x["pos"],
            "definition": x["definition"],
            "example": x["examples"][0] if x["examples"] else "",
            "source": x["source"],
            "synset": x["synset"],
        } for x in results]
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=export_data[0].keys())
        writer.writeheader()
        writer.writerows(export_data)
        st.download_button("⬇️ Download results as CSV", csv_buffer.getvalue(), file_name="synonym_results.csv", mime="text/csv")

    with st.expander("View retrieved evidence (RAG transparency)"):
        if evidence:
            for doc in evidence[:12]:
                st.markdown(f"**{doc['term']}** · `{doc['id']}` · {doc['pos']}")
                st.write(doc["definition"])
                st.caption("WordNet lemmas: " + ", ".join(doc["lemmas"]))
                if doc["examples"]:
                    st.caption("Examples: " + " | ".join(doc["examples"][:2]))
                st.divider()
        else:
            st.write("No evidence documents retrieved.")

with st.expander("Search history"):
    if st.session_state.history:
        for entry in st.session_state.history:
            st.write(f"**{entry['query']}** — {entry['count']} suggestions")
    else:
        st.caption("Search history is kept for this browser session only.")

st.markdown("---")
st.caption("LexiFind is a vocabulary aid, not a perfect synonym oracle. WordNet is a lexical database; semantic similarity is not the same as synonymy. Verify the sense before using a suggested word.")
