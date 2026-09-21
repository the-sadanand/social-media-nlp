# topic labeling pipeline 
# LDA and CountVectorizer

"""Outputs (all saved to output/)

  lda_model.pkl           — fitted LDA model
  count_vectorizer.pkl    — fitted CountVectorizer (needed for pyLDAvis)
  topics.json             — top words per topic
  lda_visualization.html  — interactive pyLDAvis visualisation
"""

import json
import pickle
import warnings
from pathlib import Path

import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# configuration

N_TOPICS = 5 

N_TOP_WORDS = 10

MAX_FEATURES = 5000

MAX_ITER = 20

def train_topic_model(
    preprocessed_path: str = "output/preprocessed_data.csv",
    output_dir : str = "output" ,
    n_topics : int = N_TOPICS,
    n_top_words : int = N_TOP_WORDS,
) -> tuple:
    print("\n" + "=" * 60)
    print("Step 3 - Topic Modeling(LDA)")
    print("=" * 60)
    
    out = Path(output_dir)
    out.mkdir(parents=True , exist_ok=True)
    
    # Load preprocessed text
    
    df = pd.read_csv(preprocessed_path)
    texts = df["cleaned_text"].dropna().tolist()
    print(f"Loaded{len(texts):,} documents")
    
    # 2 Build document term matrix with countVectorizer
    
    count_vec = CountVectorizer(
        max_features= MAX_FEATURES,
        max_df= 0.95,
        min_df=2,
    )
    
    dtm = count_vec.fit_transform(texts)
    print(f"DTM shape : {dtm.shape}(documents x vocabulary)")
    
    # trian lda
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter= MAX_ITER,
        learning_method="online",
        random_state=42,
        n_jobs= - 1,
    )
    
    lda.fit(dtm)
    
    print(f"LDA trianed with {n_topics} topics")
    
    # extract top words per topic
    
    feature_names = count_vec.get_feature_names_out()
    topics:dict[str,list[str]]={}
    
    print(f"\n {'-' * 50}")
    
    for idx, component in enumerate(lda.components_):
        top_indices = component.argsort()[: -n_top_words - 1 : -1]
        top_words = [feature_names[i] for i in top_indices]
        topic_key = f"topic_{idx}"
        topics[topic_key] = top_words
        print(f"  Topic {idx}: {', '.join(top_words)}")
    print(f"  {'-' * 50}")
    
        # 5.  Save model artifacts
    with open(out / "lda_model.pkl", "wb") as f:
        pickle.dump(lda, f)

    with open(out / "count_vectorizer.pkl", "wb") as f:
        pickle.dump(count_vec, f)

    with open(out / "topics.json", "w") as f:
        json.dump(topics, f, indent=2)
        

    # 6.  Generate pyLDAvis interactive visualisation
    
    _generate_pyldavis(lda, dtm, count_vec, out / "lda_visualization.html")

    print(f"\n  [OK] Saved: lda_model.pkl, count_vectorizer.pkl, "
          f"topics.json, lda_visualization.html")

    return lda, count_vec, topics


# ======================== pyLDAvis Helper ==================================


def _generate_pyldavis(lda, dtm, vectorizer, output_path: Path) -> None:
    try:
        import numpy as np
        import pyLDAvis

        # 1. Topic-term distributions (normalise rows to sum to 1)
        topic_term_dists = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
        topic_term_dists = np.float64(topic_term_dists)

        # 2. Document-topic distributions
        doc_topic_dists = np.float64(lda.transform(dtm))

        # 3. Document lengths (total words per doc)
        doc_lengths = np.asarray(dtm.sum(axis=1), dtype=np.float64).flatten()

        # 4. Vocabulary
        vocab = vectorizer.get_feature_names_out().tolist()

        # 5. Term frequencies (total count of each word across corpus)
        term_frequency = np.asarray(dtm.sum(axis=0), dtype=np.float64).flatten()

        # Suppress deprecation warnings from pyLDAvis internals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            prepared = pyLDAvis.prepare(
                topic_term_dists=topic_term_dists,
                doc_topic_dists=doc_topic_dists,
                doc_lengths=doc_lengths,
                vocab=vocab,
                term_frequency=term_frequency,
                sort_topics=False,
                mds="mmds",   # metric MDS — avoids complex eigenvalues
            )

        pyLDAvis.save_html(prepared, str(output_path))
        print(f"  pyLDAvis visualisation saved -> {output_path}")

    except Exception as exc:
        print(f"  [WARN] pyLDAvis generation failed: {exc}")
        print("  Generating fallback HTML instead...")
        _generate_fallback_html(lda, vectorizer, output_path)


def _generate_fallback_html(lda, vectorizer, output_path: Path) -> None:
    """Create a simple HTML page showing topics when pyLDAvis fails."""
    feature_names = vectorizer.get_feature_names_out()
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>LDA Topics</title>",
        "<style>body{font-family:sans-serif;max-width:800px;margin:2em auto}",
        ".topic{background:#f4f4f4;padding:1em;margin:1em 0;border-radius:8px}",
        "h2{color:#333}</style></head><body>",
        "<h1>LDA Topic Model Results</h1>",
        "<p><em>pyLDAvis interactive visualisation was unavailable. "
        "Showing static topic summaries instead.</em></p>",
    ]
    for idx, component in enumerate(lda.components_):
        top_indices = component.argsort()[: -11 : -1]
        words = [feature_names[i] for i in top_indices]
        html_parts.append(
            f'<div class="topic"><h2>Topic {idx}</h2>'
            f'<p>{", ".join(words)}</p></div>'
        )
    html_parts.append("</body></html>")

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"  Fallback HTML saved -> {output_path}")


# ================================ CLI ======================================


if __name__ == "__main__":
    train_topic_model()
