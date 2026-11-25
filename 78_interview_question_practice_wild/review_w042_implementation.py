import numpy as np
import pandas as pd


class RetrievalIndex:
    def __init__(self, df, embed, text_column="text", id_column="id"):
        if text_column not in df or id_column not in df:
            raise ValueError("map actual schema columns first")
        if df[id_column].duplicated().any() or df[id_column].isna().any():
            raise ValueError("unique nonmissing ids required")
        if not df[text_column].map(lambda v: isinstance(v, str)).all():
            raise ValueError("text must be strings")
        self.df = df.copy().reset_index(drop=True)
        self.embed = embed
        self.id_column = id_column
        E = np.asarray(embed(self.df[text_column].tolist()), dtype=float)
        if (
            E.ndim != 2
            or E.shape[0] != len(df)
            or not E.shape[1]
            or not np.isfinite(E).all()
        ):
            raise ValueError("invalid embedding output")
        self.E = np.divide(
            E,
            np.linalg.norm(E, axis=1, keepdims=True),
            out=np.zeros_like(E),
            where=np.linalg.norm(E, axis=1, keepdims=True) > 0,
        )

    def retrieve(self, query, k=10):
        if k < 0:
            raise ValueError("negative k")
        q = np.asarray(self.embed([query]), dtype=float)
        if q.shape != (1, self.E.shape[1]) or not np.isfinite(q).all():
            raise ValueError("query embedding mismatch")
        norm = np.linalg.norm(q)
        if norm == 0:
            return self.df.iloc[:0].assign(score=np.array([], dtype=float))
        scores = self.E @ (q[0] / norm)
        order = np.argsort(-scores, kind="stable")[: min(k, len(self.df))]
        result = self.df.iloc[order].copy()
        result["score"] = scores[order]
        return result

    def evaluate(self, queries):
        rows = []
        for query, relevant in queries:
            relevant = set(relevant)
            if not relevant:
                raise ValueError("recall undefined without relevant labels")
            ids = self.retrieve(query, 10)[self.id_column].tolist()
            rows.append(
                {
                    "query": query,
                    "recall_at_10": len(set(ids) & relevant) / len(relevant),
                    "reciprocal_rank": next(
                        (1 / r for r, i in enumerate(ids, 1) if i in relevant), 0.0
                    ),
                }
            )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Deterministic small embedding fixture; replace with the notebook's supplied encoder.
    def embed(texts):
        return np.array(
            [
                [text.lower().split().count(w) for w in ["cat", "dog", "bird"]]
                for text in texts
            ],
            dtype=float,
        )

    df = pd.DataFrame(
        {"id": range(12), "text": ["cat", "dog", "bird", "cat dog"] + ["bird"] * 8},
        index=range(20, 32),
    )
    index = RetrievalIndex(df, embed)
    found = index.retrieve("cat")
    assert len(found) == 10 and found.id.iloc[0] == 0 and found.id.iloc[1] == 3
    assert index.retrieve("unknown").empty
    results = index.evaluate([("cat", {0, 3}), ("dog", {1, 3})])
    assert np.allclose(results.recall_at_10, 1) and np.allclose(
        results.reciprocal_rank, 1
    )
    assert df.index.tolist() == list(range(20, 32))  # Preserve caller's table.
    print(
        "W042: declared fixture retrieval/evaluation:\n", results.to_string(index=False)
    )
