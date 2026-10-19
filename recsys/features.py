"""Признаки книг, пользователей и взаимодействий.

* ``Catalog`` — признаки, не зависящие от рейтингов (title, authors, теги, год). Их можно строить
  по всему каталогу: в них нет информации из validation/test.
* ``build_item_features`` / ``build_user_features`` — признаки, зависящие от рейтингов. Они
  принимают только матрицу обучающей части; функция-обёртка ``fit_context`` гарантирует это.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import NOISE_TAGS, Config
from .data import Index


# ----------------------------------------------------------------------------------------------
# Каталог: контентные признаки книг (не используют рейтинги)
# ----------------------------------------------------------------------------------------------
@dataclass
class Catalog:
    books: pd.DataFrame            # строки упорядочены по item_idx = books.id - 1
    tfidf_base: csr_matrix         # HW_1_FINAL: original_title + top-25 tags
    tfidf_enhanced: csr_matrix     # title + authors + top-25 tags без служебных
    topics: np.ndarray             # (n_items, n_topics), NMF по тегам, строки суммируются в 1
    topic_names: list[str]
    content_emb: np.ndarray        # (n_items, d), L2-нормированный SVD-эмбеддинг tfidf_enhanced
    author_idx: np.ndarray         # главный автор; 0 = "редкий автор" (меньше min_author_books книг)
    n_authors: int
    year_norm: np.ndarray          # стандартизованный год публикации (пропуски -> 0)

    @property
    def n_items(self) -> int:
        return len(self.books)


def _clean_text(s: pd.Series) -> pd.Series:
    return (
        s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        .replace({"nan": "", "None": "", "<NA>": ""})
    )


def _tfidf() -> TfidfVectorizer:
    # Параметры TF-IDF из HW_1_FINAL.
    return TfidfVectorizer(lowercase=True, stop_words="english", min_df=2, max_df=0.7,
                           max_features=20000, ngram_range=(1, 2))


def build_catalog(raw: dict[str, pd.DataFrame], cfg: Config) -> Catalog:
    books = raw["books"].sort_values("id").reset_index(drop=True).copy()
    bt = raw["book_tags"].drop_duplicates().merge(raw["tags"], on="tag_id", how="left")
    bt = bt.sort_values(["goodreads_book_id", "count"], ascending=[True, False], kind="mergesort")
    top = bt.groupby("goodreads_book_id", group_keys=False).head(cfg.top_n_tags)
    top_clean = top[~top["tag_name"].isin(NOISE_TAGS)]

    def join_tags(df):
        return df.groupby("goodreads_book_id")["tag_name"].apply(
            lambda x: " ".join(dict.fromkeys(x.astype(str))))

    tags_base = books["book_id"].map(join_tags(top)).fillna("")
    tags_clean = books["book_id"].map(join_tags(top_clean)).fillna("")

    title = _clean_text(books["original_title"])
    title = title.where(title.ne(""), _clean_text(books["title"]))
    authors = _clean_text(books["authors"])
    books["content_base"] = (title + " " + tags_base).str.strip()
    books["content_enhanced"] = (title + " " + authors + " " + tags_clean).str.strip()
    books["tags_clean"] = tags_clean

    tfidf_base = _tfidf().fit_transform(books["content_base"]).tocsr().astype(np.float32)
    tfidf_enh = _tfidf().fit_transform(books["content_enhanced"]).tocsr().astype(np.float32)

    # Тематические признаки: NMF по тегам (без служебных). Теги-токены целиком, без разбиения по "-".
    tag_vec = TfidfVectorizer(token_pattern=r"[^ ]+", min_df=5, max_df=0.5, sublinear_tf=True)
    tag_matrix = tag_vec.fit_transform(tags_clean)
    nmf = NMF(n_components=cfg.n_topics, init="nndsvda", random_state=cfg.random_state, max_iter=400)
    w = nmf.fit_transform(tag_matrix)
    topics = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    vocab = np.array(tag_vec.get_feature_names_out())
    topic_names = [", ".join(vocab[np.argsort(-comp)[:4]]) for comp in nmf.components_]

    svd = TruncatedSVD(n_components=cfg.content_emb_dim, random_state=cfg.random_state)
    content_emb = normalize(svd.fit_transform(tfidf_enh)).astype(np.float32)

    main_author = authors.str.split(",").str[0].str.strip()
    counts = main_author.value_counts()
    frequent = counts[counts >= cfg.min_author_books].index
    author_map = {a: i + 1 for i, a in enumerate(sorted(frequent))}
    author_idx = main_author.map(author_map).fillna(0).astype(np.int64).to_numpy()
    books["main_author"] = main_author

    year = books["original_publication_year"].where(
        books["original_publication_year"].between(1000, 2017))
    year_norm = ((year - year.mean()) / year.std()).fillna(0.0).to_numpy(np.float32)

    books["dominant_topic"] = topics.argmax(axis=1)
    return Catalog(books=books, tfidf_base=tfidf_base, tfidf_enhanced=tfidf_enh,
                   topics=topics.astype(np.float32), topic_names=topic_names,
                   content_emb=content_emb, author_idx=author_idx,
                   n_authors=len(author_map) + 1, year_norm=year_norm)


# ----------------------------------------------------------------------------------------------
# Признаки, зависящие от рейтингов: строятся только по переданной обучающей части
# ----------------------------------------------------------------------------------------------
def ratings_matrix(train: pd.DataFrame, index: Index) -> csr_matrix:
    u = index.users_to_idx(train["user_id"].to_numpy())
    i = index.items_to_idx(train["book_id"].to_numpy())
    assert (u >= 0).all() and (i >= 0).all() and (i < index.n_items).all()
    R = csr_matrix((train["rating"].to_numpy(np.float32), (u, i)), shape=(index.n_users, index.n_items))
    R.sum_duplicates()
    assert R.nnz == len(train), "повторные пары user-book в train"
    return R


def _row_stats(M: csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.diff(M.indptr).astype(np.float64)
    s = np.asarray(M.sum(axis=1)).ravel()
    s2 = np.asarray(M.multiply(M).sum(axis=1)).ravel()
    mean = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    var = np.divide(s2, n, out=np.zeros_like(s2), where=n > 0) - mean ** 2
    return n, mean, np.sqrt(np.clip(var, 0, None))


def build_item_features(R: csr_matrix, catalog: Catalog) -> pd.DataFrame:
    Rt = R.T.tocsr()
    n, mean, std = _row_stats(Rt)
    global_mean = float(R.data.mean())
    m = float(np.median(n))  # MIN_RATINGS из HW_1_FINAL: медиана поддержки книги в train
    pos = np.asarray((Rt >= 4).sum(axis=1)).ravel()
    wr = (n / (n + m)) * np.where(n > 0, mean, global_mean) + (m / (n + m)) * global_mean
    feats = pd.DataFrame({
        "item_n_ratings": n,
        "item_log_n": np.log1p(n),
        "item_mean": np.where(n > 0, mean, global_mean),
        "item_std": std,
        "item_share_pos": np.divide(pos, n, out=np.zeros_like(pos, dtype=float), where=n > 0),
        "item_weighted_rating": wr,
        "item_pop_pct": pd.Series(n).rank(pct=True).to_numpy(),
        "item_year_norm": catalog.year_norm,
    })
    feats.attrs["min_ratings"] = m
    feats.attrs["global_mean"] = global_mean
    return feats


def segment_labels(n_ratings: np.ndarray, edges: tuple) -> np.ndarray:
    lo, hi = edges
    return np.select([n_ratings == 0, n_ratings <= lo, n_ratings <= hi], ["new", "short", "medium"], "active")


SEGMENTS = ("new", "short", "medium", "active")


def liked_matrix(R: csr_matrix, threshold: float) -> csr_matrix:
    """Бинарная матрица понравившихся книг; если таких нет — вся история пользователя."""
    L = (R >= threshold).astype(np.float32).tocsr()
    has_like = np.diff(L.indptr) > 0
    B = (R > 0).astype(np.float32).tocsr()
    fallback = diags((~has_like).astype(np.float32)) @ B
    return (L + fallback).tocsr()


def build_user_features(R: csr_matrix, item_feats: pd.DataFrame, catalog: Catalog, cfg: Config):
    n, mean, std = _row_stats(R)
    global_mean = item_feats.attrs["global_mean"]
    B = (R > 0).astype(np.float32).tocsr()
    pos = np.asarray((R >= cfg.relevance_threshold).sum(axis=1)).ravel()
    sum_r = np.asarray(R.sum(axis=1)).ravel()

    pop_pct = item_feats["item_pop_pct"].to_numpy()
    mainstream = np.divide(B @ pop_pct, n, out=np.full_like(n, 0.5), where=n > 0)
    centroid = B @ catalog.content_emb
    centroid_norm = np.linalg.norm(centroid, axis=1)
    taste_diversity = np.where(n > 0, 1.0 - np.divide(centroid_norm, n, out=np.zeros_like(n), where=n > 0), 0.0)

    L = liked_matrix(R, cfg.relevance_threshold)
    topic_profile = np.asarray(L @ catalog.topics)
    topic_profile /= np.maximum(topic_profile.sum(axis=1, keepdims=True), 1e-12)

    feats = pd.DataFrame({
        "user_n_ratings": n,
        "user_log_n": np.log1p(n),
        "user_mean": np.where(n > 0, mean, global_mean),
        "user_std": std,
        "user_share_pos": np.divide(pos, n, out=np.zeros_like(n), where=n > 0),
        "user_bias_shrunk": (sum_r - n * global_mean) / (n + 5.0),
        "user_mainstreamness": mainstream,
        "user_taste_diversity": taste_diversity,
        "segment": segment_labels(n, cfg.segment_edges),
    })
    return feats, topic_profile.astype(np.float32)


FEATURE_DESCRIPTIONS = [
    ("user", "user_n_ratings / user_log_n", "число оценок в train (активность)",
     "сегментация new/short/medium/active и веса гибрида; признак башни пользователя"),
    ("user", "user_mean, user_std", "средняя и разброс оценок пользователя в train",
     "строгость и «шкала» пользователя; признаки башни пользователя"),
    ("user", "user_share_pos", "доля оценок >= 4", "склонность ставить высокие оценки"),
    ("user", "user_bias_shrunk", "(sum r - n*mu)/(n+5)", "смещение пользователя, сглаженное к 0 при короткой истории"),
    ("user", "user_mainstreamness", "средний популярностный перцентиль прочитанных книг",
     "насколько пользователю подходит популярное: баланс персонализации и популярности"),
    ("user", "user_taste_diversity", "1 - ||среднее контентных эмбеддингов истории||",
     "широта вкуса (0 = одна тематика)"),
    ("user", "segment", "new / short (1-5) / medium (6-20) / active (>20)", "отдельные веса гибрида по сегментам"),
    ("user", "topic_profile", "среднее тематик NMF понравившихся книг", "тематический профиль; вход башни пользователя"),
    ("item", "item_n_ratings / item_log_n", "число оценок книги в train", "популярность внутри обучающей выборки"),
    ("item", "item_mean, item_std", "средняя и разброс оценок книги в train", "качество и спорность книги"),
    ("item", "item_share_pos", "доля оценок >= 4", "вероятность понравиться случайному читателю"),
    ("item", "item_weighted_rating", "v/(v+m)*R + m/(v+m)*C, m = медиана поддержки", "score модели Popularity (HW_1_FINAL)"),
    ("item", "item_pop_pct", "перцентиль числа оценок", "popularity bias, novelty-метрика"),
    ("item", "item_year_norm", "стандартизованный год публикации", "контентный признак башни книги"),
    ("item", "topics (20)", "NMF по тегам без служебных", "тематические категории; разнообразие; башня книги"),
    ("item", "main_author", "первый автор (редкие -> 0)", "author affinity; эмбеддинг автора в башне книги"),
    ("item", "content_emb (64)", "TruncatedSVD от TF-IDF enhanced", "сходство книг для MMR и ILD"),
    ("interaction", "cb", "cos(профиль TF-IDF понравившихся книг, книга)", "контентная релевантность"),
    ("interaction", "icf", "агрегат сходств книги с историей (top-k соседей)", "коллаборативное сходство с историей"),
    ("interaction", "author", "доля истории пользователя у автора книги", "сходство с историей по автору"),
    ("interaction", "topic", "topic_profile · topics(книга)", "тематическое соответствие профилю"),
]


def feature_descriptions() -> pd.DataFrame:
    return pd.DataFrame(FEATURE_DESCRIPTIONS, columns=["group", "feature", "definition", "purpose"])
