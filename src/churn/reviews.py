"""Phase 8: why customers leave, from Netflix app reviews on Google Play.

The churn datasets say *who* leaves but never *why*. Public app reviews are customers explaining it in their own
words. This module turns them into three results:

  * Sentiment: the star rating is the label (1-2 negative, 3 neutral, 4-5 positive). A TF-IDF + Logistic
    Regression model checks how well the text alone separates negative from positive reviews, and its
    coefficients show which words mark a complaint. The monthly share of negative reviews tracks sentiment over time.
  * Topics: NMF on TF-IDF of the negative reviews finds REVIEW_N_TOPICS topics. NMF splits some complaints into
    near-synonyms ("stopped working", "won't open"), so each topic is mapped to one of a few named THEMES with a
    small keyword lexicon. The same lexicon also flags every review directly (multi-label), which (a) checks the
    NMF result and (b) gives themes NMF merges away (ads, password sharing) a timeline.
  * Cancel intent: reviews that mention cancelling or switching. Their theme mix is the closest thing to a churn
    reason we have.

Caveat for every result: reviewers choose to write and cannot be linked to the customers in S or A, so the link to
the churn drivers (THEMES[...]["s_driver"]) is qualitative support, not proof.

Run `python -m churn.reviews` to write reports/results/reviews_topics.csv, reviews_sentiment.csv,
reviews_theme_trends.csv, reviews_examples.csv and reviews_cleaning_log.csv.
"""

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from churn import config
from churn.evaluate import metrics_from_proba

COLUMN_MAP = {
    "reviewId": "review_id", "content": "content", "score": "score", "thumbsUpCount": "thumbs_up",
    "appVersion": "app_version", "at": "date",
}  # userName is dropped on load: it is personal data and not needed

SENTIMENT_LABELS = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}

CANCEL_PATTERN = (r"\bcancel|unsubscri|switch(?:ing|ed)? to\b|not worth|\brefund|leaving netflix|"
                  r"\bquit(?:ting)? netflix|end(?:ed|ing)? my (?:subscription|membership)")

# Themes: name, keyword regex (matched on cleaned text and on NMF topic terms), and the closest S churn driver.
THEMES = {
    "price_billing": {
        "label": "Price & billing",
        "pattern": r"\bpric|expensive|costly|money|\bpay|payment|subscri|charg|\bcard\b|refund|\bmonth|\bfee|"
                   r"afford|increase|\bplan\b|\bwaste|\bfree\b",
        "description": "Price rises, being charged, payment failures, not worth the money.",
        "s_driver": "monthly_charges",
    },
    "account_sharing": {
        "label": "Account, login & password sharing",
        "pattern": r"account|sign|login|log in|\blog\b|password|household|sharing|\bshare|email|verif|"
                   r"profile|extra member|location",
        "description": "Login loops, locked accounts, the household / paid-sharing rules.",
        "s_driver": "multi_device_access",
    },
    "playback": {
        "label": "Playback & streaming quality",
        "pattern": r"screen|video|\bplay|black|audio|sound|loading|buffer|quality|subtitle|lag|\bhd\b|"
                   r"resolution|watch|stuck|brightness",
        "description": "Black screen, no sound, buffering, low resolution, videos that will not play.",
        "s_driver": "support_tickets_per_month",
    },
    "app_broken": {
        "label": "App crashes or won't open",
        "pattern": r"\bopen|crash|working|stopped|\bwork\b|\bworks\b|does work|\bload|uninstall|reinstall|"
                   r"restart|android|phone|device|\btv\b",
        "description": "The app crashes, freezes, will not open or stopped working on a device.",
        "s_driver": "support_tickets_per_month",
    },
    "bad_update": {
        "label": "A bad update",
        "pattern": r"updat|version|latest|recent",
        "description": "Something broke after a new app version.",
        "s_driver": "support_tickets_per_month",
    },
    "server_error": {
        "label": "Connection & server errors",
        "pattern": r"error|reach|service later|sorry|connect|network|server|internet|wifi",
        "description": "\"Sorry, we couldn't reach the Netflix service\" and other connection errors.",
        "s_driver": "support_tickets_per_month",
    },
    "content": {
        "label": "Content library",
        "pattern": r"movie|shows?\b|series|content|available|\bold\b|language|dubb|library|season|episode|"
                   r"catalog|anime|removed",
        "description": "Not enough new or local-language titles, shows removed, old catalogue.",
        "s_driver": "viewing_hours_per_week",
    },
    "downloads": {
        "label": "Downloads",
        "pattern": r"download|offline|storage|\bdelete",
        "description": "Downloads fail, expire or are limited.",
        "s_driver": "content_downloads_per_month",
    },
    "ads": {
        "label": "Ads",
        "pattern": r"\bads?\b|advert|commercial",
        "description": "Adverts in the cheaper plan, ads for Netflix's own titles.",
        "s_driver": None,  # S has no ad-supported plan
    },
}

# Generic words that say "this is bad" but not *what* is bad. Removing them stops NMF from building a
# "bad / worst / experience" topic out of pure sentiment.
TOPIC_STOP_WORDS = sorted(ENGLISH_STOP_WORDS | set("""
netflix app apps application just like dont doesnt didnt cant isnt wont wasnt arent im ive id youre theyre thats
theres lets really please pls plz fix use using used even also get got gets one will would could still now time times
every make makes many much thing things way want wants know say said going go goes need needs tried try bad worst
good great nice best love awesome terrible horrible poor experience ok okay hate disappointed stupid sucks useless
rubbish trash garbage worse star stars review rating give""".split()))

# Real-world events to check the timeline against (dates are approximate global rollouts; Play Store reviews are global).
EVENTS = [
    ("2022-01", "US price rise"),
    ("2022-11", "ad-supported plan launches"),
    ("2023-02", "paid sharing in first countries"),
    ("2023-05", "paid sharing in the US and 100+ countries"),
    ("2023-10", "price rise (Basic, Premium)"),
    ("2025-01", "price rise"),
]


# ---------------------------------------------------------------- load and clean
def load_reviews() -> pd.DataFrame:
    """Raw reviews with snake_case columns, `date` parsed and the reviewer name dropped."""
    df = pd.read_csv(config.RAW_REVIEWS, usecols=list(COLUMN_MAP)).rename(columns=COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"])
    return df


def normalize_text(s: pd.Series) -> pd.Series:
    """Lowercase, drop apostrophes (so "doesn't" -> "doesnt" stays one token) and collapse whitespace."""
    return (s.fillna("").str.lower().str.replace(r"[’']", "", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def clean_reviews(df: pd.DataFrame, min_words: int = config.REVIEW_MIN_WORDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the raw reviews and add the analysis columns.

    Steps: drop empty reviews, duplicate review ids and duplicate texts (copy-pasted spam and one-word repeats),
    reviews shorter than `min_words`, and reviews that are probably not English (5+ words and not a single English
    stop word, e.g. Hindi or Tagalog written in Latin script).
    Adds: text, n_words, sentiment, negative (1-2 stars), cancel_intent, month.
    Returns (clean_df, log_df) with the log in the same format as preprocessing's cleaning log.
    """
    log = []

    def step(d, mask_drop, name, note):
        before = len(d)
        d = d[~mask_drop]
        log.append({"dataset": "reviews", "step": name, "rows_before": before, "rows_after": len(d),
                    "n_affected": before - len(d), "note": note})
        return d

    d = df.copy()
    d["text"] = normalize_text(d["content"])
    d = step(d, d["text"].eq(""), "drop_empty", "review has no text")
    d = step(d, d.duplicated("review_id"), "drop_duplicate_ids", "same review scraped twice")
    d = step(d, d.duplicated("text"), "drop_duplicate_text", "identical text, first kept (spam, 'good app' repeats)")
    tokens = d["text"].str.findall(r"[a-z]+")
    d["n_words"] = tokens.str.len()
    d = step(d, d["n_words"] < min_words, "drop_short", f"fewer than {min_words} words, too short for a topic")
    tokens = d["text"].str.findall(r"[a-z]+")
    en_share = tokens.map(lambda t: sum(w in ENGLISH_STOP_WORDS for w in t) / max(len(t), 1))
    d = step(d, (d["n_words"] >= 5) & (en_share == 0), "drop_non_english",
             "5+ words but no English stop word (other languages in Latin script)")

    d["sentiment"] = d["score"].map(SENTIMENT_LABELS)
    d["negative"] = (d["score"] <= 2).astype(int)
    d["cancel_intent"] = d["text"].str.contains(CANCEL_PATTERN, regex=True).astype(int)
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    return d.reset_index(drop=True), pd.DataFrame(log)


def load_clean_reviews() -> pd.DataFrame:
    """Cleaned reviews from data/interim/, created first if missing."""
    if not config.CLEAN_REVIEWS.exists():
        config.ensure_dirs()
        clean_reviews(load_reviews())[0].to_csv(config.CLEAN_REVIEWS, index=False)
    return pd.read_csv(config.CLEAN_REVIEWS, parse_dates=["date", "month"])


# ---------------------------------------------------------------- sentiment
def monthly_sentiment(df: pd.DataFrame, min_n: int = config.REVIEW_MIN_MONTH_N) -> pd.DataFrame:
    """One row per month: n_reviews, negative_share (1-2 stars), mean_score, cancel_share.
    Months with fewer than `min_n` reviews (gaps in the scrape) are left out."""
    g = df.groupby("month")
    out = pd.DataFrame({"n_reviews": g.size(), "negative_share": g["negative"].mean(),
                        "mean_score": g["score"].mean(), "cancel_share": g["cancel_intent"].mean()})
    return out[out["n_reviews"] >= min_n].reset_index()


def _sentiment_split(d: pd.DataFrame, seed: int):
    """70 / 15 / 15 split (config.SPLIT) of the non-neutral reviews, stratified on `negative`."""
    train_frac, val_frac, test_frac = config.SPLIT
    train, rest = train_test_split(d, train_size=train_frac, stratify=d["negative"], random_state=seed)
    val, test = train_test_split(rest, test_size=test_frac / (val_frac + test_frac), stratify=rest["negative"],
                                 random_state=seed)
    return train, val, test


def _sentiment_vectorizer() -> TfidfVectorizer:
    # Keeps stop words on purpose: "not", "no" and "never" matter for sentiment.
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50_000, sublinear_tf=True)


def sentiment_model(df: pd.DataFrame, seeds=config.SEEDS, c_grid=(0.5, 1, 2, 4, 8)):
    """Predict negative (1-2 stars) vs positive (4-5 stars) reviews from their text.

    C is tuned on the validation split of seed 0 (ROC-AUC), then every seed is evaluated once on its test split.
    Returns (metrics_df with one row per seed, coef_df: term, coef from the seed-0 model, best_c).
    """
    d = df[df["sentiment"] != "neutral"]
    train, val, _ = _sentiment_split(d, seeds[0])
    vec = _sentiment_vectorizer()
    Xtr, Xval = vec.fit_transform(train["text"]), vec.transform(val["text"])
    val_auc = {}
    for c in c_grid:
        m = LogisticRegression(C=c, max_iter=2000, class_weight="balanced").fit(Xtr, train["negative"])
        val_auc[c] = metrics_from_proba(val["negative"], m.predict_proba(Xval)[:, 1])["roc_auc"]
    best_c = max(val_auc, key=val_auc.get)

    rows, coef = [], None
    for seed in seeds:
        train, _, test = _sentiment_split(d, seed)
        vec = _sentiment_vectorizer()
        m = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced").fit(
            vec.fit_transform(train["text"]), train["negative"])
        proba = m.predict_proba(vec.transform(test["text"]))[:, 1]
        rows.append({"seed": seed, "C": best_c, **metrics_from_proba(test["negative"], proba)})
        if coef is None:
            coef = pd.DataFrame({"term": vec.get_feature_names_out(), "coef": m.coef_[0]})
    return pd.DataFrame(rows), coef.sort_values("coef", ascending=False, ignore_index=True), best_c


# ---------------------------------------------------------------- topics
def topic_vectorizer() -> TfidfVectorizer:
    """TF-IDF for topic modelling: unigrams + bigrams, generic sentiment words removed, very rare and very common
    terms dropped."""
    return TfidfVectorizer(stop_words=TOPIC_STOP_WORDS, token_pattern=r"(?u)\b[a-z][a-z]+\b", ngram_range=(1, 2),
                           min_df=20, max_df=0.4, max_features=20_000, sublinear_tf=True)


def fit_topics(texts: pd.Series, n_topics: int = config.REVIEW_N_TOPICS, seed: int = config.RANDOM_STATE):
    """Fit NMF on TF-IDF of `texts`. Returns (vectorizer, nmf, W): W[i, t] is how much review i is about topic t."""
    vec = topic_vectorizer()
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, init="nndsvdar", random_state=seed, max_iter=500)
    W = nmf.fit_transform(X)
    return vec, nmf, W


def top_terms(vec: TfidfVectorizer, nmf: NMF, n: int = 10) -> list[list[str]]:
    """The `n` highest-weight terms of every topic."""
    terms = np.asarray(vec.get_feature_names_out())
    return [list(terms[np.argsort(h)[::-1][:n]]) for h in nmf.components_]


def npmi_coherence(X_binary, term_idx) -> float:
    """Mean normalised PMI of all pairs of the given terms (document co-occurrence). -1 = never together,
    0 = independent, 1 = always together. Higher means a more interpretable topic."""
    n_docs = X_binary.shape[0]
    p = np.asarray(X_binary[:, term_idx].mean(axis=0)).ravel()
    sub = X_binary[:, term_idx]
    co = (sub.T @ sub).toarray() / n_docs
    scores = []
    for i in range(len(term_idx)):
        for j in range(i + 1, len(term_idx)):
            if co[i, j] == 0:
                scores.append(-1.0)
            else:
                scores.append(np.log(co[i, j] / (p[i] * p[j])) / -np.log(co[i, j]))
    return float(np.mean(scores))


def choose_k(texts: pd.Series, ks=(6, 8, 10, 12, 14), seeds=(0, 1, 2), n_terms: int = 10) -> pd.DataFrame:
    """Compare numbers of topics. For each k: mean NPMI coherence of the topics' top terms, and stability = mean
    cosine similarity between the topics of the RANDOM_STATE fit and refits with other seeds (topics paired by the
    Hungarian algorithm). 1.0 means the same topics every time."""
    from scipy.optimize import linear_sum_assignment

    vec = topic_vectorizer()
    X = vec.fit_transform(texts)
    Xb = (X > 0).astype(np.float32).tocsc()
    rows = []
    for k in ks:
        H = NMF(k, init="nndsvdar", random_state=config.RANDOM_STATE, max_iter=500).fit(X).components_
        coherence = np.mean([npmi_coherence(Xb, np.argsort(h)[::-1][:n_terms]) for h in H])
        stab = []
        for s in seeds:
            H2 = NMF(k, init="nndsvdar", random_state=s, max_iter=500).fit(X).components_
            sim = normalize(H) @ normalize(H2).T
            r, c = linear_sum_assignment(-sim)
            stab.append(sim[r, c].mean())
        rows.append({"k": k, "coherence": coherence, "stability": float(np.mean(stab))})
    return pd.DataFrame(rows)


def theme_stability(neg: pd.DataFrame, cancel: pd.DataFrame, ks=(8, 10, 12), seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Theme shares (negative and cancel-intent reviews) for every k and seed. Topic-level stability can be low
    when NMF splits one complaint into near-synonyms differently per seed; what the results rely on is that the
    *theme* shares stay put. Returns columns k, seed, group (negative / cancel), theme, share."""
    rows = []
    for k in ks:
        for seed in seeds:
            vec, nmf, W = fit_topics(neg["text"], k, seed)
            topics = topic_table(vec, nmf, W)
            shares = {"negative": assign_themes(neg, vec, nmf, W, topics).value_counts(normalize=True),
                      "cancel": cancel_theme_shares(cancel, vec, nmf, topics)}
            for group, s in shares.items():
                for theme in [*THEMES, "other"]:
                    rows.append({"k": k, "seed": seed, "group": group, "theme": theme,
                                 "share": float(s.get(theme, 0.0))})
    return pd.DataFrame(rows)


def theme_of_terms(terms: list[str], weights=None) -> str:
    """The THEMES key whose keyword pattern matches the most (weighted) of a topic's top terms."""
    weights = np.ones(len(terms)) if weights is None else np.asarray(weights)
    scores = {name: sum(w for t, w in zip(terms, weights) if re.search(th["pattern"], t))
              for name, th in THEMES.items()}
    return max(scores, key=scores.get)


def topic_table(vec: TfidfVectorizer, nmf: NMF, W: np.ndarray, n_terms: int = 20) -> pd.DataFrame:
    """One row per NMF topic: share of reviews it dominates, top terms and the theme it is mapped to."""
    terms = np.asarray(vec.get_feature_names_out())
    dominant = W.argmax(axis=1)
    rows = []
    for t, h in enumerate(nmf.components_):
        idx = np.argsort(h)[::-1][:n_terms]
        rows.append({"nmf_topic": t, "share": float(np.mean(dominant == t)),
                     "top_terms": ", ".join(terms[idx[:10]]), "theme": theme_of_terms(list(terms[idx]), h[idx])})
    return pd.DataFrame(rows).sort_values("share", ascending=False, ignore_index=True)


def keyword_themes(texts: pd.Series) -> pd.DataFrame:
    """Multi-label theme flags (0/1) for every review, straight from the THEMES keyword patterns."""
    return pd.DataFrame({name: texts.str.contains(th["pattern"], regex=True).astype(int)
                         for name, th in THEMES.items()}, index=texts.index)


def assign_themes(neg: pd.DataFrame, vec, nmf, W, topics: pd.DataFrame) -> pd.Series:
    """Theme of each negative review = the theme of its dominant NMF topic. Reviews no topic explains
    (all weights ~0) get "other"."""
    topic_theme = topics.set_index("nmf_topic")["theme"]
    theme = pd.Series(topic_theme.reindex(W.argmax(axis=1)).to_numpy(), index=neg.index)
    return theme.where(W.max(axis=1) > 1e-3, "other")


def cancel_theme_shares(cancel: pd.DataFrame, vec, nmf, topics: pd.DataFrame) -> pd.Series:
    """Theme shares among cancel-intent reviews, scored with the NMF model fitted on negative reviews."""
    W = nmf.transform(vec.transform(cancel["text"]))
    return assign_themes(cancel, vec, nmf, W, topics).value_counts(normalize=True)


def theme_summary(neg: pd.DataFrame, cancel: pd.DataFrame, vec, nmf, topics: pd.DataFrame) -> pd.DataFrame:
    """reviews_topics.csv: one row per theme (the app reads `topic` and `share`).

    share = share of negative reviews whose dominant NMF topic maps to the theme (single label);
    cancel_share = the same among cancel-intent reviews; keyword_share = share of negative reviews mentioning the
    theme's keywords (multi-label check, so it does not sum to 1).
    """
    share = neg["theme"].value_counts(normalize=True)
    cancel_share = cancel_theme_shares(cancel, vec, nmf, topics)
    kw = keyword_themes(neg["text"]).mean()
    rows = []
    for name, th in THEMES.items():
        rows.append({
            "theme": name, "topic": th["label"], "description": th["description"],
            "share": float(share.get(name, 0.0)), "cancel_share": float(cancel_share.get(name, 0.0)),
            "keyword_share": float(kw[name]), "n_reviews": int((neg["theme"] == name).sum()),
            "nmf_topics": " ".join(str(t) for t in topics.loc[topics["theme"] == name, "nmf_topic"]),
            "s_driver": th["s_driver"] or "",
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["share", "keyword_share"], ascending=False, ignore_index=True)


def theme_trends(neg: pd.DataFrame, min_n: int = config.REVIEW_MIN_MONTH_N) -> pd.DataFrame:
    """Long table quarter x theme: share of that quarter's negative reviews that mention the theme (keyword flags,
    so ads and password sharing are visible even though NMF merges them). Quarters with fewer than `min_n`
    negative reviews are left out."""
    flags = keyword_themes(neg["text"])
    flags["cancel_intent"] = neg["cancel_intent"]
    q = neg["date"].dt.to_period("Q").astype(str)
    shares = flags.groupby(q).mean()
    n = q.value_counts()
    shares = shares[n.reindex(shares.index) >= min_n]
    out = shares.rename_axis("quarter").reset_index().melt("quarter", var_name="theme", value_name="share")
    return out.assign(n_negative=out["quarter"].map(n).astype(int))


def theme_examples(neg: pd.DataFrame, W: np.ndarray, per_theme: int = 5) -> pd.DataFrame:
    """The most-upvoted clear examples of every theme (theme weight in the top quarter of its reviews)."""
    strength = pd.Series(W.max(axis=1), index=neg.index)
    rows = []
    for name, g in neg.groupby("theme"):
        if name == "other":
            continue
        clear = g[strength[g.index] >= strength[g.index].quantile(0.75)]
        top = clear[clear["n_words"].between(8, 80)].nlargest(per_theme, "thumbs_up")
        rows.append(top.assign(theme=name, topic=THEMES[name]["label"])[
            ["theme", "topic", "score", "thumbs_up", "date", "content"]])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- pipeline
def run(n_topics: int = config.REVIEW_N_TOPICS) -> dict:
    """Clean the reviews, model sentiment and topics, and write the Phase 8 outputs to reports/results/."""
    config.ensure_dirs()
    df, log = clean_reviews(load_reviews())
    df.to_csv(config.CLEAN_REVIEWS, index=False)

    neg = df[df["negative"] == 1].copy()
    cancel = df[df["cancel_intent"] == 1]
    vec, nmf, W = fit_topics(neg["text"], n_topics)
    topics = topic_table(vec, nmf, W)
    neg["theme"] = assign_themes(neg, vec, nmf, W, topics)

    out = {
        "clean": df, "log": log, "neg": neg, "cancel": cancel, "vec": vec, "nmf": nmf, "W": W, "topics": topics,
        "themes": theme_summary(neg, cancel, vec, nmf, topics),
        "sentiment": monthly_sentiment(df),
        "trends": theme_trends(neg),
        "examples": theme_examples(neg, W),
    }
    res = config.RESULTS_DIR
    out["themes"].to_csv(res / "reviews_topics.csv", index=False)
    out["sentiment"].assign(month=out["sentiment"]["month"].dt.strftime("%Y-%m-%d")).to_csv(
        res / "reviews_sentiment.csv", index=False)
    out["trends"].to_csv(res / "reviews_theme_trends.csv", index=False)
    out["examples"].to_csv(res / "reviews_examples.csv", index=False)
    out["log"].to_csv(res / "reviews_cleaning_log.csv", index=False)
    return out


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    out = run()
    print(out["log"].to_string(index=False))
    print(out["topics"].to_string(index=False))
    print(out["themes"].drop(columns="description").round(3).to_string(index=False))
