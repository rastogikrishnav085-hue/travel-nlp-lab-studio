"""Self-test for the Studio backend: runs the real engine on the bundled ATIS example."""
import io
import re
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.server import app, ROOT

EXAMPLE = next((ROOT / "examples").glob("*.zip"))
client = TestClient(app)


def open_bundle(data: bytes, name="b.zip"):
    return client.post("/api/upload", files={"file": (name, data, "application/zip")})


@pytest.fixture(scope="module")
def sid():
    r = client.post("/api/load", json={"key": f"example:{EXAMPLE.name}"})
    assert r.status_code == 200
    return r.json()["id"]


def test_health():
    assert client.get("/api/health").json()["ok"] is True


def test_overview_is_verified_and_recomputed(sid):
    o = client.get(f"/api/s/{sid}/overview").json()
    assert o["integrity"]["status"] == "verified"
    assert o["metric_status"] == "recomputed"
    assert abs(o["metrics"]["recomputed"]["accuracy"] - o["metrics"]["reported"]["accuracy"]) < 6e-4
    assert o["metrics"]["ci"]["accuracy"][0] < o["metrics"]["recomputed"]["accuracy"] < o["metrics"]["ci"]["accuracy"][1]


@pytest.mark.parametrize("page", ["data", "models", "errors", "explain", "files"])
def test_pages_respond(sid, page):
    assert client.get(f"/api/s/{sid}/{page}").status_code == 200


def test_error_filter_and_csv(sid):
    r = client.get(f"/api/s/{sid}/errors", params={"q": "zzzz-no-such-text"}).json()
    assert r["matched"] == 0
    assert client.get(f"/api/s/{sid}/errors.csv").headers["content-type"].startswith("text/csv")


def test_autocomplete(sid):
    r = client.post(f"/api/s/{sid}/autocomplete", json={"text": "show me flights fro"}).json()
    assert r["partial"] == "fro" and r["word_completions"][0]["word"] == "from"


def test_tampered_bundle_fails_verification():
    src = zipfile.ZipFile(EXAMPLE)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
        for info in src.infolist():
            blob = src.read(info)
            if info.filename.endswith("results/predictions.csv"):
                blob = blob.replace(b"flight", b"fl1ght", 5)
            out.writestr(info, blob)
    sid2 = open_bundle(buf.getvalue()).json()["id"]
    o = client.get(f"/api/s/{sid2}/overview").json()
    assert o["integrity"]["status"] == "failed"


def test_rejects_csv_and_garbage_and_missing_manifest():
    assert open_bundle(b"a,b\n1,2", "x.csv").status_code == 400
    assert open_bundle(b"not a zip").status_code == 400
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("hello.txt", "hi")
    r = open_bundle(buf.getvalue())
    assert r.status_code == 400 and "run_manifest" in r.json()["detail"]


def test_load_and_file_endpoints_block_path_tricks(sid):
    assert client.post("/api/load", json={"key": "example:../../etc/passwd.zip"}).status_code == 400
    assert client.post("/api/load", json={"key": "nope:x.zip"}).status_code == 400
    assert client.get(f"/api/s/{sid}/file", params={"path": "../../etc/passwd"}).status_code == 404
    assert client.get("/api/s/doesnotexist/overview").status_code == 404


def test_security_headers_and_local_only_api():
    r = client.get("/api/health")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert "streamlit" in client.get("/").json()["ui"]


def test_no_cloud_or_ai_services_in_studio_code():
    banned = re.compile(r"gemini|openai|anthropic|googleapis|cdn\.|https?://(?!127\.0\.0\.1|localhost|www\.w3\.org|www\.python\.org)", re.I)
    code = [*Path("app").glob("*.py"), *Path("studio").rglob("*.py"), Path("streamlit_app.py"), Path(".streamlit/config.toml"),
            *Path("static").glob("*.svg")]
    for p in code:
        assert not banned.search(p.read_text(encoding="utf-8")), p


# ----------------------------------------------------------------------------- Studio 2: new features
def _tweak_bundle(edit):
    """Copy the example bundle, letting `edit(name, blob) -> blob` change files. Returns a new session id."""
    src = zipfile.ZipFile(EXAMPLE)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
        for info in src.infolist():
            out.writestr(info, edit(info.filename, src.read(info)))
    return open_bundle(buf.getvalue()).json()["id"]


def ac(sid, text):
    return client.post(f"/api/s/{sid}/autocomplete", json={"text": text}).json()


def test_ghost_text_keeps_what_was_typed_and_finishes_the_word(sid):
    r = ac(sid, "show me flights fro")
    assert r["level"] in ("high", "medium")
    assert r["ghost"].startswith("m ")  # finishes "fro" -> "from", never rewrites the typed letters
    assert ac(sid, "flights to")["ghost"].startswith(" ")  # next word needs a leading space


def test_no_ghost_text_after_a_word_the_model_has_never_seen(sid):
    for text in ("zzzx qq", "flights from dnever to", "show me flihts from bostn"):
        r = ac(sid, text)
        assert r["ghost"] == "", text
        assert r["level"] in ("careful", "unsure"), text


def test_typo_help_offers_real_words(sid):
    fixes = {f["typed"]: [o["word"] for o in f["options"]] for f in ac(sid, "show me flihts from bostn")["fixes"]}
    assert "flights" in fixes["flihts"] and "boston" in fixes["bostn"]
    assert ac(sid, "show me flights from boston")["fixes"] == []  # known words never get "corrections"


def test_autocomplete_empty_and_odd_input(sid):
    assert ac(sid, "")["ghost"] == ""
    r = ac(sid, "   ")
    assert r["ghost"] == "" and isinstance(r["next_words"], list)
    assert ac(sid, "x" * 5000)["partial"] is not None  # capped, does not crash


def test_autocomplete_accuracy_is_recomputed_and_matches_the_notebook(sid):
    c = client.get(f"/api/s/{sid}/autocomplete/check").json()
    assert c["compare"]["status"] == "verified"
    assert all(r["match"] for r in c["compare"]["rows"] if r["match"] is not None)
    assert c["fresh"]["positions"] == 9770 and 0.42 < c["fresh"]["top1"] < 0.43
    assert c["fresh"]["top1"] <= c["fresh"]["top3"] <= c["fresh"]["top5"]
    assert 0 < c["fresh"]["keystroke_saving"] < 1


def test_autocomplete_check_catches_a_doctored_accuracy_claim():
    changed = []

    def edit(name, blob):
        if name.endswith("next_word_model.json"):
            new = re.sub(rb'"top3_accuracy":\s*0\.62\d+', b'"top3_accuracy":0.9523132036847492', blob)
            changed.append(new != blob)  # guard: the test must really doctor the file
            return new
        return blob
    sid2 = _tweak_bundle(edit)
    assert changed == [True]
    c = client.get(f"/api/s/{sid2}/autocomplete/check").json()
    assert c["compare"]["status"] == "mismatch"


def test_sample_queries_come_from_the_test_set(sid):
    import pandas as pd
    z = zipfile.ZipFile(EXAMPLE)
    test = pd.read_csv(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith("test_processed.csv")))))
    got = client.get(f"/api/s/{sid}/autocomplete/samples", params={"n": 8, "seed": 3}).json()["samples"]
    assert len(got) == 8 and set(s["text"] for s in got) <= set(test["text"])


def test_predictions_browser_is_complete_and_consistent(sid):
    o = client.get(f"/api/s/{sid}/overview").json()
    p = client.get(f"/api/s/{sid}/predictions", params={"limit": 500}).json()
    assert o["has_predictions"] and p["total"] == o["metrics"]["recomputed"]["rows"] == 850
    assert p["wrong_total"] == o["errors_exact"] == 39
    assert round(1 - p["wrong_total"] / p["total"], 6) == round(o["metrics"]["recomputed"]["accuracy"], 6)
    assert all(r["text"] and r["actual"] and r["predicted"] for r in p["rows"])
    w = client.get(f"/api/s/{sid}/predictions", params={"status": "wrong", "actual": "meal", "predicted": "flight"}).json()
    assert w["matched"] == 4 and all(not r["correct"] for r in w["rows"])
    right = client.get(f"/api/s/{sid}/predictions", params={"status": "right", "limit": 500}).json()
    assert right["matched"] == 850 - 39 and all(r["actual"] == r["predicted"] for r in right["rows"])


def test_errors_page_now_lists_every_mistake_not_a_sample(sid):
    e = client.get(f"/api/s/{sid}/errors", params={"limit": 500}).json()
    assert e["complete"] and e["total_errors"] == e["matched"] == len(e["rows"]) == 39


def test_prediction_text_is_never_shown_when_the_join_is_unsafe():
    sid2 = _tweak_bundle(lambda n, b: b.replace(b"test_atis.csv::", b"other.csv::") if n.endswith("results/predictions.csv") else b)
    assert client.get(f"/api/s/{sid2}/overview").json()["has_predictions"] is False
    assert client.get(f"/api/s/{sid2}/predictions").status_code == 409


def test_intent_cards_agree_with_the_models_page(sid):
    cards = {i["intent"]: i for i in client.get(f"/api/s/{sid}/intents").json()["intents"]}
    per = {r["intent"]: r for r in client.get(f"/api/s/{sid}/models").json()["per_intent"]}
    for name, r in per.items():
        assert abs(cards[name]["f1"] - r["f1"]) < 1e-9 and cards[name]["support"] == r["support"]
    assert cards["meal"]["confused_with"][0]["intent"] == "flight"
    assert all(len(c["examples"]) <= 6 for c in cards.values())


def test_compare_with_itself_shows_no_change(sid):
    sid_b = client.post("/api/load", json={"key": f"example:{EXAMPLE.name}"}).json()["id"]
    c = client.get("/api/compare", params={"a": sid, "b": sid_b}).json()
    assert c["same_data"] and c["warning"] is None
    assert c["a"]["accuracy"] == c["b"]["accuracy"] and all(abs(d["delta"]) < 1e-12 for d in c["deltas"])
    assert client.get("/api/compare", params={"a": sid, "b": "nope"}).status_code == 404


def test_edit_distance_and_typo_logic():
    from app.autocomplete_plus import _edit_distance
    assert _edit_distance("flihts", "flights") == 1 and _edit_distance("bostn", "boston") == 1
    assert _edit_distance("form", "from") == 1  # a swap counts as one edit
    assert _edit_distance("abc", "xyzabc", cap=2) > 2


def test_streamlit_theme_is_local_and_dark():
    cfg = Path(".streamlit/config.toml").read_text(encoding="utf-8")
    assert 'base = "dark"' in cfg and "enableStaticServing = true" in cfg and 'address = "127.0.0.1"' in cfg
    for url in re.findall(r'url = "app/static/([^"]+)"', cfg):
        assert (Path("static") / url).is_file(), url
    assert "gatherUsageStats = false" in cfg


# ---- Studio 3 analytics
def test_confusion_totals(sid):
    c = client.get(f"/api/s/{sid}/confusion").json()
    ov = client.get(f"/api/s/{sid}/overview").json()
    assert c["total"] == sum(map(sum, c["matrix"])) == ov["metrics"]["recomputed"]["rows"]
    assert c["errors"] == ov["errors_exact"] and c["diagonal"] + c["errors"] == c["total"]


def test_confidence_and_profile(sid):
    c = client.get(f"/api/s/{sid}/confidence")
    if c.status_code == 200:
        c = c.json()
        assert sum(h["correct"] + h["wrong"] for h in c["histogram"]) == c["n"]
        assert all(0 <= x["coverage"] <= 1 for x in c["curve"])
    p = client.get(f"/api/s/{sid}/profile").json()
    assert p["vocab_size"] > 0 and p["words"]["min"] <= p["words"]["max"]


def test_summary_domains_entities(sid):
    s = client.get(f"/api/s/{sid}/summary").json()
    assert s["capabilities"]["predictions"] is True
    d = {x["domain"]: x for x in client.get("/api/domains").json()["domains"]}
    assert {"flight", "hotel", "car", "rail"} <= set(d)
    assert d["flight"]["has_ner"] is True and d["hotel"]["rows"] > 0
    e = client.get("/api/entities").json()
    assert e["types"] and e["destinations"]


def test_bad_session_still_404():
    assert client.get("/api/s/nope/confusion").status_code == 404


def test_error_lab_matches_confusion(sid):
    e = client.get(f"/api/s/{sid}/error-lab").json()
    c = client.get(f"/api/s/{sid}/confusion").json()
    assert e["errors"] == c["errors"] and e["shown"] == len(e["items"]) == min(120, c["errors"])
    assert all(isinstance(i["categories"], list) for i in e["items"])
    assert "not the model" in e["note"]


def test_compare_verdict_and_self_compare(sid):
    r = client.get(f"/api/compare?a={sid}&b={sid}").json()
    assert r["verdict"] == "no_clear_change" and r["regressions"] == []


def test_zip_with_too_many_files_or_a_bomb_is_rejected():
    import io, zipfile
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for k in range(300):
            z.writestr(f"x/{k}.txt", "a")
    assert open_bundle(b.getvalue()).status_code == 400
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("travel_nlp_run_bundle/run_manifest.json", "{}")
        z.writestr("travel_nlp_run_bundle/big.csv", "a" * 60_000_000)
    assert open_bundle(b.getvalue()).status_code == 400


# ---------------------------------------------------------------- Query Lab (live inference from bundled JSON model)
def test_summary_reports_live_classifier(sid):
    s = client.get(f"/api/s/{sid}/summary").json()
    assert s["capabilities"]["live_classifier"] is True


def test_live_classifier_reproduces_every_stored_prediction(sid):
    """The JSON model must agree with the notebook's scikit-learn predictions on the whole test set."""
    from app import server as S
    sess = S.SESSIONS[sid]
    j = sess["joined"]
    assert j is not None and len(j) > 100
    bad = [t for t, p in zip(j["text"], j["predicted"]) if S.E.clf_classify(sess["clf"], t)["intent"] != p]
    assert bad == []


def test_classify_endpoint_and_benchmark_flag(sid):
    from app import server as S
    j = S.SESSIONS[sid]["joined"]
    q = j["text"].iloc[0]
    r = client.post(f"/api/s/{sid}/classify", json={"text": q}).json()
    assert r["mode"] == "live_inference" and r["intent"] == j["predicted"].iloc[0]
    assert r["stored_benchmark"]["predicted"] == r["intent"]       # a known test query is linked to its stored result
    assert r["band"] in ("low", "medium", "high") and len(r["top"]) == 5
    new = client.post(f"/api/s/{sid}/classify", json={"text": "zzz qqq xyzzy plugh"}).json()
    assert new["stored_benchmark"] is None and new["known_features"] == 0


@pytest.mark.parametrize("text", ["", "   ", "x" * 501])
def test_classify_rejects_bad_input(sid, text):
    assert client.post(f"/api/s/{sid}/classify", json={"text": text}).status_code == 400


def test_robustness_variants(sid):
    r = client.post(f"/api/s/{sid}/robustness", json={"text": "what is the cheapest airfare from boston to denver"}).json()
    names = [v["variant"] for v in r["variants"]]
    assert names[0] == "Original" and "UPPERCASE" in names and "One typo" in names
    up = next(v for v in r["variants"] if v["variant"] == "UPPERCASE")
    assert up["same_as_original"] and up["neutralised_by_preprocessing"]


def test_classifier_model_parser_rejects_bad_files():
    from app import engine as E
    import json as _j
    good = {"format": E.CLF_FORMAT, "classes": ["a", "b"], "vocab": {"x": 0}, "idf": [1.0],
            "weights": [[0.1], [0.2]], "bias": [0.0, 0.0], "ngram_range": [1, 2], "sublinear_tf": True}
    assert E.parse_classifier_model(_j.dumps(good).encode())["classes"] == ["a", "b"]
    for patch in ({"format": "pickle"}, {"idf": [1.0, 2.0]}, {"weights": [[0.1]]}, {"vocab": {"x": 5}},
                  {"ngram_range": [1, 9]}, {"bias": [0.0]}, {"classes": ["a"]}, {"idf": ["nope"]}):
        with pytest.raises(E.BundleError):
            E.parse_classifier_model(_j.dumps({**good, **patch}).encode())
    with pytest.raises(E.BundleError):
        E.parse_classifier_model(b"\x80not json")


def test_bundle_without_model_returns_no_model_code():
    src = zipfile.ZipFile(EXAMPLE)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
        for info in src.infolist():
            if not info.filename.endswith("classifier_model.json"):
                out.writestr(info, src.read(info))
    r = open_bundle(buf.getvalue())
    if r.status_code == 200:                      # (a missing declared file may also fail verification; both are safe)
        new = r.json()["id"]
        assert client.get(f"/api/s/{new}/summary").json()["capabilities"]["live_classifier"] is False
        c = client.post(f"/api/s/{new}/classify", json={"text": "hello flight"})
        assert c.status_code == 409 and "NO_MODEL" in c.text


def test_leakage_endpoint_near_duplicates(sid):
    r = client.get(f"/api/s/{sid}/leakage").json()
    nd = r["near_duplicates"]
    assert nd["threshold"] == 0.9 and "test_queries" not in nd
    a = nd["accuracy"]
    assert a["near_duplicate_rows"] + a["other_rows"] == r["test_rows"]
    assert a["near_duplicate_rows"] == nd["test_rows_with_near_train_match"]
    assert r["train_test_exact_overlap"] == 0


# ----------------------------------------------------------------------------- Studio 3.5
def _forge_bundle(transform):
    """Rewrite the example bundle with `transform(rel_name, bytes) -> bytes|None` and re-hash the manifest so it still verifies."""
    import hashlib
    import json as _j
    src = zipfile.ZipFile(EXAMPLE)
    prefix = "travel_nlp_run_bundle/"
    new = {}
    for info in src.infolist():
        if info.is_dir():
            continue
        rel = info.filename.split(prefix, 1)[-1]
        data = transform(rel, src.read(info))
        if data is not None:
            new[info.filename] = data
    mkey = next(k for k in new if k.endswith("run_manifest.json"))
    man = _j.loads(new[mkey])
    for rel in list((man.get("files") or {}).keys()):
        blob = new.get(prefix + rel)
        if blob is None:
            man["files"].pop(rel)
        else:
            man["files"][rel] = {**man["files"][rel], "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
    new[mkey] = _j.dumps(man).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
        for k, v in new.items():
            out.writestr(k, v)
    return buf.getvalue()


def test_versions_are_kept_apart(sid):
    o = client.get(f"/api/s/{sid}/overview").json()
    v = o["versions"]
    assert v["studio"] == "4.3.1" and set(v) == {"studio", "pipeline", "contract", "bundle_schema"}
    assert v["pipeline"] == o["run"]["pipeline"] and v["contract"] == o["run"]["contract"]


def test_capability_map_is_complete_and_matches_summary(sid):
    o = client.get(f"/api/s/{sid}/overview").json()["capabilities"]
    s = client.get(f"/api/s/{sid}/summary").json()["capabilities"]
    assert o == s
    for k in ("benchmark", "error_analysis", "confidence_scores", "autocomplete", "live_inference", "robustness",
              "ner_annotations", "leakage_check", "figures", "run_comparison"):
        assert isinstance(o[k], bool), k
    assert o["entities"] is o["ner_annotations"] and o["live_classifier"] is o["live_inference"]


def test_entities_come_from_the_bundle_and_reconcile(sid):
    e = client.get(f"/api/s/{sid}/entities").json()
    assert "this bundle" in e["source"] and e["alignment"]["misaligned"] == 0
    assert e["by_intent"] and all(x["types"] for x in e["by_intent"])
    ee = e["errors_by_entity_count"]
    o = client.get(f"/api/s/{sid}/overview").json()
    assert ee["matched"] == o["metrics"]["recomputed"]["rows"]
    assert sum(r["rows"] for r in ee["rows"]) == ee["matched"]
    assert sum(r["errors"] for r in ee["rows"]) == o["errors_exact"]          # same mistakes as the confusion matrix


def test_bundle_without_annotations_does_not_claim_entities():
    def drop_ner(rel, data):
        if rel.startswith("data/raw/") and rel.endswith(".csv"):
            import pandas as pd
            return pd.read_csv(io.BytesIO(data)).drop(columns=["ner"], errors="ignore").to_csv(index=False).encode()
        return data
    r = open_bundle(_forge_bundle(drop_ner))
    assert r.status_code == 200
    new = r.json()["id"]
    o = client.get(f"/api/s/{new}/overview").json()
    assert o["integrity"]["status"] == "verified"                 # a clean bundle, simply without annotations
    assert o["capabilities"]["ner_annotations"] is False and o["capabilities"]["entities"] is False
    assert o["capabilities"]["benchmark"] is True                 # everything else still works
    r2 = client.get(f"/api/s/{new}/entities")                     # the Studio folder HAS ner data; it must not leak in
    assert r2.status_code == 409 and "NO_NER" in r2.text


def test_confidence_focus_filters_rows(sid):
    allc = client.get(f"/api/s/{sid}/confidence").json()
    f = client.get(f"/api/s/{sid}/confidence", params={"focus": "airfare"}).json()
    assert 0 < f["n"] < allc["n"]
    assert all("airfare" in (r["actual"], r["predicted"]) for r in f["lowest"])


def test_entity_analytics_guards():
    import pandas as pd
    from app import analytics as AN
    df = pd.DataFrame({"text": ["a b c", "d e"], "intent": ["x", "y"], "ner": ["O B-fromloc.city_name O", "O"]})
    e = AN.entities_from_df(df, "t")
    assert e["alignment"] == {**e["alignment"], "checked": 2, "misaligned": 1} and e["queries"] == 1
    assert AN.entities_from_df(df.drop(columns=["ner"]), "t") is None
    j = pd.DataFrame({"id": ["f::0"], "text": ["something else"], "correct": [True]})
    raw = pd.DataFrame({"id": [0], "text": ["a b c"], "ner": ["O O O"]})
    assert AN.errors_by_entity_count(j, raw, "f") is None          # text mismatch -> refuse to show numbers


def test_compare_bundles_detects_a_changed_prediction(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import compare_bundles as CB

    rows, _ = CB.compare(EXAMPLE, EXAMPLE)
    assert rows and all(ok for _, ok, _ in rows)                      # a bundle equals itself

    def flip_first_prediction(rel, data):
        if rel == "results/predictions.csv":
            import pandas as pd
            df = pd.read_csv(io.BytesIO(data))
            col = next(c for c in df.columns if c.lower() == "predicted_intent")
            df.loc[0, col] = "definitely_not_an_intent"
            return df.to_csv(index=False).encode()
        return data
    other = tmp_path / "changed.zip"
    other.write_bytes(_forge_bundle(flip_first_prediction))
    rows, _ = CB.compare(EXAMPLE, other)
    failed = [n for n, ok, _ in rows if not ok]
    assert "results/predictions.csv" in failed


# ---------------------------------------------------------------------------------------------
# Studio 3.5.2: a second, very different run (hotel queries: other columns, other intents, no NER,
# unsplit file, Logistic Regression wins so the notebook stores probabilities, not margins).
HOTEL = Path(__file__).parent / "fixtures" / "hotel_run_bundle.zip"
ATIS_ONLY = re.compile(r"\b(atis|flight_time|airfare|ground_service|fromloc|toloc)\b", re.I)


@pytest.fixture(scope="module")
def hsid():
    r = open_bundle(HOTEL.read_bytes(), "hotel.zip")
    assert r.status_code == 200
    return r.json()["id"]


def test_other_dataset_is_verified_and_has_no_atis_content(hsid):
    o = client.get(f"/api/s/{hsid}/overview").json()
    assert o["integrity"]["status"] == "verified"
    for page in ("overview", "data", "models", "intents", "predictions", "confusion", "error-lab", "profile", "explain", "autocomplete/samples"):
        text = client.get(f"/api/s/{hsid}/{page}").text
        assert not ATIS_ONLY.search(text), page


def test_other_dataset_without_ner_disables_entities(hsid):
    caps = client.get(f"/api/s/{hsid}/overview").json()["capabilities"]
    assert caps["ner_annotations"] is False and caps["entities"] is False
    r = client.get(f"/api/s/{hsid}/entities")
    assert r.status_code == 409 and "NO_NER" in r.text


def test_stored_probability_is_used_and_labelled_as_probability(hsid):
    r = client.get(f"/api/s/{hsid}/confidence")
    assert r.status_code == 200
    d = r.json()
    assert d["kind"] == "probability" and "not been calibrated" in d["note"] and 0 <= d["min"] <= d["max"] <= 1
    assert client.get(f"/api/s/{hsid}/predictions").json()["score_kind"] == "probability"
    assert client.get(f"/api/s/{hsid}/overview").json()["capabilities"]["confidence_scores"] is True


def test_probability_scale_is_never_compared_with_live_margin(hsid):
    r = client.post(f"/api/s/{hsid}/classify", json={"text": "does the hotel have a pool"}).json()
    assert r["mode"] == "live_inference" and r["intent"] == "hotel_amenities"
    assert r["band"] is None and r["bands"] is None      # stored 0-1 probabilities vs a live margin of ~2


def test_margin_run_keeps_margin_kind(sid):
    d = client.get(f"/api/s/{sid}/confidence").json()
    assert d["kind"] == "margin" and "not a probability" in d["note"]


def test_small_test_set_is_warned_about(hsid, sid):
    w = " ".join(client.get(f"/api/s/{hsid}/overview").json()["warnings"])
    assert "Small test set" in w and "fewer than 5 test queries" in w
    assert "Small test set" not in " ".join(client.get(f"/api/s/{sid}/overview").json()["warnings"])


# ---------------------------------------------------------------------------------------------
# Studio 3.6.0: intent map (3D view). Every number on it comes from the bundled classifier and test results.
def test_intent_map_is_derived_from_the_bundled_classifier(sid):
    import math
    d = client.get(f"/api/s/{sid}/intent-map").json()
    names = [x["intent"] for x in d["intents"]]
    clf = client.get(f"/api/s/{sid}/classifier").json() if False else None
    assert len(names) == len(set(names)) == 22
    for x in d["intents"]:
        assert all(math.isfinite(x[k]) and abs(x[k]) <= 1.0000001 for k in ("x", "y", "z"))
        assert len(x["top_terms"]) == 6 and len(x["nearest"]) == 3 and all(n["intent"] != x["intent"] for n in x["nearest"])
    ex = d["explained"]
    assert all(0 <= v <= 1 for v in ex) and ex[0] >= ex[1] >= ex[2] and abs(d["explained_total"] - sum(ex)) < 1e-9 and d["explained_total"] <= 1
    # point size and colour come from the recomputed per-intent table, not from the classifier
    per = {r["intent"]: r for r in client.get(f"/api/s/{sid}/models").json()["per_intent"]}
    for x in d["intents"]:
        if x["intent"] in per:
            assert x["support"] == per[x["intent"]]["support"] and abs(x["f1"] - per[x["intent"]]["f1"]) < 1e-9
    # red lines are exactly the confusion matrix's mixed-up pairs
    mp = {tuple(sorted((p["a"], p["b"]))): p["count"] for p in client.get(f"/api/s/{sid}/confusion?top=50").json()["mutual_pairs"]}
    for e in d["edges"]:
        assert mp[tuple(sorted((e["a"], e["b"])))] == e["count"] and -1.0000001 <= e["similarity"] <= 1.0000001
    sc = d["similarity_check"]
    assert sc["confused_pairs"] == len(d["edges"]) and sc["confused_pairs"] + sc["other_pairs"] == 22 * 21 // 2


def test_intent_map_works_on_a_different_dataset(hsid):
    d = client.get(f"/api/s/{hsid}/intent-map").json()
    assert len(d["intents"]) == 8 and not ATIS_ONLY.search(str(d))


def test_intent_map_is_unavailable_without_a_classifier():
    import zipfile as _z, io as _io
    src = _z.ZipFile(next(iter((ROOT / "examples").glob("*.zip"))))
    buf = _io.BytesIO()
    with _z.ZipFile(buf, "w") as out:
        for info in src.infolist():
            if not info.filename.endswith("classifier_model.json"):
                out.writestr(info, src.read(info))
    r = open_bundle(buf.getvalue(), "noclf.zip"); assert r.status_code == 200
    c = client.get(f"/api/s/{r.json()['id']}/intent-map")
    assert c.status_code == 409 and "NO_MODEL" in c.text


def test_old_html_frontend_is_gone():
    assert not (ROOT / "app" / "static").exists()


# ---------- Studio 4: every Streamlit page renders on both bundles ----------
VIEWS = ["home", "overview", "colab", "dataset", "intents", "models", "confusion", "predictions", "errors", "confidence", "querylab", "autocomplete",
         "model_lab", "compare", "files"]


def _page(view, bundle):
    import sys, importlib
    from pathlib import Path as _P
    root = _P.cwd(); sys.path.insert(0, str(root))
    import streamlit as st
    from studio import core, ui
    ui.css()
    if "run" not in st.session_state:
        if bundle == "example":
            core.ensure_run()
        else:
            core.open_upload((root / bundle).read_bytes(), _P(bundle).name)
    mod = importlib.import_module(f"studio.views.{view}")
    try:
        mod.render()
    except core.Unavailable as exc:
        ui.unavailable(str(exc))


@pytest.mark.parametrize("bundle", ["example", "tests/fixtures/hotel_run_bundle.zip"])
@pytest.mark.parametrize("view", VIEWS)
def test_every_streamlit_page_renders(view, bundle):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_page, args=(view, bundle), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "undefined" not in text and "NaN" not in text


def test_hotel_run_pages_never_show_atis_text():
    from streamlit.testing.v1 import AppTest
    for view in ("intents", "confusion", "errors", "dataset"):
        at = AppTest.from_function(_page, args=(view, "tests/fixtures/hotel_run_bundle.zip"), default_timeout=120)
        at.run()
        text = " ".join(m.value for m in at.markdown) + " ".join(str(d.value) for d in at.dataframe)
        assert not ATIS_ONLY.search(text), view


def test_colour_themes_never_change_the_meaning_colours():
    from studio import ui
    assert len(ui.THEMES) == 7 and ui.DEFAULT_THEME in ui.THEMES
    semantic = {"#2DD4BF", "#FF7A85", "#F5B93E"}          # correct, wrong, warning
    for name, t in ui.THEMES.items():
        assert {"brand", "accent", "glow", "bg", "side", "s2", "blurb"} <= set(t), name
        assert t["brand"].upper() not in semantic, name
        assert not ({"teal", "red", "amber"} & set(t)), name
    assert "{brand}" in ui.CSS and "{accent}" in ui.CSS and "#5EA8FF" not in ui.CSS
