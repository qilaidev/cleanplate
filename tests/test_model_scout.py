from __future__ import annotations

from datetime import UTC, datetime

from wm_platform.model_scout import (
    Candidate,
    classify_license,
    parse_arxiv_feed,
    parse_hf_models,
    rank,
    render_markdown,
)

SINCE = datetime(2026, 9, 1, tzinfo=UTC)


def test_classify_license():
    assert classify_license("apache-2.0") == "permissive"
    assert classify_license("MIT") == "permissive"
    assert classify_license("cc-by-nc-4.0") == "non_commercial"
    assert classify_license("agpl-3.0") == "copyleft"
    assert classify_license("other") == "unknown"
    assert classify_license(None) == "unknown"


def test_parse_hf_models_filters_window_and_reads_license():
    items = [
        {"id": "a/video-remover", "lastModified": "2026-09-20T00:00:00.000Z", "likes": 3,
         "tags": ["license:apache-2.0", "video", "inpainting"], "pipeline_tag": "image-to-video"},
        {"id": "b/old", "lastModified": "2026-06-01T00:00:00.000Z", "tags": []},
    ]
    found = parse_hf_models(items, SINCE)
    assert [c.name for c in found] == ["a/video-remover"]
    assert found[0].license == "apache-2.0" and found[0].license_class == "permissive"


def test_parse_arxiv_feed_and_rank():
    feed = """<?xml version='1.0'?><feed xmlns="http://www.w3.org/2005/Atom">
    <entry><id>http://arxiv.org/abs/2609.00001v1</id><published>2026-09-10T00:00:00Z</published>
    <title>Mask-free video subtitle removal</title><summary>Code at github.com/x/y.</summary></entry>
    <entry><id>http://arxiv.org/abs/2609.00002v1</id><published>2026-09-11T00:00:00Z</published>
    <title>Invisible watermark removal attacks</title><summary>video watermark removal</summary></entry>
    <entry><id>http://arxiv.org/abs/2605.00003v1</id><published>2026-05-01T00:00:00Z</published>
    <title>Old video inpainting</title><summary>video</summary></entry></feed>"""
    found = parse_arxiv_feed(feed, SINCE)
    assert len(found) == 2
    ranked = rank(found)
    assert [c.name for c in ranked] == ["Mask-free video subtitle removal"]
    assert ranked[0].tags == ["code-link"]
    text = render_markdown(ranked, days=30, now=datetime(2026, 9, 30, tzinfo=UTC), errors=["github: 403"])
    assert "github: 403" in text and "Mask-free video subtitle removal" in text
    assert "_none in window_" in text  # empty HF/GitHub sections still render


def test_rank_prefers_relevance_then_popularity():
    a = Candidate("github", "x/video-watermark-remover", "u", SINCE, popularity=1)
    b = Candidate("github", "y/video-inpainting", "u", SINCE, popularity=100)
    assert rank([b, a])[0].name == "x/video-watermark-remover"
