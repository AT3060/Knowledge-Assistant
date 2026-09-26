"""Step 15 - Citation parsing, abstention detection and prompt building (src/generate.py)."""
from generate import NO_ANSWER, build_messages, is_abstention, parse_citations


def test_parse_citations_all_formats():
    valid, invalid = parse_citations("A [1]. B [2, 3]. C [1][4].", n_passages=5)
    assert valid == [1, 2, 3, 4]          # in order of first appearance, no duplicates
    assert invalid == []


def test_parse_citations_out_of_range_numbers_are_invalid():
    valid, invalid = parse_citations("Siehe [9] und [0], sonst [2].", n_passages=5)
    assert valid == [2]
    assert invalid == [0, 9]


def test_parse_citations_no_citation():
    assert parse_citations("Nein, der PACS ist nicht verfügbar.", 5) == ([], [])


def test_abstention_detection():
    assert is_abstention(NO_ANSWER)
    assert is_abstention("Leider: dazu finde ich in den Dokumenten keine Information.")
    assert not is_abstention("Der Antrag wird von der Leitung der Radiologie genehmigt [1].")


def test_build_messages_numbers_passages_and_keeps_rules(chunks):
    passages = chunks[:3]
    msgs = build_messages("Wer genehmigt den PACS-Zugang?", passages)
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert NO_ANSWER in system                          # the abstention rule is in the prompt
    for n, c in enumerate(passages, start=1):
        assert f"[{n}]" in user and f"Seite {c['page']}" in user
    assert user.rstrip().endswith("Frage: Wer genehmigt den PACS-Zugang?")
