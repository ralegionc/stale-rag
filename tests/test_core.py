from datetime import date

from stale_rag.conditions import conditions, render
from stale_rag.facts import parse_day, pick_change
from stale_rag.passages import find_passage, name_patterns, wikitext_paragraphs
from stale_rag.readers import match_answer

TODAY = date(2026, 10, 10)


def st(v, start, end=None):
    return {"subject": "Q1", "value": v, "start": date.fromisoformat(start),
            "end": date.fromisoformat(end) if end else None}


def test_parse_day():
    assert parse_day("2019-07-24T00:00:00Z") == date(2019, 7, 24)
    assert parse_day("t12345") is None
    assert parse_day(None) is None


def test_pick_change_clean_succession():
    cur, old = pick_change([st("A", "2010-01-01", "2019-07-24"), st("B", "2019-07-24")], TODAY)
    assert (cur["value"], old["value"]) == ("B", "A")


def test_pick_change_rejects_two_current_holders():
    assert pick_change([st("A", "2010-01-01"), st("B", "2019-07-24")], TODAY) is None


def test_pick_change_rejects_gap_and_too_recent():
    assert pick_change([st("A", "2001-01-01", "2005-01-01"), st("B", "2019-07-24")], TODAY) is None
    assert pick_change([st("A", "2010-01-01", "2026-09-01"), st("B", "2026-09-01")], TODAY) is None


def test_pick_change_rejects_short_predecessor_term():
    assert pick_change([st("A", "2019-05-01", "2019-07-24"), st("B", "2019-07-24")], TODAY) is None


def test_wikitext_paragraphs_strip_markup():
    wt = ("{{Infobox country|leader=Someone}}\n'''Ruritania''' is led by "
          "[[Prime Minister of Ruritania|Prime Minister]] [[Theresa May]].<ref>{{cite web|x=1}}</ref>"
          " She took office in 2016.\n\n== History ==\n* a list item that is long enough to count\n")
    assert wikitext_paragraphs(wt) == [
        "Ruritania is led by Prime Minister Theresa May. She took office in 2016."]


def test_wikitext_paragraphs_unwrap_inline_templates():
    wt = ("The city covers {{convert|156|km2|sqmi}} and its parliament, the "
          "{{lang|de|Landtag}}, sits in the old town hall by the river.")
    assert wikitext_paragraphs(wt) == [
        "The city covers 156 km2 and its parliament, the Landtag, sits in the old town hall by the river."]


LONG = " The office carries wide executive powers under the national constitution."


def test_find_passage_prefers_current_tense_cue():
    paras = ["In 2017 the party, led by Andrej Babis, won the election outright." + LONG,
             "The incumbent prime minister is Andrej Babis, who took office in 2025." + LONG]
    text, _ = find_passage(paras, name_patterns(["Andrej Babis"]), name_patterns(["Petr Fiala"]))
    assert text.startswith("The incumbent")


def test_find_passage_excludes_other_holder():
    paras = ["Theresa May replaced David Cameron in 2016. She was born in Eastbourne.",
             "The current prime minister is Theresa May, leader of the Conservatives." + LONG]
    text, kind = find_passage(paras, name_patterns(["Theresa May"]), name_patterns(["David Cameron"]))
    assert "Cameron" not in text and "Theresa May" in text and kind == "full"


def test_find_passage_skips_hatnotes_and_fragments():
    paras = ["For the current state government, see Cabinet Laschet.", "Armin Laschet leads."]
    assert find_passage(paras, name_patterns(["Armin Laschet"]), name_patterns(["Hendrik Wust"])) \
        == (None, None)


def test_match_answer():
    task = {"new": {"aliases": ["Boris Johnson", "Alexander Boris de Pfeffel Johnson"]},
            "old": {"aliases": ["Theresa May"]}}
    assert match_answer("Boris Johnson.", task) == "new"
    assert match_answer("It was Theresa May, then Johnson", task) == "both"
    assert match_answer("Keir Starmer", task) == "neither"


def test_conditions_and_render():
    cs = conditions()
    assert len(cs) == 2 + 4 * 3 * 2
    task = {"question": "Who is the head of government of X?",
            "old": {"passage": "OLD TEXT", "rev_date": "2019-03-01"},
            "new": {"passage": "NEW TEXT", "rev_date": "2026-10-01"}}
    _, user = render(task, {"context": "old_new", "dates": "swapped", "instruct": "plain"}, "2026-10-10")
    assert user.index("OLD TEXT") < user.index("NEW TEXT")
    assert "Source 1 (Wikipedia, revision of 2026-10-01):\nOLD TEXT" in user
    _, closed = render(task, {"context": "closed", "dates": "none", "instruct": "recency"}, "2026-10-10")
    assert "TEXT" not in closed and "2026-10-10" in closed


def test_sentences_keep_initials_together():
    from stale_rag.passages import sentences
    text = "The bank is led by C. S. Venkatakrishnan. He took over in 2021. Dr. Lee advises."
    assert sentences(text) == ["The bank is led by C. S. Venkatakrishnan.",
                               "He took over in 2021.", "Dr. Lee advises."]
