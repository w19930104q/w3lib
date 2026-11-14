from __future__ import annotations

# The memory-boundedness tests read the parser buffer, and "== []" asserts
# the exact result, not a falsey one.
# pylint: disable=protected-access,use-implicit-booleaness-not-comparison
import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from w3lib.links import _LinkParser, iter_links

if TYPE_CHECKING:
    from collections.abc import Iterator

FIXTURES = Path(__file__).parent / "fixtures" / "links"
BASE = "http://example.com/"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def chunked(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


class TestFixtures:
    def test_catalog(self):
        assert list(iter_links(fixture("catalog.html"), BASE)) == [
            "http://example.com/archive/page-1",
            "http://example.com/archive/page-2?b=1&c=2",
            "mailto:someone@example.com",
            "http://example.com/archive/page-3",
            "http://example.com/archive/page-4",
            "http://example.com/archive/",
        ]

    def test_malformed(self):
        assert list(iter_links(fixture("malformed.html"), BASE)) == [
            f"http://example.com/{name}"
            for name in ("first", "second", "third", "fourth", "fifth", "sixth")
        ]

    def test_unicode(self):
        assert list(iter_links(fixture("unicode.html"), BASE)) == [
            "http://example.com/r%C3%A9sum%C3%A9",
            "http://example.com/caf%C3%A9?q=caf%C3%A9",
        ]

    @pytest.mark.parametrize("name", ["catalog.html", "malformed.html", "unicode.html"])
    @pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 64])
    def test_chunk_size_does_not_change_links(self, name, size):
        text = fixture(name)
        whole = list(iter_links(text, BASE))
        assert list(iter_links(chunked(text, size), BASE)) == whole

    @pytest.mark.parametrize("name", ["catalog.html", "malformed.html", "unicode.html"])
    @pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 64])
    def test_byte_chunks(self, name, size):
        data = (FIXTURES / name).read_bytes()
        whole = list(iter_links(data, BASE))
        chunks = [data[i : i + size] for i in range(0, len(data), size)]
        assert list(iter_links(chunks, BASE)) == whole

    def test_mixed_str_and_bytes_chunks(self):
        text = fixture("catalog.html")
        midpoint = len(text) // 2
        chunks: list[str | bytes] = [text[:midpoint], text[midpoint:].encode()]
        assert list(iter_links(chunks, BASE)) == list(iter_links(text, BASE))


class TestStreaming:
    def test_first_link_comes_from_the_first_chunk(self):
        consumed = []

        def chunks() -> Iterator[str]:
            for i in itertools.count():
                consumed.append(i)
                yield f'<a href="/{i}">x</a>'

        links = iter_links(chunks(), BASE)
        assert next(links) == "http://example.com/0"
        # The first link is out before the second chunk is even read.
        assert consumed == [0]
        assert next(links) == "http://example.com/1"

    def test_input_is_not_consumed_ahead_of_output(self):
        def chunks() -> Iterator[str]:
            yield '<a href="/a">x</a>'
            raise AssertionError("input consumed past the first link")

        links = iter_links(chunks(), BASE)
        assert next(links) == "http://example.com/a"

    def test_memory_does_not_grow_with_text_nodes(self):
        parser = _LinkParser(BASE, "utf-8", 65536)
        for i in range(20000):
            assert parser.feed(f"<p>node {i} with some text</p>") == []
            assert len(parser._buf) < 1024
        assert parser.close() == []

    def test_memory_does_not_grow_with_comment_or_script_content(self):
        parser = _LinkParser(BASE, "utf-8", 65536)
        parser.feed("<!--")
        for _ in range(10000):
            parser.feed("filler " * 100)
            assert len(parser._buf) < 1024
        parser.feed("--><script>")
        for _ in range(10000):
            parser.feed("filler " * 100)
            assert len(parser._buf) < 1024
        assert parser.feed("</script>") == []

    def test_links_are_yielded_per_feed(self):
        parser = _LinkParser(BASE, "utf-8", 65536)
        assert parser.feed('<a href="/a">') == ["http://example.com/a"]
        assert parser.feed("text without markup") == []
        assert parser.feed('<a href="/b">') == ["http://example.com/b"]


class TestBaseUrl:
    def test_first_base_wins(self):
        html = (
            '<base href="http://a.example/">'
            '<base href="http://b.example/">'
            '<a href="x">1</a>'
        )
        assert list(iter_links(html, BASE)) == ["http://a.example/x"]

    def test_relative_base_resolves_against_document_url(self):
        html = '<base href="sub/"><a href="x">1</a>'
        assert list(iter_links(html, "http://example.com/dir/page")) == [
            "http://example.com/dir/sub/x"
        ]

    def test_empty_base_href_freezes_the_fallback(self):
        html = '<base href><base href="http://b.example/"><a href="x">1</a>'
        assert list(iter_links(html, "http://example.com/dir/")) == [
            "http://example.com/dir/x"
        ]

    def test_unusable_base_href_freezes_the_fallback(self):
        for href in ('href="javascript:x"', 'href="data:x"'):
            html = f"<base {href}><a href='/x'>1</a>"
            assert list(iter_links(html, BASE)) == ["http://example.com/x"]

    def test_base_without_href_does_not_freeze(self):
        html = '<base target="_blank"><base href="http://b.example/"><a href="x">1</a>'
        assert list(iter_links(html, BASE)) == ["http://b.example/x"]

    def test_base_in_comment_or_script_is_ignored(self):
        html = (
            '<!-- <base href="http://c.example/"> -->'
            '<script><base href="http://s.example/"></script>'
            '<base href="http://b.example/">'
            '<a href="x">1</a>'
        )
        assert list(iter_links(html, BASE)) == ["http://b.example/x"]

    def test_links_before_base_use_the_document_url(self):
        html = (
            '<a href="/early">1</a><base href="http://b.example/"><a href="late">2</a>'
        )
        assert list(iter_links(html, BASE)) == [
            "http://example.com/early",
            "http://b.example/late",
        ]

    def test_base_href_entities_are_decoded(self):
        html = '<base href="http://example.com/?a=1&amp;b=2"><a href="">1</a>'
        assert list(iter_links(html, BASE)) == ["http://example.com/?a=1&b=2"]


class TestRecovery:
    def test_oversized_unterminated_tag_is_given_up_on(self):
        # No second quote: the value of title never closes.
        html = '<a title="' + "x" * 1000 + "<a href=/recovered>"
        assert list(iter_links(html, BASE, max_tag_size=64)) == [
            "http://example.com/recovered"
        ]

    def test_oversized_tag_name_is_given_up_on(self):
        html = "<" + "a" * 1000 + '<a href="/recovered">'
        assert list(iter_links(html, BASE, max_tag_size=64)) == [
            "http://example.com/recovered"
        ]

    def test_oversized_declaration_is_given_up_on(self):
        # No ">" in the first chunk: the declaration never closes, the parser
        # recovers and the next chunk is read normally.
        chunks = ["<!" + "x" * 1000, '<a href="/recovered">']
        assert list(iter_links(chunks, BASE, max_tag_size=64)) == [
            "http://example.com/recovered"
        ]

    def test_unterminated_tag_at_eof_is_dropped(self):
        html = '<a href="/kept">1</a><a href="/dropped"'
        assert list(iter_links(html, BASE)) == ["http://example.com/kept"]

    def test_unterminated_quote_at_eof_is_dropped(self):
        html = '<a href="/kept">1</a><a href="dropped'
        assert list(iter_links(html, BASE)) == ["http://example.com/kept"]

    def test_unterminated_comment_swallows_the_rest(self):
        html = '<a href="/kept">1</a><!-- <a href="/dropped">'
        assert list(iter_links(html, BASE)) == ["http://example.com/kept"]

    def test_unterminated_script_swallows_the_rest(self):
        html = '<a href="/kept">1</a><script><a href="/dropped">'
        assert list(iter_links(html, BASE)) == ["http://example.com/kept"]

    def test_stray_markup_characters(self):
        html = 'a < b, </, <?, <!x, <3, < a href="/not-a-link">, <a href="/kept">'
        assert list(iter_links(html, BASE)) == ["http://example.com/kept"]

    def test_undecodable_bytes_do_not_abort_the_scan(self):
        data = b'<a href="/kept">\xff\xfe</a>'
        assert list(iter_links([data], BASE)) == ["http://example.com/kept"]

    def test_unknown_encoding_falls_back_to_utf8(self):
        data = '<a href="/café">x</a>'.encode()
        assert list(iter_links([data], BASE, encoding="not-an-encoding")) == [
            "http://example.com/caf%C3%A9"
        ]

    def test_empty_and_text_only_input(self):
        assert list(iter_links([], BASE)) == []
        assert list(iter_links("", BASE)) == []
        assert list(iter_links("plain text, no markup", BASE)) == []


class TestLinkSemantics:
    def test_duplicate_href_attributes_first_wins(self):
        html = '<a href="/first" href="/second">x</a>'
        assert list(iter_links(html, BASE)) == ["http://example.com/first"]

    def test_entities_in_href(self):
        html = '<a href="/?a=1&amp;b=2&#63;c=3">x</a>'
        assert list(iter_links(html, BASE)) == ["http://example.com/?a=1&b=2?c=3"]

    def test_entity_decoded_whitespace_is_stripped(self):
        html = '<a href="&#32;/padded&#32;">x</a>'
        assert list(iter_links(html, BASE)) == ["http://example.com/padded"]

    def test_quoted_angle_brackets_in_attributes(self):
        html = '<a href="/a?x=>y">1</a><a href="/b">2</a>'
        assert list(iter_links(html, BASE)) == [
            "http://example.com/a?x=%3Ey",
            "http://example.com/b",
        ]

    def test_unquoted_href(self):
        html = "<a href=/bare>1</a>"
        assert list(iter_links(html, BASE)) == ["http://example.com/bare"]

    def test_single_quoted_href(self):
        html = "<a href='/single'>1</a>"
        assert list(iter_links(html, BASE)) == ["http://example.com/single"]

    def test_fragment_only_and_empty_href_resolve_to_the_document(self):
        html = '<a href="#top">1</a><a href="">2</a>'
        assert list(iter_links(html, "http://example.com/page")) == [
            "http://example.com/page"
        ]

    def test_skip_content_elements(self):
        for tag in (
            "script",
            "style",
            "template",
            "title",
            "textarea",
            "iframe",
            "noembed",
            "noframes",
            "xmp",
        ):
            html = f"<{tag}><a href='/hidden-{tag}'>x</{tag}><a href='/seen-{tag}'>y"
            assert list(iter_links(html, BASE)) == [f"http://example.com/seen-{tag}"], (
                tag
            )

    def test_plaintext_swallows_the_rest(self):
        html = '<a href="/kept">1</a><plaintext><a href="/dropped">'
        assert list(iter_links(html, BASE)) == ["http://example.com/kept"]

    def test_noscript_content_is_parsed(self):
        html = '<noscript><a href="/kept">1</a></noscript>'
        assert list(iter_links(html, BASE)) == ["http://example.com/kept"]

    def test_self_closing_script_still_skips_content(self):
        # HTML has no self-closing script element.
        html = '<script/><a href="/dropped">x</script><a href="/kept">y'
        assert list(iter_links(html, BASE)) == ["http://example.com/kept"]

    def test_end_tag_with_attributes_closes_script(self):
        html = '<script>x</script foo="bar"><a href="/kept">y'
        assert list(iter_links(html, BASE)) == ["http://example.com/kept"]

    def test_case_insensitive_markup(self):
        html = '<A HREF="/up">1</A><ScRiPt><a href="/hidden"></sCrIpT>'
        assert list(iter_links(html, BASE)) == ["http://example.com/up"]

    def test_area_links(self):
        html = '<map><area href="/a"><area href="/b"></map>'
        assert list(iter_links(html, BASE)) == [
            "http://example.com/a",
            "http://example.com/b",
        ]

    def test_javascript_and_data_links_are_dropped(self):
        html = (
            '<a href="javascript:void(0)">1</a>'
            '<a href="JaVaScRiPt:void(0)">2</a>'
            '<a href="data:text/html,x">3</a>'
            '<a href="/kept">4</a>'
        )
        assert list(iter_links(html, BASE)) == ["http://example.com/kept"]

    def test_relative_links_against_relative_baseurl_are_kept_relative(self):
        # Without an absolute document URL there is nothing to resolve
        # against; absolute-path links come out as they are.
        assert list(iter_links('<a href="/x">1</a>')) == ["/x"]

    def test_document_order_is_preserved(self):
        html = "".join(f'<a href="/{i}">{i}</a>' for i in range(100))
        assert list(iter_links(html, BASE)) == [
            f"http://example.com/{i}" for i in range(100)
        ]
