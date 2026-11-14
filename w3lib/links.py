"""
Streaming extraction of crawlable links from HTML documents.

:func:`iter_links` reads a document as an iterable of chunks and yields the
absolute URLs of its ``<a>`` and ``<area>`` links as soon as they are parsed,
without ever building a DOM or holding the whole document in memory. It is
meant for a first pass over large HTML archives, to collect the links worth
fetching before any heavier processing.

The scan follows the same tag grammar as the rest of :mod:`w3lib.html`:

-   ``<base>`` sets the base URL the way :func:`w3lib.html.get_base_url`
    reads it: the first ``<base>`` with an ``href`` attribute wins, and later
    ones are ignored. Links that appear before it resolve against the
    document URL alone, so a ``<base>`` should sit in ``<head>``, as in any
    well-formed document.
-   Character references in attribute values are decoded, e.g. ``&amp;``.
-   Comments, markup declarations and the content of ``<script>``,
    ``<style>``, ``<template>``, ``<title>``, ``<textarea>``, ``<iframe>``,
    ``<noembed>``, ``<noframes>``, ``<xmp>`` and ``<plaintext>`` elements are
    skipped, the way a browser does not parse links out of them. Nested
    ``<template>`` elements are not tracked: the content up to the first
    ``</template>`` is skipped. ``<noscript>`` content is parsed, as a
    crawler without JavaScript sees it.
-   An unterminated tag, comment or element content runs to the end of the
    document, as browsers treat it. A tag that grows past *max_tag_size*
    without closing is given up on: its ``<`` is taken as text and the scan
    resumes after it, so a single malformed tag cannot stall the extraction
    nor grow the memory bound.

Each link is made absolute with :func:`w3lib.url.safe_url_string`, stripped
of its fragment, and yielded only the first time it occurs, in document
order. ``javascript:`` and ``data:`` links are not yielded: nothing can be
fetched from them. Links of other schemes (``mailto:``, ``tel:``, …) are
yielded; filter them in the consumer if they are not wanted.

Memory use is bounded by the largest single tag (capped at *max_tag_size*),
the largest chunk, and the set of links already yielded; it does not grow
with the number of text nodes of the document.
"""

from __future__ import annotations

import codecs
import re
from typing import TYPE_CHECKING

from w3lib._util import _attr_re, iter_tag_attributes
from w3lib.html import HTML5_WHITESPACE, replace_entities
from w3lib.url import _urljoin, safe_url_string

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = ["iter_links"]


# Elements whose content a browser does not parse as markup, and the pattern
# of the end tag of each: the name followed by HTML whitespace, "/" or ">".
# Their content is consumed in runs of characters that cannot start the end
# tag, so a large inline script costs a search per chunk, not per character.
_SKIP_CONTENT_END = {
    name: re.compile(rf"</{name}[\t\n\f\r />]", re.IGNORECASE | re.ASCII)
    for name in (
        "script",
        "style",
        "template",
        "title",
        "textarea",
        "iframe",
        "noembed",
        "noframes",
        "xmp",
    )
}
# "<plaintext>" has no end tag: the rest of the document is text.
_PLAINTEXT = "plaintext"

_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
# Characters that can decide how a tag body continues: "<" ends it without
# closing it, ">" closes it, "=" can open a quoted value, and a quote that
# follows no "=" is an ordinary character of the tag.
_TAG_META_RE = re.compile(r"[<>=\"']")
_TAG_WS_RE = re.compile(r"[\t\n\f\r ]*")

# Parser states.
_TEXT = 0
_COMMENT = 1
_RAWTEXT = 2


def _scan_tag_end(buf: str, start: int) -> int | None:
    """Return the index just after the ``>`` that ends the tag whose body
    starts at *start* in *buf*.

    Return ``None`` if *buf* does not hold the whole tag yet, and ``~i`` if
    the ``<`` at index *i* ends the tag without closing it, as
    :data:`w3lib.html._TAG_BODY` reads it: a quoted attribute value is
    consumed whole, so that an angle bracket in it does not end the tag, and
    a quote opens a value only right after the ``=`` of an attribute.
    """
    n = len(buf)
    i = start
    while True:
        m = _TAG_META_RE.search(buf, i)
        if m is None:
            return None
        c = m.group()
        if c == ">":
            return m.end()
        if c == "<":
            return ~m.start()
        if c == "=":
            ws_match = _TAG_WS_RE.match(buf, m.end())
            j = ws_match.end() if ws_match is not None else m.end()
            if j >= n:
                return None
            q = buf[j]
            if q in {"'", '"'}:
                k = buf.find(q, j + 1)
                if k == -1:
                    return None
                i = k + 1
            else:
                # An unquoted value, read as ordinary tag text.
                i = j
        else:
            # A quote that follows no "=" is an ordinary character.
            i = m.end()


class _LinkParser:  # pylint: disable=too-many-instance-attributes
    """Incremental scanner behind :func:`iter_links`.

    ``feed()`` takes decoded text and returns the links it completes;
    ``close()`` flushes the end of the document. Only unprocessed markup is
    kept between feeds: at most one tag of up to *max_tag_size* bytes, or a
    few characters that may start a comment or an end tag.
    """

    def __init__(self, baseurl: str, encoding: str, max_tag_size: int) -> None:
        self._encoding = encoding
        self._max_tag_size = max_tag_size
        self._document_url = safe_url_string(baseurl) if baseurl else ""
        self._base = self._document_url
        self._base_frozen = False
        self._seen: set[str] = set()
        self._buf = ""
        self._state = _TEXT
        self._rawtext = ""
        self._out: list[str] = []

    def feed(self, text: str) -> list[str]:
        """Consume *text* and return the links it completes, in order."""
        if not text:
            return []
        self._buf += text
        self._process(final=False)
        out, self._out = self._out, []
        return out

    def close(self) -> list[str]:
        """Flush the end of the document and return the remaining links."""
        self._process(final=True)
        self._buf = ""
        out, self._out = self._out, []
        return out

    def _process(self, final: bool) -> None:  # pylint: disable=too-many-statements
        buf = self._buf
        n = len(buf)
        pos = 0
        while pos < n:
            if self._state == _COMMENT:
                # buf[pos:] is comment content, discarded as it is scanned.
                k = buf.find("--", pos)
                if k == -1:
                    # Keep the last character: it may be the first "-" of the
                    # closer, split across chunks.
                    pos = n if final else max(pos, n - 1)
                    break
                if k + 2 >= n:
                    pos = n if final else k
                    break
                c = buf[k + 2]
                if c == ">":
                    pos = k + 3
                    self._state = _TEXT
                    continue
                if c == "!":
                    if k + 3 >= n:
                        pos = n if final else k
                        break
                    if buf[k + 3] == ">":
                        pos = k + 4
                        self._state = _TEXT
                        continue
                pos = k + 2
                continue
            if self._state == _RAWTEXT:
                name = self._rawtext
                if name == _PLAINTEXT:
                    pos = n
                    break
                m = _SKIP_CONTENT_END[name].search(buf, pos)
                if m is None:
                    # Keep a tail as long as a partial end tag can be.
                    pos = n if final else max(pos, n - (len(name) + 3))
                    break
                gt = buf.find(">", m.end() - 1)
                if gt == -1:
                    if final:
                        pos = n
                        break
                    if n - m.start() > self._max_tag_size:
                        # The end tag never closes: keep scanning its text as
                        # element content, as a browser does.
                        pos = m.end()
                        continue
                    pos = m.start()
                    break
                pos = gt + 1
                self._state = _TEXT
                continue
            i = buf.find("<", pos)
            if i == -1:
                pos = n
                break
            if i + 1 >= n:
                # A "<" at the end of the buffer may start markup.
                pos = n if final else i
                break
            c = buf[i + 1]
            if c == "!":
                if buf.startswith("<!--", i):
                    if n - i < 6:
                        # Not enough input yet to tell the abrupt closers
                        # "<!-->" and "<!--->" apart from comment content.
                        pos = n if final else i
                        break
                    if buf[i + 4] == ">":
                        pos = i + 5
                        continue
                    if buf[i + 4] == "-" and buf[i + 5] == ">":
                        pos = i + 6
                        continue
                    self._state = _COMMENT
                    pos = i + 4
                    continue
                if n - i < 4 and not final and buf[i:] == "<!--"[: n - i]:
                    # A prefix of "<!--" split across chunks.
                    pos = i
                    break
                # A markup declaration or bogus comment, up to the next ">".
                gt = buf.find(">", i + 2)
                if gt == -1:
                    if final:
                        pos = n
                        break
                    if n - i > self._max_tag_size:
                        pos = i + 1  # recover: the "<" is text
                        continue
                    pos = i
                    break
                pos = gt + 1
                continue
            if c in {"?", "/"}:
                # A processing instruction or an end tag: nothing to extract,
                # skip to the next ">".
                gt = buf.find(">", i + 2)
                if gt == -1:
                    if final:
                        pos = n
                        break
                    if n - i > self._max_tag_size:
                        pos = i + 1  # recover: the "<" is text
                        continue
                    pos = i
                    break
                pos = gt + 1
                continue
            if c.isascii() and c.isalpha():
                k = i + 1
                while k < n and buf[k] in _NAME_CHARS:
                    k += 1
                if k == n:
                    # The tag name may continue in the next chunk; at the end
                    # of the document the tag is unterminated and dropped.
                    if final:
                        pos = n
                        break
                    if n - i > self._max_tag_size:
                        pos = i + 1  # recover: the "<" is text
                        continue
                    pos = i
                    break
                if buf[k] not in "\t\n\f\r </>":
                    # Not the name of an HTML element, e.g. "<3" or "<a!>":
                    # read the tag plainly, up to the next ">".
                    gt = buf.find(">", k)
                    if gt == -1:
                        if final:
                            pos = n
                            break
                        if n - i > self._max_tag_size:
                            pos = i + 1  # recover: the "<" is text
                            continue
                        pos = i
                        break
                    pos = gt + 1
                    continue
                end = _scan_tag_end(buf, k)
                if end is None:
                    if final:
                        # EOF in a tag drops it, as in a browser.
                        pos = n
                        break
                    if n - i > self._max_tag_size:
                        pos = i + 1  # recover: the "<" is text
                        continue
                    pos = i
                    break
                if end < 0:
                    # A "<" ends the tag without closing it; it may itself
                    # start markup, so the scan resumes from it.
                    pos = ~end
                    continue
                name = buf[i + 1 : k].lower()
                self._handle_tag(name, buf[k : end - 1])
                if name in _SKIP_CONTENT_END or name == _PLAINTEXT:
                    self._state = _RAWTEXT
                    self._rawtext = name
                pos = end
                continue
            # A "<" that starts no markup is text.
            pos = i + 1
        self._buf = buf[pos:]

    def _handle_tag(self, name: str, body: str) -> None:
        if name == "base":
            self._handle_base(body)
        elif name in {"a", "area"}:
            self._handle_link(body)

    def _handle_base(self, body: str) -> None:
        if self._base_frozen:
            return
        # iter_tag_attributes drops valueless attributes, but a valueless
        # href counts as one, so read the attributes here to see it, as
        # w3lib.html.get_base_url does.
        for attr in _attr_re.finditer(body):
            if attr["name"].lower() != "href":
                continue
            # The first <base> with an href attribute freezes the base URL,
            # even an empty or unusable one.
            self._base_frozen = True
            value = attr["double"] or attr["single"] or attr["bare"] or ""
            if "&" in value:
                value = replace_entities(value)
            if url := value.strip(HTML5_WHITESPACE):
                try:
                    url = safe_url_string(
                        _urljoin(self._document_url, url), encoding=self._encoding
                    )
                except ValueError:
                    return
                if not url.startswith(("data:", "javascript:")):
                    self._base = url
            return

    def _handle_link(self, body: str) -> None:
        for name, raw_value in iter_tag_attributes(body):
            if name != "href":
                continue
            # A reference is decoded into the value of its own attribute, so
            # a decoded quote or "=" is data.
            value = replace_entities(raw_value) if "&" in raw_value else raw_value
            url = value.strip(HTML5_WHITESPACE)
            try:
                url = safe_url_string(
                    _urljoin(self._base, url), encoding=self._encoding
                )
            except ValueError:
                # A URL that cannot be resolved, e.g. a relative URL against
                # an opaque base such as "data:…", is not a link to follow.
                return
            if not url:
                return
            scheme = url.partition(":")[0].lower()
            if scheme in ("javascript", "data"):
                return
            url = url.partition("#")[0]
            if url and url not in self._seen:
                self._seen.add(url)
                self._out.append(url)
            return


def _resolve_encoding(encoding: str) -> str:
    """Return *encoding* if it names a codec, else ``"utf-8"``."""
    encoding = encoding or "utf-8"
    try:
        codecs.lookup(encoding)
    except LookupError:
        return "utf-8"
    return encoding


def _incremental_decoder(encoding: str) -> codecs.IncrementalDecoder:
    return codecs.getincrementaldecoder(encoding)("replace")


def iter_links(
    chunks: Iterable[str | bytes] | str | bytes,
    baseurl: str = "",
    encoding: str = "utf-8",
    *,
    max_tag_size: int = 131072,
) -> Iterator[str]:
    """Yield the crawlable links of an HTML document, in document order.

    *chunks* is the document as an iterable of text or byte chunks (a single
    string or bytes object also works), read as a stream: the first link is
    yielded as soon as the tag that carries it is complete, and the input is
    consumed only as far as the links yielded so far require. Byte chunks are
    decoded incrementally with *encoding* (UTF-8 by default), so a multi-byte
    character split across chunks is read whole; undecodable bytes are
    replaced, they do not abort the scan.

    *baseurl* is the absolute URL of the document, used to resolve relative
    links, as the document URL of :func:`w3lib.html.get_base_url`. A
    ``<base href>`` of the document overrides it for the links that follow.

    Each link is yielded once, as an absolute URL without a fragment, made
    safe with :func:`w3lib.url.safe_url_string`. ``javascript:`` and
    ``data:`` links are not yielded.

    *max_tag_size* bounds how much markup a single unterminated tag (or
    comment-like construct) may span before the parser gives up on it and
    recovers by taking its ``<`` as text. Memory use is bounded by this
    value, the chunk size and the links already yielded; it does not grow
    with the number of text nodes of the document.

    >>> list(iter_links('<a href="/a">x</a><a href="/a#f">y</a>', "http://example.com/"))
    ['http://example.com/a']
    >>> list(iter_links(['<a href="', '/b">x</a>'], "http://example.com/"))
    ['http://example.com/b']
    """
    if isinstance(chunks, str):
        stream: Iterable[str | bytes] = (chunks,)
    elif isinstance(chunks, (bytes, bytearray, memoryview)):
        stream = (bytes(chunks),)
    else:
        stream = chunks
    encoding = _resolve_encoding(encoding)
    parser = _LinkParser(baseurl, encoding, max_tag_size)
    decoder: codecs.IncrementalDecoder | None = None
    for chunk in stream:
        if isinstance(chunk, str):
            text = chunk
        else:
            if decoder is None:
                decoder = _incremental_decoder(encoding)
            text = decoder.decode(bytes(chunk))
        yield from parser.feed(text)
    if decoder is not None:
        yield from parser.feed(decoder.decode(b"", final=True))
    yield from parser.close()
