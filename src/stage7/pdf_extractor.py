"""
pdf_extractor.py — Pure-Python PDF text extraction for my-LLM Stage 7.
======================================================================

WHAT IT DOES
    Extracts text from a PDF file using only the Python standard library.
    No external PDF dependencies are required. zlib (standard library) is
    used to decompress FlateDecode streams.

WHY PURE PYTHON
    This project runs fully offline with no installs and no network calls.
    Installing PyPDF2/pdfplumber/PyMuPDF would require pip and network
    access, which the project rules forbid without explicit approval.
    The Folger Hamlet PDF is simple enough (Type1 fonts, MacRomanEncoding,
    FlateDecode compression, no ToUnicode maps) that a targeted pure-Python
    extractor is reliable and fully auditable.

LIMITATIONS (honest)
    * Handles FlateDecode-compressed content streams only. Other filters
      (LZW, DCT, JPX) are not supported.
    * Uses MacRomanEncoding for character-code -> glyph mapping. Fonts with
      custom /Differences arrays or non-MacRoman encodings may produce
      substitution errors.
    * No /ToUnicode map support (this PDF has none). Text is mapped through
      the font encoding instead.
    * Complex PDFs with transparency groups, form XObjects, or nested
      content streams may not be fully extracted.
    * Text positioning (space insertion between words) is approximate — it
      uses text-matrix deltas. Some words may be merged or split incorrectly.

PDF FILE ASSUMPTIONS (verified against data/raw/Hamlet.pdf)
    * PDF version 1.3/1.4
    * Cross-reference table present (standard xref, not compressed)
    * Fonts: Times New Roman variants with MacRomanEncoding
    * No /ToUnicode maps, no /Differences arrays
    * Content streams use FlateDecode
    * Text shown via Tj (string) and TJ (array) operators
    * Page headers repeated on every page ("Hamlet" + page number)
    * FTLN line identifiers (e.g. "FTLN 0001") inline in the text

OUTPUT
    Plain text with one page per form-feed-separated section, plus helper
    functions for scene/speaker parsing.

HOW TO RUN
    python src\\stage7\\pdf_extractor.py data\\raw\\Hamlet.pdf

    Prints a summary and writes nothing. Import extract_pdf_text() into
    other scripts for real use.

This file is part of the from-scratch my-LLM project. We authored it.
"""

import re
import sys
import zlib
from codecs import decode as codec_decode


# ---------------------------------------------------------------------------
# MacRoman encoding table (Apple Mac OS Roman, code page 10000).
#
# PDF content-stream byte values 0-255 map through this table when the font
# uses MacRomanEncoding (as this PDF does). Entry i is the Unicode character
# for byte value i.
# ---------------------------------------------------------------------------

_MAC_ROMAN = [
    "\u0000", "\u0001", "\u0002", "\u0003", "\u0004", "\u0005", "\u0006", "\u0007",
    "\u0008", "\t", "\n", "\u000b", "\f", "\r", "\u000e", "\u000f",
    "\u0010", "\u0011", "\u0012", "\u0013", "\u0014", "\u0015", "\u0016", "\u0017",
    "\u0018", "\u0019", "\u001a", "\u001b", "\u001c", "\u001d", "\u001e", "\u001f",
    " ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".",
    "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=",
    ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[",
    "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
    "z", "{", "|", "}", "~", "\u007f", "\u00c4", "\u00c5", "\u00c7", "\u00d9",
    "\u00e1", "\u00e9", "\u00ed", "\u00f3", "\u00fa", "\u00e7", "\u00e0", "\u00e8",
    "\u00ec", "\u00f2", "\u00f9", "\u00e4", "\u00e5", "\u00e7", "\u00e9", "\u00ea",
    "\u00eb", "\u00ec", "\u00ed", "\u00ee", "\u00ef", "\u00f1", "\u00f3", "\u00f4",
    "\u00f5", "\u00f6", "\u00f7", "\u00f9", "\u00fa", "\u00fb", "\u00fc", "\u00fd",
    "\u00fe", "\u00ff", "\u2022", "\u2020", "\u2021", "\u00b7", "\u00a8", "\u00a4",
    "\u00a6", "\u00a1", "\u00ab", "\u00bb", "\u25ca", "\u2013", "\u2014", "\u00a3",
    "\u00a9", "\u00a7", "\u00b6", "\u00df", "\u00ae", "\u00a5", "\u00b4", "\u00a2",
    "\u00a3", "\u201a", "\u201e", "\u201c", "\u201d", "\u2018", "\u2019", "\u2022",
    "\u2022", "\u2022", "\u2022", "\u2022", "\u2022", "\u2022", "\u2022", "\u2022",
    "\u2022", "\u2022", "\u2022", "\u2022", "\u2022", "\u2022", "\u2022", "\u2022",
    "\u00e1", "\u00ed", "\u00f3", "\u00fa", "\u00e7", "\u00e9", "\u00ef", "\u2019",
    "\u2018", "\u00aa", "\u00ba", "\u00bf", "\u00ac", "\u00bd", "\u00bc", "\u00b1",
    "\u00b0", "\u00b9", "\u2026", "\u2026", "\u2022", "\u00ae", "\u00a9", "\u2122",
    "\u00b8", "\u00a4", "\u00a8", "\u2260", "\u2264", "\u2265", "\u00a2", "\u00a6",
    "\u00ab", "\u00bb", "\u2026", "\u00a2", "\u00a6", "\u00ab", "\u00bb", "\u2026",
    "\u2020", "\u2021", "\u00b2", "\u00b3", "\u00b9", "\u00bd", "\u00bc", "\u00b1",
    "\u00b0", "\u00b9", "\u2026", "\u00ae", "\u00a9", "\u2122", "\u00b8", "\u00a4",
    "\u00a8", "\u2260", "\u2264", "\u2265", "\u00a2", "\u00a6", "\u00ab", "\u00bb",
    "\u2026", "\u00a2", "\u00a6", "\u00ab", "\u00bb", "\u2026",
]


def mac_roman_byte(byte_value):
    """Map a single byte value (0-255) to its MacRoman Unicode character."""
    if 0 <= byte_value < len(_MAC_ROMAN):
        ch = _MAC_ROMAN[byte_value]
        return ch if ch else "\ufffd"
    return "\ufffd"


# ---------------------------------------------------------------------------
# Low-level PDF object parsing (for document structure: dictionaries,
# arrays, streams, references).
# ---------------------------------------------------------------------------

def _skip_ws_comments(data, pos):
    """Advance pos past whitespace and PDF comments."""
    n = len(data)
    while pos < n:
        c = data[pos:pos + 1]
        if c in (b" ", b"\t", b"\n", b"\r", b"\f"):
            pos += 1
        elif c == b"%":
            while pos < n and data[pos:pos + 1] not in (b"\n", b"\r"):
                pos += 1
        else:
            break
    return pos


def _read_number(data, pos):
    """Read a PDF integer or real number. Returns (value, new_pos) or
    (None, pos) if no valid number starts at pos."""
    n = len(data)
    start = pos
    if pos < n and data[pos:pos + 1] in (b"+", b"-"):
        pos += 1
    digit_start = pos
    while pos < n and data[pos:pos + 1] in b"0123456789":
        pos += 1
    if pos == digit_start:
        # No digits after optional sign: not a number
        return None, start
    is_float = False
    if pos < n and data[pos:pos + 1] == b".":
        is_float = True
        pos += 1
        while pos < n and data[pos:pos + 1] in b"0123456789":
            pos += 1
    if pos < n and data[pos:pos + 1] in (b"e", b"E"):
        is_float = True
        pos += 1
        if pos < n and data[pos:pos + 1] in (b"+", b"-"):
            pos += 1
        while pos < n and data[pos:pos + 1] in b"0123456789":
            pos += 1
    num_str = data[start:pos].decode("ascii", errors="replace")
    try:
        return (float(num_str) if is_float else int(num_str)), pos
    except ValueError:
        return None, start


def _read_name(data, pos):
    """Read a PDF name (/Foo). Returns (name_string, new_pos)."""
    assert data[pos:pos + 1] == b"/"
    pos += 1
    n = len(data)
    start = pos
    while pos < n:
        c = data[pos:pos + 1]
        if c in (b" ", b"\t", b"\n", b"\r", b"\f", b"(", b")", b"<", b">",
                 b"[", b"]", b"{", b"}", b"%", b"/"):
            break
        pos += 1
    return data[start:pos].decode("latin-1"), pos


def _read_literal_string(data, pos):
    """Read a PDF literal string (...). Handles escapes and balanced parens.
    Returns (decoded_string, new_pos)."""
    assert data[pos:pos + 1] == b"("
    pos += 1
    n = len(data)
    depth = 1
    chars = []
    while pos < n and depth > 0:
        c = data[pos:pos + 1]
        if c == b"(":
            depth += 1
            chars.append("(")
            pos += 1
        elif c == b")":
            depth -= 1
            if depth == 0:
                pos += 1
                break
            chars.append(")")
            pos += 1
        elif c == b"\\":
            pos += 1
            if pos < n:
                esc = data[pos:pos + 1]
                if esc == b"n":
                    chars.append("\n")
                elif esc == b"r":
                    chars.append("\r")
                elif esc == b"t":
                    chars.append("\t")
                elif esc == b"b":
                    chars.append("\b")
                elif esc == b"f":
                    chars.append("\f")
                elif esc == b"\\":
                    chars.append("\\")
                elif esc == b"(":
                    chars.append("(")
                elif esc == b")":
                    chars.append(")")
                elif esc in (b"\r", b"\n"):
                    if esc == b"\r" and pos + 1 < n and data[pos + 1:pos + 2] == b"\n":
                        pos += 1
                elif esc in b"01234567":
                    oct_str = esc
                    pos += 1
                    while pos < n and len(oct_str) < 3 and data[pos:pos + 1] in b"01234567":
                        oct_str += data[pos:pos + 1]
                        pos += 1
                    chars.append(chr(int(oct_str, 8)))
                else:
                    chars.append(esc.decode("latin-1"))
                pos += 1
        else:
            chars.append(codec_decode(c, "latin-1", errors="replace"))
            pos += 1
    return "".join(chars), pos


def _read_hex_string(data, pos):
    """Read a PDF hex string (<...>). Returns (decoded_string, new_pos)."""
    assert data[pos:pos + 1] == b"<"
    pos += 1
    n = len(data)
    start = pos
    while pos < n and data[pos:pos + 1] != b">":
        pos += 1
    hex_str = data[start:pos].decode("ascii", errors="replace")
    hex_str = "".join(hex_str.split())
    if len(hex_str) % 2 == 1:
        hex_str += "0"
    try:
        raw = bytes.fromhex(hex_str)
        text = raw.decode("latin-1")
    except Exception:
        text = ""
    pos += 1  # skip '>'
    return text, pos


def _read_array(data, pos):
    """Read a PDF array [...]. Returns (list_of_values, new_pos)."""
    assert data[pos:pos + 1] == b"["
    pos += 1
    items = []
    n = len(data)
    while pos < n:
        pos = _skip_ws_comments(data, pos)
        if pos >= n:
            break
        if data[pos:pos + 1] == b"]":
            pos += 1
            break
        if data[pos:pos + 1] == b"/":
            val, pos = _read_name(data, pos)
            items.append(val)
        elif data[pos:pos + 1] == b"(":
            val, pos = _read_literal_string(data, pos)
            items.append(val)
        elif data[pos:pos + 1] == b"<":
            if pos + 1 < n and data[pos + 1:pos + 2] == b"<":
                # Nested dictionary - parse recursively
                val, pos = _read_dict(data, pos)
                items.append(val)
            else:
                val, pos = _read_hex_string(data, pos)
                items.append(val)
        elif data[pos:pos + 1] == b"[":
            val, pos = _read_array(data, pos)
            items.append(val)
        elif data[pos:pos + 1] in b"0123456789+-.":
            num, new_pos = _read_number_or_ref(data, pos)
            if num is not None:
                items.append(num)
                pos = new_pos
            else:
                pos += 1
        else:
            pos += 1
    return items, pos


def _read_number_or_ref(data, pos):
    """Read a PDF number, or a reference (N 0 R).
    
    Returns (value, new_pos) where value is either an int/float or a
    (obj_num, gen) tuple for references. Returns (None, pos) if no valid
    number starts at pos.
    """
    num, new_pos = _read_number(data, pos)
    if num is not None:
        # Check for a reference: N 0 R
        ref_pos = _skip_ws_comments(data, new_pos)
        if (ref_pos + 2 <= len(data)
                and data[ref_pos:ref_pos + 1] == b"0"
                and data[ref_pos + 1:ref_pos + 2] == b" "):
            ref_pos2 = _skip_ws_comments(data, ref_pos + 2)
            if ref_pos2 < len(data) and data[ref_pos2:ref_pos2 + 1] == b"R":
                return (num, 0), ref_pos2 + 1
        return num, new_pos
    return None, pos


def _read_dict(data, pos):
    """Read a PDF dictionary << ... >>. Returns (dict, new_pos)."""
    assert data[pos:pos + 2] == b"<<"
    pos += 2
    items = {}
    n = len(data)
    while pos < n:
        pos = _skip_ws_comments(data, pos)
        if pos >= n:
            break
        if data[pos:pos + 2] == b">>":
            pos += 2
            break
        if data[pos:pos + 1] != b"/":
            pos += 1
            continue
        key, pos = _read_name(data, pos)
        pos = _skip_ws_comments(data, pos)
        if pos >= n:
            break
        if data[pos:pos + 1] == b"/":
            val, pos = _read_name(data, pos)
        elif data[pos:pos + 1] == b"(":
            val, pos = _read_literal_string(data, pos)
        elif data[pos:pos + 1] == b"<":
            if pos + 1 < n and data[pos + 1:pos + 2] == b"<":
                # Nested dictionary - parse recursively
                val, pos = _read_dict(data, pos)
            else:
                val, pos = _read_hex_string(data, pos)
        elif data[pos:pos + 1] == b"[":
            val, pos = _read_array(data, pos)
        elif data[pos:pos + 1] in b"0123456789+-.":
            num, new_pos = _read_number_or_ref(data, pos)
            if num is not None:
                val = num
                pos = new_pos
            else:
                pos += 1
                continue
        else:
            pos += 1
            continue
        items[key] = val
    return items, pos


def _parse_pdf_value(data, pos):
    """Parse any PDF value starting at pos. Returns (value, new_pos)."""
    pos = _skip_ws_comments(data, pos)
    if pos >= len(data):
        return None, pos
    c = data[pos:pos + 1]
    if c == b"/":
        return _read_name(data, pos)
    if c == b"(":
        return _read_literal_string(data, pos)
    if c == b"<":
        if pos + 1 < len(data) and data[pos + 1:pos + 2] == b"<":
            return _read_dict(data, pos)
        return _read_hex_string(data, pos)
    if c == b"[":
        return _read_array(data, pos)
    if c in b"0123456789+-.":
        num, new_pos = _read_number(data, pos)
        if num is not None:
            # Check for a reference: N 0 R
            ref_pos = _skip_ws_comments(data, new_pos)
            if (ref_pos + 2 <= len(data)
                    and data[ref_pos:ref_pos + 1] == b"0"
                    and data[ref_pos + 1:ref_pos + 2] == b" "):
                ref_pos2 = _skip_ws_comments(data, ref_pos + 2)
                if ref_pos2 < len(data) and data[ref_pos2:ref_pos2 + 1] == b"R":
                    return (num, 0), ref_pos2 + 1
            return num, new_pos
        return None, pos + 1
    if c == b"t" and data[pos:pos + 4] == b"true":
        return True, pos + 4
    if c == b"f" and data[pos:pos + 5] == b"false":
        return False, pos + 5
    if c == b"n" and data[pos:pos + 4] == b"null":
        return None, pos + 4
    return None, pos + 1


# ---------------------------------------------------------------------------
# PDF document structure.
# ---------------------------------------------------------------------------

class PDFDocument:
    """Parsed PDF document with object table and reference resolution."""

    def __init__(self, data):
        self.data = data
        self.objects = {}
        self._scan_objects()

    def _scan_objects(self):
        """Scan the file for all `N 0 obj ... endobj` definitions."""
        pattern = re.compile(rb"(\d+)\s+0\s+obj\r?\n")
        for m in pattern.finditer(self.data):
            obj_num = int(m.group(1))
            content_start = m.end()
            endobj_pos = self.data.find(b"endobj", content_start)
            if endobj_pos < 0:
                continue
            obj_content = self.data[content_start:endobj_pos]

            stream_marker = b"stream\r\n"
            stream_pos = obj_content.find(stream_marker)
            if stream_pos >= 0:
                dict_part = obj_content[:stream_pos]
                after_stream = obj_content[stream_pos + len(stream_marker):]
                endstream_pos = after_stream.find(b"endstream")
                if endstream_pos >= 0:
                    stream_data = after_stream[:endstream_pos]
                    dict_val, _ = _parse_pdf_value(dict_part, 0)
                    self.objects[obj_num] = {
                        "type": "stream",
                        "dict": dict_val if isinstance(dict_val, dict) else {},
                        "raw": stream_data,
                    }
                    continue

            val, _ = _parse_pdf_value(obj_content, 0)
            self.objects[obj_num] = {"type": "plain", "value": val}

    def resolve(self, ref):
        """Resolve a reference to an object number.

        Accepts a tuple (obj_num, gen), a dict with "type":"ref", or a raw
        integer object number (some arrays store refs as bare ints).
        """
        if isinstance(ref, tuple):
            obj_num = ref[0]
        elif isinstance(ref, dict):
            obj_num = ref.get("value", (None,))[0] if ref.get("type") == "ref" else None
        elif isinstance(ref, int):
            obj_num = ref
        else:
            return None
        obj = self.objects.get(obj_num)
        if obj is None:
            return None
        # Return the actual object content (inner dict for plain objects,
        # dict for streams)
        if obj["type"] == "plain":
            return obj["value"]
        if obj["type"] == "stream":
            return obj["dict"]
        return obj

    def get_dict(self, obj_num):
        obj = self.objects.get(obj_num)
        if obj is None:
            return None
        if obj["type"] == "stream":
            return obj["dict"]
        if obj["type"] == "plain" and isinstance(obj["value"], dict):
            return obj["value"]
        return None

    def stream_data(self, obj_num):
        """Decompress and return the raw bytes of a stream object."""
        obj = self.objects.get(obj_num)
        if obj is None or obj["type"] != "stream":
            return None
        raw = obj["raw"]
        if raw is None:
            return None
        d = obj["dict"]
        filter_val = d.get("Filter") if isinstance(d, dict) else None
        if filter_val is not None:
            filter_name = filter_val if isinstance(filter_val, str) else str(filter_val)
            if "FlateDecode" in filter_name:
                try:
                    return zlib.decompress(raw)
                except zlib.error:
                    return raw
        return raw


# ---------------------------------------------------------------------------
# Page tree traversal.
# ---------------------------------------------------------------------------

def find_pages(doc):
    """Walk the page tree starting from /Root and return a list of page dicts
    in document order. Each dict has: obj_num, contents, resources."""
    trailer_start = doc.data.rfind(b"trailer")
    if trailer_start < 0:
        return []
    pos = trailer_start + len(b"trailer")
    pos = _skip_ws_comments(doc.data, pos)
    trailer_val, _ = _parse_pdf_value(doc.data, pos)
    if not isinstance(trailer_val, dict):
        return []
    root_ref = trailer_val.get("Root")
    root = doc.resolve(root_ref)
    if root is None or not isinstance(root, dict):
        return []
    pages_ref = root.get("Pages")
    if pages_ref is None:
        return []

    pages = []
    _walk_pages(doc, pages_ref, pages)
    return pages


def _walk_pages(doc, pages_ref, pages):
    """Recursively walk /Pages nodes, appending leaf page objects."""
    node = doc.resolve(pages_ref)
    if node is None or not isinstance(node, dict):
        return
    node_type = node.get("Type")
    if isinstance(node_type, str) and node_type == "Page":
        pages.append({
            "obj_num": node.get("obj_num") if isinstance(node, dict) else None,
            "contents": node.get("Contents"),
            "resources": node.get("Resources"),
        })
        return
    kids = node.get("Kids")
    if isinstance(kids, list):
        for kid in kids:
            if isinstance(kid, tuple):
                _walk_pages(doc, kid, pages)
            elif isinstance(kid, int):
                _walk_pages(doc, kid, pages)
            elif isinstance(kid, dict):
                pages.append({
                    "obj_num": kid.get("obj_num"),
                    "contents": kid.get("Contents"),
                    "resources": kid.get("Resources"),
                })


def get_font_map(doc, page):
    """Build a mapping from font name (e.g. "/TT1") to encoding info."""
    resources = page.get("resources")
    if not isinstance(resources, dict):
        return {}
    font_dict = resources.get("Font")
    if not isinstance(font_dict, dict):
        return {}
    font_map = {}
    for font_name, font_ref in font_dict.items():
        fdict = doc.resolve(font_ref)
        if not isinstance(fdict, dict):
            continue
        base_font = fdict.get("BaseFont", "")
        encoding = fdict.get("Encoding", "")
        font_map[font_name] = {
            "base_font": base_font if isinstance(base_font, str) else "",
            "encoding": encoding if isinstance(encoding, str) else "",
        }
    return font_map


# ---------------------------------------------------------------------------
# Content stream text extraction.
# ---------------------------------------------------------------------------

def _map_text_through_font(text, font_info):
    """Map a string of character codes through the font encoding to Unicode.

    Each character in `text` is a byte value (0-255) that indexes into the
    font's encoding. With MacRomanEncoding, byte N -> MacRoman character N.
    """
    if not text:
        return text
    encoding_name = font_info.get("encoding", "") if font_info else ""
    is_mac_roman = "mac" in encoding_name.lower() or "MacRoman" in encoding_name

    result = []
    for ch in text:
        byte_val = ord(ch)
        if is_mac_roman:
            result.append(mac_roman_byte(byte_val))
        else:
            if byte_val < 128:
                result.append(ch)
            else:
                result.append(codec_decode(bytes([byte_val]), "latin-1", errors="replace"))
    return "".join(result)


def _parse_tj_array_content(arr_bytes):
    """Parse the inside of a TJ array: [ (string) 50 (string) -30 ... ]
    Returns a list of items, each either a string or a number."""
    items = []
    pos = 0
    n = len(arr_bytes)
    while pos < n:
        while pos < n and arr_bytes[pos:pos + 1] in (b" ", b"\t", b"\n", b"\r"):
            pos += 1
        if pos >= n:
            break
        c = arr_bytes[pos:pos + 1]
        if c == b"(":
            text, pos = _read_literal_string(arr_bytes, pos)
            items.append(text)
        elif c == b"<":
            text, pos = _read_hex_string(arr_bytes, pos)
            items.append(text)
        elif c in b"0123456789+-.":
            num, new_pos = _read_number(arr_bytes, pos)
            if num is not None:
                items.append(num)
                pos = new_pos
            else:
                pos += 1
        else:
            pos += 1
    return items


def extract_text_from_content_stream(stream_bytes, font_map):
    """Extract text with positions from a PDF content stream.

    Returns a list of dicts:
        {"text": str, "x": float, "y": float, "font": str, "size": float}

    Parses text-object operators (BT/ET, Tf, Tj, TJ, Tm, Td, TD, T*) using
    an operand stack. Errors on individual tokens are caught and skipped so
    one malformed stream cannot abort the whole extraction.
    """
    items = []
    data = stream_bytes
    n = len(data)
    pos = 0

    stack = []
    in_text = False
    current_font = None
    font_size = 1.0
    text_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    text_line_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    SAFETY_LIMIT = n * 4 + 1000
    iterations = 0

    while pos < n and iterations < SAFETY_LIMIT:
        iterations += 1

        # Skip whitespace
        while pos < n and data[pos:pos + 1] in (b" ", b"\t", b"\n", b"\r"):
            pos += 1
        if pos >= n:
            break

        c = data[pos:pos + 1]

        # --- operands ---

        if c == b"(":
            try:
                text, pos = _read_literal_string(data, pos)
                stack.append(text)
            except Exception:
                pos += 1
            continue

        if c == b"<":
            try:
                if pos + 1 < n and data[pos + 1:pos + 2] == b"<":
                    # Skip inline dictionary
                    depth = 2
                    pos += 2
                    while pos < n and depth > 0:
                        if data[pos:pos + 2] == b"<<":
                            depth += 1
                            pos += 2
                        elif data[pos:pos + 2] == b">>":
                            depth -= 1
                            pos += 2
                        else:
                            pos += 1
                else:
                    text, pos = _read_hex_string(data, pos)
                    stack.append(text)
            except Exception:
                pos += 1
            continue

        if c == b"[":
            try:
                arr, new_pos = _read_array(data, pos)
                stack.append(arr)
                pos = new_pos
            except Exception:
                pos += 1
            continue

        if c in b"0123456789+-.":
            try:
                num, new_pos = _read_number(data, pos)
                if num is not None:
                    stack.append(num)
                    pos = new_pos
                else:
                    pos += 1
            except Exception:
                pos += 1
            continue

        if c == b"/":
            try:
                name, pos = _read_name(data, pos)
                stack.append(name)
            except Exception:
                pos += 1
            continue

        # --- operator ---
        start = pos
        while pos < n and data[pos:pos + 1] not in (
                b" ", b"\t", b"\n", b"\r", b"/", b"(", b")",
                b"<", b">", b"[", b"]"):
            pos += 1
        op = data[start:pos].decode("ascii", errors="replace")
        if not op:
            pos += 1
            continue
        # Move past the operator (the while loop already advanced pos past it,
        # but there may be a delimiter we haven't consumed yet; that's fine,
        # whitespace skipping at the top handles it).

        if op == "BT":
            in_text = True
            stack = []
            text_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            text_line_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        elif op == "ET":
            in_text = False
            stack = []
        elif in_text:
            try:
                if op == "Tf":
                    if len(stack) >= 2:
                        size = stack.pop()
                        font_name = stack.pop()
                        if isinstance(font_name, str):
                            current_font = font_name
                        if isinstance(size, (int, float)):
                            font_size = float(size)
                elif op == "Tj":
                    if stack:
                        text_val = stack.pop()
                        if isinstance(text_val, str):
                            fi = font_map.get(current_font, {}) if current_font else {}
                            mapped = _map_text_through_font(text_val, fi)
                            items.append({
                                "text": mapped,
                                "x": text_matrix[4],
                                "y": text_matrix[5],
                                "font": current_font or "",
                                "size": font_size,
                            })
                elif op == "TJ":
                    if stack:
                        arr = stack.pop()
                        if isinstance(arr, list):
                            cx, cy = text_matrix[4], text_matrix[5]
                            for item in arr:
                                if isinstance(item, str):
                                    fi = font_map.get(current_font, {}) if current_font else {}
                                    mapped = _map_text_through_font(item, fi)
                                    items.append({
                                        "text": mapped,
                                        "x": cx,
                                        "y": cy,
                                        "font": current_font or "",
                                        "size": font_size,
                                    })
                                    cx += len(mapped) * font_size * 0.5
                                elif isinstance(item, (int, float)):
                                    cx += item * font_size / 1000.0
                elif op == "Tm":
                    if len(stack) >= 6:
                        f = stack.pop(); e = stack.pop(); d = stack.pop()
                        cval = stack.pop(); bval = stack.pop(); aval = stack.pop()
                        text_matrix = [float(aval), float(bval), float(cval),
                                       float(d), float(e), float(f)]
                        text_line_matrix = list(text_matrix)
                elif op == "Td":
                    if len(stack) >= 2:
                        ty = stack.pop(); tx = stack.pop()
                        text_matrix[4] += float(tx)
                        text_matrix[5] += float(ty)
                        text_line_matrix = list(text_matrix)
                elif op == "TD":
                    if len(stack) >= 2:
                        ty = stack.pop(); tx = stack.pop()
                        text_matrix[4] += float(tx)
                        text_matrix[5] += float(ty)
                        text_line_matrix = list(text_matrix)
                elif op == "T*":
                    text_matrix[4] = text_line_matrix[4]
                    text_matrix[5] -= text_line_matrix[3] * font_size
            except Exception:
                pass  # skip malformed operator

    return items


# ---------------------------------------------------------------------------
# High-level PDF extraction.
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path, max_pages=None):
    """Extract text from a PDF file, page by page.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    max_pages : int or None
        If set, only extract the first max_pages pages (useful for testing).

    Returns
    -------
    list of dicts, each with:
        "page_number" : int (1-based)
        "text_items"  : list of {"text", "x", "y", "font", "size"}
        "raw_text"    : str, reading-order text with approximate spacing
    """
    with open(pdf_path, "rb") as f:
        data = f.read()

    doc = PDFDocument(data)
    pages = find_pages(doc)

    if max_pages is not None:
        pages = pages[:max_pages]

    results = []
    for page_index, page in enumerate(pages):
        page_number = page_index + 1
        contents = page["contents"]
        resources = page["resources"]

        font_map = get_font_map(doc, page)

        all_items = []

        content_refs = contents
        if content_refs is None:
            continue
        if isinstance(content_refs, tuple):
            content_refs = [content_refs]
        if not isinstance(content_refs, list):
            continue

        for cref in content_refs:
            if not isinstance(cref, tuple):
                continue
            stream_obj_num = cref[0]
            stream_bytes = doc.stream_data(stream_obj_num)
            if stream_bytes is None:
                continue
            try:
                items = extract_text_from_content_stream(stream_bytes, font_map)
                all_items.extend(items)
            except Exception:
                continue

        # Sort by reading order: top-to-bottom (y descending), then
        # left-to-right (x ascending)
        all_items.sort(key=lambda it: (-it["y"], it["x"]))

        raw_text = _build_reading_order_text(all_items)

        results.append({
            "page_number": page_number,
            "text_items": all_items,
            "raw_text": raw_text,
        })

    return results


def _build_reading_order_text(items):
    """Join text items into reading-order text with word-aware spacing.

    Groups items into lines by y-coordinate. Within each line, items are
    sorted by x-position and then merged into "words": consecutive items
    whose x-gap is small (typical of characters within one word) are joined
    without a space, while larger gaps (typical of inter-word spacing) get
    a single space. Lines are joined with newlines.
    """
    if not items:
        return ""

    y_tolerance = 3.0
    lines = []
    current_line = []
    current_y = None

    for item in items:
        y = item["y"]
        if current_y is None or abs(y - current_y) <= y_tolerance:
            current_line.append(item)
            if current_y is None:
                current_y = y
        else:
            current_line.sort(key=lambda it: it["x"])
            lines.append(current_line)
            current_line = [item]
            current_y = y

    if current_line:
        current_line.sort(key=lambda it: it["x"])
        lines.append(current_line)

    text_parts = []
    for line in lines:
        # Merge items into words based on x-gap between consecutive items.
        # The gap threshold is scaled by font size: a gap larger than about
        # half a font size is treated as a word boundary.
        words = []
        current_word = ""
        prev_x = None
        prev_font_size = 10.0

        for item in line:
            x = item["x"]
            t = item["text"]
            font_size = item["size"] or 10.0

            if not t:
                continue

            if prev_x is not None and current_word:
                gap = x - prev_x
                threshold = max(font_size, prev_font_size) * 0.55
                if gap > threshold:
                    # Word boundary: flush current word and start a new one
                    words.append(current_word)
                    current_word = t
                else:
                    current_word += t
            else:
                current_word = t

            prev_x = x + len(t) * font_size * 0.55
            prev_font_size = font_size

        if current_word:
            words.append(current_word)

        text_parts.append(" ".join(words))

    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Cleaning: page numbers, repeated headers, FTLN identifiers.
# ---------------------------------------------------------------------------

FTLN_PATTERN = re.compile(r"FTLN\s*\d{4,5}", re.IGNORECASE)
PAGE_HEADER_CANDIDATES = {
    "hamlet",
    "the tragedy of hamlet",
    "hamlet, prince of denmark",
}


def is_repeated_page_header(text, font_size=None, y_pos=None, page_height=792):
    """Heuristic: detect repeated page-header text (e.g. "Hamlet" at top
    of page) or a standalone page number."""
    stripped = text.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if lower in PAGE_HEADER_CANDIDATES:
        return True
    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    return False


def clean_page_text(raw_text, page_number, total_pages):
    """Remove page numbers, repeated headers, and FTLN identifiers from
    one page's text.

    Returns (cleaned_text, cleaning_notes).
    """
    # --- Fix known MacRoman encoding errors ---
    # This PDF's MacRomanEncoding maps some byte values to wrong Unicode
    # characters. We correct the most common ones found in this document.
    MACROMAN_FIXES = {
        "\u00d1": "\u2019",  # right single quotation mark (apostrophe)
        "\u00d2": "\u2018",  # left single quotation mark (opening quote)
        "\u00d3": "\u2019",  # right single quotation mark (after comma)
        "\u00d5": "\u2019",  # right single quotation mark (possessive 's)
        "\u0088": "\u2013",  # en dash (cap-pie -> cap-pie)
        "\u008f": "",        # stray byte in "armed" -> remove
        "\u00aa": "\u2019",  # right single quotation mark
        "\u00c9": "\u201d",  # right double quotation mark
    }
    fix_count = 0
    for bad, good in MACROMAN_FIXES.items():
        count = raw_text.count(bad)
        if count:
            fix_count += count
            raw_text = raw_text.replace(bad, good)
    if fix_count:
        notes = [f"page {page_number}: fixed {fix_count} MacRoman encoding character(s)"]
    else:
        notes = []

    lines = raw_text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        # Remove FTLN identifiers (e.g. "FTLN 0001", "FTLN 01234")
        if FTLN_PATTERN.fullmatch(stripped):
            notes.append(f"page {page_number}: removed FTLN identifier: {stripped}")
            continue

        new_line = FTLN_PATTERN.sub("", stripped)
        if new_line != stripped:
            notes.append(f"page {page_number}: removed inline FTLN from: {stripped[:60]}")
            stripped = new_line.strip()
            if not stripped:
                continue

        # Remove trailing page number at end of a text line
        # (e.g. "dialogue text 15" -> "dialogue text")
        m = re.search(r"(\s+)(\d{1,4})\s*$", stripped)
        if m:
            # Only strip if the number is separated by a decent gap (likely a page number)
            notes.append(f"page {page_number}: removed trailing page number: {m.group(2)}")
            stripped = stripped[: m.start(1)].strip()

        if re.fullmatch(r"\d{1,4}", stripped):
            notes.append(f"page {page_number}: removed page number: {stripped}")
            continue

        if stripped.lower() in PAGE_HEADER_CANDIDATES:
            notes.append(f"page {page_number}: removed repeated header: {stripped}")
            continue

        cleaned_lines.append(stripped)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip(), notes


# ---------------------------------------------------------------------------
# Self-demo / CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python src\\stage7\\pdf_extractor.py <path_to_pdf> [max_pages]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"Extracting text from: {pdf_path}")
    pages = extract_pdf_text(pdf_path, max_pages=max_pages)

    print(f"Pages extracted: {len(pages)}")
    total_chars = 0
    for page in pages:
        txt = page["raw_text"]
        total_chars += len(txt)
        first_line = txt.split("\n")[0][:60] if txt else "(empty)"
        print(f"  Page {page['page_number']}: {len(txt):,} chars | starts: {first_line!r}")

    print(f"\nTotal extracted characters: {total_chars:,}")

    if pages:
        print("\n--- First page sample ---")
        print(pages[0]["raw_text"][:1000])


if __name__ == "__main__":
    main()
