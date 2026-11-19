"""Targeted reading: send the parts of a big source file that matter, not all of it.

A whole-file Read result is split into units (imports/constants, classes,
functions, methods). Units are scored against what the conversation is about
(your latest prompt plus the assistant's recent notes and tool calls). The result
keeps, verbatim and with their original line numbers:

  * the header (imports, constants) when it is small
  * every unit that matches the task
  * class headers that hold kept methods

Every other unit is replaced by one marker naming what it holds, so the model
still sees the file's whole outline and can Read any range it needs. Nothing is
ever rewritten: a kept line is byte-for-byte the line the tool returned, which
is what Edit needs.

Pure functions, no I/O. Returns None whenever the file should go through whole.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from shrink import count_tokens

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".kts",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".lua", ".sh",
}

NOTE = ("[Condensed by proxy: this {total}-line file is shown only where it matches the task. Elided ranges "
        "are marked with their line numbers and contents. Read with offset/limit to see any of them, and never "
        "rewrite the whole file from this view.]")
MISC_KEEP_LINES = 60        # imports / constants kept whole up to this many lines
CLASS_HEAD_KEEP_LINES = 40  # class signature + docstring kept whole up to this many lines
SPLIT_MIN_LINES = 50        # a generic block this long is split into its members
MAX_NAMES_IN_MARKER = 14
MIN_ELIDE_TOKENS = 90       # smaller gaps are kept: the marker would cost about as much

_LINE = re.compile(r"^(\s*\d+)([→\t])(.*)$", re.S)
_TRAILER = re.compile(r"\n*<system-reminder>.*</system-reminder>\s*$", re.S)
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_STOP = set("""
a an and are as at be but by can could do does for from has have how i if in into is it its me my no not of on
or our please so that the their then there these this to up us use was we what when where which who will with
would you your def class return self cls this import from pass none true false null var let const new void int
str string bool function fn func pub public private static async await try except catch raise throw else elif
file code line lines change changes make fix add update look need want get set
everything anything something thing things stuff bug bugs issue issues problem problems error errors work works
working also just like check help implement remove some any all more most other should now still
function functions method methods
""".split())
_COMMENT = ("#", "//", "/*", "*", "@", "///", "--")
_START_WORD = re.compile(
    r"^(?:(?:export|default|public|private|protected|static|async|abstract|final|pub|unsafe|extern|inline)\s+)*"
    r"(?:def|class|function|fn|func|impl|struct|enum|interface|trait|type|module|namespace|object|const|let|var|"
    r"fun|sub|proc|package)\b"
)


@dataclass
class Unit:
    start: int      # index into the file's lines
    end: int        # inclusive
    sig: str        # first line, for the outline
    name: str
    kind: str       # "misc" | "def" | "head"
    text: str = ""

    @property
    def size(self):
        return self.end - self.start + 1


# --------------------------------------------------------------- parsing


def parse_read_result(content: str):
    """Split a Read result into (rows, trailer); rows = [(prefix, sep, text)]. None if unrecognised."""
    trailer = ""
    match = _TRAILER.search(content)
    if match:
        trailer, content = match.group(0), content[:match.start()]
    rows, expect = [], None
    for raw in content.split("\n"):
        m = _LINE.match(raw)
        if not m:
            return None
        number = int(m.group(1))
        if expect is not None and number != expect:
            return None
        expect = number + 1
        rows.append((m.group(1), m.group(2), m.group(3)))
    return (rows, trailer) if rows else None


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _attach_comments(lines, start, floor):
    """Pull comment/decorator lines sitting directly above a definition into it."""
    while start - 1 >= floor and lines[start - 1].strip().startswith(_COMMENT):
        start -= 1
    return start


def _name_of(sig: str) -> str:
    s = sig.strip()
    m = re.search(r"\b(?:def|class|function|fn|func|struct|enum|interface|trait|type|impl|fun|object)\s+([A-Za-z_]\w*)", s)
    if m:
        return m.group(1)
    m = re.match(r"(?:export\s+|pub\s+|public\s+|private\s+|static\s+|async\s+)*(?:const|let|var)\s+([A-Za-z_]\w*)", s)
    if m:
        return m.group(1)
    m = re.search(r"([A-Za-z_]\w*)\s*\(", s)
    return m.group(1) if m else s[:40]


def _python_units(lines):
    try:
        tree = ast.parse("\n".join(lines))
    except (SyntaxError, ValueError):
        return None
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]

    def span(node, floor):
        first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        return _attach_comments(lines, first, floor), node.end_lineno - 1

    units, cursor = [], 0
    for node in nodes:
        start, end = span(node, cursor)
        if start > cursor:
            units.append(Unit(cursor, start - 1, lines[cursor].strip(), "", "misc"))
        if isinstance(node, ast.ClassDef):
            members = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            inner, pos = [], start
            for member in members:
                m_start, m_end = span(member, pos + 1)
                inner.append((m_start, m_end, member))
                pos = m_end
            if inner and (end - start + 1) >= SPLIT_MIN_LINES:
                units.append(Unit(start, inner[0][0] - 1, lines[start].strip(), node.name, "head"))
                last = inner[0][0]
                for m_start, m_end, member in inner:
                    if m_start > last:  # attributes between methods stay with the class
                        units.append(Unit(last, m_start - 1, lines[last].strip(), node.name, "head"))
                    units.append(Unit(m_start, m_end, lines[m_start].strip(), f"{node.name}.{member.name}", "def"))
                    last = m_end + 1
                if last <= end:
                    units.append(Unit(last, end, lines[last].strip(), node.name, "head"))
            else:
                units.append(Unit(start, end, lines[start].strip(), node.name, "def"))
        else:
            units.append(Unit(start, end, lines[start].strip(), node.name, "def"))
        cursor = end + 1
    if cursor < len(lines):
        units.append(Unit(cursor, len(lines) - 1, lines[cursor].strip(), "", "misc"))
    return units


def _is_start(line: str) -> bool:
    if not line or line[0] in " \t}" or line.lstrip().startswith(_COMMENT[:3]):
        return False
    stripped = line.rstrip()
    return bool(_START_WORD.match(stripped)) or stripped.endswith(("{", ":", "("))


def _generic_units(lines):
    starts = [i for i, line in enumerate(lines) if _is_start(line)]
    if not starts:
        return None
    bounds = [_attach_comments(lines, s, (starts[k - 1] + 1) if k else 0) for k, s in enumerate(starts)]
    units = []
    if bounds[0] > 0:
        units.append(Unit(0, bounds[0] - 1, lines[0].strip(), "", "misc"))
    for k, start in enumerate(bounds):
        end = (bounds[k + 1] - 1) if k + 1 < len(bounds) else len(lines) - 1
        units += _split_members(lines, start, end, lines[starts[k]])
    return units


def _split_members(lines, start, end, sig):
    """One top-level block -> a single unit, or head + member units when it is big."""
    outer = _name_of(sig)
    whole = [Unit(start, end, sig.strip(), outer, "def")]
    if end - start + 1 < SPLIT_MIN_LINES:
        return whole
    heads = [i for i in range(start + 1, end + 1)
             if lines[i].strip() and _indent(lines[i]) > 0 and lines[i].rstrip().endswith(("{", ":", "("))
             and not lines[i].strip().startswith(_COMMENT)]
    if not heads:
        return whole
    level = min(_indent(lines[i]) for i in heads)
    sigs = [i for i in heads if _indent(lines[i]) == level]
    if len(sigs) < 2:
        return whole
    firsts = [_attach_comments(lines, i, start + 1) for i in sigs]
    units = [Unit(start, firsts[0] - 1, sig.strip(), outer, "head")]
    for k, i in enumerate(sigs):
        stop = (firsts[k + 1] - 1) if k + 1 < len(sigs) else end
        units.append(Unit(firsts[k], stop, lines[i].strip(), f"{outer}.{_name_of(lines[i])}", "def"))
    return units


def split_units(lines, ext):
    units = _python_units(lines) if ext == ".py" else None
    units = units or _generic_units(lines)
    if not units:
        return None
    for u in units:
        u.text = "\n".join(lines[u.start:u.end + 1])
        u.sig = next((lines[i].strip() for i in range(u.start, u.end + 1)
                      if lines[i].strip() and not lines[i].strip().startswith(_COMMENT)), u.sig)
    return units


# --------------------------------------------------------------- scoring


def terms(text: str):
    """Lowercased identifier words: parse_config / parseConfig -> parse, config (+ the whole identifier)."""
    out = []
    for ident in _WORD.findall(text):
        parts = [p.lower() for p in _CAMEL.findall(ident.replace("_", " "))]
        kept = [p for p in parts if len(p) > 2 and p not in _STOP]
        out += kept
        if len(ident) > 2 and ident.lower() not in _STOP and kept != [ident.lower()]:
            out.append(ident.lower())  # the whole identifier, e.g. handler3 or parse_config
    return out


def score_units(units, query):
    """query = [(text, weight)]. Returns one score per unit (0 = no evidence)."""
    weights = Counter()
    for text, weight in query:
        for t in terms(text):
            weights[t] += weight
    if not weights:
        return [0.0] * len(units)
    bags = [Counter(terms(u.text)) for u in units]
    whole = [u.name.lower().split(".")[-1] for u in units]
    parts = [set(terms(u.name.replace(".", " "))) for u in units]
    df = Counter(t for bag in bags for t in bag)
    n = len(units)
    scores = []
    for bag, own, pieces in zip(bags, whole, parts):
        s = 0.0
        for t, w in weights.items():
            idf = math.log(1 + n / (1 + df[t])) if df[t] else 0.0
            if t in bag:
                s += w * idf * (1 + math.log(bag[t]))
            if t == own:
                s += w * 6.0              # the exact name of this unit is mentioned
            elif t in pieces:
                s += w * idf * 1.5        # a word of its name is, weighted by how rare the word is
        scores.append(s)
    return scores


# ------------------------------------------------------------- rendering


def _marker(units, lines_total):
    first, last = units[0].start + 1, units[-1].end + 1
    names = []
    for u in units:
        if u.kind == "misc":
            label = "top-level statements"
        else:
            label = u.sig[:70] + ("…" if len(u.sig) > 70 else "")
        if label:
            names.append(label)
    shown = "; ".join(dict.fromkeys(names[:MAX_NAMES_IN_MARKER]))
    more = f"; +{len(names) - MAX_NAMES_IN_MARKER} more" if len(names) > MAX_NAMES_IN_MARKER else ""
    tokens = sum(count_tokens(u.text) for u in units)
    return f"[… lines {first}-{last} elided by proxy (~{tokens} tokens): {shown}{more} — Read offset={first} limit={last - first + 1}]"


def targeted(content: str, path: str, query, min_tokens=1500, max_ratio=0.7) -> Optional[str]:
    """Return the condensed Read result, or None to send the file whole."""
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
    if ext not in CODE_EXTENSIONS:
        return None
    original = count_tokens(content)
    if original < min_tokens:
        return None
    parsed = parse_read_result(content)
    if not parsed:
        return None
    rows, trailer = parsed
    lines = [text for _, _, text in rows]
    units = split_units(lines, ext)
    if not units or len(units) < 3:
        return None

    scores = score_units(units, query)
    best = max(scores)
    if best <= 0:
        return None  # nothing ties this file to the task: send it whole
    top = scores.index(best)
    wanted = {t for text, _ in query for t in terms(text)}
    if len(wanted & set(terms(units[top].text + " " + units[top].name))) < 2 \
            and units[top].name.lower().split(".")[-1] not in wanted:
        return None  # one stray shared word is not enough evidence to hide the rest of the file

    keep = [False] * len(units)
    for i, u in enumerate(units):
        if u.kind == "misc" and u.size <= MISC_KEEP_LINES:
            keep[i] = True
    budget = max(400, int(original * 0.45))
    spent = sum(count_tokens(u.text) for u, k in zip(units, keep) if k)
    for i in sorted(range(len(units)), key=lambda i: -scores[i]):
        if scores[i] < 0.3 * best or scores[i] <= 0 or keep[i]:
            continue
        cost = count_tokens(units[i].text)
        if spent + cost > budget and any(keep[j] and units[j].kind == "def" for j in range(len(units))):
            continue
        keep[i], spent = True, spent + cost
    for i, u in enumerate(units):  # a kept member needs its class header around it
        if u.kind == "def" and keep[i]:
            for j in range(i - 1, -1, -1):
                if units[j].kind == "head" and units[j].size <= CLASS_HEAD_KEEP_LINES:
                    keep[j] = True
                if units[j].kind != "head":
                    break
            for j in range(i + 1, len(units)):
                if units[j].kind != "head":
                    break
                if units[j].size <= CLASS_HEAD_KEEP_LINES:
                    keep[j] = True

    i = 0
    while i < len(units):  # a marker costs ~40 tokens: keep tiny gaps instead of eliding them
        if keep[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(units) and not keep[j + 1]:
            j += 1
        if sum(count_tokens(u.text) for u in units[i:j + 1]) < MIN_ELIDE_TOKENS:
            keep[i:j + 1] = [True] * (j - i + 1)
        i = j + 1

    out, i = [], 0
    while i < len(units):
        if keep[i]:
            u = units[i]
            out += [f"{rows[k][0]}{rows[k][1]}{rows[k][2]}" for k in range(u.start, u.end + 1)]
            i += 1
        else:
            j = i
            while j + 1 < len(units) and not keep[j + 1]:
                j += 1
            out.append(_marker(units[i:j + 1], len(lines)))
            i = j + 1
    if not any(not k for k in keep):
        return None
    result = NOTE.format(total=len(lines)) + "\n" + "\n".join(out) + trailer
    if count_tokens(result) > original * max_ratio:
        return None
    return result
