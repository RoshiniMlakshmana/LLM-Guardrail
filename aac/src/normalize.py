"""
src/normalize.py  —  shared input normalisation (defense vs evasion, Block 7/8)
Decode base64 blobs, de-leet, de-space (re-segment with an attack wordlist) so that
obfuscated attacks resurface for the signature layer. Used by red_team.py and the
real-time engine. High-precision: aimed at recovering ATTACK vocabulary.
"""
import re, base64

_SEG_WORDS = sorted({
    "ignore","disregard","forget","previous","prior","earlier","above","everything","all",
    "instruction","instructions","rule","rules","task","tasks","prompt","prompts","system",
    "reveal","show","print","repeat","output","display","reset","override","safety","policy",
    "developer","hidden","initial","original","configuration","admin","password","credentials",
    "anything","now","dan","unrestricted","uncensored","unfiltered","jailbreak","act","you",
    "are","verbatim","instead","and","the","your","with","restrictions","guidelines","before",
    "do","no","an","as","ai",
}, key=len, reverse=True)


def _segment(compact):
    words, i = [], 0
    while i < len(compact):
        for w in _SEG_WORDS:
            if compact.startswith(w, i):
                words.append(w); i += len(w); break
        else:
            i += 1
    return " ".join(words)


def normalise(s: str) -> str:
    def _b64(m):
        try:
            dec = base64.b64decode(m.group(0)).decode("utf-8", "ignore")
            return dec if (len(dec) > 4 and dec.isprintable()) else m.group(0)
        except Exception:
            return m.group(0)
    out = re.sub(r"[A-Za-z0-9+/]{16,}={0,2}", _b64, s)
    out = re.sub(r"(?:\b\w\s){2,}\w\b", lambda m: m.group(0).replace(" ", ""), out)
    out = out.translate(str.maketrans("13405", "ieaos"))
    toks = s.split()
    if len(toks) >= 6 and sum(len(t) == 1 for t in toks) / len(toks) > 0.5:
        out += " " + _segment(re.sub(r"\s+", "", s.translate(str.maketrans("13405", "ieaos"))).lower())
    return out
