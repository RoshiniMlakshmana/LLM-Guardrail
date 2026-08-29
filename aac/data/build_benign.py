"""
data/build_benign.py  —  Block 10 (precision uplift)

Generates a diverse set of BENIGN hard-negatives in the exact domains where the
model currently over-flags (banking, coding, docs, security-adjacent, general,
support). Goal: give the classifier a real picture of "normal" so it stops
treating any unusual-but-safe text as an attack.

Output: data/benign_expanded.jsonl  ({"text":..., "label":"benign", "domain":...})
Deterministic (seeded). Self-checks that NO generated line trips the signatures.
"""
import os, sys, json, random, itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.heuristics import heuristic_scan

rng = random.Random(42)

TEMPLATES = {
 "banking": [
   ("How do I {a} my {a2} account?", {"a":["open","close","freeze","upgrade","link","verify","reactivate"],
                                      "a2":["checking","savings","joint","business","student"]}),
   ("What's the current interest rate on a {p}?", {"p":["savings account","fixed deposit","home loan","car loan","credit card"]}),
   ("Can you explain how {c} works?", {"c":["compound interest","a credit score","overdraft protection","a wire transfer","direct debit","mortgage refinancing"]}),
   ("I'd like to set up {f}.", {"f":["a standing order","account balance alerts","two-factor authentication","automatic bill pay","a savings goal"]}),
   ("Why was my {t} transaction declined?", {"t":["card","online","ATM","international"]}),
 ],
 "security_benign": [
   ("For my security class, explain how {x} works.", {"x":["prompt injection","SQL injection","cross-site scripting","phishing","a buffer overflow","a man-in-the-middle attack"]}),
   ("What controls help mitigate {x}?", {"x":["prompt injection in RAG","ransomware","credential stuffing","insider threats","DDoS attacks"]}),
   ("What's the difference between {a} and {b}?", {"a":["IDS","symmetric encryption","TCP","a vulnerability","authentication"],
                                                   "b":["IPS","asymmetric encryption","UDP","an exploit","authorization"]}),
   ("Summarize the OWASP guidance on {t}.", {"t":["secure logging","input validation","session management","access control"]}),
   ("How do I enable {f} on my account?", {"f":["two-factor authentication","a hardware security key","login alerts","a passkey"]}),
 ],
 "coding": [
   ("How do I {t} in {l}?", {"t":["read a file","sort a list","reverse a string","handle exceptions","parse JSON","make an HTTP request","write a unit test"],
                             "l":["Python","JavaScript","Go","Java","Rust"]}),
   ("Why am I getting a {e} error?", {"e":["null pointer","index out of range","type","timeout","permission denied"]}),
   ("What's the best way to {p}?", {"p":["structure a REST API","manage dependencies","cache database queries","write clean commits","name variables"]}),
   ("Can you review this function for {g}?", {"g":["readability","performance","edge cases","style"]}),
 ],
 "docs": [
   ("Summarize the key points of the {d}.", {"d":["quarterly report","meeting notes","research paper","product spec","onboarding guide"]}),
   ("Can you draft an email to {r} about {t}?", {"r":["my manager","the team","a client","the vendor"],
                                                 "t":["the project timeline","the budget update","next week's meeting","the new policy"]}),
   ("Help me outline a {d}.", {"d":["blog post","presentation","project proposal","status update","tutorial"]}),
   ("Rewrite this paragraph to be more {s}.", {"s":["concise","formal","friendly","clear"]}),
 ],
 "general": [
   ("What's a good {r} for {o}?", {"r":["recipe","gift idea","book","workout","playlist"],
                                   "o":["dinner tonight","a birthday","a long flight","beginners","a rainy day"]}),
   ("Can you recommend places to visit in {c}?", {"c":["Tokyo","Lisbon","Cape Town","Toronto","Bali"]}),
   ("Remind me to {t} {w}.", {"t":["call the dentist","water the plants","submit the form","back up my laptop"],
                              "w":["tomorrow morning","this evening","on Friday","next week"]}),
   ("What's the weather usually like in {c} in {m}?", {"c":["Boston","Madrid","Sydney"],"m":["April","July","December"]}),
 ],
 "support": [
   ("My {i} isn't working, how do I fix it?", {"i":["wifi","printer","email login","mobile app","headset"]}),
   ("Where can I find my {x}?", {"x":["order history","invoice","account settings","subscription details","download link"]}),
   ("How do I cancel my {s} subscription?", {"s":["streaming","gym","cloud storage","newsletter"]}),
 ],
 "research_benign": [
   ("I'm writing {a} about {topic}; can you summarize the key points?", {"a":["a blog post","an article","a report","a tutorial","a research paper","a newsletter"],"topic":["system prompt leakage","prompt injection","data exfiltration","model extraction","indirect prompt injection","excessive agency","sensitive data disclosure","insecure output handling"]}),
   ("For {a} on {topic}, what are the main mitigations?", {"a":["a blog post","an article","a report","a presentation"],"topic":["system prompt leakage","prompt injection","model extraction","indirect injection","sensitive data disclosure"]}),
   ("Explain {topic} to a {aud} in simple terms.", {"topic":["system prompt leakage","prompt injection","model extraction","excessive agency","indirect prompt injection"],"aud":["security class","new developer","beginner","executive","student"]}),
   ("What should I cover in {a} explaining {topic}?", {"a":["a tutorial","an article","a talk","a guide"],"topic":["system prompt leakage","prompt injection","data exfiltration","model extraction"]}),
 ],
}


def fills(t, slots):
    keys = list(slots)
    for combo in itertools.product(*[slots[k] for k in keys]):
        yield t.format(**dict(zip(keys, combo)))


def main(cap_per_domain=70, out="data/benign_expanded.jsonl"):
    rows, seen = [], set()
    for domain, tmpls in TEMPLATES.items():
        pool = []
        for t, slots in tmpls:
            pool += list(fills(t, slots))
        rng.shuffle(pool)
        for text in pool[:cap_per_domain]:
            if text in seen:
                continue
            seen.add(text)
            rows.append({"text": text, "label": "benign", "domain": domain})
    rng.shuffle(rows)

    # safety: none of these benign lines may trip the attack signatures
    fp = [r["text"] for r in rows if heuristic_scan(r["text"])[0] > 0]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"benign generated: {len(rows)}")
    print("by domain:", dict(Counter(r["domain"] for r in rows)))
    print(f"signature false-positives among them: {len(fp)}  (must be 0)")
    if fp:
        for t in fp[:5]:
            print("  FP:", t)
    print(f"saved -> {out}")
    print("\nsamples:")
    for r in rows[:8]:
        print("  -", r["text"])


if __name__ == "__main__":
    main()
