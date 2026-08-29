"""
src/multiturn.py  —  Block 3 (working scaffold)
Accumulates an attack score across conversation turns so a slow-burn jailbreak
that looks clean turn-by-turn still trips the gate. Uses an exponential
moving-average plus a boost when consecutive turns are suspicious.
"""
class ConversationState:
    def __init__(self, decay=0.6, escalation=0.15):
        self.decay = decay            # weight on history vs. current turn
        self.escalation = escalation  # bonus for back-to-back suspicious turns
        self.cum = 0.0
        self.prev_suspicious = False
        self.turns = []

    def update(self, turn_attack_score: float, label: str = None) -> float:
        suspicious = turn_attack_score >= 0.5
        self.cum = self.decay * self.cum + (1 - self.decay) * turn_attack_score
        if suspicious and self.prev_suspicious:
            self.cum = min(1.0, self.cum + self.escalation)
        self.prev_suspicious = suspicious
        self.turns.append({"turn": len(self.turns) + 1,
                           "turn_score": round(turn_attack_score, 4),
                           "cumulative": round(self.cum, 4), "label": label})
        return self.cum
