"""W194 executable answer; demonstration fixtures are explicitly labeled."""
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Email:
    message_id: str
    subject: str
    body: str
    sender_reputation: float
    authentication_failed: bool

class SpamDetector:
    def __init__(self, model: Callable[[dict], float], threshold=0.99):
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must lie in [0,1]')
        self.model, self.threshold = model, threshold

    def features(self, email):
        return {'text': email.subject + '\n' + email.body,
                'sender_reputation': email.sender_reputation,
                'authentication_failed': email.authentication_failed}

    def classify(self, email):
        p = float(self.model(self.features(email)))
        if not 0 <= p <= 1:
            raise ValueError('model must return a finite probability')
        return {'message_id': email.message_id, 'spam_probability': p,
                'action': 'QUARANTINE' if p >= self.threshold else 'DELIVER'}
