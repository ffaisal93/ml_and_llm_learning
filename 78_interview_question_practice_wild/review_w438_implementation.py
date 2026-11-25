"""Educational OOD model, integer cents; simplified daily interest and statement policy."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from threading import RLock
import uuid


@dataclass
class Card:
    id: str
    account_id: str
    active: bool = True


@dataclass
class Account:
    id: str
    credit_limit: int
    settled: int = 0
    pending: int = 0


@dataclass
class Transaction:
    id: str
    account_id: str
    card_id: str
    amount: int
    state: str
    posted_on: date | None = None
    kind: str = "PURCHASE"


@dataclass(frozen=True)
class Statement:
    id: str
    account_id: str
    start: date
    end: date
    transaction_ids: tuple
    net_activity: int
    closing_balance: int


class CreditCardService:
    def __init__(self):
        self.accounts = {}
        self.cards = {}
        self.transactions = {}
        self.statements = {}
        self.keys = {}
        self.lock = RLock()

    def _once(self, key, signature, action):
        if key in self.keys:
            old, result = self.keys[key]
            if old != signature:
                raise ValueError("idempotency key reused with different parameters")
            return result
        result = action()
        self.keys[key] = (signature, result)
        return result

    @staticmethod
    def _amount(value):
        if type(value) is not int or value <= 0:
            raise ValueError("positive integer cents required")

    def issueCard(self, account):
        with self.lock:
            if account.credit_limit < 0:
                raise ValueError("invalid credit limit")
            if account.id in self.accounts and self.accounts[account.id] is not account:
                raise ValueError("conflicting account")
            self.accounts[account.id] = account
            card = Card(uuid.uuid4().hex, account.id)
            self.cards[card.id] = card
            return card

    def authorize(self, card, amount, key):
        self._amount(amount)
        with self.lock:

            def apply():
                if self.cards.get(card.id) is not card or not card.active:
                    raise ValueError("inactive or unknown card")
                account = self.accounts[card.account_id]
                if account.settled + account.pending + amount > account.credit_limit:
                    raise ValueError("credit limit exceeded")
                tx = Transaction(
                    uuid.uuid4().hex, account.id, card.id, amount, "PENDING"
                )
                self.transactions[tx.id] = tx
                account.pending += amount
                return tx

            return self._once(("authorize", key), (card.id, amount), apply)

    def _posting_allowed(self, account_id, day):
        if any(
            s.account_id == account_id and day < s.end for s in self.statements.values()
        ):
            raise ValueError("posting date belongs to a closed statement period")

    def settle(self, transaction, today):
        with self.lock:
            tx = self.transactions[transaction.id]
            if tx.state == "SETTLED":
                return tx
            if tx.state != "PENDING":
                raise ValueError("not pending")
            self._posting_allowed(tx.account_id, today)
            account = self.accounts[tx.account_id]
            account.pending -= tx.amount
            account.settled += tx.amount
            tx.state = "SETTLED"
            tx.posted_on = today
            return tx

    def void(self, transaction):
        with self.lock:
            tx = self.transactions[transaction.id]
            if tx.state == "VOID":
                return
            if tx.state != "PENDING":
                raise ValueError("only pending authorization may be voided")
            self.accounts[tx.account_id].pending -= tx.amount
            tx.state = "VOID"

    def repay(self, card, amount, key, today):
        self._amount(amount)
        with self.lock:

            def apply():
                if self.cards.get(card.id) is not card:
                    raise ValueError("unknown card")
                self._posting_allowed(card.account_id, today)
                account = self.accounts[card.account_id]
                if amount > account.settled:
                    raise ValueError("overpayment not supported in this demo")
                account.settled -= amount
                tx = Transaction(
                    uuid.uuid4().hex,
                    account.id,
                    card.id,
                    -amount,
                    "SETTLED",
                    today,
                    "REPAYMENT",
                )
                self.transactions[tx.id] = tx
                return tx

            return self._once(
                ("repay", key), (card.id, amount, today.isoformat()), apply
            )

    def accrue_interest(self, account_id, day, daily_rate_ppm):
        # Assumption: all unpaid settled principal immediately accrues, no grace; half-up cents.
        if type(daily_rate_ppm) is not int or daily_rate_ppm < 0:
            raise ValueError("invalid rate")
        with self.lock:

            def apply():
                self._posting_allowed(account_id, day)
                account = self.accounts[account_id]
                interest = (
                    max(0, account.settled) * daily_rate_ppm + 500000
                ) // 1000000
                account.settled += interest
                tx = Transaction(
                    uuid.uuid4().hex,
                    account_id,
                    "",
                    interest,
                    "SETTLED",
                    day,
                    "INTEREST",
                )
                self.transactions[tx.id] = tx
                return tx

            return self._once(("interest", account_id, day), (daily_rate_ppm,), apply)

    def generateStatement(self, card, period):
        start, end = period
        if not start < end:
            raise ValueError("period must be nonempty [start,end)")
        with self.lock:
            if self.cards.get(card.id) is not card:
                raise ValueError("unknown card")
            key = (card.account_id, start, end)
            if key in self.statements:
                return self.statements[key]
            txs = [
                t
                for t in self.transactions.values()
                if t.account_id == card.account_id
                and t.state == "SETTLED"
                and start <= t.posted_on < end
            ]
            closing = sum(
                t.amount
                for t in self.transactions.values()
                if t.account_id == card.account_id
                and t.state == "SETTLED"
                and t.posted_on < end
            )
            result = Statement(
                uuid.uuid4().hex,
                card.account_id,
                start,
                end,
                tuple(t.id for t in txs),
                sum(t.amount for t in txs),
                closing,
            )
            self.statements[key] = result
            return result
