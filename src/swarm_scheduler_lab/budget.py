"""A shared, thread-safe ledger for generation and candidate evaluation budgets.

Reservations happen before work starts. A request whose provider does not return
complete usage retains its conservative token reservation, including on failure.
There are no retries here: a retry must reserve another call like any other call.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any


class BudgetExceeded(RuntimeError):
    """The requested work would exceed a configured hard resource limit."""


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class CallReservation:
    """Opaque reservation returned by :meth:`BudgetLedger.reserve_call`."""

    number: int
    input_token_estimate: int
    max_output_tokens: int

    @property
    def total(self) -> int:
        return self.input_token_estimate + self.max_output_tokens


class BudgetLedger:
    """Atomically account for all requests sharing one experimental method.

    ``input_token_estimate`` must be a conservative upper bound supplied by the
    backend (rather than a typical characters-per-token approximation). Once
    complete provider usage arrives the reservation is replaced by actual usage.
    A provider reporting usage above the reserved bounds is recorded explicitly;
    such usage cannot be prevented retroactively, and no further request can
    overrun the remaining local allowance.
    """

    def __init__(self, max_calls: int, max_tokens: int, max_evaluations: int):
        self.max_calls = _nonnegative_integer(max_calls, "max_calls")
        self.max_tokens = _nonnegative_integer(max_tokens, "max_tokens")
        self.max_evaluations = _nonnegative_integer(
            max_evaluations, "max_evaluations"
        )
        self._lock = Lock()
        self._calls = 0
        self._failures = 0
        self._evaluations = 0
        self._instance_schedules = 0
        self._charged_tokens = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._known_input_calls = 0
        self._known_output_calls = 0
        self._complete_usage_calls = 0
        self._unknown_usage_calls = 0
        self._provider_token_overrun_calls = 0
        self._pending: dict[int, CallReservation] = {}

    def reserve_call(
        self, input_token_estimate: int, max_output_tokens: int
    ) -> CallReservation:
        """Reserve one call and its maximum token allowance before sending it."""
        input_token_estimate = _nonnegative_integer(
            input_token_estimate, "input_token_estimate"
        )
        max_output_tokens = _nonnegative_integer(
            max_output_tokens, "max_output_tokens"
        )
        requested = input_token_estimate + max_output_tokens
        with self._lock:
            if self._calls >= self.max_calls:
                raise BudgetExceeded("generation call budget exhausted")
            if self._charged_tokens + requested > self.max_tokens:
                raise BudgetExceeded("total token budget cannot cover this request")
            self._calls += 1
            reservation = CallReservation(
                self._calls, input_token_estimate, max_output_tokens
            )
            self._pending[reservation.number] = reservation
            self._charged_tokens += requested
            return reservation

    def finish_call(
        self,
        reservation: CallReservation,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        failed: bool = False,
    ) -> None:
        """Settle a request once, retaining reservations for unmetered usage.

        Partially reported usage is exposed as known partial totals. Its charged
        allowance is at least the original reservation, never a fabricated actual
        usage. Unknown failed calls therefore cannot release budget for retries.
        """
        if input_tokens is not None:
            input_tokens = _nonnegative_integer(input_tokens, "input_tokens")
        if output_tokens is not None:
            output_tokens = _nonnegative_integer(output_tokens, "output_tokens")
        if not isinstance(failed, bool):
            raise ValueError("failed must be a boolean")
        with self._lock:
            if (
                not isinstance(reservation, CallReservation)
                or self._pending.get(reservation.number) is not reservation
            ):
                raise ValueError("unknown, foreign, or already settled reservation")
            del self._pending[reservation.number]
            if failed:
                self._failures += 1
            if input_tokens is not None:
                self._input_tokens += input_tokens
                self._known_input_calls += 1
            if output_tokens is not None:
                self._output_tokens += output_tokens
                self._known_output_calls += 1
            if input_tokens is not None and output_tokens is not None:
                charged = input_tokens + output_tokens
                self._complete_usage_calls += 1
            else:
                charged = max(
                    reservation.total,
                    (input_tokens or 0) + (output_tokens or 0),
                )
                self._unknown_usage_calls += 1
            self._charged_tokens += charged - reservation.total
            if (
                input_tokens is not None
                and input_tokens > reservation.input_token_estimate
            ) or (
                output_tokens is not None
                and output_tokens > reservation.max_output_tokens
            ):
                self._provider_token_overrun_calls += 1

    def reserve_evaluation(self) -> int:
        """Reserve one candidate across the complete training set, atomically."""
        with self._lock:
            if self._evaluations >= self.max_evaluations:
                raise BudgetExceeded("candidate evaluation budget exhausted")
            self._evaluations += 1
            return self._evaluations

    def record_instance_schedules(self, count: int) -> None:
        """Record CPU scheduling attempts, distinct from policy evaluations."""
        count = _nonnegative_integer(count, "count")
        if count == 0:
            raise ValueError("count must be positive")
        with self._lock:
            self._instance_schedules += count

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-serializable counts without exposing backend credentials."""
        with self._lock:
            if not self._calls:
                accounting = "not_used"
            elif self._complete_usage_calls == self._calls:
                accounting = "provider_reported"
            elif self._complete_usage_calls:
                accounting = "mixed_actual_and_conservative_reservation"
            else:
                accounting = "conservative_reservation"
            return {
                "calls": self._calls,
                "failures": self._failures,
                "evaluations": self._evaluations,
                "instance_schedules": self._instance_schedules,
                "input_tokens": (
                    self._input_tokens if self._known_input_calls or not self._calls else None
                ),
                "output_tokens": (
                    self._output_tokens if self._known_output_calls or not self._calls else None
                ),
                "known_input_calls": self._known_input_calls,
                "known_output_calls": self._known_output_calls,
                "unknown_usage_calls": self._unknown_usage_calls,
                "pending_calls": len(self._pending),
                "reserved_or_charged_tokens": self._charged_tokens,
                "token_accounting": accounting,
                "provider_token_overrun_calls": self._provider_token_overrun_calls,
                "token_limit_exceeded": self._charged_tokens > self.max_tokens,
                "limits": {
                    "calls": self.max_calls,
                    "tokens": self.max_tokens,
                    "evaluations": self.max_evaluations,
                },
            }
