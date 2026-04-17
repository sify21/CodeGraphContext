from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class BackendCapabilities:
    """Backend execution capabilities used for strategy decisions."""

    backend: str
    supports_transactions: bool
    supports_concurrent_writes: bool
    supports_session_scoped_reads: bool


@runtime_checkable
class ExecutionContext(Protocol):
    """Minimal context contract shared by sessions and transactions."""

    def run(self, query: str, **parameters: Any) -> Any:
        ...


@runtime_checkable
class TransactionContext(ExecutionContext, Protocol):
    """Transaction contract used by orchestrators and write services."""

    def commit(self) -> None:
        ...

    def rollback(self) -> None:
        ...

    def close(self) -> None:
        ...


class NoopTransactionContext:
    """
    Fallback transaction facade for backends without explicit tx support.

    It preserves call compatibility for upper layers but does not provide
    rollback semantics.
    """

    def __init__(self, execution_context: ExecutionContext):
        self._execution_context = execution_context

    def run(self, query: str, **parameters: Any) -> Any:
        return self._execution_context.run(query, **parameters)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None
