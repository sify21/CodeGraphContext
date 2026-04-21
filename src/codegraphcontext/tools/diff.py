from __future__ import annotations

import asyncio
import re
import subprocess
import pprint
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, override

from ..core.jobs import JobManager, JobStatus
from ..utils.debug_log import debug_log

from .graph_builder import GraphBuilder
from .code_finder import CodeFinder

ChangeType = Literal["added", "modified", "deleted", "renamed"]


@dataclass
class HunkRange:
    old_start: int
    old_count: int
    new_start: int
    new_count: int


@dataclass
class FileChange:
    change_type: ChangeType
    old_path: str|None
    new_path: str|None
    hunks: list[HunkRange] = field(default_factory=list)


@dataclass
class DiffResult:
    old_root: str
    new_root: str
    files: list[FileChange]


@dataclass
class SymbolRef:
    kind: Literal["Function", "Class", "File", "Module"]
    name: str|None
    path: str
    line_number: int|None


@dataclass
class CallEdgeRef:
    line_number: int|None
    full_call_name: str|None
    is_rpc: bool|None


@dataclass
class CallChain:
    chain_id: str
    nodes: list[SymbolRef]
    edges: list[CallEdgeRef]
    length: int


@dataclass
class ImpactReport:
    old_path: str
    new_path: str
    changed_files: list[FileChange]
    added_chains: list[CallChain]
    removed_chains: list[CallChain]
    affected_existing_chains: list[CallChain]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiffProvider:
    def collect_diff(self, old_root: str, new_root: str) -> DiffResult:  # pyright: ignore[reportUnusedParameter]
        raise NotImplementedError


class UnifiedDiffParser:
    _HUNK_RE: re.Pattern[str] = re.compile(r"^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@")

    def parse(self, patch_text: str, old_root: str, new_root: str) -> DiffResult:
        files: list[FileChange] = []
        current: FileChange|None = None

        for line in patch_text.splitlines():
            if line.startswith("--- "):
                old_path = line[4:].split("\t", 1)[0].strip()
                if old_path == "/dev/null" or "\t1970-01-01" in line:
                    old_path = None
                current = FileChange(change_type="modified", old_path=old_path, new_path=None, hunks=[])
                continue

            if line.startswith("+++ ") and current is not None:
                new_path = line[4:].split("\t", 1)[0].strip()
                if new_path == "/dev/null" or "\t1970-01-01" in line:
                    new_path = None
                current.new_path = new_path

                if current.old_path is None and current.new_path is not None:
                    current.change_type = "added"
                elif current.old_path is not None and current.new_path is None:
                    current.change_type = "deleted"
                elif current.old_path is not None and current.new_path is not None and current.old_path[len(old_root):] == current.new_path[len(new_root):]:
                    current.change_type = "modified"
                else:
                    current.change_type = "renamed"

                files.append(current)
                continue

            if line.startswith("@@ ") and current is not None:
                match = self._HUNK_RE.match(line)
                if not match:
                    continue
                old_start = int(match.group(1))
                old_count = int(match.group(2) or 1)
                new_start = int(match.group(3))
                new_count = int(match.group(4) or 1)
                current.hunks.append(HunkRange(old_start, old_count, new_start, new_count))

        return DiffResult(old_root=old_root, new_root=new_root, files=files)


class GnuDiffProvider(DiffProvider):
    def __init__(self, parser: UnifiedDiffParser|None = None):
        self.parser: UnifiedDiffParser = parser or UnifiedDiffParser()

    @override
    def collect_diff(self, old_root: str, new_root: str) -> DiffResult:
        cmd = [
            "diff",
            "-ruN",
            "--strip-trailing-cr",
            "--exclude=.git",
            "--exclude=.svn",
            old_root,
            new_root,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"gnu diff failed: {proc.stderr.strip()}")
        return self.parser.parse(proc.stdout, old_root=old_root, new_root=new_root)


class ChangeMapper:
    def __init__(self, graph_builder: GraphBuilder):
        self.graph_builder: GraphBuilder = graph_builder

    @staticmethod
    def _resolve_file_path(repo_root: str, raw_path: str|None) -> str|None:
        if not raw_path:
            return None
        text = raw_path.strip()
        if not text:
            return None
        path_obj = Path(text)
        if path_obj.is_absolute():
            return str(path_obj.resolve())
        return str((Path(repo_root).resolve() / path_obj).resolve())

    @staticmethod
    def _hunk_ranges(fc: FileChange, side: Literal["old", "new"]) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for h in fc.hunks:
            start = h.old_start if side == "old" else h.new_start
            count = h.old_count if side == "old" else h.new_count
            if count <= 0:
                continue
            ranges.append((start, start + count - 1))
        return ranges

    @staticmethod
    def _to_int(value: Any) -> int|None:
        return value if isinstance(value, int) else None

    @classmethod
    def _line_overlaps_ranges(
        cls,
        start_line: Any,
        end_line: Any,
        ranges: list[tuple[int, int]],
    ) -> bool:
        if not ranges:
            return True
        start = cls._to_int(start_line)
        if start is None or start <= 0:
            return False
        end = cls._to_int(end_line)
        if end is None or end < start:
            end = start
        return any(start <= r_end and r_start <= end for r_start, r_end in ranges)

    @staticmethod
    def _symbol_key(symbol: SymbolRef) -> tuple[str, str, str, int|None]:
        return (symbol.kind, symbol.path, symbol.name or "", symbol.line_number)

    def _collect_symbols_from_file(
        self,
        file_path: str,
        file_data: dict[str, Any],
        hunk_ranges: list[tuple[int, int]],
    ) -> list[SymbolRef]:
        symbols: list[SymbolRef] = [SymbolRef(kind="File", name=Path(file_path).name, path=file_path, line_number=None)]
        seen: set[tuple[str, str, str, int|None]] = set()

        def add_symbol(kind: Literal["Function", "Class", "File", "Module"], name: str|None, line: Any, end_line: Any) -> None:
            if not self._line_overlaps_ranges(line, end_line, hunk_ranges):
                return
            line_number = self._to_int(line)
            symbol = SymbolRef(kind=kind, name=name, path=file_path, line_number=line_number)
            key = self._symbol_key(symbol)
            if key in seen:
                return
            seen.add(key)
            symbols.append(symbol)

        for fn in file_data.get("functions", []):
            if isinstance(fn, dict):
                name = fn.get("name")
                if isinstance(name, str) and name.strip():
                    add_symbol("Function", name, fn.get("line_number"), fn.get("end_line"))

        for cls in file_data.get("classes", []):
            if isinstance(cls, dict):
                name = cls.get("name")
                if isinstance(name, str) and name.strip():
                    add_symbol("Class", name, cls.get("line_number"), cls.get("end_line"))

        for mod in file_data.get("modules", []):
            if isinstance(mod, dict):
                name = mod.get("name")
                if isinstance(name, str) and name.strip():
                    add_symbol("Module", name, mod.get("line_number"), mod.get("end_line"))

        return symbols

    def _map_changes_to_symbols(
        self,
        diff: DiffResult,
        repo_root: str,
        side: Literal["old", "new"],
    ) -> list[SymbolRef]:
        symbols: list[SymbolRef] = []
        seen: set[tuple[str, str, str, int|None]] = set()
        repo_path = Path(repo_root).resolve()

        for fc in diff.files:
            raw_path = fc.old_path if side == "old" else fc.new_path
            file_path = self._resolve_file_path(repo_root, raw_path)
            if not file_path:
                continue

            hunk_ranges = self._hunk_ranges(fc, side)
            parsed = self.graph_builder.parse_file(repo_path=repo_path, path=Path(file_path))
            file_symbols: list[SymbolRef]
            if parsed.get("error"):
                file_symbols = [SymbolRef(kind="File", name=Path(file_path).name, path=file_path, line_number=None)]
            else:
                file_symbols = self._collect_symbols_from_file(
                    file_path=file_path,
                    file_data=parsed,
                    hunk_ranges=hunk_ranges,
                )

            for symbol in file_symbols:
                key = self._symbol_key(symbol)
                if key in seen:
                    continue
                seen.add(key)
                symbols.append(symbol)

        return symbols

    def map_changes_to_symbols_before(self, diff: DiffResult, repo_path_old: str) -> list[SymbolRef]:
        return self._map_changes_to_symbols(diff=diff, repo_root=repo_path_old, side="old")

    def map_changes_to_symbols_after(self, diff: DiffResult, repo_path_new: str) -> list[SymbolRef]:
        return self._map_changes_to_symbols(diff=diff, repo_root=repo_path_new, side="new")


class ChainSnapshotService:
    def __init__(self, code_finder: CodeFinder, job_manager: JobManager):
        self.code_finder: CodeFinder = code_finder
        self.job_manager: JobManager = job_manager

    @staticmethod
    def _symbol_lookup_props(symbol: SymbolRef) -> dict[str, Any]:
        if symbol.kind == "File":
            return {"path": symbol.path}
        if symbol.kind == "Module":
            if symbol.name:
                return {"name": symbol.name}
            return {}
        props: dict[str, Any] = {"path": symbol.path}
        if symbol.name:
            props["name"] = symbol.name
        if symbol.line_number is not None:
            props["line_number"] = symbol.line_number
        return props

    @staticmethod
    def _format_symbol(symbol: SymbolRef) -> str:
        line_number = symbol.line_number if symbol.line_number is not None else "-"
        return f"{symbol.path}:{line_number}:{symbol.kind}:{symbol.name}"

    @staticmethod
    def _chain_signature(nodes: list[SymbolRef], edges: list[CallEdgeRef]) -> str:
        node_sig = "->".join(
            f"{n.kind}:{n.name or ''}:{n.path}:{n.line_number or 0}"
            for n in nodes
        )
        edge_sig = "->".join(
            f"{e.line_number or 0}:{e.full_call_name or ''}:{bool(e.is_rpc)}"
            for e in edges
        )
        return f"{node_sig}|{edge_sig}"

    @staticmethod
    def _to_symbol_ref(label: str, props: dict[str, Any]) -> SymbolRef:
        name = props.get("name")
        path = props.get("path") or ""
        line_number = props.get("line_number")
        if not isinstance(line_number, int):
            line_number = None
        kind = label if label in ("Function", "Class", "File", "Module") else "File"
        return SymbolRef(kind=kind, name=name, path=path, line_number=line_number)

    @staticmethod
    def _to_call_edge_ref(detail: dict[str, Any]) -> CallEdgeRef:
        line_number = detail.get("line_number")
        if not isinstance(line_number, int):
            line_number = None
        return CallEdgeRef(
            line_number=line_number,
            full_call_name=detail.get("full_call_name"),
            is_rpc=detail.get("is_rpc"),
        )

    def snapshot_around_symbols(
        self,
        symbols: list[SymbolRef],
        max_depth: int = 6,
        chain_limit: int = 200,
        execution_context=None,
        job_id: str = "",
    ) -> list[CallChain]:
        if not symbols:
            return []

        chains: list[CallChain] = []
        used_ids: set[str] = set()
        chain_sigs: set[str] = set()
        emitted = 0
        total_symbols = len(symbols)
        self.job_manager.update_job(job_id, status=JobStatus.RUNNING, processed_files=0, total_files=total_symbols)
        for idx, symbol in enumerate(symbols):
            self.job_manager.update_job(job_id, processed_files=idx+1, current_file=self._format_symbol(symbol))
            if emitted >= chain_limit:
                break
            label = symbol.kind
            props = self._symbol_lookup_props(symbol)
            debug_log(f"snapshot around symbol : {label} {props}")
            if not props:
                debug_log(f"{symbol} no _symbol_lookup_props")
                continue

            try:
                result = self.code_finder.find_call_paths_around_node(
                    label=label,
                    props=props,
                    max_depth=max_depth,
                    limit=max(1, chain_limit - emitted),
                    execution_context=execution_context,
                )
            except ValueError as e:
                debug_log(f"{symbol} ERROR: {e}")
                continue
            for chain_idx, chain in enumerate(result.get("chains", [])):
                node_chain = chain.get("node_chain") or []
                call_details = chain.get("call_details") or []
                nodes = [
                    self._to_symbol_ref(n.get("label", "File"), n.get("props") or {})
                    for n in node_chain
                    if isinstance(n, dict)
                ]
                if not nodes:
                    continue
                edges = [
                    self._to_call_edge_ref(detail)
                    for detail in call_details
                    if isinstance(detail, dict)
                ]
                sig = self._chain_signature(nodes, edges)
                if sig in chain_sigs:
                    continue
                chain_sigs.add(sig)
                chain_id_base = (
                    f"{label}:{props.get('name', '')}:{props.get('path', '')}:"
                    f"{props.get('line_number', 0)}:{idx}:{chain_idx}"
                )
                chain_id = chain_id_base
                suffix = 1
                while chain_id in used_ids:
                    suffix += 1
                    chain_id = f"{chain_id_base}:{suffix}"
                used_ids.add(chain_id)
                raw_len = chain.get("chain_length", len(edges))
                chain_len = raw_len if isinstance(raw_len, int) else len(edges)
                chains.append(
                    CallChain(
                        chain_id=chain_id,
                        nodes=nodes,
                        edges=edges,
                        length=chain_len,
                    )
                )
                emitted += 1
                if emitted >= chain_limit:
                    break

        self.job_manager.update_job(job_id, status=JobStatus.COMPLETED)
        return chains


class ReindexService:
    def __init__(self, graph_builder: GraphBuilder, code_finder: CodeFinder):
        self.graph_builder = graph_builder
        self.code_finder = code_finder

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        msg = str(error).lower()
        retry_keywords = ("deadlock", "lock", "timeout", "temporarily unavailable")
        return any(k in msg for k in retry_keywords)

    async def reindex_repo(
        self,
        old_root: str,
        new_root: str,
        execution_context=None,
        max_retries: int = 2,
    ) -> None:
        attempt = 0
        while True:
            try:
                _ = self.graph_builder.delete_repository_from_graph(
                    old_root,
                    execution_context=execution_context,
                )
                await self.graph_builder.build_graph_from_path_async(
                    Path(new_root).resolve(),
                    is_dependency=False,
                    execution_context=execution_context,
                )
                return
            except Exception as e:
                attempt += 1
                if attempt > max_retries or not self._is_retryable_error(e):
                    raise
                await asyncio.sleep(0.2 * (2 ** (attempt - 1)))


class ImpactAnalyzer:
    @staticmethod
    def _normalize_path(path: str, repo_root: str) -> str:
        if not path:
            return ""

        path_obj = Path(path).resolve()
        try:
            return path_obj.relative_to(Path(repo_root).resolve()).as_posix()
        except ValueError:
            pass
        return path_obj.as_posix()

    @classmethod
    def _chain_key(cls, chain: CallChain, repo_root: str) -> str:
        node_key = "->".join(
            f"{n.kind}:{n.name or ''}:{cls._normalize_path(n.path, repo_root)}:{n.line_number or 0}"
            for n in chain.nodes
        )
        edge_key = "->".join(
            f"{e.line_number or 0}:{e.full_call_name or ''}:{bool(e.is_rpc)}" for e in chain.edges
        )
        return f"{node_key}|{edge_key}"

    def diff_chains(
        self,
        before_chains: list[CallChain],
        after_chains: list[CallChain],
        changed_symbols_after: list[SymbolRef],
        old_root: str,
        new_root: str,
        changed_files: list[FileChange],
    ) -> ImpactReport:
        before_map = {self._chain_key(c, old_root): c for c in before_chains}
        after_map = {self._chain_key(c, new_root): c for c in after_chains}

        before_keys = set(before_map.keys())
        after_keys = set(after_map.keys())

        added = [after_map[k] for k in sorted(after_keys - before_keys)]
        removed = [before_map[k] for k in sorted(before_keys - after_keys)]
        common = [after_map[k] for k in sorted(before_keys & after_keys)]

        stats = {
            "changed_files": len(changed_files),
            "changed_symbols_after": len(changed_symbols_after),
            "added_chains": len(added),
            "removed_chains": len(removed),
            "affected_existing_chains": len(common),
        }
        return ImpactReport(
            old_path=old_root,
            new_path=new_root,
            changed_files=changed_files,
            added_chains=added,
            removed_chains=removed,
            affected_existing_chains=common,
            stats=stats,
        )


class DiffOrchestrator:
    def __init__(
        self,
        diff_provider: DiffProvider,
        change_mapper: ChangeMapper,
        chain_snapshot_service: ChainSnapshotService,
        reindex_service: ReindexService,
        impact_analyzer: ImpactAnalyzer,
    ):
        self.diff_provider: DiffProvider = diff_provider
        self.change_mapper: ChangeMapper = change_mapper
        self.chain_snapshot_service: ChainSnapshotService = chain_snapshot_service
        self.reindex_service: ReindexService = reindex_service
        self.impact_analyzer: ImpactAnalyzer = impact_analyzer

    async def run(
        self,
        old_path: str,
        new_path: str,
        max_depth: int = 6,
        chain_limit: int = 200,
        commit: bool = False,
        old_job_id: str = "",
        new_job_id: str = "",
    ) -> ImpactReport:
        old_root = str(Path(old_path).resolve())
        new_root = str(Path(new_path).resolve())
        diff_result = self.diff_provider.collect_diff(old_root=old_root, new_root=new_root)
        debug_log(f"diff_result:\n "+pprint.pformat(diff_result))

        changed_symbols_before = self.change_mapper.map_changes_to_symbols_before(diff_result, old_root)
        debug_log(f"changed_symbols_before:\n"+pprint.pformat(changed_symbols_before))
        before_chains = self.chain_snapshot_service.snapshot_around_symbols(
            changed_symbols_before,
            max_depth=max_depth,
            chain_limit=chain_limit,
            job_id=old_job_id,
        )

        tx_context = None
        db_manager = getattr(self.reindex_service.graph_builder, "db_manager")
        capabilities = getattr(db_manager, "get_capabilities")()
        supports_tx = bool(capabilities and capabilities.supports_transactions)

        if not commit: 
            if supports_tx and hasattr(db_manager, "begin_transaction"):
                tx_context = db_manager.begin_transaction()
            else:
                raise RuntimeError("tx not supported")

        try:
            await self.reindex_service.reindex_repo(
                old_root,
                new_root,
                execution_context=tx_context,
            )
            changed_symbols_after = self.change_mapper.map_changes_to_symbols_after(diff_result, new_root)
            debug_log(f"changed_symbols_after:\n"+pprint.pformat(changed_symbols_after))
            after_chains = self.chain_snapshot_service.snapshot_around_symbols(
                changed_symbols_after,
                max_depth=max_depth,
                chain_limit=chain_limit,
                execution_context=tx_context,
                job_id=new_job_id,
            )
        finally:
            if tx_context is not None:
                tx_context.rollback()
                tx_context.close()

        return self.impact_analyzer.diff_chains(
            before_chains=before_chains,
            after_chains=after_chains,
            changed_symbols_after=changed_symbols_after,
            old_root=old_root,
            new_root=new_root,
            changed_files=diff_result.files,
        )
