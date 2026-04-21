from codegraphcontext.tools.diff import (
    CallChain,
    CallEdgeRef,
    FileChange,
    ImpactAnalyzer,
    SymbolRef,
)


def _make_chain(repo_root: str) -> CallChain:
    return CallChain(
        chain_id=f"chain:{repo_root}",
        nodes=[
            SymbolRef(
                kind="Function",
                name="entrypoint",
                path=f"{repo_root}/pkg/service.py",
                line_number=10,
            ),
            SymbolRef(
                kind="Function",
                name="helper",
                path=f"{repo_root}/pkg/helpers.py",
                line_number=42,
            ),
        ],
        edges=[
            CallEdgeRef(
                line_number=11,
                full_call_name="pkg.helpers.helper",
                is_rpc=False,
            )
        ],
        length=1,
    )


def test_diff_chains_matches_same_relative_chain_across_repo_roots():
    analyzer = ImpactAnalyzer()
    old_root = "/tmp/old-project"
    new_root = "/var/tmp/new-project"
    before_chain = _make_chain(old_root)
    after_chain = _make_chain(new_root)

    report = analyzer.diff_chains(
        before_chains=[before_chain],
        after_chains=[after_chain],
        changed_symbols_after=[],
        old_path=old_root,
        new_path=new_root,
        changed_files=[
            FileChange(
                change_type="modified",
                old_path=f"{old_root}/pkg/service.py",
                new_path=f"{new_root}/pkg/service.py",
            )
        ],
    )

    assert report.added_chains == []
    assert report.removed_chains == []
    assert report.affected_existing_chains == [after_chain]
