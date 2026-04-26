"""Verify that streaming code never imports from labeling or uses joint-angle data.
Joint angles are training-time supervision only; the deployed model must work
from biosignals alone.
"""
import ast
from pathlib import Path
import pytest


FORBIDDEN_NAMES = {
    'joint_angle', 'joint_angles', 'load_joint',
    'knee_angle', 'elbow_angle', 'hip_angle', 'shoulder_angle', 'ankle_angle',
    'phase_label', 'rep_index', 'rep_count_in_set',
    'rpe_for_this_set', 'load_participants',
}

FORBIDDEN_MODULES = {
    'src.labeling', 'labeling',
    'src.data.participants', 'src.data.joint',
}


def _python_files_under(p: Path):
    if not p.exists():
        return []
    return list(p.rglob('*.py'))


def _check_file(path: Path):
    """Return list of violations as (line, kind, name)."""
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_MODULES or any(
                    alias.name.startswith(m + '.') for m in FORBIDDEN_MODULES
                ):
                    violations.append((node.lineno, 'import', alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if node.module in FORBIDDEN_MODULES or any(
                    node.module.startswith(m + '.') for m in FORBIDDEN_MODULES
                ):
                    violations.append((node.lineno, 'from-import', node.module))
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                violations.append((node.lineno, 'name', node.id))
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_NAMES:
                violations.append((node.lineno, 'attr', node.attr))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if 'participants.xlsx' in node.value or 'joint_angles.csv' in node.value:
                violations.append((node.lineno, 'string', node.value))
    return violations


def test_no_joint_or_labeling_imports_in_streaming():
    streaming_dir = Path('src/streaming')
    if not streaming_dir.exists():
        pytest.skip('src/streaming does not exist yet')

    all_violations = {}
    for f in _python_files_under(streaming_dir):
        v = _check_file(f)
        if v:
            all_violations[str(f)] = v

    if all_violations:
        msg = ['Streaming code references offline-only labeling artifacts:']
        for f, vs in all_violations.items():
            msg.append(f'  {f}:')
            for ln, kind, name in vs:
                msg.append(f'    line {ln}: {kind} {name!r}')
        pytest.fail('\n'.join(msg))
