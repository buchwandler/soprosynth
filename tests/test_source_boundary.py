import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_product_source_has_no_upstream_sopro_import_or_version_boundary():
    assert not (ROOT / "soprosynth" / "_native.py").exists()

    for path in (ROOT / "soprosynth").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name != "sopro" and not alias.name.startswith("sopro.")
                    for alias in node.names
                ), path
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "sopro" and not (node.module or "").startswith("sopro."), path
            elif isinstance(node, ast.Call):
                function = node.func
                name = (
                    function.id
                    if isinstance(function, ast.Name)
                    else getattr(function, "attr", None)
                )
                if name in {"import_module", "version"} and node.args:
                    argument = node.args[0]
                    assert not (isinstance(argument, ast.Constant) and argument.value == "sopro"), (
                        path
                    )
