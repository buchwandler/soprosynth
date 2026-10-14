import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_product_source_has_no_non_onnxvoice_inference_engine_imports():
    forbidden = {"torch", "sopro", "onnxruntime"}
    assert not (ROOT / "soprosynth" / "_native.py").exists()

    for path in (ROOT / "soprosynth").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".", 1)[0] not in forbidden for alias in node.names), (
                    path
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module.split(".", 1)[0] not in forbidden, path
            elif isinstance(node, ast.Call):
                function = node.func
                name = (
                    function.id
                    if isinstance(function, ast.Name)
                    else getattr(function, "attr", None)
                )
                if name in {"import_module", "version"} and node.args:
                    argument = node.args[0]
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        assert argument.value.split(".", 1)[0] not in forbidden, path
