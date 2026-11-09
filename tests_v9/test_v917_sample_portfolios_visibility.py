"""Regression coverage for the V9.17 Portfolio Builder sample chooser.

Uses real function bodies from app.py with a tiny Streamlit fake. This test can
run without market-data access or the Streamlit dependency installed.
"""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

APP = Path(__file__).resolve().parents[1] / "portfolio_risk_ui" / "app.py"


def _load_function(name, namespace):
    parsed = ast.parse(APP.read_text(encoding="utf-8"))
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP), "exec"), namespace)
    return namespace[name]


class _FakeStreamlit:
    def __init__(self, selected=None):
        ticker = "AAPL" if selected else ""
        self.session_state = SimpleNamespace(
            base="USD", entry_mode="values", ticker_version=0,
            last_search_query=ticker, selected_ticker=selected, holdings=[],
        )
        self.query = ticker
        self.rerun_count = 0
        self.load_button = False

    def segmented_control(self, *_args, **_kwargs):
        return "Actual holdings"

    def text_input(self, label, **_kwargs):
        return self.query if label == "Search ticker or company name" else ""

    def expander(self, *_args, **_kwargs):
        return nullcontext()

    def columns(self, sizes, **_kwargs):
        return [nullcontext() for _ in sizes]

    def button(self, label, **_kwargs):
        return self.load_button and label.startswith("Load this example")

    def selectbox(self, _label, options, **kwargs):
        return options[kwargs.get("index", 0)]

    def number_input(self, *_args, **kwargs):
        return kwargs["value"]

    def rerun(self):
        self.rerun_count += 1

    def __getattr__(self, name):
        if name in ("caption", "divider", "subheader", "success", "write", "error"):
            return lambda *_args, **_kwargs: None
        raise AttributeError(name)


def test_samples_are_rendered_for_empty_builder_without_ticker():
    st = _FakeStreamlit()
    calls = []
    page_builder = _load_function("page_builder", {
        "st": st, "_header": lambda *_a: None,
        "_render_portfolio_list": lambda: calls.append("portfolio"),
        "_render_sample_portfolios": lambda: calls.append("samples"),
    })
    page_builder()
    assert calls == ["portfolio", "samples"]


def test_samples_are_rendered_when_ticker_is_selected():
    st = _FakeStreamlit(selected={"symbol": "AAPL", "name": "Apple"})
    calls = []
    page_builder = _load_function("page_builder", {
        "st": st, "_header": lambda *_a: None,
        "SUPPORTED_CURRENCIES": ("USD", "CAD"), "MAX_HOLDINGS": 40,
        "_render_portfolio_list": lambda: calls.append("portfolio"),
        "_render_sample_portfolios": lambda: calls.append("samples"),
    })
    page_builder()
    assert calls == ["portfolio", "samples"]


def test_chooser_can_load_an_example_into_session():
    st = _FakeStreamlit()
    st.load_button = True
    demo = SimpleNamespace(
        key="starter", label="Starter: AAPL, MSFT, SPY, GLD",
        purpose="Basic portfolio", inspect="Risk and return",
        positions=[("AAPL", "Apple", 25.0, "USD", "USD")], note=None,
    )
    calls = []
    render = _load_function("_render_sample_portfolios", {
        "st": st, "EXAMPLES": [demo], "get_example": lambda key: demo,
        "load_example": lambda state, key: calls.append((state, key)),
    })
    render()
    assert calls == [(st.session_state, "starter")]
    assert st.rerun_count == 1
