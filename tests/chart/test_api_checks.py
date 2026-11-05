"""global.apiChecks: required-API checks are on by default and can be turned off for offline linting."""
import pytest

from conftest import RenderError, render

NEEDS_API = [
    {"gateway": {"enabled": True, "gatewayClassName": "x"}},
    {"ollama": {"enabled": True, "gpu": {"enabled": True, "dra": {"enabled": True}}}},
    {"semantic": {"enabled": True}},
]


@pytest.mark.parametrize("values", NEEDS_API)
def test_missing_api_fails_by_default(values):
    with pytest.raises(RenderError):
        render(values)


@pytest.mark.parametrize("values", NEEDS_API)
def test_api_checks_off_renders(values):
    render({**values, "global": {"apiChecks": False}})
