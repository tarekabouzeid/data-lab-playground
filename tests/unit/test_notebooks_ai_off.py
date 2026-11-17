"""With no LLM configured (llm.provider=none) every notebook still runs to the end: an AI cell either guards itself
with one of lab_utils' skip helpers or the notebook stops on purpose (require_llm), never fails on a connection."""
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parents[2] / "jupyter" / "notebooks"
AI = re.compile(r"langchain_chat|langchain_embeddings|openai_client|embeddings_client|\b_chat\(|\bembed\(|lu\.ask_raw_sql|ask_raw_sql\(|semantic_agent|ops_agent")
GUARD = re.compile(r"skip_unless_llm|skip_unless_embeddings|llm_available|embeddings_available|require_llm|if store is None|llm_ready")


def cells():
    for f in sorted(NB.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in f.parts:
            continue
        for i, c in enumerate(json.loads(f.read_text())["cells"]):
            if c["cell_type"] == "code":
                yield f.relative_to(NB), i, "".join(c["source"])


def test_every_ai_cell_is_guarded_for_ai_off():
    bad = [f"{f} cell {i}" for f, i, src in cells() if AI.search(src) and not GUARD.search(src)]
    assert not bad, "guard these cells with lab_utils.skip_unless_llm(...) (or require_llm):\n" + "\n".join(bad)


def test_no_notebook_talks_to_ollama_directly():
    bad = [f"{f} cell {i}" for f, i, src in cells() if re.search(r"\bollama\b|OLLAMA_URL|ChatOllama|OllamaEmbeddings", src, re.I)]
    assert not bad, "use lab_utils (OpenAI-compatible API) instead of Ollama's own client:\n" + "\n".join(bad)
