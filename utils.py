"""Pure helpers shared by the app — kept free of streamlit/llama-index
imports so they can be unit-tested without heavy dependencies."""


def format_sources(source_nodes):
    """Format source nodes into readable citations."""
    if not source_nodes:
        return ""
    lines = ["\n\n---\n**📚 Sources:**"]
    for i, node in enumerate(source_nodes, 1):
        meta = node.node.metadata or {}
        file_name = meta.get("file_name", "Unknown")
        page = meta.get("page_label", "?")
        score = node.score if node.score is not None else 0.0
        lines.append(
            f"{i}. **{file_name}** — Page {page} (relevance: {score:.2f})"
        )
    return "\n".join(lines)
