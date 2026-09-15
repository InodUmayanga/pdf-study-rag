"""Prompt text for grounded answers.

Kept as plain strings (no llama-index import) so the templates can be
unit-tested and inspected without loading the heavy runtime dependencies.
``rag.py`` wraps them in ``PromptTemplate`` objects.
"""

# The exact sentence the model must return when the retrieved passages do
# not contain the answer. Fixed text so it can be tested and evaluated.
NOT_COVERED = "The notes don't cover this."

_QA_PROMPT = """\
You are a study assistant. Answer the question using ONLY the context \
passages below, which were retrieved from the user's own notes.

Rules:
1. Use only the context passages. Do not use outside knowledge and do not \
guess.
2. After each claim, cite the passage it came from as [<file_name> p.<page>] \
using the file_name and page_label shown with that passage.
3. If the passages do not contain the answer, reply with exactly this \
sentence and nothing else: {not_covered}
4. Be concise. No preamble, no restating the question.

Context passages:
---------------------
{context_str}
---------------------
Question: {query_str}
Answer:"""

_REFINE_PROMPT = """\
You are refining an existing answer to a question using additional context \
passages from the user's own notes.

Rules:
1. Use only the existing answer and the new context passages. Do not use \
outside knowledge.
2. Keep citations in the form [<file_name> p.<page>] and add a citation for \
any new claim.
3. If the new context adds nothing useful, return the existing answer \
unchanged.
4. If neither the existing answer nor the new context answers the question, \
reply with exactly this sentence and nothing else: {not_covered}

Question: {query_str}
Existing answer: {existing_answer}
New context passages:
---------------------
{context_msg}
---------------------
Refined answer:"""

# Bake the refusal sentence in; the remaining ``{...}`` fields are the
# variables llama-index fills at query time.
GROUNDED_QA_PROMPT = _QA_PROMPT.replace("{not_covered}", NOT_COVERED)
GROUNDED_REFINE_PROMPT = _REFINE_PROMPT.replace("{not_covered}", NOT_COVERED)


def is_not_covered(answer):
    """True if ``answer`` is the fixed refusal sentence."""
    return (answer or "").strip().rstrip(".").lower() == (
        NOT_COVERED.rstrip(".").lower()
    )
