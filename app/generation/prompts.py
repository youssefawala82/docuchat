"""
Prompt template for the QA chain. This is the single most important piece
for controlling hallucination: it explicitly instructs the model to refuse
when the answer isn't in the provided context.
"""

from langchain_core.prompts import PromptTemplate

QA_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions
using ONLY the context provided below, which comes from the user's own
documents. Do not use any outside knowledge.

If the answer is not contained in the context, say clearly that you don't
know based on the provided documents. Do not make anything up.

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)