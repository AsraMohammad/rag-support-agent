"""
Streamlit web UI for the RAG documentation assistant.
Run with: streamlit run app.py
"""
import streamlit as st
from rag import answer_question

st.set_page_config(page_title="Documentation RAG Assistant", layout="wide")

st.title("Documentation RAG Assistant")
st.caption("Ask questions about Anthropic's API. Answers grounded in source docs with hallucination verification.")

# Sidebar with info
with st.sidebar:
    st.header("About this system")
    st.markdown("""
    **Architecture:**
    1. Question is embedded with `all-MiniLM-L6-v2`
    2. Top 3 most similar chunks retrieved from ChromaDB
    3. Sonnet 4.5 generates an answer using only those chunks
    4. Haiku 4.5 verifies every claim is grounded in the sources
    5. If verification fails, response is suppressed
    
    **Stack:** Python · Anthropic API · ChromaDB · sentence-transformers · LangChain · Streamlit
    
    [GitHub repo](https://github.com/AsraMohammad/rag-support-agent)
    """)
    
    st.divider()
    st.subheader("Try these")
    st.markdown("""
    - How is API usage billed?
    - What models does Claude offer?
    - What is Constitutional AI?
    - What's the population of Tokyo? *(should refuse)*
    """)

# Main interface
question = st.text_input("Ask a question:", placeholder="How does API pricing work?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving and answering..."):
        result = answer_question(question)
    
    # Verdict badge
    if result["verification"]["verdict"] == "PASS":
        st.success(f"✓ Verified — {result['verification'].get('summary', '')}")
    else:
        st.error(f"✗ Verification failed — {result['verification'].get('summary', '')}")
    
    # Answer
    st.subheader("Answer")
    st.write(result["answer"])
    
    # Sources
    with st.expander("Sources used"):
        for i, c in enumerate(result["sources"], 1):
            st.markdown(f"**{i}. `{c['source']}`** — similarity {c['similarity']:.3f}")
            st.text(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
    
    # Tokens
    with st.expander("Token usage"):
        v_t = result["verification"].get("tokens", {})
        st.markdown(f"""
        - **Answerer:** {result['tokens']['input']} input + {result['tokens']['output']} output
        - **Verifier:** {v_t.get('input', 0)} input + {v_t.get('output', 0)} output
        """)