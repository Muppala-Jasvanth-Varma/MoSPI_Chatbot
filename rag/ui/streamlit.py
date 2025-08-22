import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="📊 MoSPI RAG", layout="wide")

st.title("📊 MoSPI RAG Assistant")
st.markdown("Ask questions on MoSPI reports or search retrieved chunks.")

if "history" not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["💬 Ask (Gemini)", "🔎 Search (Chunks)"])




with tab1:
    query = st.text_input("Enter your question:", key="ask_input")

    if st.button("Ask", key="ask_button"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Contacting backend..."):
                try:
                    res = requests.get(f"{API_URL}/ask", params={"query": query})
                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer", "")
                        sources = data.get("sources", [])

                        st.session_state.history.append({
                            "query": query,
                            "answer": answer,
                            "sources": sources
                        })

                        st.subheader("Answer")
                        st.markdown(answer)

                        st.subheader("Sources")
                        for src in sources:
                            doc_id = src.get("doc_id")
                            chunk_id = src.get("chunk_id")
                            dist = src.get("dist")
                            st.markdown(
                                f"- 📄 Doc {doc_id} | Chunk {chunk_id} | (dist={dist:.3f})"
                            )
                    else:
                        st.error(f"Backend error: {res.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    if st.session_state.history:
        st.subheader("Conversation History")
        for h in st.session_state.history[::-1]:  # latest first
            st.markdown(f"**Q:** {h['query']}")
            st.markdown(f"**A:** {h['answer']}")
            st.markdown("---")




with tab2:
    search_query = st.text_input("Enter a search query:", key="search_input")

    if st.button("Search", key="search_button"):
        if not search_query.strip():
            st.warning("Please enter a search term.")
        else:
            with st.spinner("Searching FAISS index..."):
                try:
                    res = requests.get(f"{API_URL}/search", params={"query": search_query})
                    if res.status_code == 200:
                        data = res.json()
                        results = data.get("results", [])

                        if not results:
                            st.info("No chunks found.")
                        else:
                            for r in results:
                                st.markdown(f"### 📄 {r['title'] or 'Untitled'}")
                                if r["url"]:
                                    st.markdown(f"[Open Document]({r['url']})")
                                st.markdown(f"**Chunk ID:** {r['chunk_id']} | **Distance:** {r['distance']:.3f}")
                                st.markdown(f"**Snippet:** {r['text']}")
                                st.markdown("---")
                    else:
                        st.error(f"Backend error: {res.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")