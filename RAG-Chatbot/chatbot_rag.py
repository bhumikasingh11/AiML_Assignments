# # DocQuery – AI-powered document search and analysis
# # =============================================================
# # Converted from RAG Chatbot demo to a professional document
# # intelligence platform.
# #
# # What changed vs the previous version:
# #   - Rebranding: "RAG Chatbot" → "DocQuery"
# #   - Sidebar: professional stack table, PDF upload, doc status card
# #   - PDF upload pipeline (reuses exact ingestion.py settings)
# #   - Relevance guard: no-LLM response when retriever finds nothing
# #   - Expander renamed to "Retrieved Evidence", display improved
# #   - Empty state when no document has been uploaded this session
# #   - CSS refreshed to a clean, SaaS-grade dark palette
# #
# # What is UNCHANGED (RAG pipeline):
# #   - Embedding model (all-MiniLM-L6-v2)
# #   - Pinecone configuration
# #   - Retrieval logic (similarity_score_threshold, k=3)
# #   - Chunking strategy (chunk_size=800, chunk_overlap=400)
# #   - LLM invocation (TinyLlama via HuggingFacePipeline)
# #   - Prompt template
# #   - Session state keys: messages, citations, llm
# # =============================================================

# import streamlit as st
# import os
# import tempfile
# from dotenv import load_dotenv
# from transformers import pipeline

# # Pinecone
# from pinecone import Pinecone, ServerlessSpec

# # LangChain core (unchanged)
# from langchain_pinecone import PineconeVectorStore
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFacePipeline
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# # PDF ingestion — reuses same loader + splitter as ingestion.py
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# load_dotenv()

# # ── Page config ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="DocQuery",
#     page_icon="📄",
#     layout="centered",
#     initial_sidebar_state="expanded",
# )

# # ── CSS ───────────────────────────────────────────────────────────────────────
# # Professional SaaS-grade dark theme. Targets Streamlit data-testid attributes
# # so no external framework is needed.
# st.markdown("""
# <style>
# /* ── Google Fonts ── */
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

# html, body, [class*="css"], [class*="st-"] {
#     font-family: 'Inter', sans-serif !important;
# }

# /* ── Background ── */
# [data-testid="stAppViewContainer"] {
#     background: #080810;
#     min-height: 100vh;
# }

# [data-testid="stHeader"] {
#     background: transparent !important;
#     box-shadow: none !important;
# }

# .block-container {
#     padding-top: 2rem !important;
#     padding-bottom: 5.5rem !important;
#     max-width: 780px !important;
#     margin: 0 auto !important;
# }

# /* ── DocQuery title ── */
# .dq-wordmark {
#     font-size: 1.75rem;
#     font-weight: 700;
#     color: #e4e4f0;
#     letter-spacing: -0.6px;
#     margin-bottom: 2px;
#     line-height: 1;
# }
# .dq-wordmark span { color: #4f7cff; }

# .dq-tagline {
#     font-size: 0.82rem;
#     color: #4a4a6a;
#     font-weight: 400;
#     margin-bottom: 1.6rem;
#     letter-spacing: 0.15px;
# }

# /* ── Sidebar ── */
# [data-testid="stSidebar"] {
#     background: #060610 !important;
#     border-right: 1px solid #141428 !important;
# }

# [data-testid="stSidebar"] .block-container {
#     padding-top: 1rem !important;
#     max-width: 100% !important;
# }

# /* Suppress Streamlit's default sidebar text styling */
# [data-testid="stSidebar"] p,
# [data-testid="stSidebar"] span,
# [data-testid="stSidebar"] li,
# [data-testid="stSidebar"] label {
#     color: #7070a0 !important;
#     font-size: 0.82rem !important;
# }

# /* ── Sidebar brand ── */
# .sb-brand {
#     display: flex;
#     align-items: center;
#     gap: 9px;
#     padding-bottom: 14px;
#     border-bottom: 1px solid #141428;
#     margin-bottom: 16px;
# }
# .sb-brand-dot {
#     width: 28px; height: 28px;
#     background: linear-gradient(135deg, #4f7cff, #7c5cfc);
#     border-radius: 7px;
#     display: flex; align-items: center; justify-content: center;
#     font-size: 14px; flex-shrink: 0;
# }
# .sb-brand-name {
#     font-size: 0.95rem; font-weight: 700;
#     color: #dde0f5; letter-spacing: -0.3px;
# }

# /* ── Sidebar section labels ── */
# .sb-section {
#     font-size: 0.63rem;
#     font-weight: 700;
#     color: #2e2e50;
#     text-transform: uppercase;
#     letter-spacing: 1.1px;
#     margin: 18px 0 10px 0;
# }

# /* ── Stack table ── */
# .stack-row {
#     display: flex;
#     justify-content: space-between;
#     align-items: center;
#     padding: 5px 0;
#     border-bottom: 1px solid #0f0f20;
# }
# .stack-key {
#     font-size: 0.72rem; color: #3e3e60; font-weight: 500;
# }
# .stack-val {
#     font-size: 0.75rem; color: #8890cc;
#     font-family: 'Inter', monospace; font-weight: 500;
# }

# /* ── Doc status card ── */
# .doc-card {
#     background: #0c0c1e;
#     border: 1px solid #181830;
#     border-radius: 9px;
#     padding: 11px 13px;
#     margin-top: 6px;
# }
# .doc-card-name {
#     font-size: 0.80rem; font-weight: 600;
#     color: #c8ccf0; margin-bottom: 8px;
#     word-break: break-all; line-height: 1.4;
# }
# .doc-card-row {
#     display: flex; justify-content: space-between;
#     margin-bottom: 3px;
# }
# .doc-card-key { font-size: 0.70rem; color: #2e2e50; }
# .doc-card-val { font-size: 0.70rem; color: #7070a0; font-weight: 500; }
# .doc-badge {
#     display: inline-block;
#     background: #08200f; color: #2dd65a;
#     font-size: 0.65rem; font-weight: 700;
#     padding: 2px 8px; border-radius: 20px;
#     border: 1px solid #124d24; margin-top: 8px;
#     letter-spacing: 0.3px;
# }
# .doc-none {
#     font-size: 0.77rem; color: #2e2e50;
#     font-style: italic; padding: 6px 0;
# }

# /* ── Upload area ── */
# [data-testid="stFileUploader"] {
#     background: #0a0a1c !important;
#     border: 1px dashed #1e1e3c !important;
#     border-radius: 9px !important;
# }
# [data-testid="stFileUploader"] p,
# [data-testid="stFileUploader"] span {
#     font-size: 0.76rem !important;
#     color: #3a3a60 !important;
# }

# /* ── Process button ── */
# .stButton > button {
#     background: #4f7cff !important;
#     color: #ffffff !important;
#     border: none !important;
#     border-radius: 8px !important;
#     font-size: 0.82rem !important;
#     font-weight: 600 !important;
#     padding: 7px 14px !important;
#     width: 100% !important;
#     transition: background 0.15s ease, transform 0.1s ease !important;
#     letter-spacing: 0.2px !important;
#     margin-top: 6px !important;
# }
# .stButton > button:hover {
#     background: #3a68f0 !important;
#     transform: translateY(-1px) !important;
# }
# .stButton > button:active { transform: translateY(0) !important; }

# /* ── Chat bubbles ── */
# [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
#     background: #0e0e20 !important;
#     border: 1px solid #181830 !important;
#     border-radius: 12px !important;
#     box-shadow: none !important;
#     margin-bottom: 10px !important;
# }

# [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
#     background: #0c1422 !important;
#     border: 1px solid #172033 !important;
#     border-radius: 12px !important;
#     box-shadow: 0 2px 16px rgba(79, 124, 255, 0.05) !important;
#     margin-bottom: 10px !important;
#     transition: box-shadow 0.15s ease !important;
# }

# [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]):hover {
#     box-shadow: 0 4px 20px rgba(79, 124, 255, 0.1) !important;
# }

# [data-testid="stMarkdownContainer"] p {
#     color: #c0c4de !important;
#     line-height: 1.72 !important;
#     font-size: 0.91rem !important;
# }

# /* ── Message sender labels ── */
# .msg-sender-user {
#     font-size: 0.68rem; font-weight: 700;
#     color: #4f7cff; text-transform: uppercase;
#     letter-spacing: 0.9px; margin-bottom: 4px;
# }
# .msg-sender-ai {
#     font-size: 0.68rem; font-weight: 700;
#     color: #2ea84f; text-transform: uppercase;
#     letter-spacing: 0.9px; margin-bottom: 4px;
# }

# /* ── Chat input ── */
# [data-testid="stChatInput"] {
#     background: #0c0c1e !important;
#     border: 1px solid #1e1e38 !important;
#     border-radius: 12px !important;
#     box-shadow: none !important;
#     transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
# }
# [data-testid="stChatInput"]:focus-within {
#     border-color: #4f7cff !important;
#     box-shadow: 0 0 0 3px rgba(79, 124, 255, 0.1) !important;
# }
# [data-testid="stChatInputTextArea"] {
#     font-family: 'Inter', sans-serif !important;
#     font-size: 0.91rem !important;
#     color: #d0d4f0 !important;
#     background: transparent !important;
# }

# /* ── Source citation block ── */
# .src-block {
#     background: #0a0a1c;
#     border: 1px solid #141428;
#     border-left: 2px solid #4f7cff;
#     border-radius: 7px;
#     padding: 7px 11px;
#     margin: 4px 0;
# }
# .src-filename { font-size: 0.79rem; font-weight: 600; color: #6080ff; }
# .src-page { font-size: 0.70rem; color: #30305a; margin-top: 2px; }

# /* ── Retrieved Evidence expander ── */
# [data-testid="stExpander"] {
#     background: #09091a !important;
#     border: 1px solid #131326 !important;
#     border-radius: 10px !important;
#     margin-top: 8px !important;
#     overflow: hidden !important;
# }
# [data-testid="stExpander"] summary {
#     font-size: 0.73rem !important;
#     font-weight: 600 !important;
#     color: #2e2e55 !important;
#     letter-spacing: 0.5px !important;
#     padding: 8px 12px !important;
#     text-transform: uppercase !important;
# }
# [data-testid="stExpander"]:hover {
#     border-color: #1e1e40 !important;
# }

# /* ── Chunk display inside expander ── */
# .chunk-header {
#     font-size: 0.72rem; font-weight: 700;
#     color: #3a3a66; text-transform: uppercase;
#     letter-spacing: 0.8px; margin-bottom: 4px;
# }
# .chunk-meta-row {
#     display: flex; gap: 16px; margin-bottom: 6px;
# }
# .chunk-meta-item { font-size: 0.71rem; color: #2e2e55; }
# .chunk-meta-item span { color: #5858a0; font-weight: 500; }
# .chunk-content {
#     font-size: 0.82rem; color: #8888b0;
#     line-height: 1.65; padding: 6px 0;
# }

# /* ── Empty state ── */
# .empty-state-wrap {
#     text-align: center;
#     padding: 52px 20px 40px 20px;
#     border: 1px dashed #141428;
#     border-radius: 16px;
#     margin: 8px 0 24px 0;
#     background: #09091a;
# }
# .empty-icon { font-size: 2.2rem; opacity: 0.35; margin-bottom: 14px; }
# .empty-title {
#     font-size: 1.15rem; font-weight: 600;
#     color: #8080a8; margin-bottom: 8px; letter-spacing: -0.3px;
# }
# .empty-body {
#     font-size: 0.82rem; color: #34345a;
#     line-height: 1.75; margin-bottom: 0;
# }

# /* ── Relevance guard info box ── */
# .no-context-box {
#     background: #0a0a1c;
#     border: 1px solid #1a1a30;
#     border-left: 3px solid #4f7cff;
#     border-radius: 8px;
#     padding: 12px 15px;
#     font-size: 0.87rem;
#     color: #6068a0;
#     line-height: 1.7;
# }

# /* ── Inline code ── */
# code {
#     background: #10101e !important;
#     color: #5a7aff !important;
#     border-radius: 5px !important;
#     padding: 1px 6px !important;
#     font-size: 0.82em !important;
# }

# /* ── Divider ── */
# hr {
#     border: none !important;
#     border-top: 1px solid #111128 !important;
#     margin: 10px 0 !important;
# }

# /* ── Caption / metadata ── */
# [data-testid="stCaptionContainer"] p {
#     color: #252545 !important;
#     font-size: 0.70rem !important;
#     line-height: 1.5 !important;
# }

# /* ── Error alert ── */
# [data-testid="stAlert"] {
#     border-radius: 9px !important;
#     border-left: 3px solid #ef4444 !important;
#     background: #140a0a !important;
# }

# /* ── Spinner ── */
# [data-testid="stStatusWidget"] {
#     background: #0c0c1e !important;
#     border: 1px solid #1e1e38 !important;
#     border-radius: 9px !important;
# }

# /* ── Scrollbar ── */
# ::-webkit-scrollbar { width: 4px; }
# ::-webkit-scrollbar-track { background: #080810; }
# ::-webkit-scrollbar-thumb { background: #1e1e38; border-radius: 10px; }
# ::-webkit-scrollbar-thumb:hover { background: #4f7cff; }
# </style>
# """, unsafe_allow_html=True)

# # ── Initialize Pinecone + vector store (UNCHANGED) ────────────────────────────
# pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# index_name = os.environ.get("PINECONE_INDEX_NAME")
# index = pc.Index(index_name)

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# # ── Session state initialization ──────────────────────────────────────────────
# # Existing keys (messages, citations, llm) are preserved as-is.
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#     st.session_state.messages.append(
#         SystemMessage("You are an assistant for question-answering tasks.")
#     )

# if "citations" not in st.session_state:
#     st.session_state.citations = []

# # New keys for document upload tracking (no impact on RAG pipeline)
# if "doc_uploaded" not in st.session_state:
#     st.session_state.doc_uploaded = False
# if "doc_info" not in st.session_state:
#     st.session_state.doc_info = {}

# # ── LLM — cached, unchanged ───────────────────────────────────────────────────
# if "llm" not in st.session_state:
#     pipe = pipeline(
#         "text-generation",
#         model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#         max_new_tokens=200,
#         temperature=0.7,
#         return_full_text=False,
#     )
#     st.session_state.llm = HuggingFacePipeline(pipeline=pipe)

# llm = st.session_state.llm

# # ── Sidebar ───────────────────────────────────────────────────────────────────
# # Must come AFTER vector_store is initialized so upload processing can use it.
# with st.sidebar:

#     # ── Brand mark ──────────────────────────────────────────────────────────
#     st.markdown("""
#     <div class="sb-brand">
#         <div class="sb-brand-dot">📄</div>
#         <div class="sb-brand-name">DocQuery</div>
#     </div>
#     """, unsafe_allow_html=True)

#     # ── Upload Document ──────────────────────────────────────────────────────
#     # PDF upload pipeline: reuses same PyPDFLoader + RecursiveCharacterTextSplitter
#     # settings as ingestion.py (chunk_size=800, chunk_overlap=400).
#     st.markdown('<div class="sb-section">Upload Document</div>', unsafe_allow_html=True)

#     uploaded_file = st.file_uploader(
#         "Choose a PDF",
#         type=["pdf"],
#         label_visibility="collapsed",
#     )

#     if uploaded_file is not None:
#         if st.button("Process Document"):
#             with st.spinner("Indexing…"):
#                 # Step 1: save upload bytes to a temp file so PyPDFLoader can read it
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#                     tmp.write(uploaded_file.read())
#                     tmp_path = tmp.name

#                 # Step 2: load PDF — same loader family as ingestion.py
#                 loader = PyPDFLoader(tmp_path)
#                 raw_docs = loader.load()

#                 # Step 3: split into chunks — exact same settings as ingestion.py
#                 splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=800,
#                     chunk_overlap=400,
#                     length_function=len,
#                     is_separator_regex=False,
#                 )
#                 chunks = splitter.split_documents(raw_docs)

#                 # Step 4: replace temp-file path with the real filename in metadata
#                 # so citations display the original name, not a system temp path.
#                 for chunk in chunks:
#                     chunk.metadata["source"] = uploaded_file.name

#                 # Step 5: upsert into Pinecone using existing vector_store + embeddings
#                 uuids = [f"upload_{i}" for i in range(len(chunks))]
#                 vector_store.add_documents(documents=chunks, ids=uuids)

#                 # Step 6: clean up temp file
#                 os.remove(tmp_path)

#                 # Step 7: persist document metadata for the status card
#                 st.session_state.doc_info = {
#                     "filename": uploaded_file.name,
#                     "pages":    len(raw_docs),
#                     "chunks":   len(chunks),
#                 }
#                 st.session_state.doc_uploaded = True

#             st.success(f"✓ Indexed {len(chunks)} chunks from {uploaded_file.name}")

#     # ── Document Status ──────────────────────────────────────────────────────
#     st.markdown('<div class="sb-section">Document Status</div>', unsafe_allow_html=True)

#     if st.session_state.doc_uploaded:
#         info = st.session_state.doc_info
#         st.markdown(f"""
#         <div class="doc-card">
#             <div class="doc-card-name">📄 {info['filename']}</div>
#             <div class="doc-card-row">
#                 <span class="doc-card-key">Pages</span>
#                 <span class="doc-card-val">{info['pages']}</span>
#             </div>
#             <div class="doc-card-row">
#                 <span class="doc-card-key">Chunks</span>
#                 <span class="doc-card-val">{info['chunks']}</span>
#             </div>
#             <div class="doc-card-row">
#                 <span class="doc-card-key">Vector Status</span>
#                 <span class="doc-card-val">Pinecone</span>
#             </div>
#             <span class="doc-badge">✓ Indexed</span>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown('<p class="doc-none">No document uploaded.</p>', unsafe_allow_html=True)

#     # ── Technology Stack ─────────────────────────────────────────────────────
#     st.markdown('<div class="sb-section">Technology Stack</div>', unsafe_allow_html=True)
#     st.markdown("""
#     <div class="stack-row">
#         <span class="stack-key">LLM</span>
#         <span class="stack-val">TinyLlama-1.1B</span>
#     </div>
#     <div class="stack-row">
#         <span class="stack-key">Embedding Model</span>
#         <span class="stack-val">all-MiniLM-L6-v2</span>
#     </div>
#     <div class="stack-row">
#         <span class="stack-key">Vector Database</span>
#         <span class="stack-val">Pinecone</span>
#     </div>
#     <div class="stack-row">
#         <span class="stack-key">Framework</span>
#         <span class="stack-val">LangChain</span>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
#     st.caption("API keys are loaded from .env and never stored.")

# # ── Main area ─────────────────────────────────────────────────────────────────

# # DocQuery wordmark (replaces old "🤖 RAG Chatbot" title)
# st.markdown("""
#     <h1 class="dq-wordmark">Doc<span>Query</span></h1>
#     <p class="dq-tagline">AI-powered document search and analysis</p>
# """, unsafe_allow_html=True)

# # ── Empty state — shown when no document has been uploaded this session ────────
# if not st.session_state.doc_uploaded:
#     st.markdown("""
#     <div class="empty-state-wrap">
#         <div class="empty-icon">📂</div>
#         <div class="empty-title">Welcome to DocQuery</div>
#         <div class="empty-body">
#             Upload a PDF document using the sidebar to get started.<br>
#             Once indexed, you can ask questions about its content.<br><br>
#             <strong style="color:#2e2e55;">Supported format: PDF</strong>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# # ── Chat history display ──────────────────────────────────────────────────────
# # Logic is unchanged. ai_idx pairs each AIMessage with its stored citations.
# ai_idx = 0
# for message in st.session_state.messages:
#     if isinstance(message, HumanMessage):
#         with st.chat_message("user"):
#             # Sender label
#             st.markdown('<div class="msg-sender-user">You</div>', unsafe_allow_html=True)
#             st.markdown(message.content)

#     elif isinstance(message, AIMessage):
#         with st.chat_message("assistant"):
#             # Sender label
#             st.markdown('<div class="msg-sender-ai">DocQuery</div>', unsafe_allow_html=True)
#             st.markdown(message.content)

#             # ── Feature 1: Source Citations (history) ────────────────────────
#             if ai_idx < len(st.session_state.citations):
#                 turn_citations = st.session_state.citations[ai_idx]
#                 if turn_citations:
#                     st.markdown("---")
#                     st.markdown(
#                         "<div style='font-size:0.68rem;font-weight:700;color:#2e2e55;"
#                         "text-transform:uppercase;letter-spacing:0.9px;margin-bottom:6px'>"
#                         "Sources</div>",
#                         unsafe_allow_html=True,
#                     )
#                     seen = set()
#                     for cite in turn_citations:
#                         key = (cite["source"], cite["page"])
#                         if key not in seen:
#                             seen.add(key)
#                             # Show only the filename, not the full temp path
#                             fname = cite["source"].split("/")[-1].split("\\")[-1]
#                             page_line = (
#                                 f'<div class="src-page">Page {cite["page"]}</div>'
#                                 if cite["page"] is not None else ""
#                             )
#                             st.markdown(
#                                 f'<div class="src-block">'
#                                 f'<div class="src-filename">📄 {fname}</div>'
#                                 f'{page_line}</div>',
#                                 unsafe_allow_html=True,
#                             )

#             # ── Feature 2: Retrieved Evidence expander (history) ─────────────
#             if ai_idx < len(st.session_state.citations):
#                 turn_citations = st.session_state.citations[ai_idx]
#                 if turn_citations:
#                     with st.expander("Retrieved Evidence"):
#                         for i, cite in enumerate(turn_citations, start=1):
#                             fname = cite["source"].split("/")[-1].split("\\")[-1]
#                             page_str = (
#                                 f'<span>{cite["page"]}</span>' if cite["page"] is not None
#                                 else "<span>—</span>"
#                             )
#                             st.markdown(
#                                 f'<div class="chunk-header">Chunk {i}</div>'
#                                 f'<div class="chunk-meta-row">'
#                                 f'<div class="chunk-meta-item">Source: <span>{fname}</span></div>'
#                                 f'<div class="chunk-meta-item">Page: {page_str}</div>'
#                                 f'</div>',
#                                 unsafe_allow_html=True,
#                             )
#                             st.markdown(
#                                 f'<div class="chunk-content">{cite["content"]}</div>',
#                                 unsafe_allow_html=True,
#                             )
#                             if i < len(turn_citations):
#                                 st.markdown("---")

#         ai_idx += 1

# # ── Chat input ────────────────────────────────────────────────────────────────
# prompt = st.chat_input("Ask a question about your document…")

# # ── RAG pipeline ─────────────────────────────────────────────────────────────
# if prompt:

#     # Display user message
#     with st.chat_message("user"):
#         st.markdown('<div class="msg-sender-user">You</div>', unsafe_allow_html=True)
#         st.markdown(prompt)

#     st.session_state.messages.append(HumanMessage(prompt))

#     try:
#         # ── Retriever (UNCHANGED) ─────────────────────────────────────────────
#         retriever = vector_store.as_retriever(
#             search_type="similarity_score_threshold",
#             search_kwargs={"k": 3, "score_threshold": 0.1},
#         )

#         docs = retriever.invoke(prompt)

#         # ── Relevance guard ───────────────────────────────────────────────────
#         # If the retriever finds nothing above the score threshold, the query is
#         # off-topic or unrelated to indexed documents. Do NOT call the LLM to
#         # prevent hallucination. Return a polite no-context message instead.
#         if not docs:
#             no_ctx_msg = (
#                 "I couldn't find relevant information in the uploaded documents. "
#                 "Please ask questions related to the document content."
#             )
#             with st.chat_message("assistant"):
#                 st.markdown('<div class="msg-sender-ai">DocQuery</div>', unsafe_allow_html=True)
#                 st.markdown(
#                     f'<div class="no-context-box">{no_ctx_msg}</div>',
#                     unsafe_allow_html=True,
#                 )
#             st.session_state.messages.append(AIMessage(no_ctx_msg))
#             # Append empty citations list to keep ai_idx alignment intact
#             st.session_state.citations.append([])

#         else:
#             # ── Full RAG pipeline (UNCHANGED) ─────────────────────────────────
#             docs_text = "".join(d.page_content for d in docs)

#             # Prompt template (UNCHANGED)
#             system_prompt = """You are an assistant for question-answering tasks. 
#     Use the following pieces of retrieved context to answer the question. 
#     If you don't know the answer, just say that you don't know. 
#     Use three sentences maximum and keep the answer concise.
#     Context: {context}:"""

#             system_prompt_fmt = system_prompt.format(context=docs_text)

#             print("-- SYS PROMPT --")
#             print(system_prompt_fmt)

#             # Replace system prompt at index 0 (Bug 5 fix — unchanged)
#             st.session_state.messages[0] = SystemMessage(system_prompt_fmt)

#             # Serialize messages to plain string (Bug 2 fix — unchanged)
#             prompt_text = "\n".join(
#                 f"{'System' if isinstance(m, SystemMessage) else 'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
#                 for m in st.session_state.messages
#             )

#             # LLM invocation (Bug 3 fix — unchanged). Spinner updated to DocQuery copy.
#             with st.spinner("Searching documents…"):
#                 result = llm.invoke(prompt_text)

#             # ── Extract citation data ─────────────────────────────────────────
#             # Stored as plain dicts so metadata survives Streamlit reruns.
#             turn_citations = []
#             for doc in docs:
#                 turn_citations.append({
#                     "source":   doc.metadata.get("source", "Unknown"),
#                     "page":     doc.metadata.get("page", None),
#                     "content":  doc.page_content,
#                     "metadata": doc.metadata,
#                 })

#             # ── Display assistant response ────────────────────────────────────
#             with st.chat_message("assistant"):
#                 st.markdown('<div class="msg-sender-ai">DocQuery</div>', unsafe_allow_html=True)
#                 st.markdown(result)

#                 # ── Feature 1: Source Citations (live turn) ───────────────────
#                 if turn_citations:
#                     st.markdown("---")
#                     st.markdown(
#                         "<div style='font-size:0.68rem;font-weight:700;color:#2e2e55;"
#                         "text-transform:uppercase;letter-spacing:0.9px;margin-bottom:6px'>"
#                         "Sources</div>",
#                         unsafe_allow_html=True,
#                     )
#                     seen = set()
#                     for cite in turn_citations:
#                         key = (cite["source"], cite["page"])
#                         if key not in seen:
#                             seen.add(key)
#                             fname = cite["source"].split("/")[-1].split("\\")[-1]
#                             page_line = (
#                                 f'<div class="src-page">Page {cite["page"]}</div>'
#                                 if cite["page"] is not None else ""
#                             )
#                             st.markdown(
#                                 f'<div class="src-block">'
#                                 f'<div class="src-filename">📄 {fname}</div>'
#                                 f'{page_line}</div>',
#                                 unsafe_allow_html=True,
#                             )

#                 # ── Feature 2: Retrieved Evidence expander (live turn) ────────
#                 if turn_citations:
#                     with st.expander("Retrieved Evidence"):
#                         for i, cite in enumerate(turn_citations, start=1):
#                             fname = cite["source"].split("/")[-1].split("\\")[-1]
#                             page_str = (
#                                 f'<span>{cite["page"]}</span>'
#                                 if cite["page"] is not None else "<span>—</span>"
#                             )
#                             st.markdown(
#                                 f'<div class="chunk-header">Chunk {i}</div>'
#                                 f'<div class="chunk-meta-row">'
#                                 f'<div class="chunk-meta-item">Source: <span>{fname}</span></div>'
#                                 f'<div class="chunk-meta-item">Page: {page_str}</div>'
#                                 f'</div>',
#                                 unsafe_allow_html=True,
#                             )
#                             st.markdown(
#                                 f'<div class="chunk-content">{cite["content"]}</div>',
#                                 unsafe_allow_html=True,
#                             )
#                             if i < len(turn_citations):
#                                 st.markdown("---")

#                 st.session_state.messages.append(AIMessage(result))
#                 # Persist citations for future reruns (Feature 1 — unchanged logic)
#                 st.session_state.citations.append(turn_citations)

#     # Error handling (Bug 7 fix — unchanged)
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
# DocQuery – AI-powered document search and analysis
# =============================================================
# Rebranded from RAG Chatbot → DocQuery
#
# Changes made in this file (focused, minimal):
#  - Rebrand UI and header to "DocQuery" + subtitle.
#  - Professional sidebar with Technology Stack and Document Status card.
#  - PDF upload + indexing flow (reuses PyPDFLoader + RecursiveCharacterTextSplitter + Pinecone upsert).
#  - Relevance guard: return a no-context message and DO NOT call the LLM if no relevant chunks are found.
#  - Rename "Retrieved Context" → "Retrieved Evidence".
#  - Improved chat styling and assistant name "DocQuery".
#  - Empty-state before upload with requested message.
# =============================================================

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from transformers import pipeline

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# LangChain core (unchanged)
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# PDF ingestion — reuses same loader + splitter as ingestion.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocQuery",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
# Clean, professional SaaS-style dark theme. Targets Streamlit data-testid attributes.
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Background ── */
[data-testid="stAppViewContainer"] {
    background: #080810;
    min-height: 100vh;
}

[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 5.5rem !important;
    max-width: 780px !important;
    margin: 0 auto !important;
}

/* ── DocQuery title ── */
.dq-wordmark {
    font-size: 1.75rem;
    font-weight: 700;
    color: #e4e4f0;
    letter-spacing: -0.6px;
    margin-bottom: 2px;
    line-height: 1;
}
.dq-wordmark span { color: #4f7cff; }

.dq-tagline {
    font-size: 0.82rem;
    color: #4a4a6a;
    font-weight: 400;
    margin-bottom: 1.6rem;
    letter-spacing: 0.15px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #060610 !important;
    border-right: 1px solid #141428 !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1rem !important;
    max-width: 100% !important;
}

/* Suppress Streamlit's default sidebar text styling */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label {
    color: #7070a0 !important;
    font-size: 0.82rem !important;
}

/* ── Sidebar brand ── */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 9px;
    padding-bottom: 14px;
    border-bottom: 1px solid #141428;
    margin-bottom: 16px;
}
.sb-brand-dot {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #4f7cff, #7c5cfc);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
}
.sb-brand-name {
    font-size: 0.95rem; font-weight: 700;
    color: #dde0f5; letter-spacing: -0.3px;
}

/* ── Sidebar section labels ── */
.sb-section {
    font-size: 0.63rem;
    font-weight: 700;
    color: #2e2e50;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    margin: 18px 0 10px 0;
}

/* ── Stack table ── */
.stack-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #0f0f20;
}
.stack-key {
    font-size: 0.72rem; color: #3e3e60; font-weight: 500;
}
.stack-val {
    font-size: 0.75rem; color: #8890cc;
    font-family: 'Inter', monospace; font-weight: 500;
}

/* ── Doc status card ── */
.doc-card {
    background: #0c0c1e;
    border: 1px solid #181830;
    border-radius: 9px;
    padding: 11px 13px;
    margin-top: 6px;
}
.doc-card-title {
    font-size: 0.72rem; color: #2e2e50; font-weight: 700;
    margin-bottom: 6px;
    text-transform: uppercase;
}
.doc-card-name {
    font-size: 0.80rem; font-weight: 600;
    color: #c8ccf0; margin-bottom: 8px;
    word-break: break-all; line-height: 1.4;
}
.doc-card-row {
    display: flex; justify-content: space-between;
    margin-bottom: 3px;
}
.doc-card-key { font-size: 0.70rem; color: #2e2e50; }
.doc-card-val { font-size: 0.70rem; color: #7070a0; font-weight: 500; }
.doc-badge {
    display: inline-block;
    background: #08200f; color: #2dd65a;
    font-size: 0.65rem; font-weight: 700;
    padding: 2px 8px; border-radius: 20px;
    border: 1px solid #124d24; margin-top: 8px;
    letter-spacing: 0.3px;
}
.doc-none {
    font-size: 0.77rem; color: #2e2e50;
    font-style: italic; padding: 6px 0;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #0a0a1c !important;
    border: 1px dashed #1e1e3c !important;
    border-radius: 9px !important;
}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span {
    font-size: 0.76rem !important;
    color: #3a3a60 !important;
}

/* ── Process button ── */
.stButton > button {
    background: #4f7cff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 7px 14px !important;
    width: 100% !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
    letter-spacing: 0.2px !important;
    margin-top: 6px !important;
}
.stButton > button:hover {
    background: #3a68f0 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Chat bubbles ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #0e0e20 !important;
    border: 1px solid #181830 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    margin-bottom: 10px !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #0c1422 !important;
    border: 1px solid #172033 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 16px rgba(79, 124, 255, 0.05) !important;
    margin-bottom: 10px !important;
    transition: box-shadow 0.15s ease !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]):hover {
    box-shadow: 0 4px 20px rgba(79, 124, 255, 0.1) !important;
}

[data-testid="stMarkdownContainer"] p {
    color: #c0c4de !important;
    line-height: 1.72 !important;
    font-size: 0.91rem !important;
}

/* ── Message sender labels ── */
.msg-sender-user {
    font-size: 0.68rem; font-weight: 700;
    color: #4f7cff; text-transform: uppercase;
    letter-spacing: 0.9px; margin-bottom: 4px;
}
.msg-sender-ai {
    font-size: 0.68rem; font-weight: 700;
    color: #2ea84f; text-transform: uppercase;
    letter-spacing: 0.9px; margin-bottom: 4px;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #0c0c1e !important;
    border: 1px solid #1e1e38 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #4f7cff !important;
    box-shadow: 0 0 0 3px rgba(79, 124, 255, 0.1) !important;
}
[data-testid="stChatInputTextArea"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.91rem !important;
    color: #d0d4f0 !important;
    background: transparent !important;
}

/* ── Source citation block ── */
.src-block {
    background: #0a0a1c;
    border: 1px solid #141428;
    border-left: 2px solid #4f7cff;
    border-radius: 7px;
    padding: 7px 11px;
    margin: 4px 0;
}
.src-filename { font-size: 0.79rem; font-weight: 600; color: #6080ff; }
.src-page { font-size: 0.70rem; color: #30305a; margin-top: 2px; }

/* ── Retrieved Evidence expander ── */
[data-testid="stExpander"] {
    background: #09091a !important;
    border: 1px solid #131326 !important;
    border-radius: 10px !important;
    margin-top: 8px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.73rem !important;
    font-weight: 600 !important;
    color: #2e2e55 !important;
    letter-spacing: 0.5px !important;
    padding: 8px 12px !important;
    text-transform: uppercase !important;
}
[data-testid="stExpander"]:hover {
    border-color: #1e1e40 !important;
}

/* ── Chunk display inside expander ── */
.chunk-header {
    font-size: 0.72rem; font-weight: 700;
    color: #3a3a66; text-transform: uppercase;
    letter-spacing: 0.8px; margin-bottom: 4px;
}
.chunk-meta-row {
    display: flex; gap: 16px; margin-bottom: 6px;
}
.chunk-meta-item { font-size: 0.71rem; color: #2e2e55; }
.chunk-meta-item span { color: #5858a0; font-weight: 500; }
.chunk-content {
    font-size: 0.82rem; color: #8888b0;
    line-height: 1.65; padding: 6px 0;
}

/* ── Empty state ── */
.empty-state-wrap {
    text-align: center;
    padding: 52px 20px 40px 20px;
    border: 1px dashed #141428;
    border-radius: 16px;
    margin: 8px 0 24px 0;
    background: #09091a;
}
.empty-icon { font-size: 2.2rem; opacity: 0.35; margin-bottom: 14px; }
.empty-title {
    font-size: 1.15rem; font-weight: 600;
    color: #8080a8; margin-bottom: 8px; letter-spacing: -0.3px;
}
.empty-body {
    font-size: 0.82rem; color: #34345a;
    line-height: 1.75; margin-bottom: 0;
}

/* ── Relevance guard info box ── */
.no-context-box {
    background: #0a0a1c;
    border: 1px solid #1a1a30;
    border-left: 3px solid #4f7cff;
    border-radius: 8px;
    padding: 12px 15px;
    font-size: 0.87rem;
    color: #6068a0;
    line-height: 1.7;
}

/* ── Inline code ── */
code {
    background: #10101e !important;
    color: #5a7aff !important;
    border-radius: 5px !important;
    padding: 1px 6px !important;
    font-size: 0.82em !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #111128 !important;
    margin: 10px 0 !important;
}

/* ── Caption / metadata ── */
[data-testid="stCaptionContainer"] p {
    color: #252545 !important;
    font-size: 0.70rem !important;
    line-height: 1.5 !important;
}

/* ── Error alert ── */
[data-testid="stAlert"] {
    border-radius: 9px !important;
    border-left: 3px solid #ef4444 !important;
    background: #140a0a !important;
}

/* ── Spinner ── */
[data-testid="stStatusWidget"] {
    background: #0c0c1e !important;
    border: 1px solid #1e1e38 !important;
    border-radius: 9px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080810; }
::-webkit-scrollbar-thumb { background: #1e1e38; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #4f7cff; }
</style>
""", unsafe_allow_html=True)

# ── Initialize Pinecone + vector store (UNCHANGED) ────────────────────────────
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ── Session state initialization ──────────────────────────────────────────────
# Existing keys (messages, citations, llm) are preserved as-is.
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        SystemMessage("You are an assistant for question-answering tasks.")
    )

if "citations" not in st.session_state:
    st.session_state.citations = []

# New keys for document upload tracking (no impact on RAG pipeline)
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False
if "doc_info" not in st.session_state:
    st.session_state.doc_info = {}

# ── LLM — cached, unchanged ───────────────────────────────────────────────────
if "llm" not in st.session_state:
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=200,
        temperature=0.7,
        return_full_text=False,
    )
    st.session_state.llm = HuggingFacePipeline(pipeline=pipe)

llm = st.session_state.llm

# ── Sidebar ───────────────────────────────────────────────────────────────────
# Must come AFTER vector_store is initialized so upload processing can use it.
with st.sidebar:

    # ── Brand mark ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-dot">📄</div>
        <div class="sb-brand-name">DocQuery</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload Document ──────────────────────────────────────────────────────
    # PDF upload pipeline: reuses same PyPDFLoader + RecursiveCharacterTextSplitter
    # settings as ingestion.py (chunk_size=800, chunk_overlap=400).
    st.markdown('<div class="sb-section">Upload Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Indexing…"):
                # Step 1: save upload bytes to a temp file so PyPDFLoader can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Step 2: load PDF — same loader family as ingestion.py
                loader = PyPDFLoader(tmp_path)
                raw_docs = loader.load()

                # Step 3: split into chunks — exact same settings as ingestion.py
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=400,
                    length_function=len,
                    is_separator_regex=False,
                )
                chunks = splitter.split_documents(raw_docs)

                # Step 4: replace temp-file path with the real filename in metadata
                # so citations display the original name, not a system temp path.
                for chunk in chunks:
                    chunk.metadata["source"] = uploaded_file.name

                # Step 5: upsert into Pinecone using existing vector_store + embeddings
                uuids = [f"upload_{i}" for i in range(len(chunks))]
                vector_store.add_documents(documents=chunks, ids=uuids)

                # Step 6: clean up temp file
                os.remove(tmp_path)

                # Step 7: persist document metadata for the status card
                st.session_state.doc_info = {
                    "filename": uploaded_file.name,
                    "pages":    len(raw_docs),
                    "chunks":   len(chunks),
                }
                st.session_state.doc_uploaded = True

            st.success(f"✓ Indexed {len(chunks)} chunks from {uploaded_file.name}")

    # ── Document Status ──────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Document Status</div>', unsafe_allow_html=True)

    if st.session_state.doc_uploaded:
        info = st.session_state.doc_info
        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-card-title">Current Document</div>
            <div class="doc-card-name">📄 {info['filename']}</div>
            <div class="doc-card-row">
                <span class="doc-card-key">Pages</span>
                <span class="doc-card-val">{info['pages']}</span>
            </div>
            <div class="doc-card-row">
                <span class="doc-card-key">Chunks</span>
                <span class="doc-card-val">{info['chunks']}</span>
            </div>
            <div class="doc-card-row">
                <span class="doc-card-key">Vector Status</span>
                <span class="doc-card-val">Indexed</span>
            </div>
            <span class="doc-badge">✓ Indexed</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<p class="doc-none">No document uploaded.</p>', unsafe_allow_html=True)

    # ── Technology Stack ─────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Technology Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stack-row">
        <span class="stack-key">LLM</span>
        <span class="stack-val">TinyLlama-1.1B</span>
    </div>
    <div class="stack-row">
        <span class="stack-key">Embedding Model</span>
        <span class="stack-val">all-MiniLM-L6-v2</span>
    </div>
    <div class="stack-row">
        <span class="stack-key">Vector Database</span>
        <span class="stack-val">Pinecone</span>
    </div>
    <div class="stack-row">
        <span class="stack-key">Framework</span>
        <span class="stack-val">LangChain</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.caption("API keys are loaded from .env and never stored.")

# ── Main area ─────────────────────────────────────────────────────────────────

# DocQuery wordmark (replaces old "🤖 RAG Chatbot" title)
st.markdown("""
    <h1 class="dq-wordmark">Doc<span>Query</span></h1>
    <p class="dq-tagline">AI-powered document search and analysis</p>
""", unsafe_allow_html=True)

# ── Empty state — shown when no document has been uploaded this session ────────
if not st.session_state.doc_uploaded:
    st.markdown("""
    <div class="empty-state-wrap">
        <div class="empty-icon">📂</div>
        <div class="empty-title">Welcome to DocQuery</div>
        <div class="empty-body">
            Upload a PDF to begin asking questions about your documents.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat history display ──────────────────────────────────────────────────────
# Logic is unchanged. ai_idx pairs each AIMessage with its stored citations.
ai_idx = 0
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            # Sender label
            st.markdown('<div class="msg-sender-user">You</div>', unsafe_allow_html=True)
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            # Sender label
            st.markdown('<div class="msg-sender-ai">DocQuery</div>', unsafe_allow_html=True)
            st.markdown(message.content)

            # ── Feature 1: Source Citations (history) ────────────────────────
            if ai_idx < len(st.session_state.citations):
                turn_citations = st.session_state.citations[ai_idx]
                if turn_citations:
                    st.markdown("---")
                    st.markdown(
                        "<div style='font-size:0.68rem;font-weight:700;color:#2e2e55;"
                        "text-transform:uppercase;letter-spacing:0.9px;margin-bottom:6px'>"
                        "Sources</div>",
                        unsafe_allow_html=True,
                    )
                    seen = set()
                    for cite in turn_citations:
                        key = (cite["source"], cite["page"])
                        if key not in seen:
                            seen.add(key)
                            # Show only the filename, not the full temp path
                            fname = cite["source"].split("/")[-1].split("\\")[-1]
                            page_line = (
                                f'<div class="src-page">Page {cite["page"]}</div>'
                                if cite["page"] is not None else ""
                            )
                            st.markdown(
                                f'<div class="src-block">'
                                f'<div class="src-filename">📄 {fname}</div>'
                                f'{page_line}</div>',
                                unsafe_allow_html=True,
                            )

            # ── Feature 2: Retrieved Evidence expander (history) ─────────────
            if ai_idx < len(st.session_state.citations):
                turn_citations = st.session_state.citations[ai_idx]
                if turn_citations:
                    with st.expander("Retrieved Evidence"):
                        for i, cite in enumerate(turn_citations, start=1):
                            fname = cite["source"].split("/")[-1].split("\\")[-1]
                            page_str = (
                                f'<span>{cite["page"]}</span>' if cite["page"] is not None
                                else "<span>—</span>"
                            )
                            st.markdown(
                                f'<div class="chunk-header">Chunk {i}</div>'
                                f'<div class="chunk-meta-row">'
                                f'<div class="chunk-meta-item">Source: <span>{fname}</span></div>'
                                f'<div class="chunk-meta-item">Page: {page_str}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="chunk-content">{cite["content"]}</div>',
                                unsafe_allow_html=True,
                            )
                            if i < len(turn_citations):
                                st.markdown("---")

        ai_idx += 1

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask a question about your document…")

# ── RAG pipeline ─────────────────────────────────────────────────────────────
if prompt:

    # Display user message
    with st.chat_message("user"):
        st.markdown('<div class="msg-sender-user">You</div>', unsafe_allow_html=True)
        st.markdown(prompt)

    st.session_state.messages.append(HumanMessage(prompt))

    try:
        # ── Retriever (UNCHANGED) ─────────────────────────────────────────────
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )

        docs = retriever.invoke(prompt)

        # ── Relevance guard ───────────────────────────────────────────────────
        # If the retriever finds nothing above the score threshold, the query is
        # off-topic or unrelated to indexed documents. Do NOT call the LLM to
        # prevent hallucination. Return a polite no-context message instead.
        if not docs:
            no_ctx_msg = (
                "I couldn't find relevant information in the uploaded document.\n\n"
                "Please ask questions related to the uploaded document."
            )
            with st.chat_message("assistant"):
                st.markdown('<div class="msg-sender-ai">DocQuery</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="no-context-box">{no_ctx_msg.replace(chr(10), "<br><br>")}</div>',
                    unsafe_allow_html=True,
                )
            st.session_state.messages.append(AIMessage(no_ctx_msg))
            # Append empty citations list to keep ai_idx alignment intact
            st.session_state.citations.append([])

        else:
            # ── Full RAG pipeline (UNCHANGED) ─────────────────────────────────
            docs_text = "".join(d.page_content for d in docs)

            # Prompt template (UNCHANGED)
            system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Context: {context}:"""

            system_prompt_fmt = system_prompt.format(context=docs_text)

            print("-- SYS PROMPT --")
            print(system_prompt_fmt)

            # Replace system prompt at index 0 (Bug 5 fix — unchanged)
            st.session_state.messages[0] = SystemMessage(system_prompt_fmt)

            # Serialize messages to plain string (Bug 2 fix — unchanged)
            prompt_text = "\n".join(
                f"{'System' if isinstance(m, SystemMessage) else 'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in st.session_state.messages
            )

            # LLM invocation (Bug 3 fix — unchanged). Spinner updated to DocQuery copy.
            with st.spinner("Searching documents…"):
                result = llm.invoke(prompt_text)

            # ── Extract citation data ─────────────────────────────────────────
            # Stored as plain dicts so metadata survives Streamlit reruns.
            turn_citations = []
            for doc in docs:
                turn_citations.append({
                    "source":   doc.metadata.get("source", "Unknown"),
                    "page":     doc.metadata.get("page", None),
                    "content":  doc.page_content,
                    "metadata": doc.metadata,
                })

            # ── Display assistant response ────────────────────────────────────
            with st.chat_message("assistant"):
                st.markdown('<div class="msg-sender-ai">DocQuery</div>', unsafe_allow_html=True)
                st.markdown(result)

                # ── Feature 1: Source Citations (live turn) ───────────────────
                if turn_citations:
                    st.markdown("---")
                    st.markdown(
                        "<div style='font-size:0.68rem;font-weight:700;color:#2e2e55;"
                        "text-transform:uppercase;letter-spacing:0.9px;margin-bottom:6px'>"
                        "Sources</div>",
                        unsafe_allow_html=True,
                    )
                    seen = set()
                    for cite in turn_citations:
                        key = (cite["source"], cite["page"])
                        if key not in seen:
                            seen.add(key)
                            fname = cite["source"].split("/")[-1].split("\\")[-1]
                            page_line = (
                                f'<div class="src-page">Page {cite["page"]}</div>'
                                if cite["page"] is not None else ""
                            )
                            st.markdown(
                                f'<div class="src-block">'
                                f'<div class="src-filename">📄 {fname}</div>'
                                f'{page_line}</div>',
                                unsafe_allow_html=True,
                            )

                # ── Feature 2: Retrieved Evidence expander (live turn) ────────
                if turn_citations:
                    with st.expander("Retrieved Evidence"):
                        for i, cite in enumerate(turn_citations, start=1):
                            fname = cite["source"].split("/")[-1].split("\\")[-1]
                            page_str = (
                                f'<span>{cite["page"]}</span>'
                                if cite["page"] is not None else "<span>—</span>"
                            )
                            st.markdown(
                                f'<div class="chunk-header">Chunk {i}</div>'
                                f'<div class="chunk-meta-row">'
                                f'<div class="chunk-meta-item">Source: <span>{fname}</span></div>'
                                f'<div class="chunk-meta-item">Page: {page_str}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="chunk-content">{cite["content"]}</div>',
                                unsafe_allow_html=True,
                            )
                            if i < len(turn_citations):
                                st.markdown("---")

                st.session_state.messages.append(AIMessage(result))
                # Persist citations for future reruns (Feature 1 — unchanged logic)
                st.session_state.citations.append(turn_citations)

    # Error handling (unchanged)
    except Exception as e:
        st.error(f"An error occurred: {e}")