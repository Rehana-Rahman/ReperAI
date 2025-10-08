"""
ReperAI - Smart Contextual Summarizer for Research Papers
A clean RAG implementation for analyzing research documents
"""

# File: config/settings.py
import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model settings - tweak these based on your needs"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.3  # lower = more deterministic
    max_tokens: int = 500
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class AppConfig:
    """App configuration"""
    data_dir: str = "data/uploads"
    index_dir: str = "data/indices"
    max_file_size_mb: int = 10
    supported_formats: tuple = (".pdf", ".txt", ".md")
    
    def __post_init__(self):
        # Make sure data directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

model_config = ModelConfig()
app_config = AppConfig()


# File: utils/document_processor.py
"""Handles loading and chunking documents"""

import re
from typing import List, Dict
from pathlib import Path
import PyPDF2

class DocumentProcessor:
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, file_path):
        """Extract text from PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return self._clean_text(text)
        except Exception as e:
            raise ValueError(f"Failed to read PDF: {e}")
    
    def load_text(self, file_path):
        """Load plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return self._clean_text(f.read())
        except Exception as e:
            raise ValueError(f"Failed to read file: {e}")
    
    def _clean_text(self, text):
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def chunk_text(self, text):
        """Split text into overlapping chunks for better context"""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'content': chunk_text,
                'index': len(chunks),
                'word_count': len(chunk_words)
            })
            
            i += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def process_document(self, file_path):
        """Main processing pipeline"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            text = self.load_pdf(file_path)
        elif ext in ['.txt', '.md']:
            text = self.load_text(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        
        return self.chunk_text(text)


# File: models/embeddings.py
"""Vector embeddings and similarity search"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

class EmbeddingManager:
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def create_embeddings(self, chunks):
        """Generate embeddings for all chunks"""
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.astype('float32')
    
    def build_index(self, embeddings):
        """Build FAISS index for fast search"""
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
    
    def search_similar(self, query, k=5):
        """Find k most similar chunks to query"""
        query_vec = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'chunk': self.chunks[idx],
                'score': float(dist),
                'index': int(idx)
            })
        
        return results
    
    def save_index(self, path):
        """Save index to disk"""
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
    
    def load_index(self, path):
        """Load saved index"""
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)


# File: models/summarizer.py
"""RAG-based summarization using retrieved context"""

import openai
from typing import List, Dict

class ContextualSummarizer:
    
    def __init__(self, api_key, model="gpt-3.5-turbo", temperature=0.3):
        openai.api_key = api_key
        self.model = model
        self.temperature = temperature
    
    def _build_prompt(self, query, context_chunks):
        """Create prompt with retrieved context"""
        context = "\n\n".join([
            f"[Chunk {i+1}]\n{chunk}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""Based on the following excerpts from a research paper, answer this question: "{query}"

Context from paper:
{context}

Provide a clear, concise answer. Reference chunk numbers when citing specific info. If the context doesn't fully answer the question, say so.

Answer:"""
        
        return prompt
    
    def generate_summary(self, query, retrieved_chunks, max_tokens=500):
        """Generate summary from retrieved chunks"""
        
        # Extract just the text content
        context_texts = [chunk['chunk']['content'] for chunk in retrieved_chunks]
        
        # Build the prompt
        prompt = self._build_prompt(query, context_texts)
        
        # Call OpenAI
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research assistant helping analyze academic papers."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=max_tokens
        )
        
        summary = response.choices[0].message.content
        
        return {
            'summary': summary,
            'sources': [c['index'] for c in retrieved_chunks],
            'scores': [c['score'] for c in retrieved_chunks]
        }
    
    def extract_key_points(self, text, num_points=5):
        """Pull out main takeaways from text"""
        # Limit context to avoid token limits
        truncated = text[:3000]
        
        prompt = f"""List the {num_points} most important points from this text. Just return a numbered list, nothing else.

Text:
{truncated}

Key points:"""
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        
        points = response.choices[0].message.content
        return [line.strip() for line in points.split('\n') if line.strip()]


# File: ui/app.py
"""Streamlit UI for ReperAI"""

import streamlit as st
from pathlib import Path
import sys

# Add parent dir to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import model_config, app_config
from utils.document_processor import DocumentProcessor
from models.embeddings import EmbeddingManager
from models.summarizer import ContextualSummarizer

def init_session():
    """Set up session state variables"""
    if 'doc_ready' not in st.session_state:
        st.session_state.doc_ready = False
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'filename' not in st.session_state:
        st.session_state.filename = None

def main():
    st.set_page_config(
        page_title="ReperAI",
        page_icon="📚",
        layout="wide"
    )
    
    init_session()
    
    # Header
    st.title("📚 ReperAI")
    st.markdown("*Smart summarizer for research papers*")
    st.markdown("---")
    
    # Sidebar config
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key", type="password", 
                                help="Get yours at platform.openai.com")
        
        st.markdown("---")
        st.info("""
        **How it works:**
        1. Upload a research paper
        2. Ask questions about it
        3. Get contextual summaries with sources
        """)
    
    # Main tabs
    upload_tab, query_tab, insights_tab = st.tabs(["📄 Upload", "🔍 Query", "📊 Insights"])
    
    # Tab 1: Upload
    with upload_tab:
        st.header("Upload Research Paper")
        
        file = st.file_uploader(
            "Choose file (PDF, TXT, or MD)",
            type=['pdf', 'txt', 'md']
        )
        
        if file and st.button("Process"):
            with st.spinner("Processing..."):
                # Save file
                save_path = Path(app_config.data_dir) / file.name
                with open(save_path, 'wb') as f:
                    f.write(file.getbuffer())
                
                # Process
                processor = DocumentProcessor(
                    chunk_size=model_config.chunk_size,
                    chunk_overlap=model_config.chunk_overlap
                )
                chunks = processor.process_document(str(save_path))
                
                # Create embeddings
                emb_mgr = EmbeddingManager(model_config.embedding_model)
                embeddings = emb_mgr.create_embeddings(chunks)
                emb_mgr.build_index(embeddings)
                emb_mgr.chunks = chunks
                
                # Save to session
                st.session_state.embeddings = emb_mgr
                st.session_state.filename = file.name
                st.session_state.doc_ready = True
                
                st.success(f"✅ Processed {len(chunks)} chunks from {file.name}")
    
    # Tab 2: Query
    with query_tab:
        st.header("Ask Questions")
        
        if not st.session_state.doc_ready:
            st.warning("Upload a document first")
        else:
            st.info(f"📄 Current: {st.session_state.filename}")
            
            query = st.text_area(
                "Your question:",
                placeholder="e.g., What methodology did the authors use?"
            )
            
            n_chunks = st.slider("Chunks to retrieve", 3, 10, 5)
            
            if query and st.button("Get Answer"):
                if not api_key:
                    st.error("Add your API key in sidebar")
                else:
                    with st.spinner("Thinking..."):
                        # Search
                        results = st.session_state.embeddings.search_similar(
                            query, k=n_chunks
                        )
                        
                        # Summarize
                        summarizer = ContextualSummarizer(api_key, model_config.llm_model)
                        output = summarizer.generate_summary(query, results)
                        
                        # Display
                        st.markdown("### Answer")
                        st.markdown(output['summary'])
                        
                        with st.expander("Source Chunks"):
                            for i, r in enumerate(results):
                                st.markdown(f"**Chunk {i+1}** (distance: {r['score']:.3f})")
                                st.text(r['chunk']['content'][:300] + "...")
                                st.markdown("---")
    
    # Tab 3: Insights
    with insights_tab:
        st.header("Key Insights")
        
        if not st.session_state.doc_ready:
            st.warning("Upload a document first")
        else:
            if st.button("Extract Main Points") and api_key:
                with st.spinner("Extracting..."):
                    # Get some text
                    chunks = st.session_state.embeddings.chunks[:5]
                    text = " ".join([c['content'] for c in chunks])
                    
                    summarizer = ContextualSummarizer(api_key)
                    points = summarizer.extract_key_points(text)
                    
                    st.markdown("### Main Points")
                    for point in points:
                        st.markdown(f"- {point}")

if __name__ == "__main__":
    main()
