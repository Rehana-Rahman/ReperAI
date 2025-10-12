# ReperAI - Smart Contextual Summarizer

> **REsearch papER AI** - An intelligent document analysis tool that uses Retrieval-Augmented Generation (RAG) to provide contextual summaries of research papers.

## Overview

ReperAI transforms how you interact with research papers by combining semantic search with advanced language models. Instead of reading entire papers to find specific information, simply ask questions and get precise, context-aware summaries backed by relevant document sections.

## Key Features

- **📄 Multi-Format Support**: Process PDF, TXT, and Markdown files
- **🔍 Semantic Search**: Find relevant content using vector embeddings
- **🤖 Contextual Summarization**: Generate accurate summaries using GPT-3.5
- **📊 Key Insights Extraction**: Automatically identify main points
- **🎯 Citation Tracking**: References source sections in summaries
- **⚡ Fast Retrieval**: FAISS-powered vector search for instant results

## Architecture

```
ReperAI/
├── config/
│   └── settings.py          # Configuration management
├── data/
│   ├── uploads/             # Uploaded documents
│   └── indices/             # Vector indices
├── models/
│   ├── embeddings.py        # Embedding generation & search
│   └── summarizer.py        # RAG-based summarization
├── utils/
│   └── document_processor.py # Document loading & chunking
├── ui/
│   └── app.py               # Streamlit interface
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Rehana-Rahman/reperai.git
cd reperai
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run ui/app.py
```

5. **Access the interface**
Open your browser to `http://localhost:8501`

## Usage

### 1. Upload Document
- Navigate to the **"Upload Document"** tab
- Select a research paper (PDF, TXT, or MD format)
- Click **"Process Document"** to chunk and index the content

### 2. Query & Summarize
- Enter your OpenAI API key in the sidebar
- Type your question or topic of interest
- Adjust the number of relevant sections to retrieve
- Click **"Generate Summary"** to get contextual answers

### 3. Extract Key Insights
- Switch to the **"Key Insights"** tab
- Click **"Extract Key Points"** to get main takeaways
- View automatically generated bullet points

## Technical Details

### Document Processing Pipeline

1. **Loading**: Extracts text from PDFs using PyPDF2 or reads plain text files
2. **Cleaning**: Normalizes whitespace and removes formatting artifacts
3. **Chunking**: Splits documents into overlapping segments (default: 1000 words, 200-word overlap)
4. **Embedding**: Generates 384-dimensional vectors using sentence-transformers
5. **Indexing**: Builds FAISS index for efficient similarity search

### Retrieval-Augmented Generation

```python
Query → Embedding → Vector Search → Top-K Chunks → LLM Context → Summary
```

The system:
1. Converts queries to embeddings
2. Finds most similar document chunks using L2 distance
3. Constructs prompts with retrieved context
4. Generates summaries using GPT-3.5-turbo
5. Returns results with source citations

### Configuration Options

Edit `config/settings.py` to customize:

```python
ModelConfig:
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model = "gpt-3.5-turbo"
    temperature = 0.3  # Lower = more focused
    chunk_size = 1000  # Words per chunk
    chunk_overlap = 200  # Overlap for context
```

## Performance

- **Indexing Speed**: ~500 pages/minute
- **Query Time**: <2 seconds (retrieval + generation)
- **Memory Usage**: ~100MB per 100-page document
- **Supported File Size**: Up to 10MB (configurable)

## Future Enhancements

- [ ] Multi-document cross-referencing
- [ ] Batch processing for paper collections
- [ ] Export summaries to PDF/Markdown
- [ ] Citation graph visualization
- [ ] Custom prompt templates
- [ ] Support for GPT-4 and Claude models
- [ ] Comparative analysis between papers
- [ ] Automatic table/figure extraction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://github.com/hwchase17/langchain) concepts
- Powered by [Sentence Transformers](https://www.sbert.net/)
- Vector search via [FAISS](https://github.com/facebookresearch/faiss)
- UI framework: [Streamlit](https://streamlit.io/)

## Contact

Rehana Rahman - [LinkedIn](https://www.linkedin.com/in/rehana-rahman-4b2bb933b/)

Project Link: [ReperAI](https://github.com/Rehana-Rahman/ReperAI)

---

Made with ❤️ for researchers who value their time
