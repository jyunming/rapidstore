# TurboQuantDB

A high-performance, embedded vector database engine written in Rust, powered by Google DeepMind's **TurboQuant** algorithm. It provides near-optimal, data-oblivious vector quantization for ultra-fast, memory-efficient Retrieval-Augmented Generation (RAG) applications in Python.

Unlike traditional vector databases that require expensive training phases (like FAISS) to build codebooks, TurboQuantDB uses provably near-optimal randomized rotations and scalar quantization to compress vectors down to 2-bit or 4-bit representations *on the fly*, with zero index training.

## Key Features

*   **Zero Training Time:** Insert vectors immediately. No `train()` step required.
*   **Extreme Compression:** Quantizes float32 embeddings down to 2 or 4 bits per coordinate, saving 8x to 16x in RAM overhead.
*   **Unbiased Inner Products:** Uses Quantized Johnson-Lindenstrauss (QJL) transforms to guarantee unbiased cosine similarity/inner product estimation.
*   **Python Native:** Built with `PyO3` and `maturin`, giving you the speed of Rust directly inside your Python REPL.

---

## Installation Guide

### Prerequisites (Crucial for Windows Users)

Because TurboQuantDB is a Rust extension built tightly into Python's C-API, your system **must have a C++ build chain installed**.

#### **Windows**
You must install the **Visual Studio Build Tools**:
1. Download the [Visual Studio Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe).
2. Run the installer and select the **"Desktop development with C++"** workload.
3. Ensure the Windows 10/11 SDK and MSVC v143 build tools are checked.
4. Restart your terminal.

#### **macOS / Linux**
You typically already have the necessary compiler tools. If not:
*   **macOS:** Run `xcode-select --install`
*   **Ubuntu/Debian:** Run `sudo apt-get install build-essential`

### Building the Package

1. Ensure you have [Rust installed](https://rustup.rs/).
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install `maturin`, the Rust-to-Python build system:
   ```bash
   pip install maturin
   ```
4. Build and install the TurboQuantDB module into your environment:
   ```bash
   maturin develop --release
   ```

---

## Usage: RAG Integration

TurboQuantDB comes with a LangChain-style retriever interface out of the box.

```python
import numpy as np
from turboquantdb.rag import TurboQuantRetriever

# 1. Initialize the Database (e.g., d=1536 for OpenAI embeddings, 4-bit compression)
retriever = TurboQuantRetriever(dimension=1536, bits=4, seed=42)

# 2. Prepare your data and embeddings
texts = [
    "TurboQuant is a revolutionary data-oblivious quantization algorithm.",
    "Rust provides memory safety without a garbage collector.",
    "Vector databases are essential for RAG pipelines."
]

# Generate embeddings (mocked here, use OpenAI/HuggingFace in reality)
embeddings = [np.random.randn(1536).tolist() for _ in texts]
metadatas = [{"source": "paper"}, {"source": "manual"}, {"source": "blog"}]

# 3. Ingest data instantly (No training required!)
retriever.add_texts(texts=texts, embeddings=embeddings, metadatas=metadatas)

# 4. Search for the most relevant context
query_emb = np.random.randn(1536).tolist()
results = retriever.similarity_search(query_embedding=query_emb, k=2)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text']}")
    print(f"Metadata: {result['metadata']}\n")
```

## Architecture & Roadmap

TurboQuantDB is currently architected as an **embedded database** (similar to LanceDB or DuckDB). It runs in the same process as your Python application and performs highly optimized, SIMD-accelerated brute-force scans over deeply compressed vector representations.

While it is phenomenally fast for datasets up to 1 million vectors due to its tiny memory footprint, it does not currently feature an Approximate Nearest Neighbor (ANN) graph structure like HNSW.

*Currently implemented algorithms based on arXiv:2504.19874 ("TurboQuant: Near-Optimal Data-Oblivious Vector Quantization").*
