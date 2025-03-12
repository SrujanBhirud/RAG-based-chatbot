# RAG-based-chatbot
The chatbot system is designed to provide accurate responses to literature-related queries by integrating Mistral 7B with a Retrieval-Augmented Generation (RAG) pipeline. The backend is built using FastAPI, and the frontend is implemented in Streamlit. This can easily be changed to also include documents.

## Design Philosophy & Approach

High accuracy, scalability, speed, and cost efficiency were the core principles guiding the design. The first step was selecting key technologies (for example, the model, framework, and vector database) to ensure an optimal balance between performance and resource usage, ensuring compatibility amongst themselves.  Initially, the system was implemented with only the core functionalities working. Over time, refinements were made, continuously optimizing each section of the code. 

## Handling Scale and Performance Optimization

The system is built to handle increasing data volumes efficiently. FAISS ensures fast similarity searches, even with a large database of book embeddings. PCA-based dimensionality reduction and L2 normalization optimizes retrieval without significant accuracy loss. The backend is built using FastAPI, which supports asynchronous request handling, ensuring smooth performance even under heavy load. 
The framework, vllm + langchain, allows for efficient inference, optimized memory management and faster token generation. It enables handling larger batch sizes and reduces latency when processing multiple queries. Since Mistral 7B is a sizable model, vLLM ensures that the inference process remains smooth, preventing excessive GPU memory consumption and enabling efficient parallelization. Both FAISS and framework support vertical and horizontal scaling (explained in more detail in Documentation).
Token length management and efficient query preprocessing further contribute to scalability. The model is hosted in an environment optimized for inference speed, with parameters like temperature=0.7 and top_p=0.9 balancing response diversity and coherence.

## Key Technical Decisions

**Model Selection**: 
I debated between API-based model and Open Source self-hosting model. I went with a self-hosting model since it is cheaper when scaled and provides more customization options. Furthermore, it guarantees privacy.
Using Mistral 7B was a strategic choice, balancing model performance, efficiency, and cost-effectiveness. It is a smaller model but performs on par with larger models like LLaMa 13B, while requiring fewer resources. Furthermore it integrates well with FAISS-based retrieval.

**Vector Database**:
FAISS was selected as the vector database due to its scalability, speed, and efficiency in similarity searches. Given that the chatbot needs to retrieve relevant book/document passages quickly, FAISS provides an optimized way to search through high-dimensional embeddings with minimal latency. The only drawback is that it cannot store structured data. Since only books/documents are to be stored, FAISS can be used.
The indexing method used is Hierarchical Navigable Small World (HNSW), which significantly improves search efficiency compared to brute-force methods. HNSW allows for fast approximate nearest neighbor (ANN) searches, reducing query time while maintaining high retrieval accuracy. Additionally, L2 normalization ensures that embeddings remain consistent, enhancing the quality of similarity matching.
"sentence-transformers/all-MiniLM-L6-v2" is a lightweight transformer-based model, making it computationally feasible for large-scale embedding. It's embeddings work well with FAISS.
To further optimize performance, Principal Component Analysis (PCA) is applied for dimensionality reduction(384->256), reducing storage requirements while preserving important information. This combination of HNSW, PCA, and L2 normalization ensures that the system remains fast, memory-efficient, and scalable as the document database grows.

**Framework**:
The combination of vLLM and LangChain was chosen to balance scalability, performance, and modularity.

Why Not Just LangChain?
LangChain is great for prompt engineering, retrieval orchestration, and memory handling, but it lacks deep optimization for efficient model inference at scale. Running large LLMs directly through LangChain can be resource-intensive and slow, especially with high-concurrency demands.

Why Not Just vLLM?
vLLM is optimized for fast, efficient model inference with techniques like PagedAttention, but it does not handle retrieval, prompt structuring, or memory management natively. Without LangChain, implementing context-aware chat and retrieval-augmented generation (RAG) would require a lot of custom coding.

Why vLLM + LangChain?
By combining vLLM for optimized inference with LangChain for RAG and prompt management, the system benefits from:
- Fast inference with efficient memory handling (vLLM).
- Seamless RAG integration and structured prompt handling (LangChain).
- Scalability—vLLM allows serving multiple concurrent requests efficiently.
- Modularity—LangChain provides flexibility to swap models, retrieval methods, and memory components without affecting the core logic.
This hybrid approach ensures the chatbot is both high-performing and scalable without unnecessary overhead.

**Frontend and Backend**:
For deployment, FastAPI was selected for its speed and ability to handle concurrent requests, while Streamlit was used for its simplicity in frontend implementation. 
