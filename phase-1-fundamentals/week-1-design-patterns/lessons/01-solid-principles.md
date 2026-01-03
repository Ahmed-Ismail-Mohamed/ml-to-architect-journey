# SOLID Principles for ML Engineers

## 🎯 Why SOLID Matters for ML Systems

You've built ML models. But have you struggled with:
- 😣 Changing LLM providers takes days of refactoring?
- 😣 Adding a new embedding model breaks existing code?
- 😣 Testing requires calling actual APIs?
- 😣 One class does data loading, preprocessing, inference, and logging?

**SOLID principles solve these problems.**

---

## 1️⃣ Single Responsibility Principle (SRP)

> **"A class should have one, and only one, reason to change."**

### ❌ Bad Example (From Your Experience)

```python
# This class does EVERYTHING - violation!
class ChurnPredictor:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.clf = pickle.load(open('clf.pkl', 'rb'))
        self.encoder = pickle.load(open('encoder.pkl', 'rb'))
    
    def extract_features(self, question):
        # Extracts features using LLM
        prompt = "Extract credit_score, country..."
        return self.llm.invoke(prompt)
    
    def preprocess(self, df):
        # Encodes data
        return self.encoder.transform(df)
    
    def predict(self, data):
        # Makes prediction
        return self.clf.predict(data)
    
    def log_to_database(self, result):
        # Logs to DB
        pass
```

**Problems:**
- Changes to LLM affect the whole class
- Can't test prediction without LLM
- Can't reuse preprocessing elsewhere
- 4 reasons to change this class!

### ✅ Good Example - Separated Responsibilities

```python
# Each class has ONE job
class FeatureExtractor:
    """ONLY extracts features from text using LLM"""
    def __init__(self, llm):
        self.llm = llm
    
    def extract(self, question: str) -> dict:
        prompt = self._build_prompt(question)
        return self.llm.invoke(prompt)

class DataPreprocessor:
    """ONLY preprocesses data"""
    def __init__(self, encoder):
        self.encoder = encoder
    
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        return self.encoder.transform(df)

class ChurnClassifier:
    """ONLY makes predictions"""
    def __init__(self, model):
        self.model = model
    
    def predict(self, features: np.ndarray) -> int:
        return self.model.predict(features)

class PredictionLogger:
    """ONLY logs predictions"""
    def __init__(self, db_connection):
        self.db = db_connection
    
    def log(self, prediction, metadata):
        self.db.insert(prediction, metadata)
```

**Benefits:**
✅ Change LLM? Only touch `FeatureExtractor`  
✅ Test prediction? Mock the model, no LLM needed  
✅ Reuse preprocessing? Import `DataPreprocessor`  
✅ Each class has 1 reason to change

---

## 2️⃣ Open/Closed Principle (OCP)

> **"Software entities should be open for extension, but closed for modification."**

Add new behavior WITHOUT changing existing code.

### ❌ Bad Example

```python
class TextChunker:
    def chunk(self, text: str, method: str):
        if method == "fixed":
            return self._fixed_size_chunking(text)
        elif method == "semantic":
            return self._semantic_chunking(text)
        elif method == "recursive":
            return self._recursive_chunking(text)
        # Adding new method? Modify this class! ❌
```

**Problem**: Every new chunking method requires modifying this class.

### ✅ Good Example - Open for Extension

```python
from abc import ABC, abstractmethod

class ChunkingStrategy(ABC):
    """Abstract base - closed for modification"""
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass

# Extend by adding new classes - open for extension
class FixedSizeChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
    
    def chunk(self, text: str) -> List[str]:
        return [text[i:i+self.chunk_size] 
                for i in range(0, len(text), self.chunk_size)]

class SemanticChunking(ChunkingStrategy):
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def chunk(self, text: str) -> List[str]:
        # Semantic chunking logic
        pass

class RecursiveChunking(ChunkingStrategy):
    def chunk(self, text: str) -> List[str]:
        # Recursive logic
        pass

# TextChunker never changes!
class TextChunker:
    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy
    
    def chunk(self, text: str) -> List[str]:
        return self.strategy.chunk(text)

# Usage - add new strategies without touching TextChunker
chunker = TextChunker(SemanticChunking(embeddings))
chunks = chunker.chunk(document)

# New requirement? Just add a new class!
class AgenticChunking(ChunkingStrategy):
    def chunk(self, text: str) -> List[str]:
        # New logic here
        pass
```

**Benefits:**
✅ Add new chunking methods without modifying existing code  
✅ Test each strategy independently  
✅ Switch strategies at runtime

---

## 3️⃣ Liskov Substitution Principle (LSP)

> **"Subtypes must be substitutable for their base types."**

If class B inherits from class A, you should be able to replace A with B without breaking the program.

### ❌ Bad Example

```python
class LLMProvider:
    def generate(self, prompt: str) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        return openai.ChatCompletion.create(...)

class LocalLlamaProvider(LLMProvider):
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        # Different signature! Violates LSP ❌
        return llama.generate(prompt, temp=temperature, max_tokens=max_tokens)
```

**Problem**: Can't substitute `LocalLlamaProvider` for `LLMProvider` - different interface!

### ✅ Good Example

```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt. Kwargs for provider-specific params."""
        pass

class OpenAIProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get('temperature', 0.7)
        return openai.ChatCompletion.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )

class LocalLlamaProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 512)
        return llama.generate(prompt, temp=temperature, max_tokens=max_tokens)

# Now they're interchangeable!
def get_answer(provider: LLMProvider, question: str) -> str:
    return provider.generate(question)  # Works with ANY provider

# Both work the same way
provider1 = OpenAIProvider()
provider2 = LocalLlamaProvider()
get_answer(provider1, "What is AI?")  # ✅
get_answer(provider2, "What is AI?")  # ✅
```

---

## 4️⃣ Interface Segregation Principle (ISP)

> **"No client should be forced to depend on methods it does not use."**

Many small, specific interfaces > one large, general interface.

### ❌ Bad Example

```python
class VectorDatabase(ABC):
    @abstractmethod
    def insert(self, vectors, metadata): pass
    
    @abstractmethod
    def search(self, query, top_k): pass
    
    @abstractmethod
    def update(self, id, vector): pass
    
    @abstractmethod
    def delete(self, id): pass
    
    @abstractmethod
    def create_index(self, index_type): pass
    
    @abstractmethod
    def optimize_index(self): pass  # Not all DBs support this!
    
    @abstractmethod
    def get_statistics(self): pass

# Simple in-memory DB doesn't need indexing!
class InMemoryVectorDB(VectorDatabase):
    def optimize_index(self):
        raise NotImplementedError("In-memory DB doesn't support indexing")  # ❌
```

### ✅ Good Example - Segregated Interfaces

```python
class VectorStore(ABC):
    """Core operations - all DBs must have"""
    @abstractmethod
    def insert(self, vectors, metadata): pass
    
    @abstractmethod
    def search(self, query, top_k): pass

class Updatable(ABC):
    """Optional: For DBs that support updates"""
    @abstractmethod
    def update(self, id, vector): pass
    
    @abstractmethod
    def delete(self, id): pass

class Indexable(ABC):
    """Optional: For DBs with indexing"""
    @abstractmethod
    def create_index(self, index_type): pass
    
    @abstractmethod
    def optimize_index(self): pass

class Monitorable(ABC):
    """Optional: For DBs with stats"""
    @abstractmethod
    def get_statistics(self): pass

# Implement only what you need
class InMemoryVectorDB(VectorStore):
    """Simple DB - only core operations"""
    def insert(self, vectors, metadata): ...
    def search(self, query, top_k): ...

class QdrantDB(VectorStore, Updatable, Indexable, Monitorable):
    """Full-featured DB - all operations"""
    def insert(self, vectors, metadata): ...
    def search(self, query, top_k): ...
    def update(self, id, vector): ...
    def delete(self, id): ...
    def create_index(self, index_type): ...
    def optimize_index(self): ...
    def get_statistics(self): ...
```

**Benefits:**
✅ Clients depend only on methods they use  
✅ Small, focused interfaces  
✅ Easy to implement simple versions

---

## 5️⃣ Dependency Inversion Principle (DIP)

> **"Depend on abstractions, not concretions."**

High-level modules should not depend on low-level modules. Both should depend on abstractions.

### ❌ Bad Example - Tight Coupling

```python
class RAGSystem:
    def __init__(self):
        # Depends on concrete implementations! ❌
        self.embeddings = OpenAIEmbeddings()  # Hardcoded!
        self.vector_db = PineconeDB()  # Hardcoded!
        self.llm = ChatOpenAI()  # Hardcoded!
    
    def query(self, question: str) -> str:
        # Can't test without calling real OpenAI!
        # Can't swap to Qdrant without changing code!
        embedded = self.embeddings.embed(question)
        docs = self.vector_db.search(embedded)
        return self.llm.generate(docs, question)
```

**Problems:**
- Hardcoded dependencies
- Can't test without real services
- Can't swap providers

### ✅ Good Example - Dependency Injection

```python
from abc import ABC, abstractmethod

# Abstractions (interfaces)
class Embeddings(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]: pass

class VectorDB(ABC):
    @abstractmethod
    def search(self, vector: List[float], top_k: int) -> List[str]: pass

class LLM(ABC):
    @abstractmethod
    def generate(self, context: str, question: str) -> str: pass

# High-level module depends on abstractions
class RAGSystem:
    def __init__(self, 
                 embeddings: Embeddings,  # Injected!
                 vector_db: VectorDB,  # Injected!
                 llm: LLM):  # Injected!
        self.embeddings = embeddings
        self.vector_db = vector_db
        self.llm = llm
    
    def query(self, question: str) -> str:
        embedded = self.embeddings.embed(question)
        docs = self.vector_db.search(embedded, top_k=5)
        return self.llm.generate(docs, question)

# Production
rag = RAGSystem(
    embeddings=OpenAIEmbeddings(),
    vector_db=QdrantDB(),
    llm=ChatOpenAI()
)

# Testing with mocks
rag_test = RAGSystem(
    embeddings=MockEmbeddings(),
    vector_db=MockVectorDB(),
    llm=MockLLM()
)

# Switch providers easily
rag_local = RAGSystem(
    embeddings=SentenceTransformerEmbeddings(),
    vector_db=ChromaDB(),
    llm=Llama3Local()
)
```

**Benefits:**
✅ Testable with mocks  
✅ Swap any component easily  
✅ Loose coupling  
✅ Configuration-driven

---

## 🎯 Putting It All Together

### Real-World Example: Refactoring Your Churn Predictor

**Before (Violates all SOLID principles):**
```python
class ChurnPredictor:
    def __init__(self):
        self.google_api_key = "AIza..."
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=self.google_api_key)
        self.clf = pickle.load(open('clf.pkl', 'rb'))
        self.encoder = pickle.load(open('encoder.pkl', 'rb'))
        self.scl = pickle.load(open('scl.pkl', 'rb'))
    
    def get_answer(self, question):
        # Extract features
        prompt = f"Extract features from: {question}"
        res = self.llm.invoke(prompt).content
        
        # Parse CSV
        df_sample = pd.read_csv(StringIO(res))
        
        # Encode
        encoded_data = self.encoder.transform(df_sample[['gender', 'country']])
        
        # Scale
        scaled = self.scl.transform(df_sample)
        
        # Predict
        return self.clf.predict(scaled)
```

**After (SOLID compliant):**
```python
# S - Single Responsibility
class FeatureExtractor:
    def __init__(self, llm: LLM):
        self.llm = llm
    
    def extract(self, question: str) -> pd.DataFrame:
        prompt = self._build_extraction_prompt(question)
        response = self.llm.generate(prompt)
        return self._parse_response(response)

class DataPreprocessor:
    def __init__(self, encoder, scaler):
        self.encoder = encoder
        self.scaler = scaler
    
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        encoded = self.encoder.transform(df[['gender', 'country']])
        scaled = self.scaler.transform(df)
        return np.concatenate([encoded.toarray(), scaled], axis=1)

class ChurnClassifier:
    def __init__(self, model):
        self.model = model
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

# D - Dependency Inversion
class ChurnPredictionService:
    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 preprocessor: DataPreprocessor,
                 classifier: ChurnClassifier):
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.classifier = classifier
    
    def predict(self, question: str) -> int:
        features_df = self.feature_extractor.extract(question)
        processed = self.preprocessor.preprocess(features_df)
        return self.classifier.predict(processed)

# Usage - fully configurable!
service = ChurnPredictionService(
    feature_extractor=FeatureExtractor(GeminiLLM()),
    preprocessor=DataPreprocessor(encoder, scaler),
    classifier=ChurnClassifier(model)
)

# Testing - use mocks!
test_service = ChurnPredictionService(
    feature_extractor=FeatureExtractor(MockLLM()),
    preprocessor=DataPreprocessor(mock_encoder, mock_scaler),
    classifier=ChurnClassifier(mock_model)
)
```

---

## ✅ Checklist: Is Your Code SOLID?

- [ ] **SRP**: Each class has one reason to change
- [ ] **OCP**: Can add new behavior without modifying existing code
- [ ] **LSP**: Subtypes are interchangeable with base types
- [ ] **ISP**: Interfaces are small and focused
- [ ] **DIP**: Depend on abstractions, inject dependencies

---

## 🏋️ Practice Exercise

**Refactor this code to follow SOLID:**

```python
class MLPipeline:
    def __init__(self):
        self.openai_key = "sk-..."
        self.pinecone_key = "pk-..."
        self.embeddings = OpenAIEmbeddings(api_key=self.openai_key)
        self.vector_db = Pinecone(api_key=self.pinecone_key)
        self.llm = OpenAI(api_key=self.openai_key)
    
    def process(self, document):
        # Chunk
        chunks = document.split('\n\n')
        
        # Embed
        vectors = [self.embeddings.embed(c) for c in chunks]
        
        # Store
        self.vector_db.upsert(vectors)
        
        # Log to database
        db.insert({'doc_id': document.id, 'chunks': len(chunks)})
        
        return len(chunks)
```

**Hint**: Identify violations, create abstractions, inject dependencies!

---

## 📚 Further Reading

- [Uncle Bob's SOLID Principles](https://blog.cleancoder.com/uncle-bob/2020/10/18/Solid-Relevance.html)
- [SOLID for Python Developers](https://realpython.com/solid-principles-python/)
- "Clean Architecture" by Robert C. Martin

---

## 🚀 Next Lesson

👉 [Adapter Pattern - Swappable LLM Providers](./02-adapter-pattern.md)

---

**Remember**: SOLID isn't dogma. It's a guideline. Use judgment! 🧠