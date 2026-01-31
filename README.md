<div align="center">

<!-- Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,24&height=200&section=header&text=Nexus%20AI&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Enterprise-Grade%20Conversational%20AI%20Engine&descAlignY=55&descSize=20"/>

<!-- Animated Logo -->
```
    ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗     █████╗ ██╗
    ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝    ██╔══██╗██║
    ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗    ███████║██║
    ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║    ██╔══██║██║
    ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║    ██║  ██║██║
    ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚═╝
```

**✨ Codename: Synapse ✨**

<!-- Badges Row 1 - Core Tech -->
<p>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers"/></a>
  <a href="#"><img src="https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/></a>
</p>

<!-- Badges Row 2 - Features -->
<p>
  <a href="#"><img src="https://img.shields.io/badge/NLU-Transformer--Based-FF6B6B?style=for-the-badge&logo=openai&logoColor=white" alt="NLU"/></a>
  <a href="#"><img src="https://img.shields.io/badge/API-REST%20%2B%20WebSocket-9C27B0?style=for-the-badge&logo=socketdotio&logoColor=white" alt="API"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Docker-Production%20Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge" alt="MIT License"/></a>
</p>

<!-- Stats Badges -->
<p>
  <a href="#"><img src="https://img.shields.io/badge/Lines%20of%20Code-5,000+-blue?style=flat-square" alt="LOC"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Source%20Files-25+-orange?style=flat-square" alt="Files"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Intents-26-green?style=flat-square" alt="Intents"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Tests-42%20Passing-purple?style=flat-square" alt="Tests"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Build-Passing-brightgreen?style=flat-square" alt="Build"/></a>
</p>

<br/>

*A production-grade conversational AI platform demonstrating mastery of*<br/>
*modern NLP, transformer architectures, and enterprise software engineering.*

<br/>

[**🚀 Quick Start**](#-quick-start) · [**🏗️ Architecture**](#%EF%B8%8F-architecture) · [**✨ Features**](#-features) · [**📖 API Reference**](#-api-reference) · [**🛠️ Technical**](#%EF%B8%8F-technical-deep-dive)

<br/>

---

</div>

## 🎯 Overview

**Nexus AI** is a complete, enterprise-grade conversational AI engine built from scratch using modern NLP techniques and transformer-based models. It features real-time intent classification, named entity recognition, sentiment analysis, and intelligent multi-turn dialogue management. This project showcases:

<table>
<tr>
<td>

### 🧠 NLP Expertise
- Transformer-based embeddings
- Semantic similarity matching
- Named Entity Recognition
- Multi-class sentiment analysis

</td>
<td>

### 💬 Dialogue Systems
- Multi-turn context tracking
- Entity memory persistence
- Handler priority system
- Dynamic response generation

</td>
<td>

### ⚡ Production Engineering
- Async FastAPI backend
- WebSocket real-time chat
- Docker containerization
- CI/CD with GitHub Actions

</td>
</tr>
</table>

> *"The best way to understand conversational AI is to build one from the ground up."*

<br/>

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 🧠 Intent Classification
```
Sentence Transformers → Semantic Matching
```
- **26 custom intents** with pattern matching
- Cosine similarity scoring
- Confidence thresholding
- Fallback handling for unknown inputs

</td>
<td width="50%" valign="top">

### 🏷️ Entity Extraction
```
BERT-NER + Regex → Named Entities
```
- Transformer-based NER model
- Pattern matching (email, phone, URL)
- Entity normalization
- Multi-entity support per message

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 💭 Sentiment Analysis
```
RoBERTa → 5-Class Classification
```
- **Very Positive / Positive / Neutral**
- **Negative / Very Negative**
- Confidence scores per class
- Conversation sentiment tracking

</td>
<td width="50%" valign="top">

### 🔗 Contextual Embeddings
```
MPNet → 768-dim Vectors
```
- Semantic text representations
- Embedding cache for performance
- Batch processing support
- Similarity computation

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 💬 Dialogue Management
```
State Machine → Multi-turn Context
```
- **Session persistence** across turns
- Entity memory throughout conversation
- Topic continuity tracking
- Dynamic suggestion generation

</td>
<td width="50%" valign="top">

### ⚡ Real-Time API
```
FastAPI → REST + WebSocket
```
- Async request handling
- WebSocket bidirectional chat
- Rate limiting & CORS
- Health monitoring endpoints

</td>
</tr>
</table>

<br/>

## 🏗️ Architecture

```
                    ╔═══════════════════════════════════════════════════════════════════╗
                    ║                    NEXUS AI SYSTEM ARCHITECTURE                    ║
                    ╚═══════════════════════════════════════════════════════════════════╝
                                                    │
                                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                                    CLIENT LAYER                                      │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
    │  │   Web UI    │    │  REST API   │    │  WebSocket  │    │    CLI      │          │
    │  │  (HTML/JS)  │    │   Client    │    │   Client    │    │  Interface  │          │
    │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘          │
    └─────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    ▼                               ▼                               ▼
    ┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
    │      API LAYER        │       │    SESSION MANAGER    │       │    MIDDLEWARE         │
    │  ┌─────────────────┐  │       │  ┌─────────────────┐  │       │  ┌─────────────────┐  │
    │  │  FastAPI App    │  │       │  │  Context Track  │  │       │  │   Rate Limit    │  │
    │  │  REST Routes    │  │       │  │  Entity Memory  │  │       │  │   CORS Config   │  │
    │  │  WS Handlers    │  │       │  │  History Mgmt   │  │       │  │   Auth Layer    │  │
    │  └─────────────────┘  │       │  └─────────────────┘  │       │  └─────────────────┘  │
    └───────────────────────┘       └───────────────────────┘       └───────────────────────┘
                                                    │
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                              CONVERSATION ENGINE                                     │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
    │  │  NLU        │    │  Dialogue   │    │  Response   │    │  Suggestion │          │
    │  │  Pipeline   │───▶│  Manager    │───▶│  Generator  │───▶│  Engine     │          │
    │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘          │
    └─────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    ▼                               ▼                               ▼
    ┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
    │   INTENT CLASSIFIER   │       │   ENTITY EXTRACTOR    │       │  SENTIMENT ANALYZER   │
    │  ┌─────────────────┐  │       │  ┌─────────────────┐  │       │  ┌─────────────────┐  │
    │  │ Sentence-BERT   │  │       │  │  BERT-NER       │  │       │  │  RoBERTa        │  │
    │  │ all-MiniLM-L6   │  │       │  │  Regex Patterns │  │       │  │  5-class        │  │
    │  │ Cosine Sim      │  │       │  │  Normalization  │  │       │  │  Confidence     │  │
    │  └─────────────────┘  │       │  └─────────────────┘  │       │  └─────────────────┘  │
    └───────────────────────┘       └───────────────────────┘       └───────────────────────┘
                                                    │
                                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                            TRANSFORMER MODELS (HuggingFace)                          │
    │       Sentence-Transformers │ BERT-NER │ RoBERTa-Sentiment │ MPNet-Embeddings        │
    └─────────────────────────────────────────────────────────────────────────────────────┘
```

<br/>

## 📁 Project Structure

```
Nexus-Conversational-AI/
│
├── 📂 src/nexus/                        # Main Python Package
│   ├── __init__.py                      # Package initialization & version
│   ├── config.py                        # Pydantic configuration management
│   ├── cli.py                           # Typer CLI application
│   │
│   ├── 📂 core/                         # Core Engine Components
│   │   ├── engine.py                    # Main ConversationEngine orchestrator
│   │   ├── session.py                   # Session & context management
│   │   └── response.py                  # Response models & types
│   │
│   ├── 📂 nlu/                          # Natural Language Understanding
│   │   ├── classifier.py                # Intent classification (Sentence-BERT)
│   │   ├── extractor.py                 # Entity extraction (BERT-NER + Regex)
│   │   ├── sentiment.py                 # Sentiment analysis (RoBERTa)
│   │   └── embeddings.py                # Text embeddings (MPNet)
│   │
│   ├── 📂 dialogue/                     # Dialogue Management
│   │   ├── manager.py                   # Dialogue orchestration
│   │   ├── handlers.py                  # Intent handlers (Greeting, Help, etc.)
│   │   └── state.py                     # Dialogue state tracking
│   │
│   ├── 📂 data/                         # Data & Intent Definitions
│   │   └── intents.py                   # 26 intents with patterns & responses
│   │
│   ├── 📂 api/                          # FastAPI Backend
│   │   ├── app.py                       # Application factory & lifespan
│   │   ├── routes.py                    # REST API endpoints
│   │   └── websocket.py                 # WebSocket chat handler
│   │
│   └── 📂 training/                     # Model Training Pipeline
│       ├── trainer.py                   # PyTorch training loop
│       ├── dataset.py                   # Dataset & data augmentation
│       └── metrics.py                   # Training metrics & visualization
│
├── 📂 tests/                            # Test Suite (42 tests)
│   ├── conftest.py                      # Pytest fixtures & configuration
│   ├── test_nlu.py                      # NLU component tests
│   ├── test_api.py                      # API endpoint tests
│   └── test_engine.py                   # Engine & session tests
│
├── 📂 frontend/                         # Web Interface
│   └── index.html                       # Modern chat UI (HTML/CSS/JS)
│
├── 📂 .github/                          # CI/CD Configuration
│   ├── workflows/ci.yml                 # GitHub Actions pipeline
│   └── dependabot.yml                   # Automated dependency updates
│
├── 📂 nginx/                            # Production Proxy
│   └── nginx.conf                       # Nginx reverse proxy config
│
├── 📂 monitoring/                       # Observability
│   └── prometheus.yml                   # Metrics collection config
│
├── 📄 pyproject.toml                    # Modern Python packaging (PEP 517)
├── 📄 Dockerfile                        # Multi-stage production build
├── 📄 docker-compose.yml                # Full stack orchestration
├── 📄 CONTRIBUTING.md                   # Contribution guidelines
├── 📄 LICENSE                           # MIT License
└── 📄 README.md                         # You are here!
```

<br/>

## 🚀 Quick Start

### Prerequisites

```bash
# Required
python --version        # Python 3.10 or higher
pip --version           # pip package manager

# Optional (for production)
docker --version        # Docker for containerization
docker-compose --version
```

**On Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip
```

**On macOS:**
```bash
brew install python@3.10
```

**On Windows:**
```bash
# Download from python.org or use winget
winget install Python.Python.3.10
```

### Installation & Run

```bash
# Clone the repository
git clone https://github.com/raghavshuklaofficial/Conversation_AI_Chatbot.git
cd Conversation_AI_Chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the package
pip install -e .

# Run interactive chat
nexus chat

# Or start API server
nexus serve --port 8000
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# With monitoring stack
docker-compose --profile monitoring up -d
```

### CLI Commands

```
╔══════════════════════════════════════════════════════════════════╗
║            Nexus AI v2.0.0 - CLI Reference                       ║
╠══════════════════════════════════════════════════════════════════╣
║  nexus chat      │ Start interactive chat session                ║
║  nexus serve     │ Start REST/WebSocket API server               ║
║  nexus train     │ Train or fine-tune the model                  ║
║  nexus analyze   │ Analyze text for intent/sentiment             ║
║  nexus info      │ Display system information                    ║
╚══════════════════════════════════════════════════════════════════╝
```

<br/>

## 📖 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Send message, get AI response |
| `POST` | `/api/v1/analyze` | Analyze text (intent, entities, sentiment) |
| `GET` | `/api/v1/sessions/{id}` | Get session information |
| `DELETE` | `/api/v1/sessions/{id}` | End a conversation session |
| `GET` | `/api/v1/intents` | List all available intents |
| `GET` | `/api/v1/stats` | Get system statistics |
| `GET` | `/health` | Health check endpoint |

### Chat Request Example

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}'
```

### Response Format

```json
{
  "id": "msg_7f3a9b2c",
  "text": "Hello! I'm doing great, thank you! How can I assist you today?",
  "type": "standard",
  "session_id": "sess_4e8d1a5f",
  "suggestions": ["Tell me a joke", "What can you do?", "Help"],
  "sentiment": "positive",
  "confidence": 0.94,
  "intent": "greeting",
  "entities": [],
  "processing_time_ms": 45.2
}
```

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'message',
    payload: { text: 'Hello!' }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Response:', data.payload.text);
};
```

<br/>

## 🛠️ Technical Deep Dive

<details>
<summary><b>🧠 Intent Classification with Sentence Transformers</b></summary>

Uses semantic similarity for robust intent matching:

```python
from sentence_transformers import SentenceTransformer

class IntentClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._intent_embeddings: dict[str, np.ndarray] = {}
    
    async def classify(self, text: str) -> IntentMatch:
        # Encode input text
        text_embedding = self._model.encode(text)
        
        # Find most similar intent
        best_intent, best_score = None, 0.0
        for intent_name, intent_emb in self._intent_embeddings.items():
            similarity = cosine_similarity(text_embedding, intent_emb)
            if similarity > best_score:
                best_score = similarity
                best_intent = intent_name
        
        return IntentMatch(
            name=best_intent,
            confidence=best_score,
            is_fallback=best_score < self._threshold
        )
```

**Features:**
- Pre-computed intent embeddings for speed
- Cosine similarity scoring
- Configurable confidence threshold
- Graceful fallback handling
</details>

<details>
<summary><b>🏷️ Named Entity Recognition</b></summary>

Combines transformer NER with regex patterns:

```python
class EntityExtractor:
    # Regex patterns for common entity types
    _patterns = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "PHONE": r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b',
        "URL": r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*',
        "DATE": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    }
    
    async def extract(self, text: str) -> list[Entity]:
        entities = []
        
        # Transformer-based NER
        ner_results = self._ner_pipeline(text)
        for result in ner_results:
            entities.append(Entity(
                text=result['word'],
                type=result['entity_group'],
                confidence=result['score']
            ))
        
        # Regex pattern matching
        for entity_type, pattern in self._patterns.items():
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    text=match.group(),
                    type=entity_type,
                    confidence=0.99
                ))
        
        return entities
```
</details>

<details>
<summary><b>💭 Sentiment Analysis Pipeline</b></summary>

5-class sentiment classification using RoBERTa:

```python
class SentimentAnalyzer:
    def __init__(self):
        self._pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None  # Return all class scores
        )
    
    async def analyze(self, text: str) -> tuple[Sentiment, float]:
        results = self._pipeline(text)[0]
        
        # Map model output to sentiment enum
        label_map = {
            "positive": Sentiment.POSITIVE,
            "negative": Sentiment.NEGATIVE,
            "neutral": Sentiment.NEUTRAL,
        }
        
        best = max(results, key=lambda x: x['score'])
        return label_map[best['label']], best['score']
```
</details>

<details>
<summary><b>💬 Multi-Turn Dialogue Management</b></summary>

Stateful conversation tracking with entity memory:

```python
@dataclass
class ConversationSession:
    id: UUID
    context: ConversationContext
    entity_memory: EntityMemory
    history: list[Turn]
    
    def add_turn(self, user_input: str, response: Response):
        self.history.append(Turn(
            user_input=user_input,
            response=response,
            timestamp=datetime.utcnow()
        ))
        
        # Update context
        self.context.last_intent = response.metadata.detected_intent
        self.context.sentiment_history.append(response.metadata.sentiment)
        
        # Store extracted entities
        for entity in response.metadata.entities:
            self.entity_memory.add(entity)

class EntityMemory:
    """Persists entities across conversation turns."""
    
    def get_by_type(self, entity_type: str) -> list[Entity]:
        return self._entities.get(entity_type, [])
    
    def get_latest(self, entity_type: str) -> Entity | None:
        entities = self.get_by_type(entity_type)
        return entities[-1] if entities else None
```
</details>

<details>
<summary><b>⚡ Async FastAPI Backend</b></summary>

High-performance async API with WebSocket support:

```python
from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize conversation engine
    engine = ConversationEngine()
    await engine.initialize()
    app.state.engine = engine
    
    yield
    
    # Shutdown: Cleanup resources
    await engine.shutdown()

app = FastAPI(
    title="Nexus Conversational AI",
    lifespan=lifespan,
)

@app.post("/api/v1/chat")
async def chat(request: MessageRequest) -> MessageResponse:
    engine = request.app.state.engine
    response = await engine.process(
        text=request.text,
        session_id=request.session_id
    )
    return response.to_api_response()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    session_id = uuid4()
    
    while True:
        data = await websocket.receive_json()
        response = await engine.process(data['text'], session_id)
        await websocket.send_json(response.to_dict())
```
</details>

<br/>

## 📊 Technical Specifications

| Component | Specification |
|-----------|--------------|
| **Language** | Python 3.10+ with full type hints |
| **ML Framework** | PyTorch 2.1+ / Transformers 4.36+ |
| **Intent Model** | sentence-transformers/all-MiniLM-L6-v2 |
| **NER Model** | dslim/bert-base-NER |
| **Sentiment Model** | cardiffnlp/twitter-roberta-base-sentiment |
| **Embedding Model** | sentence-transformers/all-mpnet-base-v2 |
| **API Framework** | FastAPI 0.109+ with Uvicorn |
| **Response Time** | ~45ms average (REST API) |
| **Intents Supported** | 26 built-in, extensible |
| **Test Coverage** | 42 tests passing |

<br/>

## 🎓 Skills Demonstrated

| Category | Technologies & Concepts |
|----------|------------------------|
| **Languages** | Python 3.10+, JavaScript, HTML/CSS |
| **ML/NLP** | Transformers, Sentence-BERT, NER, Sentiment Analysis |
| **Frameworks** | PyTorch, HuggingFace, FastAPI, Pydantic |
| **Architecture** | Async/Await, WebSockets, REST API Design |
| **DevOps** | Docker, GitHub Actions, Nginx, Prometheus |
| **Testing** | Pytest, AsyncIO Testing, Mocking |
| **Patterns** | Dependency Injection, Factory Pattern, State Machine |

<br/>

## 🗺️ Future Roadmap

- [ ] **Fine-tuned Models** — Custom intent classifier trained on domain data
- [ ] **Multi-language** — Support for Spanish, French, German, Hindi
- [ ] **Voice Interface** — Speech-to-text and text-to-speech integration
- [ ] **RAG Pipeline** — Retrieval-augmented generation for knowledge base
- [ ] **LLM Integration** — Optional GPT/Claude fallback for complex queries
- [ ] **Analytics Dashboard** — Real-time conversation metrics UI
- [ ] **Kubernetes Helm** — Production Kubernetes deployment charts

<br/>

## 👨‍💻 Author

<div align="center">

**Raghav Shukla**

*AI/ML Engineer | NLP Enthusiast | Full-Stack Developer*

[![GitHub](https://img.shields.io/badge/GitHub-raghavshuklaofficial-181717?style=for-the-badge&logo=github)](https://github.com/raghavshuklaofficial)

</div>

<br/>

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Raghav Shukla

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

<br/>

---

<div align="center">

<br/>

### ⭐ If you found this project impressive, consider giving it a star!

<br/>

*"Language is the interface between human thought and machine intelligence."*

<br/>

**Built with 🧠 and ☕ by Raghav Shukla**

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,24&height=120&section=footer"/>

</div>
