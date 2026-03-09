# Nexus - Conversational AI Chatbot

A conversational AI system built with Python, using transformer models from HuggingFace for intent classification, entity extraction, and sentiment analysis. It supports multi-turn conversations with context tracking through a FastAPI backend (REST + WebSocket).

I built this project to understand how production NLP systems work end-to-end — from raw text input to intelligent responses — without relying on external LLM APIs.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

- **Intent Classification** — Uses Sentence-BERT (`all-MiniLM-L6-v2`) with cosine similarity matching against 26 predefined intents
- **Named Entity Recognition** — BERT-based NER (`dslim/bert-base-NER`) combined with regex patterns for email, phone, URLs etc.
- **Sentiment Analysis** — RoBERTa model from Cardiff NLP for 5-class sentiment detection
- **Dialogue Management** — Stateful sessions with entity memory, topic continuity, and context-aware responses
- **REST + WebSocket API** — Async FastAPI backend with live chat support
- **Web UI** — Simple chat interface with real-time WebSocket updates

## Project Structure

```
src/nexus/
├── core/           # Main engine, session management, response models
├── nlu/            # Intent classifier, entity extractor, sentiment, embeddings
├── dialogue/       # Dialogue manager, intent handlers, state tracking
├── data/           # Intent patterns and response templates (26 intents)
├── api/            # FastAPI app, REST routes, WebSocket handler
├── training/       # PyTorch training pipeline, dataset utils, metrics
├── config.py       # Pydantic-based config with env var support
└── cli.py          # CLI (chat, serve, train, analyze)

tests/              # Unit tests
frontend/           # Chat web UI
```

## Quick Start

```bash
# clone and setup
git clone https://github.com/raghavshuklaofficial/Nexus-Conversational-AI.git
cd Nexus-Conversational-AI

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .

# interactive chat mode
nexus chat

# or start the API server
nexus serve --port 8000
```

First run downloads transformer models (~500MB total) so give it a minute.

### Docker

```bash
docker-compose up -d

# with prometheus monitoring
docker-compose --profile monitoring up -d
```

## API

### Chat endpoint

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}'
```

```json
{
  "id": "msg_7f3a9b2c",
  "text": "Hello! How can I assist you today?",
  "type": "standard",
  "session_id": "sess_4e8d1a5f",
  "suggestions": ["Tell me a joke", "What can you do?"],
  "sentiment": "positive",
  "confidence": 0.94,
  "intent": "greeting",
  "entities": [],
  "processing_time_ms": 45.2
}
```

### Other endpoints

| Method | Endpoint | What it does |
|--------|----------|-------------|
| POST | `/api/v1/chat` | Send message, get response |
| POST | `/api/v1/analyze` | NLU analysis only (no response generation) |
| GET | `/api/v1/sessions/{id}` | Get session info |
| DELETE | `/api/v1/sessions/{id}` | End a session |
| GET | `/health` | Health check |

### WebSocket

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
  console.log(data.payload.text);
};
```

## How it works

1. User message goes through the NLU pipeline — intent classification, entity extraction, and sentiment analysis all run in parallel using `asyncio.gather`
2. Results get passed to the dialogue manager which picks the right handler or template
3. Response gets adapted based on conversation context (session history, sentiment trend)
4. Session state updates with the new turn for next interaction

Intent classification works by pre-computing sentence embeddings for each intent's training patterns at startup, then comparing new input via cosine similarity. Not as flexible as a full LLM but it's fast (~45ms per request) and predictable for structured conversations.

## Models

| Component | Model |
|-----------|-------|
| Intent Classification | `sentence-transformers/all-MiniLM-L6-v2` |
| Entity Extraction | `dslim/bert-base-NER` |
| Sentiment Analysis | `cardiffnlp/twitter-roberta-base-sentiment` |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` |

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Tech Stack

- Python 3.10+, PyTorch, HuggingFace Transformers
- FastAPI + Uvicorn (async), WebSockets
- Pydantic v2 for config and validation
- Docker, Nginx, Prometheus for deployment
- Pytest with async support

## TODO / Future work

- [ ] Fine-tune custom intent model on domain data
- [ ] Hindi and multilingual support
- [ ] RAG pipeline for knowledge-base queries  
- [ ] Speech interface (STT + TTS)
- [ ] Real-time analytics dashboard

## Author

**Raghav Shukla** — [GitHub](https://github.com/raghavshuklaofficial)

## License

MIT — see [LICENSE](LICENSE) for details.
