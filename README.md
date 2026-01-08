# MediChat

A conversational AI assistant that helps users understand diseases, symptoms, treatments, and health precautions. Built with LangChain, Pinecone, Flask, and deployed on AWS.

> **Important**: This tool is for informational purposes only. It does not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

---

## What It Does

MediChat allows users to have natural conversations about health topics:

- **Disease lookup** — Get details about conditions, causes, and risk factors
- **Symptom checker** — Describe what you're experiencing and learn about possible conditions
- **Treatment info** — Understand available treatment options and medications
- **Prevention tips** — Learn precautions and preventive measures

---

## Tech Stack

- **Python** — Core language
- **LangChain** — LLM orchestration
- **Pinecone** — Vector database for semantic search
- **Flask** — Web framework
- **AWS** — Cloud hosting
- **OpenAI / HuggingFace** — Language models

---

## How It Works

```
User Query → Flask API → LangChain → Pinecone (retrieval) → LLM → Response
```

The app embeds medical documents into Pinecone. When a user asks a question, it retrieves relevant context and uses an LLM to generate an informed response.

---

## Getting Started

### Prerequisites

- Python 3.9+
- Pinecone account
- OpenAI API key (or another LLM provider)
- AWS account for deployment

### Installation

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/yourusername/MediChat.git
cd MediChat

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=medichat-index
FLASK_SECRET_KEY=your_secret_key
```

### Running Locally

Initialize the vector database and start the server:

```bash
python scripts/init_pinecone.py
python app.py
```

Open `http://localhost:5000` in your browser.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/chat` | POST | Send message, get response |
| `/health` | GET | Health check |

---

## Project Structure

```
MediChat/
├── app.py                 # Entry point
├── requirements.txt
├── setup.py
├── .env                   # Not committed
├── LICENSE
├── README.md
│
├── src/
│   ├── config/            # App configuration
│   ├── chains/            # LangChain prompts and chains
│   ├── vectorstore/       # Pinecone integration
│   ├── models/            # Data models
│   └── utils/             # Helper functions
│
├── static/                # CSS, JS, images
├── templates/             # HTML templates
├── data/                  # Medical knowledge documents
├── scripts/               # Setup and utility scripts
└── tests/
```

---

## Pinecone Setup

1. Sign up at [pinecone.io](https://www.pinecone.io/)
2. Create an index:
   - Dimensions: 1536 (for OpenAI embeddings)
   - Metric: Cosine
3. Copy your API key and environment to `.env`

---

## Deployment

Can be deployed on AWS using:

- **EC2** for a simple VM setup
- **Elastic Beanstalk** for managed deployment
- **Lambda + API Gateway** for serverless
- **S3** for static assets

---

## Testing

```bash
pytest
pytest --cov=src tests/
```

---

## Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/your-feature`)
3. Commit changes
4. Push and open a PR

Please follow PEP 8 and add tests for new functionality.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

## Contact

Utsav Jain  
GitHub: UtsavJain07
