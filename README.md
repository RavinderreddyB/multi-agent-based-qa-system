Installation

Clone the repository:

git clone <your-repository-url>
cd <repository-name>

Install dependencies:

pip install -r requirements.txt
Environment Variables

Create a .env file:

HUGGINGFACEHUB_API_TOKEN=your_token_here
Run the System
python main.py

Example:

Ask your question: Who was born first, Albert Einstein or Nikola Tesla?

Output:

{
  "answer": "Nikola Tesla",
  "justification": "...",
  "confidence": 0.91,
  "verification": 0.88
}
