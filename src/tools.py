from sentence_transformers import SentenceTransformer
import chromadb
import subprocess
import ast
import operator

# Load once
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="db/")
collection = client.get_or_create_collection("juridique")


# 🔎 TOOL 1: SEARCH (RAG)
def search_tool(query: str) -> str:
    query_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    return "\n".join(results["documents"][0])


# 🧮 TOOL 2: CALCULATOR (safe — no eval)
def calculator_tool(expression: str) -> str:
    """Safely evaluate basic math expressions."""
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.n
        elif isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return allowed_ops[type(node.op)](_eval(node.operand))
        else:
            raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"Invalid expression: {e}"


# 🤖 TOOL 3: LLM (Ollama)
def llm_tool(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        capture_output=True
    )
    return result.stdout.decode()