"""The tools the agent can call, plus their schemas.

Kept separate from the agent loop so the tools can be tested on their own --
and so it's obvious that a "tool" is just a normal Python function plus a
JSON Schema describing how to call it.
"""

import ast
import operator

# ---------------------------------------------------------------------------
# calculate -- a safe arithmetic evaluator
# ---------------------------------------------------------------------------
# Deliberately NOT eval(). The model decides what goes in here, and model
# output is untrusted input: eval("__import__('os').system('rm -rf /')") is a
# real risk. Parsing to an AST and walking only the node types we allow means
# nothing outside arithmetic can execute.

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

MAX_EXPONENT = 1000


class ToolError(RuntimeError):
    """A tool failed in a way the model should be told about."""


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("only numbers are allowed")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ToolError(f"operator not allowed: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError(f"exponent too large (max {MAX_EXPONENT})")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ToolError(f"operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    raise ToolError(f"expression element not allowed: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression: {exc}") from exc

    try:
        result = _eval_node(tree)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc

    return str(result)


# ---------------------------------------------------------------------------
# get_weather -- canned data, so the project needs no API keys or network
# ---------------------------------------------------------------------------

_WEATHER = {
    "kathmandu": "18°C, partly cloudy",
    "london": "11°C, raining",
    "tokyo": "24°C, clear",
    "berlin": "15°C, overcast",
    "lagos": "31°C, humid",
}


def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    reading = _WEATHER.get(city.strip().lower())
    if reading is None:
        known = ", ".join(sorted(_WEATHER))
        raise ToolError(f"no weather data for {city!r}. Known cities: {known}")
    return f"{city.strip().title()}: {reading}"


# ---------------------------------------------------------------------------
# search_docs -- a tiny keyword search over a fixed knowledge base
# ---------------------------------------------------------------------------

_DOCS = {
    "pto": "Employees get 20 days of paid time off per year, plus public holidays.",
    "insurance": "Health insurance starts on your first day, with no waiting period.",
    "laptop": "IT ships your laptop 2-3 business days before your start date.",
    "stipend": "There is a $1,000 per year learning and development stipend.",
    "remote": "Remote employees get a one-time $500 home office allowance.",
}


def search_docs(query: str) -> str:
    """Search the internal handbook for a topic."""
    words = query.lower().split()
    hits = [text for key, text in _DOCS.items()
            if any(word.startswith(key) or key in word for word in words)]
    if not hits:
        return "No matching handbook entry found."
    return "\n".join(hits)


# ---------------------------------------------------------------------------
# Registry + schemas handed to the model
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "calculate": calculate,
    "get_weather": get_weather,
    "search_docs": search_docs,
}

TOOL_SCHEMAS = [
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression. Use this for any "
                       "math rather than working it out yourself.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "An arithmetic expression, e.g. '(3 + 4) * 12'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"}
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_docs",
        "description": "Search the internal employee handbook for policies "
                       "such as time off, insurance, equipment, or stipends.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up"}
            },
            "required": ["query"],
        },
    },
]


def run_tool(name: str, arguments: dict) -> tuple[str, bool]:
    """Execute a tool by name. Returns (result_text, is_error).

    Errors are returned rather than raised: the model needs to see what went
    wrong so it can correct itself or tell the user.
    """
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return f"Unknown tool: {name}", True

    try:
        return function(**arguments), False
    except ToolError as exc:
        return f"Error: {exc}", True
    except TypeError as exc:
        return f"Error: bad arguments for {name}: {exc}", True
