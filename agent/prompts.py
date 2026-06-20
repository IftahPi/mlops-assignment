"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Convention: the *_SYSTEM strings are sent to the model verbatim (never
`.format`-ed), so they may contain literal braces (e.g. the JSON example in
VERIFY_SYSTEM). The *_USER strings ARE `.format`-ed, so only intended
placeholders may appear in them.
"""

GENERATE_SQL_SYSTEM = """You are an expert SQLite analyst. Given a database schema and an English
question, write ONE valid SQLite query that answers it.

Rules:
- Output ONLY the SQL inside a single ```sql ... ``` fenced block. No explanation.
- Use only tables and columns that appear in the schema.
- Double-quote identifiers when needed; this is SQLite.
- A single SELECT statement. Never modify data."""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Schema:
{schema}

Question: {question}

Return only the SQL in a ```sql block."""


VERIFY_SYSTEM = """You review whether a SQL result plausibly answers a question.

Mark it NOT ok if any of these hold:
- the SQL errored,
- the result is empty (0 rows) when the question clearly implies at least one row,
- the returned columns obviously do not answer what was asked.

Respond with ONLY a JSON object, nothing else:
{"ok": true or false, "issue": "short reason, empty string if ok"}"""

# Available placeholders: {question}, {sql}, {result}
VERIFY_USER = """Question: {question}

SQL that was run:
{sql}

Execution result:
{result}

Return only the JSON object."""


REVISE_SYSTEM = """You are an expert SQLite analyst fixing a query that failed review. You get the
question, the failing SQL, its execution result, and the reviewer's complaint. Produce a corrected
SQLite query.

Rules:
- Output ONLY the corrected SQL inside a single ```sql ... ``` fenced block.
- Address the reviewer's complaint specifically.
- Use only tables and columns from the schema."""

# Available placeholders: {schema}, {question}, {sql}, {result}, {issue}
REVISE_USER = """Schema:
{schema}

Question: {question}

Previous SQL (failed review):
{sql}

Its execution result:
{result}

Reviewer complaint: {issue}

Return the corrected SQL in a ```sql block."""
