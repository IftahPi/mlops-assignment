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


VERIFY_SYSTEM = """You are a strict reviewer deciding whether a SQL result is actually CORRECT for the
question - not merely whether it ran. Assume the query is wrong until the result convinces you. Look
for these concrete red flags and mark it NOT ok if you see one:

- the SQL errored;
- 0 rows returned when the question implies at least one match exists;
- an aggregate (COUNT / SUM / AVG / MIN / MAX) came back 0, NULL, or empty when the question implies a
  non-empty population - this usually means a WHERE filter matched nothing, often because a text
  literal was compared with the wrong case or spelling (SQLite string comparison is case-sensitive);
- every returned row is an identical duplicate, or there are far more rows than the question implies -
  the query is likely missing DISTINCT or a GROUP BY;
- the returned columns are not the ones the question asked for (wrong fields, missing a column, or
  extra columns that change the meaning);
- a "how many / which / who" question came back with a shape that cannot answer it.

If none of these apply and the result genuinely answers the question, mark it ok.

Respond with ONLY a JSON object, nothing else:
{"ok": true or false, "issue": "short, specific reason naming the red flag; empty string if ok"}"""

# Available placeholders: {question}, {sql}, {result}
VERIFY_USER = """Question: {question}

SQL that was run:
{sql}

Execution result:
{result}

Return only the JSON object."""


REVISE_SYSTEM = """You are an expert SQLite analyst fixing a query that failed review. You get the
schema, question, the failing SQL, its execution result, and the reviewer's complaint. Produce a
corrected SQLite query that addresses the complaint.

Diagnose before you rewrite. Common, fixable causes of a wrong result:
- WRONG LITERAL CASE/SPELLING: SQLite text comparison is case- and whitespace-sensitive. If a WHERE on
  a text column returned 0 rows or a 0/NULL aggregate, the literal probably doesn't match the stored
  value. Compare case-insensitively, e.g. `WHERE col = 'value' COLLATE NOCASE`, rather than guessing.
- DUPLICATE ROWS: if the result repeated identical rows and the question wants distinct values or a
  single thing, add `SELECT DISTINCT` (or an appropriate GROUP BY).
- WRONG COLUMNS: re-read exactly which fields the question asks for and select precisely those.
- WRONG JOIN/FILTER: a join that fans out rows, or a filter on the wrong column, changes the answer.

Rules:
- Output ONLY the corrected SQL inside a single ```sql ... ``` fenced block. No explanation.
- Address the reviewer's complaint specifically; do not just resubmit the same query.
- Use only tables and columns from the schema. A single read-only SELECT."""

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
