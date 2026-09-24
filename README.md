# Prompt-Driven Data Analysis Assistant

**IBM SkillsBuild Academic Internship — Data Analytics Track — Final Project**

A Streamlit app that lets you ask plain-English questions about a supermarket
sales dataset. Every question is first refined by a **PromptIQ**-based
prompt-refinement layer (adapted from the author's own PromptIQ academic
project) before being sent to an LLM, which generates pandas code that is
executed to return a table and/or chart.

## How it works

```
raw question
     │
     ▼
PromptIQ refinement (promptiq_refiner.py)
  - grounds the prompt in the real dataset schema
  - rewrites vague terms ("best", "recent", "top") into precise instructions
  - structures the prompt into a fixed template
     │
     ▼
LLM call (Claude, via Anthropic API)
  - returns a pandas code snippet
     │
     ▼
Sandboxed execution (app.py)
  - runs the code against the in-memory DataFrame
  - captures `result` (table) and `fig` (chart)
     │
     ▼
Streamlit UI displays the refinement, the generated code, and the result
```

## Why prompt refinement matters

Feeding raw, ambiguous questions straight to an LLM often produces
inconsistent or wrong code (e.g. "best" branch could mean by revenue,
by rating, or by transaction count). The refinement step disambiguates
the question and grounds it in the exact column names before the model
ever sees it, which measurably improves the reliability of the generated
code — this was the whole premise of the original PromptIQ project,
applied here to a concrete data-analytics use case.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-api-key-here"
streamlit run app.py
```

## Project structure

```
.
├── app.py                          # Streamlit app + LLM call + execution sandbox
├── promptiq_refiner.py             # PromptIQ-based prompt refinement module
├── sample_supermarket_sales.csv    # Sample dataset (replace with the masterclass dataset)
├── requirements.txt
└── README.md
```

## Dataset

`sample_supermarket_sales.csv` is a synthetic placeholder with the standard
supermarket-sales schema (Branch, City, Product line, Unit price, Quantity,
Total, Date, Payment, Rating, etc.). Swap in the actual dataset provided in
the internship's Masterclass 4 workbook — the app auto-detects columns and
dtypes, so no code changes are required as long as column names stay
consistent, or the `DATA_PATH` constant is updated in `app.py`.

## Safety notes

Generated code runs in a restricted execution namespace (no `__builtins__`,
blocked imports of `os`/`sys`/`subprocess`, no `eval`/`exec`/file access) as
a first line of defense. For a production deployment this should be
hardened further (e.g. a separate subprocess sandbox or container).

## Example questions to try

- "Which branch had the best total sales?"
- "Show me the sales trend over time."
- "Compare average rating by product line."
- "What's the breakdown of payment methods?"

## Credits

Built as the final project for the IBM SkillsBuild Academic Internship
(Data Analytics track), integrating the prompt-refinement approach from
the author's PromptIQ project.
