# AI Bug Report Analyzer (LangGraph + Gemini)

Natural-language bug description -> structured bug report.

## Workflow
START -> Bug Analysis -> Severity Detection -> Root Cause Suggestion -> Format Report -> END

## Setup
1. `pip install -r requirements.txt`
2. Get a free API key from https://aistudio.google.com/apikey
3. Copy `.env.example` to `.env` and paste your key
4. `python app.py`

## Example input
The checkout page crashes with a 500 error whenever a user applies a discount code
on mobile. Customers can't complete purchases.
