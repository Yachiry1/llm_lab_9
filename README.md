# Survival AI

A university project for two developers: a conversational survival assistant built
incrementally with Python, Groq, Gradio, and six custom Python tools. Static
knowledge lives in a local JSON file. GitHub hosts collaboration; Render is the
intended deployment target.

This commit contains the scaffold and shared contracts only. The agent, UI, tools,
and survival knowledge are not implemented. No tests or deployment configuration
are included.

## Project structure

```text
survival-ai/
├── app.py
├── agent.py
├── tools.py
├── survival_data.json
├── requirements.txt
├── .gitignore
└── README.md
```

## Architecture and ownership

| File | Responsibility when implemented |
| --- | --- |
| `app.py` | Gradio chat UI, password-masked API key input, and per-session message history; calls `agent.chat`. |
| `agent.py` | Groq client/API communication, history processing, tool schemas, dispatch, and the tool-calling loop. |
| `tools.py` | All six tool functions, argument validation, and reading static knowledge. No Groq or Gradio code. |
| `survival_data.json` | Static survival guidelines. Currently an empty JSON object. |
| `requirements.txt` | Only the direct runtime dependencies: `groq` and `gradio`. |

Preferred model: `openai/gpt-oss-20b`, subject to tool-calling availability on Groq
when the agent is implemented. Model selection belongs in `agent.py`.

Data flow: Gradio → `agent.chat` → Groq → tool calls dispatched to `tools.py` →
JSON results returned to Groq → final response and executed tool names → Gradio.
The agent must handle successive tool calls until a final answer is produced,
with a bounded loop to prevent endless execution.

The project uses no Docker, database, LangChain, FastAPI, or React. Read the JSON
file relative to `tools.py`, independent of the process working directory.

## Shared agent interface

```python
chat(
    message: str,
    history: list[dict],
    api_key: str,
) -> tuple[str, list[str]]
```

- `message`: the current user message, passed separately from prior history.
- `history`: earlier user and assistant messages in chronological order. Each
  dictionary has exactly `{"role": "user" | "assistant", "content": str}`.
  The current message is not yet in history. The agent must not mutate this list.
- `api_key`: the key entered by this session's user. Require a nonempty key;
  never hardcode, commit, log, or include it in chat history or model messages.
- Return value: `(response, executed_tools)`, where `response` is displayable
  assistant text and `executed_tools` is a list of tool function names actually
  invoked, in execution order. Keep repeated names if a tool runs more than once.
  Use `[]` when no tool ran. A tool invocation that returns an error still counts;
  an unknown name or rejected arguments that never invoke a function do not.

`app.py` will own session history using Gradio session state, append the current
user message and returned assistant response once, and display executed tools
separately. Clearing a chat clears that session's history. API keys and messages
must not be shared through module-level state or saved to disk. The agent creates
its Groq messages from a copy of the supplied history; intermediate tool messages
remain local to that invocation.

The future agent will return a safe, displayable error response for missing keys,
API failures, or tool-loop exhaustion, preserving the list of tools already run.
Raw exceptions and credentials must not reach the UI.

## Shared tool contracts

All functions below live in `tools.py`. Arguments and results use only JSON types:
strings, numbers, booleans, null, lists, and dictionaries with string keys. Python
annotations below describe the intended functions; they are not implemented yet.
The schemas in `agent.py` must match these names, arguments, required fields, and
constraints. There are no optional arguments or hidden defaults in these contracts.

Every successful call returns `{"ok": true, "data": ...}` with the function's
data shape below. Every invalid argument or unavailable guide returns
`{"ok": false, "error": {"code": str, "message": str}}`; planned codes are
`invalid_arguments`, `unsupported_topic`, and `data_unavailable`. Tool errors are
data for the agent to process. Tools do not call the model or mutate session state.

Shared vocabulary:

- `environment`: one of `forest`, `mountain`, `desert`, `coastal`, `urban`.
- `weather`: one of `clear`, `rain`, `snow`, `storm`, `extreme_heat`, `extreme_cold`.
- Equipment and hazard inputs: lists of nonempty strings; empty lists are valid.
  Compare names after trimming whitespace and normalizing case.
- All numeric inputs must be finite; integer counts must be positive integers.

### 1. `assess_risk`

```python
assess_risk(environment: str, weather: str, time_of_day: str, hazards: list[str]) -> dict
```

`time_of_day` is `day` or `night`. `hazards` lists hazards reported by the user.
Assess risk using static rules and supplied context. Success `data`:

```json
{
  "risk_level": "high",
  "risk_factors": ["Reported hazard or environmental risk"],
  "recommendations": ["Suggested risk-reduction action"]
}
```

`risk_level` is `low`, `moderate`, or `high`; the other fields are lists of strings.

### 2. `check_equipment`

```python
check_equipment(environment: str, equipment: list[str]) -> dict
```

Compare available equipment against the environment's static checklist. Success
`data` (all fields are lists of strings):

```json
{
  "available": ["Reported equipment item"],
  "missing_essential": ["Missing essential item"],
  "recommendations": ["Equipment recommendation"]
}
```

`available` contains normalized, deduplicated supplied items. Unrecognized items
remain in `available`; they do not satisfy unrelated checklist requirements.

### 3. `get_survival_guide`

```python
get_survival_guide(topic: str) -> dict
```

`topic` is one of `shelter`, `water`, `fire`, `navigation`, `signaling`.
Return a static guide. Success `data`:

```json
{
  "topic": "shelter",
  "title": "Guide title",
  "steps": ["Ordered instruction"],
  "warnings": ["Relevant caution"]
}
```

`steps` and `warnings` are lists of strings; empty warnings are allowed.

### 4. `estimate_water_supply`

```python
estimate_water_supply(water_liters: float, people: int, daily_liters_per_person: float) -> dict
```

`water_liters` is nonnegative; `people` is a positive integer;
`daily_liters_per_person` is positive and supplied explicitly. This is an arithmetic
estimate using the supplied consumption rate. Success `data`:

```json
{
  "total_daily_liters": 4.0,
  "estimated_days": 1.5,
  "estimated_hours": 36.0,
  "assumptions": ["Consumption stays constant at the supplied rate."]
}
```

Calculate total daily liters as people × daily liters per person; days as water
liters ÷ total daily liters; hours as days × 24. Return unrounded numeric values
and leave display rounding to the UI. Zero water produces zero days and hours.

### 5. `create_survival_plan`

```python
create_survival_plan(environment: str, weather: str, equipment: list[str], people: int, duration_hours: float) -> dict
```

`people` is a positive integer and `duration_hours` is positive. Build a prioritized
plan from static rules and the provided context. Success `data`:

```json
{
  "summary": "Brief plan summary",
  "steps": [
    {"priority": 1, "action": "Action to take", "reason": "Reason for this priority"}
  ],
  "warnings": ["Context-specific caution"]
}
```

Steps are ordered by increasing integer priority, starting at 1 without gaps.
The tool produces structured guidance; it does not make another model request.

### 6. `get_first_aid_guide`

```python
get_first_aid_guide(topic: str) -> dict
```

`topic` is one of `bleeding`, `burns`, `sprain`, `hypothermia`, `heat_exhaustion`.
Return static first-aid information. Success `data`:

```json
{
  "topic": "sprain",
  "title": "First-aid guide title",
  "steps": ["Ordered instruction"],
  "warnings": ["Relevant caution"],
  "seek_help_if": ["Condition requiring professional help"]
}
```

`steps`, `warnings`, and `seek_help_if` are lists of strings.

## Planned knowledge file format

`survival_data.json` is currently `{}`. When knowledge is added, use these top-level
keys so both developers can work against the same layout:

- `survival_guides`: map supported survival topics to guide data objects matching
  `get_survival_guide`'s success `data` shape.
- `first_aid_guides`: map supported first-aid topics to guide data objects matching
  `get_first_aid_guide`'s success `data` shape.
- `equipment_checklists`: map environments to objects with `essential` and
  `recommended` lists of equipment strings.
- `risk_rules`: list of objects with `environment`, `weather`, `time_of_day`,
  `hazard`, `risk_level`, `risk_factor`, and `recommendations`. Context fields are
  strings or null, where null means any value. All specified context fields must
  match; a specified hazard matches an item in the supplied hazards. Combine
  matching factors and recommendations; use the highest matching risk level,
  defaulting to `low` when none match.
- `plan_rules`: list of objects with `environment`, `weather`, `required_equipment`,
  `min_people`, `max_duration_hours`, `priority`, `action`, `reason`, and `warnings`.
  Environment and weather may be null for any value; required equipment is a list
  that must be a subset of available items. The count and duration thresholds may
  be null for no bound. Combine matching rules, sort by priority then file order,
  and renumber returned steps from 1. Warnings are lists of strings.

Missing required knowledge produces `data_unavailable`; do not invent guideline
content. The water tool needs no static knowledge.

## Local setup

From this directory, create and activate a Python virtual environment, then install:

```text
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the install command after activating the environment using your shell's
activation command. Dependencies are intentionally unpinned at this scaffold stage;
record compatible versions after integration. There is no runnable app yet.

The future UI must accept the user's own Groq API key through a password-masked
input. Do not put real keys in source files, example messages, screenshots, or Git
history. `.env` files are ignored but are not required for this UI-based flow.

## Incremental collaboration

Agree on these contracts before implementation. A suggested split is Developer 1
owning `agent.py`, and Developer 2 owning `tools.py` and `survival_data.json`.
Implement `app.py` in a later integration branch using the documented `chat`
contract. Coordinate any contract changes in the README and both affected modules.

Start with a scaffold commit. For each increment, branch from the latest `main`,
make focused commits, push the branch to GitHub, and open a pull request for the
other developer to review before merging. Suggested increments:

1. Implement the tools and static knowledge.
2. Implement the Groq agent and tool-calling loop against the agreed contracts.
3. Integrate the Gradio UI, user-entered keys, and session history.
4. Prepare and verify deployment on Render after local integration works.

Do not commit secrets. Deployment setup and runtime implementation belong to later
increments.
