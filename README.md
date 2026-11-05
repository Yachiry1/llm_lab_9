# Survival AI

A university project for two developers: a conversational survival assistant built
incrementally with Python, Groq, Gradio, and six custom Python tools. Static
knowledge lives in a local JSON file. GitHub hosts collaboration; Render is the
intended deployment target.

The Gradio UI, Groq agent, all six tools, and survival/first-aid guides are
integrated. No automated tests or deployment configuration are included.

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

| File | Responsibility |
| --- | --- |
| `app.py` | Gradio chat UI, password-masked API key input, and per-session message history; calls `agent.chat`. |
| `agent.py` | Groq client/API communication, history processing, tool schemas, dispatch, and the tool-calling loop. |
| `tools.py` | All six tool functions, argument validation, and reading static knowledge. No Groq or Gradio code. |
| `survival_data.json` | Static guidelines under `survival_guides` and `first_aid_guides`. |
| `requirements.txt` | Only the direct runtime dependencies: `groq` and `gradio`. |

The agent uses `openai/gpt-oss-20b`. Model selection belongs in `agent.py`.

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

`app.py` owns session history using Gradio session state, appends the current
user message and returned assistant response once, and displays executed tools
separately. Clearing a chat clears that session's history. API keys and messages
must not be shared through module-level state or saved to disk. The agent creates
its Groq messages from a copy of the supplied history; intermediate tool messages
remain local to that invocation.

The agent returns a safe, displayable error response for missing keys,
API failures, or tool-loop exhaustion, preserving the list of tools already run.
Raw exceptions and credentials must not reach the UI.

## Shared tool contracts

All functions below live in `tools.py`. Arguments and results use only JSON types:
strings, numbers, booleans, null, lists, and dictionaries with string keys. Python
annotations below describe the shared interfaces of the six implemented functions.
The schemas in `agent.py` must match these names, arguments, required fields, and
constraints. The first three Python functions default missing arguments to `None`
so omitted required inputs return `invalid_arguments` instead of raising `TypeError`.
Agent schemas require the documented required arguments.

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
assess_risk(environment: str, weather: str, time_of_day: str, hazards: list[str],
            temperature_c: float | None = None, has_shelter: bool | None = None) -> dict
```

The first four arguments are required context. `environment` describes the survival
situation using the shared environment vocabulary. `time_of_day` is `day` or `night`.
`hazards` lists hazards reported by the user (use `[]` when none are reported).
Optional `temperature_c` is a finite number in Celsius; `has_shelter` is a boolean.
Omitted or null optional values mean unknown, not safe. Success `data`:

```json
{
  "risk_level": "high",
  "risk_factors": ["Reported hazard or environmental risk"],
  "recommendations": ["Suggested risk-reduction action"],
  "explanation": "Short rule-based explanation; educational, not a professional safety assessment."
}
```

`risk_level` is `low`, `medium`, or `high`. Factors and recommendations are lists
of strings; explanation is a string. This updates the original `moderate` label
to the requested `medium` before agent integration.

The highest matching severity wins. High: storm/extreme weather, temperature
at or below 0°C or at least 35°C, any reported hazard, or no shelter with adverse
weather/temperature. Medium: rain/snow, temperature below 10°C or at least 30°C
(outside the high bands), night, mountain/desert/coastal exposure, or no shelter.
Otherwise low. These are project heuristics, not validated safety thresholds.
An explanation flags missing temperature/shelter information and states that a
low estimate does not establish safety. Inputs cannot express every real hazard.

### 2. `check_equipment`

```python
check_equipment(environment: str, equipment: list[str]) -> dict
```

Both arguments are required. Compare equipment against the small category
dictionary in `tools.py`. Success `data`:

```json
{
  "available": ["Reported equipment item"],
  "useful_available": [
    {"item": "tarp", "possible_uses": ["Provide temporary protection from exposure."]}
  ],
  "missing_essential": ["Missing essential category"],
  "recommendations": ["Equipment recommendation"]
}
```

`available` contains normalized, deduplicated supplied items. Unrecognized items
remain in `available`; they do not satisfy unrelated checklist requirements.
`useful_available` contains recognized supplied items and their possible uses.
`missing_essential` lists categories, not exact item names: any recognized item
in a category satisfies this simplified checklist, without guaranteeing adequacy.
Core categories are water, shelter, navigation, light, first aid, and signaling.
Forest adds rain protection and water treatment; mountain adds warm clothing and
rain protection; desert adds sun protection; coastal adds rain and sun protection.
Urban uses the core categories. Matching is exact after trimming and lowercasing;
no fuzzy matching or inference about quantities, condition, or user skills occurs.

### 3. `get_survival_guide`

```python
get_survival_guide(topic: str) -> dict
```

`topic` is required. Canonical topics are `lost_in_forest`, `cold_weather`,
`shelter`, `water`, `fire`, `navigation`, and `signaling`. Case and surrounding
whitespace are ignored. Also accept `lost in the forest`, `lost in forest`,
`cold weather`, `water shortage` / `water_shortage`, and
`emergency shelter` / `emergency_shelter`. Water-shortage aliases return topic
`water`; emergency-shelter aliases return `shelter`. Read the matching guide from
`survival_data.json` on each call. Success `data`:

```json
{
  "topic": "shelter",
  "title": "Guide title",
  "steps": ["Ordered instruction"],
  "warnings": ["Relevant caution"]
}
```

`steps` and `warnings` are lists of strings; empty warnings are allowed.
Unknown topics return `unsupported_topic` with the supported topics in the message;
empty, omitted, or non-string topics return `invalid_arguments`. Missing/unreadable
files, invalid JSON, and missing or malformed guide records return `data_unavailable`.

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
plan from deterministic Python rules and the provided context. Success `data`:

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

`topic` is one of `bleeding`, `burns`, `sprain`, `hypothermia`, `heat_exhaustion`,
or `dehydration`. Case, spaces, and hyphens are normalized. Aliases include
`minor_cut` / `minor_cuts`, `minor_burn` / `minor_burns`,
`suspected_hypothermia`, and `suspected_dehydration`.
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

## Knowledge file format

`survival_data.json` contains both guide sections below. The risk rules, equipment
dictionary, and plan rules are deterministic Python code in `tools.py`; they do not
require JSON configuration.

- `survival_guides`: map supported survival topics to guide data objects matching
  `get_survival_guide`'s success `data` shape.
- `first_aid_guides`: map supported first-aid topics to guide data objects matching
  `get_first_aid_guide`'s success `data` shape except `topic`, which is added by the
  function using the normalized lookup key.

Missing required knowledge produces `data_unavailable`; do not invent guideline
content. The water tool needs no static knowledge.

The short guide content and equipment categories draw on the National Park Service
[Ten Essentials](https://www.nps.gov/articles/10essentials.htm) and
[Outdoor Emergency Plan](https://www.nps.gov/articles/gtgemergencyplan.htm).
They are educational summaries; the numerical risk bands are project rules.

## Local setup

Use Python 3.10 or newer. From the project root, run in PowerShell:

```text
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open `http://localhost:7860`, enter your Groq API key, and send a message. The UI
shows tools executed for the latest response below the conversation. Clear Chat
resets this session's conversation and tools display; it leaves the key input intact.
The direct dependencies are pinned to the versions used for integration checks.
On other platforms, use `.venv/bin/python` for the install and launch commands.

The UI accepts the user's own Groq API key through a password-masked input.
The key is passed to a per-request Groq client, not stored in conversation history,
files, or application logs. Do not put real keys in source files, example messages, screenshots, or Git
history. `.env` files are ignored but are not required for this UI-based flow.

## Incremental collaboration

Developer 1 owns the AI/backend work, including `agent.py` and this first increment
of three tools on `feature/ai-agent`. Coordinate later edits to shared `tools.py`
and `survival_data.json` with Developer 2 to avoid conflicting changes.
Developer 2 owns the Gradio UI and the other three tools. Coordinate any contract
changes in the README and both affected modules.

Start with a scaffold commit. For each increment, branch from the latest `main`,
make focused commits, push the branch to GitHub, and open a pull request for the
other developer to review before merging. Suggested increments:

1. Implement the tools and static knowledge.
2. Implement the Groq agent and tool-calling loop against the agreed contracts.
3. Integrate the Gradio UI, user-entered keys, and session history.
4. Prepare and verify deployment on Render after local integration works.

Do not commit secrets. Render deployment setup remains a later increment.
