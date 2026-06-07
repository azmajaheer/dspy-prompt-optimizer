# DSPy Prompt Optimizer

A Streamlit app that lets you upload any tabular dataset, define input features and a target column, run every DSPy optimizer against it, compare scores side-by-side, and export the final optimized prompt as JSON — all without writing a single line of code.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![DSPy](https://img.shields.io/badge/dspy-3.2.1-orange)
![Streamlit](https://img.shields.io/badge/streamlit-1.35%2B-red)
[![GitHub Stars](https://img.shields.io/github/stars/Laxminarayen/dspy-prompt-optimizer?style=social)](https://github.com/Laxminarayen/dspy-prompt-optimizer/stargazers)

---

## What it does

| Tab | Purpose |
|-----|---------|
| **Dataset** | Upload a training CSV and an optional held-out test CSV. Six built-in topic examples let you try the app instantly. |
| **Variables** | Pick input (X) columns, the target (Y) column, write a task description, and choose an evaluation metric. |
| **Optimization** | Select one optimizer or compare all seven at once. Set dataset size guardrails, review LLM-call cost estimates, then click Run. |
| **Results** | Baseline vs. optimized scores in a comparison table and bar chart. Per-run inspector with instructions, few-shot examples, and a JSON download. Live inference section to test the compiled program immediately. |

### Supported optimizers

| Optimizer | Strategy | Needs val set? |
|-----------|----------|---------------|
| **LabeledFewShot** | Selects k labeled examples as demos — fast sanity-check baseline | No |
| **BootstrapFewShot** | Teacher–student trace bootstrapping | No |
| **BootstrapFewShotWithRandomSearch** | Bootstrap + random search over candidate programs | Yes |
| **BootstrapFewShotWithOptuna** | Bootstrap + Optuna Bayesian hyperparameter search | Yes |
| **COPRO** | Coordinate-ascent instruction optimization (no demos) | No |
| **MIPROv2** | Bayesian joint optimization of instructions + demos | Yes |
| **GEPA** *(experimental)* | Evolutionary optimizer with LM-driven reflection | Yes |

### Evaluation metrics

- **Exact Match** — case-insensitive, whitespace-normalized string equality
- **Contains Answer** — checks whether the prediction contains the gold answer
- **F1 Token Overlap** — word-level F1 (good for extractive QA)
- **Always True** — scores every prediction 1.0 (useful for generation tasks)

---

## Prerequisites

- Python 3.10 or 3.11 (DSPy 3.x does **not** support Python 3.12 yet)
- An LLM API key — OpenAI, Anthropic, or any provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers)
- `conda` (recommended) or a standard `venv`

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Laxminarayen/dspy-prompt-optimizer.git
cd dspy-prompt-optimizer
```

### 2. Create a Python environment

**With conda (recommended):**

```bash
conda create -n dspy python=3.10 -y
conda activate dspy
```

**With venv:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins:

```
streamlit>=1.35.0
dspy>=3.0.0
pandas>=2.0.0
optuna>=3.0.0
```

> **Note:** DSPy 3.x pulls in LiteLLM, so nearly every major LLM provider works out of the box with no extra packages.

### 4. Run the app

**With conda:**

```bash
conda run -n dspy python -m streamlit run app.py
```

**With venv (environment already active):**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage walkthrough

### Step 1 — Configure LLM (sidebar)

Enter your provider, model name, and API key. Examples:

| Provider | Model string | Key env var |
|----------|-------------|-------------|
| OpenAI | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| Groq | `groq/llama-3.1-8b-instant` | `GROQ_API_KEY` |
| Ollama (local) | `ollama/llama3` | *(no key needed)* |

The model string follows LiteLLM's `provider/model` format. Cheaper/faster models work well for exploration; stronger models give better optimization results.

### Step 2 — Upload a dataset (Tab 1)

Upload a CSV with at least two columns — one for input(s) and one for the answer/label. Optionally upload a separate test CSV to get honest held-out evaluation scores.

If you don't have a dataset handy, click **Load example** to load one of six built-in topic examples.

> For SQuAD-style QA testing, download the validation split from HuggingFace:
> ```python
> from datasets import load_dataset
> ds = load_dataset("rajpurkar/squad", split="validation")
> ds.to_pandas().rename(columns={"answers": "answer"}) \
>   .assign(answer=lambda df: df["answer"].map(lambda a: a["text"][0])) \
>   [["title","context","question","answer"]] \
>   .to_csv("squad_validation.csv", index=False)
> ```

### Step 3 — Define variables (Tab 2)

- **X (inputs):** multi-select the columns the LLM should receive as input
- **Target (Y):** the column containing ground-truth answers
- **Task description:** one sentence describing what the LLM should do (used in instruction proposals)
- **Metric:** choose the evaluation metric that fits your task

### Step 4 — Run optimization (Tab 3)

**Single mode** — pick one optimizer, tune its parameters, run it.

**Compare mode** — tick multiple optimizers (or "Select all"), review the cost estimate table, then run all at once.

Size guardrails cap how many rows are sent to the optimizer and evaluator so you don't accidentally burn through API credits on a large dataset.

### Step 5 — Review results (Tab 4)

- Baseline score (zero-shot, no optimization) is shown as a reference
- Comparison table with Δ Baseline, demos added, instruction changed flag, and timing
- Bar chart for quick visual comparison
- Per-run inspector: expand any run to see the full instructions, few-shot examples, and input/output field names used by each predictor
- **Download prompt JSON** — save the compiled program definition for use in your own code
- **Live inference** — type an input directly in the UI and call the compiled program against the LLM in real time

---

## Using the exported JSON in your own code

```python
import dspy, json

# Load the JSON produced by the Download button
with open("my_optimized_prompt.json") as f:
    data = json.load(f)

# Reconstruct the module
lm = dspy.LM(model="openai/gpt-4o-mini", api_key="sk-...")
with dspy.context(lm=lm):
    predictor = dspy.Predict(data["signature"])
    # Inject the optimized instructions back
    for name, pred_data in data["predictors"].items():
        predictor.signature = predictor.signature.with_instructions(
            pred_data["instructions"]
        )
    result = predictor(**your_inputs)
    print(result)
```

---

## Project structure

```
dspy-prompt-optimizer/
├── app.py            # Single-file Streamlit application
├── requirements.txt  # Python dependencies
└── README.md
```

Everything lives in `app.py`. Key sections:

| Lines (approx.) | Section |
|----------------|---------|
| 1–260 | Top-level helper functions (`build_result_json`, `create_metric_fn`, `dispatch_optimizer`, `evaluate_compiled`, `record_run`, `_estimate_llm_calls`, `_df_uploader`) |
| 261–320 | Streamlit page config, sidebar LLM configuration |
| 320–560 | Tab 1 (Dataset), Tab 2 (Variables) |
| 561–960 | Tab 3 (Optimization) — OPTIMIZERS catalog, mode selector, guardrails, run loop |
| 961–end | Tab 4 (Results) — comparison table, chart, per-run inspector, live inference |

---

## Contributing

Contributions of all sizes are welcome — from fixing a typo in a tooltip to adding a full RAG pipeline. This section covers everything you need to go from zero to a merged pull request.

---

### Who should fork vs. clone directly?

| You are… | What to do |
|-----------|-----------|
| **An external contributor** (anyone without write access to this repo) | **Fork first**, then clone your fork. You cannot push branches directly to this repo. |
| **The repo owner / a collaborator with write access** | **Clone directly** — no fork needed. Create a branch and push it. |

The steps below cover both paths. Read the section that applies to you.

---

### Path A — External contributors (fork-based workflow)

This is the standard open-source path. You get your own copy of the repo on GitHub, make changes there, and raise a PR back to the original.

#### A-1. Fork the repo on GitHub

1. Go to `https://github.com/Laxminarayen/dspy-prompt-optimizer`
2. Click **Fork** (top-right corner)
3. Select your GitHub account as the destination
4. GitHub creates `https://github.com/<your-username>/dspy-prompt-optimizer` — this is **your fork**

> A fork is a full copy of the repo under your account. You have write access to it. The original repo is called **upstream**. Changes you make in your fork do not affect upstream until you open a PR and it gets merged.

#### A-2. Clone your fork locally

```bash
# Clone YOUR fork (not the upstream repo)
git clone https://github.com/<your-username>/dspy-prompt-optimizer.git
cd dspy-prompt-optimizer

# Verify the remote points to your fork
git remote -v
# origin  https://github.com/<your-username>/dspy-prompt-optimizer.git (fetch)
# origin  https://github.com/<your-username>/dspy-prompt-optimizer.git (push)
```

#### A-3. Add the upstream remote

This lets you pull future changes from the original repo into your fork:

```bash
git remote add upstream https://github.com/Laxminarayen/dspy-prompt-optimizer.git

# Verify both remotes now exist
git remote -v
# origin    https://github.com/<your-username>/dspy-prompt-optimizer.git (fetch)
# origin    https://github.com/<your-username>/dspy-prompt-optimizer.git (push)
# upstream  https://github.com/Laxminarayen/dspy-prompt-optimizer.git (fetch)
# upstream  https://github.com/Laxminarayen/dspy-prompt-optimizer.git (push)
```

> **Rule of thumb:** `origin` = your fork (you push here). `upstream` = the original repo (you only fetch/pull from here, never push).

#### A-4. Set up your dev environment

```bash
conda create -n dspy-dev python=3.10 -y
conda activate dspy-dev
pip install -r requirements.txt
```

#### A-5. Sync your fork before starting work

Always pull the latest changes from upstream before creating a branch. This avoids conflicts later:

```bash
git fetch upstream                    # download latest from upstream (no file changes yet)
git checkout main                     # switch to your local main branch
git merge upstream/main               # fast-forward your main to match upstream
git push origin main                  # push the updated main back to your fork on GitHub
```

#### A-6. Create a feature branch

Never work directly on `main`. Create a branch off the freshly-synced `main`:

```bash
git checkout main
git checkout -b feat/rouge-metric
# You are now on branch feat/rouge-metric, branched from main
```

Branch naming conventions:
- `feat/` — new feature
- `fix/` — bug fix
- `docs/` — documentation only
- `refactor/` — code cleanup with no behaviour change

#### A-7. Make your changes, commit

```bash
# Edit app.py, requirements.txt, etc.

# Stage only the files you changed (never git add -A blindly)
git add app.py requirements.txt

# Verify what is staged before committing
git status
git diff --staged

# Commit with a clear message
git commit -m "feat: add ROUGE-L metric for summarization tasks"
```

Multiple small commits are fine while working. The commit history will be visible in the PR.

#### A-8. Push the branch to your fork

```bash
git push -u origin feat/rouge-metric
# -u links your local branch to origin/feat/rouge-metric
# Future pushes on this branch only need: git push
```

#### A-9. Open the pull request

**Via GitHub CLI (recommended):**

```bash
gh pr create \
  --repo Laxminarayen/dspy-prompt-optimizer \
  --base main \
  --head <your-github-username>:feat/rouge-metric \
  --title "feat: add ROUGE-L metric for summarization tasks" \
  --body "$(cat <<'EOF'
## What this PR does
Adds ROUGE-L F-measure as a selectable evaluation metric.

## Why
Exact Match and F1 Token Overlap are too strict for summarization tasks
where multiple valid phrasings exist. Closes #<issue-number> if applicable.

## How to test it
1. Load the built-in example dataset (Tab 1 → Use example dataset)
2. Go to Tab 2 → Metric dropdown → select ROUGE-L
3. Run BootstrapFewShot
4. Confirm the score in the Results tab is a float between 0 and 1

## Checklist
- [x] Tested with built-in example (10-row training + 5-row test)
- [x] Tested with squad_sample_200.csv
- [x] requirements.txt updated
- [x] No API keys or personal data committed
EOF
)"
```

**Via the GitHub web UI:**

1. After pushing, visit `https://github.com/<your-username>/dspy-prompt-optimizer`
2. GitHub shows a yellow banner: **"feat/rouge-metric had recent pushes — Compare & pull request"** — click it
3. On the PR page, confirm:
   - **base repository:** `Laxminarayen/dspy-prompt-optimizer` → **base:** `main`
   - **head repository:** `<your-username>/dspy-prompt-optimizer` → **compare:** `feat/rouge-metric`
4. Fill in title and body, then click **Create pull request**

If the banner has disappeared: go to the **Pull requests** tab → **New pull request** → **compare across forks** → set the head fork and branch manually.

---

### Path B — Maintainer / collaborator (direct clone workflow)

If you have write access to the repo, you do not need a fork. Clone the repo directly and push branches straight to it.

#### B-1. Clone the repo directly

```bash
git clone https://github.com/Laxminarayen/dspy-prompt-optimizer.git
cd dspy-prompt-optimizer

# You have a single remote — origin points to the real repo
git remote -v
# origin  https://github.com/Laxminarayen/dspy-prompt-optimizer.git (fetch)
# origin  https://github.com/Laxminarayen/dspy-prompt-optimizer.git (push)
```

#### B-2. Set up your dev environment

```bash
conda create -n dspy-dev python=3.10 -y
conda activate dspy-dev
pip install -r requirements.txt
```

#### B-3. Create a feature branch

```bash
git checkout main
git pull origin main          # get the latest main before branching
git checkout -b feat/my-feature
```

#### B-4. Make changes, commit, push

```bash
git add app.py
git commit -m "feat: my feature"
git push -u origin feat/my-feature
```

#### B-5. Open the PR — branch to main on the same repo

```bash
gh pr create \
  --base main \
  --head feat/my-feature \
  --title "feat: my feature" \
  --body "Description of what changed and why."
# No --repo flag needed — gh detects the current repo from the git remote
```

Via the web UI, the flow is the same as Path A step A-9 except both sides are on `Laxminarayen/dspy-prompt-optimizer` (no "compare across forks" needed).

---

### Common steps (both paths)

The following applies whether you forked or cloned directly.

#### Responding to review comments

After a reviewer leaves feedback, update the code and push again. GitHub updates the PR automatically — never close and re-open.

```bash
# Make the requested changes, then:
git add app.py
git commit -m "fix: address review — switch to stemmer=False per feedback"
git push origin feat/rouge-metric   # or feat/my-feature
```

To reply to a specific comment from the terminal:

```bash
gh pr review <pr-number> --comment \
  --body "Done — switched to stemmer=False. Let me know if this looks good."
```

#### Handling merge conflicts

If GitHub shows "This branch has conflicts that must be resolved":

```bash
# External contributors (fork-based):
git fetch upstream
git rebase upstream/main

# Collaborators (direct clone):
git fetch origin
git rebase origin/main

# Git pauses at each conflicting file and shows markers:
# <<<<<<< HEAD      (your changes)
# =======
# >>>>>>> origin/main  (the incoming changes)
#
# Edit the file to keep the correct version, remove the markers, then:
git add app.py
git rebase --continue

# If the rebase gets complicated and you want to start over:
git rebase --abort

# Push the resolved branch
git push origin feat/rouge-metric --force-with-lease
```

`--force-with-lease` is safer than `--force` — it refuses to overwrite if someone else pushed to your branch after your last fetch.

#### Keeping in sync with main (fork-based contributors only)

While your PR is open, sync your fork's `main` regularly so future branches start from a current state:

```bash
git fetch upstream                  # get latest upstream changes
git checkout main
git merge upstream/main             # update your local main
git push origin main                # push the updated main to your fork
```

---

### PR description template

Use this every time you open a PR:

```markdown
## What this PR does
<!-- One specific sentence. "Adds ROUGE-L metric" not "Improves things". -->

## Why
<!-- Motivation. Link to an issue with "Closes #<number>" if one exists. -->

## How to test it
1. Load the built-in example dataset (Tab 1 → Use example dataset)
2. <!-- next step -->
3. Expected result: <!-- what the reviewer should see -->

## Screenshots (if UI changed)
<!-- Drag and drop a screenshot here -->

## Checklist
- [ ] Tested with the built-in example (10-row training + 5-row test)
- [ ] Tested with squad_sample_200.csv as training data
- [ ] requirements.txt updated if a new package was added
- [ ] No API keys, .env files, or personal CSV data committed
- [ ] Column name sanitization not broken (tested with a space in a column name)
```

#### Draft PR (work in progress)

Open a draft PR to get early feedback before your change is complete:

```bash
gh pr create \
  --base main \
  --head feat/rouge-metric \           # or <your-username>:feat/rouge-metric for forks
  --title "feat: add ROUGE-L metric [WIP]" \
  --draft \
  --body "Opening early for design feedback on the metric wrapper API."

# When the feature is complete, mark it ready:
gh pr ready <pr-number>
```

---

### `gh` command quick reference

```bash
# ── Auth ──────────────────────────────────────────────────────────��───────────
gh auth login                                   # log in once
gh auth status                                  # confirm you're authenticated

# ── Forking (external contributors only) ─────────────────────────────────────
gh repo fork Laxminarayen/dspy-prompt-optimizer --clone
# Creates the fork on GitHub AND clones it locally in one step.
# Automatically sets origin (your fork) and upstream (original repo).

# ── Creating PRs ──────────────────────────────────────────────────────────────
gh pr create                                    # interactive wizard
gh pr create --base main --head feat/x \
  --title "..." --body "..."                    # non-interactive, same repo
gh pr create --repo owner/repo \
  --head yourname:feat/x ...                    # from a fork to upstream
gh pr create --draft                            # open as draft

# ── Viewing PRs ───────────────────────────────────────────────────────────────
gh pr list                                      # open PRs in current repo
gh pr list --repo Laxminarayen/dspy-prompt-optimizer
gh pr view <number>                             # view in terminal
gh pr view <number> --web                       # open in browser
gh pr view <number> --comments                  # show review comments
gh pr diff <number>                             # show the file diff
gh pr status                                    # PRs touching your branches

# ── Updating PRs ──────────────────────────────────────────────────────────────
gh pr edit <number> --title "new title"
gh pr edit <number> --body "new description"
gh pr edit <number> --add-label "enhancement"
gh pr ready <number>                            # draft → ready for review
gh pr ready <number> --undo                     # ready → back to draft

# ── Reviewing ─────────────────────────────────────────────────────────────────
gh pr review <number> --approve
gh pr review <number> --request-changes --body "Please fix X"
gh pr review <number> --comment --body "Looks good, just one nit"
gh pr checkout <number>                         # check out PR branch locally to test it

# ── Merging / closing ─────────────────────────────────────────────────────────
gh pr merge <number> --squash                   # squash all commits into one
gh pr merge <number> --rebase                   # rebase onto main
gh pr merge <number> --merge                    # standard merge commit
gh pr close <number> --comment "reason"         # close without merging

# ── Issues ────────────────────────────────────────────────────────────────────
gh issue list
gh issue create --title "Bug: X crashes on Y" --body "Steps to reproduce..."
gh issue view <number>
gh issue close <number>
```

---

### 2. Set up your development environment

```bash
# Create an isolated conda environment (Python 3.10 or 3.11 only — DSPy 3.x does not support 3.12)
conda create -n dspy-dev python=3.10 -y
conda activate dspy-dev

# Install all dependencies
pip install -r requirements.txt
```

If you are adding a feature that requires a new package (e.g., `rouge-score` for a new metric), install it and add it to `requirements.txt` with a `>=` version pin:

```bash
pip install rouge-score
echo "rouge-score>=0.1.2" >> requirements.txt
```

---

### 3. Run the app locally

```bash
# With conda env active:
conda run -n dspy-dev python -m streamlit run app.py

# Or if the env is already activated:
streamlit run app.py
```

The app opens at `http://localhost:8501`. Streamlit hot-reloads on every file save, so you can edit `app.py` and see changes instantly without restarting.

---

### 4. Understand the codebase before you change it

Everything lives in a single file — `app.py`. Read through these sections before making changes:

| What to read first | Why |
|--------------------|-----|
| `create_metric_fn()` (line ~86) | Understand how metrics work before adding a new one |
| `make_dspy_module()` (line ~125) | Understand how DSPy signatures are built from column names |
| `dispatch_optimizer()` (line ~139) | Understand how each optimizer is called — add new ones here |
| `OPTIMIZERS` dict (line ~567) | Catalog that drives the UI — every optimizer's params are declared here |
| `_to_examples()` (inside the run button, line ~896) | How DataFrame rows become `dspy.Example` objects |
| `record_run()` (line ~234) | How results are stored in session state for the Results tab |

Key design constraints:
- **All DSPy calls must be inside `with dspy.context(lm=lm):`** — `dspy.configure()` is not thread-safe in DSPy 3.x and will raise a `RuntimeError` in Streamlit.
- **Column names are sanitized** to valid Python identifiers before building signatures. If you add a new place that uses column names as DSPy field names, apply the same `re.sub(r"[^a-zA-Z0-9_]", "_", name)` pattern.
- **`dispatch_optimizer()` receives a copy of params** (`dict(params_per_optimizer[opt_name])`). Always `pop()` values out of `p` rather than using `p.get()` then passing `**p` — leftover keys cause unexpected keyword argument errors.

---

### 5. Create a feature branch

Always branch off `main`. Use a short, descriptive name:

```bash
git checkout main
git pull upstream main           # sync with the latest upstream first
git checkout -b feat/rouge-metric
```

Branch naming conventions:
- `feat/` — new feature
- `fix/` — bug fix
- `docs/` — documentation only
- `refactor/` — code cleanup with no behaviour change

---

### 6. Make your changes

#### Adding a new evaluation metric

1. Open `create_metric_fn()` (~line 86). Add a new `elif` branch:

```python
elif metric_type == "ROUGE-L":
    from rouge_score import rouge_scorer as _rs
    _scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True)
    def metric(example, prediction, trace=None):
        gold = _get(example, target_col)
        pred = _get(prediction, target_col)
        return _scorer.score(gold, pred)["rougeL"].fmeasure
```

2. Add the new name to the `st.selectbox` list in Tab 2 (~line 543):

```python
"ROUGE-L",
```

3. Add the new package to `requirements.txt`:

```
rouge-score>=0.1.2
```

#### Adding a new optimizer

1. Add an entry to the `OPTIMIZERS` dict (~line 567). Follow the exact same structure as existing entries — `desc`, `params` (each with `type`, `default`, `min`/`max` or `options`, `help`), and `needs_valset`:

```python
"MyOptimizer": {
    "desc": "One-sentence description shown in the UI info box.",
    "params": {
        "my_param": dict(type="int", default=10, min=1, max=50,
                         help="What this param does."),
    },
    "needs_valset": True,   # set False if compile() does not accept valset
},
```

2. Add a corresponding `elif` branch to `dispatch_optimizer()` (~line 139):

```python
elif opt_name == "MyOptimizer":
    my_p = int(p.pop("my_param", 10))
    opt  = dspy.MyOptimizer(metric=metric, my_param=my_p)
    return opt.compile(module, trainset=trainset, valset=devset)
```

3. Add a cost estimate branch to `_estimate_llm_calls()` (~line 335):

```python
elif opt_name == "MyOptimizer":
    total = n_train * int(params.get("my_param", 10)) + n_eval
```

4. Run the app, switch to Compare mode, tick your new optimizer, and verify it appears in the cost estimate table and runs without errors.

#### Adding a new LLM provider to the sidebar

In the sidebar block (~line 282), add a new `elif` branch:

```python
elif provider == "Groq":
    model       = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"])
    lm_model_id = f"groq/{model}"
```

Also add the provider name to the `st.selectbox` options list above it.

---

### 7. Test your changes manually

There are no automated tests yet (contributing one would itself be a great PR). Until then, verify manually:

1. **Load the built-in example** (10-row training set, 5-row test set) and run your changed feature through the full flow: Dataset → Variables → Optimization → Results.
2. **Upload a real CSV** (use `squad_sample_200.csv` included in the repo) and repeat the flow.
3. **Check edge cases:**
   - Column names with spaces (e.g., rename a column to `"first name"` in the paste editor and confirm sanitization works)
   - Very small datasets (< 10 rows) — the guardrail `min_value` clamping must not crash
   - Missing values in the CSV — the `_to_examples()` function skips NaN cells; ensure your change does not break this
4. **Check the Results tab** — confirm the comparison table, bar chart, per-run inspector, JSON download, and live inference all still work after your change.

---

### 8. Commit your changes

Write a concise commit message that explains *why*, not just *what*:

```bash
git add app.py requirements.txt   # stage only the files you changed
git commit -m "feat: add ROUGE-L metric for summarization tasks

Adds rouge-score based ROUGE-L F-measure to the metric selector.
Useful for tasks where the target is a sentence rather than a short phrase,
since Exact Match and F1 Token Overlap are too strict for those cases."
```

Commit message guidelines:
- First line: `type: short description` (50 chars max), e.g. `feat:`, `fix:`, `docs:`, `refactor:`
- Blank line, then a paragraph explaining the motivation if needed
- Keep each commit focused on one logical change — don't bundle a bug fix and a new feature in one commit

---

### 9. Push and open a pull request

There are two ways to raise a PR: the **GitHub CLI (`gh`)** (recommended, fully from terminal) or the **GitHub web UI**. Both are covered below.

---

#### Option A — GitHub CLI (`gh`)

Install the CLI if you don't have it:

```bash
# macOS
brew install gh

# Windows (winget)
winget install --id GitHub.cli

# Linux (apt)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
  https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh
```

Authenticate once:

```bash
gh auth login
# Follow the prompts: select GitHub.com → HTTPS → authenticate via browser
# Verify it worked:
gh auth status
```

**Step 1 — Push your branch to your fork:**

```bash
git push -u origin feat/rouge-metric
# -u sets the upstream tracking so future `git push` needs no arguments
```

**Step 2 — Create the PR from the terminal:**

```bash
gh pr create \
  --repo Laxminarayen/dspy-prompt-optimizer \
  --base main \
  --head <your-github-username>:feat/rouge-metric \
  --title "feat: add ROUGE-L metric for summarization tasks" \
  --body "## What this PR does
Adds ROUGE-L F-measure as a selectable evaluation metric.

## Why
Exact Match and F1 Token Overlap are too strict for summarization tasks
where multiple valid phrasings exist. ROUGE-L rewards fluent, overlapping
n-grams and is standard in summarization benchmarks.

## How to test it
1. Load the built-in example dataset
2. Go to Tab 2 → Metric dropdown → select 'ROUGE-L'
3. Run BootstrapFewShot
4. Confirm the score in the Results tab is a float between 0 and 1

## Checklist
- [x] Tested with the built-in example (10-row dataset)
- [x] Tested with squad_sample_200.csv
- [x] requirements.txt updated (rouge-score>=0.1.2 added)
- [x] No API keys or personal data committed"
```

You will see output like:

```
https://github.com/Laxminarayen/dspy-prompt-optimizer/pull/7
```

That URL is your PR. Share it with the maintainer.

**Useful `gh pr` commands after opening:**

```bash
# View your PR in the terminal
gh pr view feat/rouge-metric

# Open your PR in the browser
gh pr view feat/rouge-metric --web

# Check CI / review status
gh pr status

# List all open PRs on the upstream repo
gh pr list --repo Laxminarayen/dspy-prompt-optimizer

# See review comments left on your PR
gh pr view feat/rouge-metric --comments

# See which files changed in your PR
gh pr diff feat/rouge-metric

# Mark your PR as ready for review (if you opened it as a draft)
gh pr ready feat/rouge-metric

# Convert an open PR back to draft while you fix things
gh pr ready feat/rouge-metric --undo

# Close (abandon) a PR without merging
gh pr close feat/rouge-metric --comment "Closing — superseded by #12"
```

---

#### Option B — GitHub web UI

1. Push your branch to your fork:

   ```bash
   git push -u origin feat/rouge-metric
   ```

2. Go to `https://github.com/<your-username>/dspy-prompt-optimizer` in your browser. GitHub will show a yellow banner:

   > **feat/rouge-metric** had recent pushes — **Compare & pull request**

   Click that button.

3. On the "Open a pull request" page:
   - **Base repository:** `Laxminarayen/dspy-prompt-optimizer` | **base:** `main`
   - **Head repository:** `<your-username>/dspy-prompt-optimizer` | **compare:** `feat/rouge-metric`
   - Fill in the title and body using the template below.
   - Click **Create pull request**.

If the yellow banner has disappeared, go to the **Pull requests** tab → click **New pull request** → click **compare across forks** → set the head fork and branch manually.

---

#### PR description template

Copy this into the PR body every time:

```markdown
## What this PR does
<!-- One sentence. Be specific: "Adds ROUGE-L metric" not "Improves metrics". -->

## Why
<!-- Motivation. Link to an issue with "Closes #<number>" if one exists. -->

## How to test it
<!-- Step-by-step so the reviewer can reproduce your change without guessing. -->
1. Load the built-in example dataset (Tab 1 → Use example dataset)
2. ...
3. Expected result: ...

## Screenshots (if UI changed)
<!-- Drag and drop a screenshot here. Helps reviewers see what changed visually. -->

## Checklist
- [ ] Tested with the built-in example (10-row training + 5-row test)
- [ ] Tested with `squad_sample_200.csv` uploaded as training data
- [ ] `requirements.txt` updated if a new package was added
- [ ] No API keys, `.env` files, or personal CSV data committed
- [ ] Column name sanitization not broken (tested with a column named "first name")
```

---

#### Opening a draft PR (work in progress)

If your feature is not finished but you want early feedback or want to show progress:

```bash
# Via CLI — add --draft flag
gh pr create \
  --repo Laxminarayen/dspy-prompt-optimizer \
  --base main \
  --head <your-username>:feat/rouge-metric \
  --title "feat: add ROUGE-L metric [WIP]" \
  --draft \
  --body "Work in progress — opening early for design feedback on the metric API."

# When ready to request a full review:
gh pr ready feat/rouge-metric
```

Via the web UI, tick **"Create draft pull request"** instead of clicking the green button.

---

### 10. Responding to review comments

After a reviewer leaves comments on your PR, update your code locally, then push again — GitHub automatically updates the PR with the new commits. **Do not close and re-open a PR to address feedback.**

```bash
# Make the requested changes in app.py, then:
git add app.py
git commit -m "fix: address review — use stemmer=False for ROUGE-L per feedback"
git push origin feat/rouge-metric
# The PR updates automatically — no further action needed
```

To reply to a specific review comment from the terminal:

```bash
gh pr review feat/rouge-metric --comment \
  --body "Done — switched to stemmer=False and added a note in the docstring."
```

To approve a PR you are reviewing (maintainer action):

```bash
gh pr review <pr-number> --approve --body "LGTM — tested locally with squad_sample_200.csv"
```

To request changes as a reviewer:

```bash
gh pr review <pr-number> --request-changes \
  --body "Please add the new package to requirements.txt before merging."
```

---

### 11. Handling merge conflicts

If `main` has moved ahead of your branch and GitHub shows "This branch has conflicts that must be resolved":

```bash
# Fetch the latest upstream changes
git fetch upstream

# Rebase your branch on top of the updated main
git rebase upstream/main

# Git will pause at each conflicting file and show conflict markers:
# <<<<<<< HEAD (your changes)
# ...
# ======= 
# ...
# >>>>>>> upstream/main (upstream changes)
#
# Edit the file to resolve conflicts, then:
git add app.py
git rebase --continue

# If rebase gets complicated and you want to start over:
git rebase --abort

# After a clean rebase, force-push (--force-with-lease is safer than --force)
git push origin feat/rouge-metric --force-with-lease
```

`--force-with-lease` refuses to overwrite if someone else pushed to your branch after your last fetch — it prevents accidentally clobbering a co-author's work.

---

### 12. Keeping your fork in sync

While your PR is under review, upstream `main` may receive other changes. Rebase your branch to keep a clean history:

```bash
# Fetch all branches from upstream (does not change your local files)
git fetch upstream

# Check what changed on upstream main since you branched
git log HEAD..upstream/main --oneline

# Rebase your branch on top of latest upstream main
git rebase upstream/main feat/rouge-metric

# Push the rebased branch (requires force since history was rewritten)
git push origin feat/rouge-metric --force-with-lease
```

To also keep your fork's `main` branch in sync (good hygiene):

```bash
git checkout main
git pull upstream main
git push origin main
```

---

#### Quick reference — all the `gh` commands you'll need

```bash
# ── Setup ─────────────────────────────────────────────────────────────────────
gh auth login                                     # authenticate once
gh auth status                                    # verify you're logged in

# ── Creating PRs ──────────────────────────────────────────────────────────────
gh pr create                                      # interactive wizard (prompts you)
gh pr create --title "..." --body "..." --base main   # non-interactive
gh pr create --draft                              # open as draft / WIP

# ── Viewing PRs ───────────────────────────────────────────────────────────────
gh pr list                                        # list PRs on current repo
gh pr list --repo Laxminarayen/dspy-prompt-optimizer  # list PRs on upstream
gh pr view <number>                               # view a specific PR
gh pr view <number> --web                         # open it in the browser
gh pr view <number> --comments                    # see all review comments
gh pr diff <number>                               # show the diff
gh pr status                                      # show PRs related to your branches

# ── Updating PRs ──────────────────────────────────────────────────────────────
gh pr edit <number> --title "new title"           # edit title
gh pr edit <number> --body "new body"             # edit description
gh pr edit <number> --add-label "feat"            # add a label
gh pr ready <number>                              # mark draft as ready for review
gh pr ready <number> --undo                       # revert back to draft

# ── Reviewing PRs ─────────────────────────────────────────────────────────────
gh pr review <number> --approve                   # approve
gh pr review <number> --request-changes           # request changes
gh pr review <number> --comment --body "..."      # leave a comment

# ── Checking out a PR locally (to test someone else's PR) ────────────────────
gh pr checkout <number>                           # switches to that branch locally

# ── Merging / closing PRs ─────────────────────────────────────────────────────
gh pr merge <number> --squash                     # squash and merge (preferred)
gh pr merge <number> --rebase                     # rebase merge
gh pr merge <number> --merge                      # regular merge commit
gh pr close <number>                              # close without merging
gh pr close <number> --comment "reason"           # close with a comment

# ── Issues (for referencing in PRs) ──────────────────────────────────────────
gh issue list                                     # list open issues
gh issue create --title "..." --body "..."        # open a new issue
gh issue view <number>                            # view an issue
```

---

### Contribution ideas

Pick any of the items below. Difficulty is approximate.

#### Good first issues — estimated 1–2 hours

| Idea | Where to look in app.py |
|------|------------------------|
| Add ROUGE / BLEU / BERTScore metrics | `create_metric_fn()` ~line 86, metric selectbox ~line 543 |
| Add Groq, Gemini, Mistral, Cohere to the sidebar | Sidebar provider selectbox ~line 285 |
| Add more built-in example datasets (medical QA, legal text, code review) | `EXAMPLE_TRAIN` / `EXAMPLE_TEST` dicts ~line 445 |
| Fix the "Always True" metric warning — it should tell the user scores will be meaningless | Results tab, `all_same` warning block ~line 1069 |
| Show a "copy to clipboard" button next to the JSON viewer | Results tab, JSON section ~line 1177 |

#### Medium complexity — estimated half a day

| Idea | Notes |
|------|-------|
| Session save / load | Serialize `st.session_state["optimization_runs"]` to JSON; add a file download and `st.file_uploader` to restore it |
| Chain-of-Thought toggle | Add a checkbox in Tab 2; swap `dspy.Predict` → `dspy.ChainOfThought` in `make_dspy_module()`; surface `rationale` in the Results inspector |
| Token usage tracker | Hook into LiteLLM's `success_callback` to count tokens per run; display tokens + estimated cost in the comparison table |
| Multi-output signatures | Allow multiple target columns in the Variables tab; build signatures like `question -> answer, rationale`; update `create_metric_fn()` to accept a list of targets |
| Custom metric code editor | Embed `st.text_area` with a Python snippet; use `exec()` in a sandbox to define the metric function; validate its signature before use |
| Streaming inference | Replace `st.success()` in the Live Inference section with DSPy's streaming API + `st.write_stream()` |
| Per-run notes | Add a `st.text_input` in the Results inspector; store notes in the run dict; include them in the downloaded JSON |

#### Larger features — estimated several days

| Idea | Notes |
|------|-------|
| Async optimizer runs | Offload each optimizer to `concurrent.futures.ThreadPoolExecutor`; use `st.status()` for live per-optimizer progress; allow multiple optimizers to run in parallel |
| Hyperparameter sweep | For MIPROv2 / Optuna optimizers, add a sweep mode with range inputs; plot score vs. LLM-call cost as a pareto scatter chart |
| RAG pipeline mode | Add a second module type that chains `dspy.Retrieve` + `dspy.ChainOfThought`; let the user configure a retriever (BM25 via `bm25s`, ChromaDB); optimize the full pipeline |
| Export to OpenAI / LangChain format | Add download buttons that convert the optimized prompt + demos to an OpenAI `messages` array or a LangChain `ChatPromptTemplate` |
| Hugging Face Hub push | Add a "Push to Hub" button that uploads the prompt JSON as a HF dataset card using `huggingface_hub` |
| Multi-step pipeline builder | UI to chain multiple Predict/ChainOfThought nodes; serialize the graph; build the composed `dspy.Module` dynamically |

#### Known bugs / limitations to fix

| Bug | Location |
|-----|----------|
| GEPA `reflection_lm` should ideally be a separate, stronger model — add a sidebar picker for it | `dispatch_optimizer()` ~line 202 |
| Cost estimator is based on heuristics — instrument LiteLLM callbacks for actual token counts | `_estimate_llm_calls()` ~line 335 |
| Very large CSVs (>50 MB) hit Streamlit's upload limit — add a "load from local file path" text input as an alternative | `_df_uploader()` ~line 382 |
| `BootstrapFewShot` passes `**p` directly to the constructor — any unexpected key from the params dict will crash it | `dispatch_optimizer()` ~line 144 |

---

### Code style guidelines

- **No comments explaining what the code does** — good names do that. Only comment when explaining *why* something non-obvious is done (a DSPy quirk, a workaround, a hidden constraint).
- **Don't add abstractions for hypothetical future use** — if you only need one optimizer to behave differently, add a targeted `if` rather than a new class hierarchy.
- **Keep the single-file structure** — `app.py` is intentionally self-contained. Don't split into modules unless a feature genuinely requires it (e.g., a separate retriever module for RAG).
- **All new packages go in `requirements.txt`** with a `>=` lower bound, not an exact pin.
- **Never commit API keys, `.env` files, or personal CSV data.** The `.gitignore` blocks `*.csv` and `.env` already, but double-check before pushing.

---

## Show your support

If this project saved you time or helped you learn DSPy, please consider giving it a star — it helps others discover it and motivates continued development.

[![Star this repo](https://img.shields.io/github/stars/Laxminarayen/dspy-prompt-optimizer?style=social)](https://github.com/Laxminarayen/dspy-prompt-optimizer/stargazers)

You can also help by:
- Sharing it with someone who works on LLM prompt engineering
- Opening an issue if you find a bug or have a feature request
- Submitting a pull request (see the Contributing section above)

---

## License

MIT
