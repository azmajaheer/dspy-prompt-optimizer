import streamlit as st
import pandas as pd
import dspy
import json
from io import StringIO
from datetime import datetime, timezone
import traceback
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# Model Pricing & Token Tracking
# ══════════════════════════════════════════════════════════════════════════════

# Pricing in USD per 1M tokens (input, output)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic
    "claude-opus-4-8": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # Groq
    "llama-3.1-8b-instant": (0.0, 0.0),  # Free tier
    # Default fallback
    "default": (1.00, 2.00),
}

def get_model_pricing(model_str: str) -> tuple[float, float]:
    """Extract pricing (input_price, output_price) per 1M tokens."""
    # Try exact match first
    if model_str in MODEL_PRICING:
        return MODEL_PRICING[model_str]
    
    # Try partial match (e.g., "gpt-4o-mini" from "openai/gpt-4o-mini")
    for key, value in MODEL_PRICING.items():
        if key in model_str.lower():
            return value
    
    return MODEL_PRICING["default"]

def calculate_cost(input_tokens: int, output_tokens: int, pricing: tuple[float, float]) -> float:
    """Calculate cost in USD given token counts and pricing."""
    input_price, output_price = pricing
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return round(cost, 4)

def get_token_usage() -> dict:
    """Extract token usage from dspy.settings.litellm_usage_logs."""
    usage_data = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }
    
    try:
        # LiteLLM logs usage in dspy.settings.litellm_usage_logs
        if hasattr(dspy.settings, "litellm_usage_logs"):
            logs = dspy.settings.litellm_usage_logs
            if logs and isinstance(logs, list):
                for log in logs:
                    if isinstance(log, dict):
                        usage_data["input_tokens"] += log.get("prompt_tokens", 0)
                        usage_data["output_tokens"] += log.get("completion_tokens", 0)
                        usage_data["calls"] += 1
                usage_data["total_tokens"] = usage_data["input_tokens"] + usage_data["output_tokens"]
    except Exception:
        pass
    
    return usage_data


# ══════════════════════════════════════════════════════════════════════════════
# Helpers  (must be defined before any tab code runs)
# ══════════════════════════════════════════════════════════════════════════════

def build_result_json(compiled, optimizer_name, sig_str, task_desc, params, metric_type):
    """Build a human-readable + machine-readable JSON of the optimized prompt."""

    result = {
        "meta": {
            "optimizer":        optimizer_name,
            "signature":        sig_str,
            "task_description": task_desc,
            "metric":           metric_type,
            "optimizer_params": {
                k: (int(v) if hasattr(v, "__index__") else v) for k, v in params.items()
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "token_usage": {},  # Will be populated after run
            "cost_usd": 0.0,    # Will be populated after run
        },
        "predictors": {},
    }

    try:
        # ── Per-predictor structured view ─────────────────────────────────
        for pname, predictor in compiled.named_predictors():
            sig = predictor.signature

            fields_meta = {}
            for fname, field in sig.fields.items():
                role = "input" if fname in sig.input_fields else "output"
                extra = field.json_schema_extra or {}
                desc  = extra.get("desc", "")
                # replace DSPy placeholder with empty string
                if desc and desc.startswith("${"):
                    desc = ""
                fields_meta[fname] = {
                    "role":        role,
                    "prefix":      extra.get("prefix", fname.capitalize() + ":"),
                    "description": desc,
                }

            instructions = ""
            try:
                instructions = sig.instructions or ""
            except Exception:
                pass

            demos = []
            try:
                for demo in (predictor.demos or []):
                    if hasattr(demo, "_store"):
                        demos.append({k: str(v) for k, v in demo._store.items()})
                    elif isinstance(demo, dict):
                        demos.append({k: str(v) for k, v in demo.items()})
            except Exception:
                pass

            result["predictors"][pname] = {
                "signature": {
                    "instructions": instructions,
                    "fields":       fields_meta,
                },
                "few_shot_examples": demos,
            }

        # ── Raw dump_state for completeness ───────────────────────────────
        try:
            result["raw_state"] = compiled.dump_state()
        except Exception:
            pass

    except Exception as exc:
        result["extraction_error"] = str(exc)

    return result


def create_metric_fn(metric_type, target_col):
    """Return a DSPy-compatible metric function based on user choice."""

    def _get(obj, key):
        if hasattr(obj, key):
            return str(getattr(obj, key))
        if hasattr(obj, "_store") and key in obj._store:
            return str(obj._store[key])
        return ""

    if metric_type == "Exact Match (case-insensitive)":
        def metric(example, prediction, trace=None):
            return _get(example, target_col).lower().strip() == \
                   _get(prediction, target_col).lower().strip()

    elif metric_type == "Contains Answer":
        def metric(example, prediction, trace=None):
            gold = _get(example, target_col).lower().strip()
            pred = _get(prediction, target_col).lower().strip()
            return gold in pred or pred in gold

    elif metric_type == "F1 Token Overlap":
        def metric(example, prediction, trace=None):
            g = set(_get(example, target_col).lower().split())
            p = set(_get(prediction, target_col).lower().split())
            if not g or not p:
                return 0.0
            tp   = len(g & p)
            prec = tp / len(p)
            rec  = tp / len(g)
            return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    else:  # Always True
        def metric(example, prediction, trace=None):
            return True

    return metric


def make_dspy_module(sig_str: str, task_d: str = ""):
    """Return a fresh DynamicModule instance for the given signature."""
    class _DynMod(dspy.Module):
        def __init__(self):
            super().__init__()
            sig_obj = dspy.Signature(sig_str, instructions=task_d) if task_d else sig_str
            self.predict = dspy.Predict(sig_obj)

        def forward(self, **kwargs):
            return self.predict(**kwargs)

    return _DynMod()


def dispatch_optimizer(opt_name: str, p: dict, module, trainset: list, devset: list, metric):
    """Compile one DSPy optimizer and return the compiled program.
    Must be called inside a dspy.context(lm=...) block.
    p is consumed (popped) by this function — pass a copy.
    """
    if opt_name == "BootstrapFewShot":
        opt = dspy.BootstrapFewShot(metric=metric, **p)
        return opt.compile(module, trainset=trainset)

    elif opt_name == "BootstrapFewShotWithRandomSearch":
        opt = dspy.BootstrapFewShotWithRandomSearch(metric=metric, **p)
        return opt.compile(module, trainset=trainset, valset=devset)

    elif opt_name == "MIPROv2":
        num_trials = int(p.pop("num_trials", 10))
        auto_val   = p.pop("auto", "light")
        n_cand     = int(p.pop("num_candidates", 5))
        mbd        = int(p.pop("max_bootstrapped_demos", 4))
        mld        = int(p.pop("max_labeled_demos", 4))
        use_auto   = auto_val in ("light", "medium", "heavy")
        opt = dspy.MIPROv2(
            metric=metric,
            auto=auto_val if use_auto else None,
            **({"num_candidates": n_cand} if not use_auto else {}),
            verbose=False,
        )
        return opt.compile(
            module,
            trainset=trainset,
            valset=devset,
            **({"num_trials": num_trials} if not use_auto else {}),
            max_bootstrapped_demos=mbd,
            max_labeled_demos=mld,
            requires_permission_to_run=False,
            minibatch=False,
        )

    elif opt_name == "COPRO":
        opt = dspy.COPRO(
            metric=metric,
            depth=int(p.pop("depth", 3)),
            breadth=int(p.pop("breadth", 10)),
            init_temperature=float(p.pop("init_temperature", 1.4)),
            verbose=False,
        )
        return opt.compile(module, trainset=trainset, eval_kwargs={})

    elif opt_name == "BootstrapFewShotWithOptuna":
        max_demos = int(p.pop("max_bootstrapped_demos", 4))
        mld       = int(p.pop("max_labeled_demos", 16))
        n_cands   = int(p.pop("num_candidate_programs", 16))
        opt = dspy.BootstrapFewShotWithOptuna(
            metric=metric,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=mld,
            num_candidate_programs=n_cands,
        )
        return opt.compile(module, trainset=trainset, valset=devset, max_demos=max_demos)

    elif opt_name == "LabeledFewShot":
        opt = dspy.LabeledFewShot(k=int(p.pop("k", 4)))
        return opt.compile(module, trainset=trainset)

    elif opt_name == "GEPA":
        auto_val   = p.pop("auto", "light")
        mini_batch = int(p.pop("reflection_minibatch_size", 3))

        # GEPA metric protocol: (gold, pred, trace, pred_name, pred_trace) -> float
        # Extra args pred_name/pred_trace are optional for score-only metrics.
        def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            return float(metric(gold, pred, trace))

        opt = dspy.GEPA(
            metric=gepa_metric,
            auto=auto_val,
            reflection_minibatch_size=mini_batch,
            reflection_lm=dspy.settings.lm,
        )
        return opt.compile(module, trainset=trainset, valset=devset)

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def evaluate_compiled(compiled, devset: list, metric) -> float | None:
    """Score a compiled program on devset; returns None on failure."""
    try:
        ev  = dspy.Evaluate(devset=devset, metric=metric, num_threads=1,
                            display_progress=False, display_table=False)
        res = ev(compiled)
        return round(float(res.score), 2)
    except Exception:
        return None


def record_run(opt_name, compiled, eval_score, trainset, devset,
               metric_t, sig_str, task_d, params):
    """Build result_json and append a run entry to session state."""
    rj = build_result_json(compiled, opt_name, sig_str, task_d, params, metric_t)
    rj["meta"]["eval_score"] = eval_score
    rj["meta"]["n_train"]    = len(trainset)
    rj["meta"]["n_eval"]     = len(devset)

    st.session_state["optimization_runs"].append({
        "optimizer":   opt_name,
        "score":       eval_score,
        "n_train":     len(trainset),
        "n_eval":      len(devset),
        "metric":      metric_t,
        "result_json": rj,
        "compiled":    compiled,
        "devset":      devset,
        "timestamp":   datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        "token_usage": {},  # Will be populated after run
        "cost_usd":    0.0,  # Will be populated after run
    })
    st.session_state["compiled_program"] = compiled
    st.session_state["result_json"]      = rj
    st.session_state["optimization_done"] = True
    return rj


# ══════════════════════════════════════════════════════════════════════════════
# Page setup
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="DSPy Prompt Optimizer", layout="wide", page_icon="⚡")
st.title("⚡ DSPy Prompt Optimizer")
st.caption("Upload a dataset · define X and Target · choose an optimizer · get your final optimized prompt JSON.")

# Session state defaults
for _k in ["df", "df_test", "x_cols", "target_col", "task_desc", "metric_type",
           "selected_optimizer", "optimizer_params", "train_ratio",
           "compiled_program", "result_json", "optimization_done", "lm_instance"]:
    if _k not in st.session_state:
        st.session_state[_k] = None

if "optimization_runs" not in st.session_state:
    st.session_state["optimization_runs"] = []   # list of run dicts, one per optimizer run


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar – LLM Configuration
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ LLM Configuration")

    provider = st.selectbox(
        "Provider",
        ["OpenAI", "Anthropic", "Gemini", "Mistral", "Groq", "Together AI", "Custom (OpenAI-compatible)"],
    )
    api_key = st.text_input("API Key", type="password", placeholder="sk-...")

    base_url_val = None
    if provider == "OpenAI":
        model       = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])
        lm_model_id = f"openai/{model}"
    elif provider == "Anthropic":
        model = st.selectbox("Model", [
            "claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
            "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307",
        ])
        lm_model_id = f"anthropic/{model}"
    elif provider == "Gemini":
        model = st.selectbox("Model", ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"])
        lm_model_id = f"gemini/{model}"
    elif provider == "Mistral":
         model = st.selectbox("Model", [
        "mistral-large-latest",
        "mistral-medium",
        "mistral-small",
        "open-mixtral-8x7b"
    ])
         lm_model_id = f"mistral/{model}"
    elif provider == "Groq":
        model = st.selectbox("Model", [
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ])
        lm_model_id = f"groq/{model}"
    elif provider == "Together AI":
        model       = st.text_input("Model name", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        lm_model_id = f"together_ai/{model}"
    else:
        base_url_val = st.text_input("Base URL", "http://localhost:11434/v1")
        model        = st.text_input("Model name", "llama3")
        lm_model_id  = f"openai/{model}"

    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    max_tokens  = st.number_input("Max Tokens", 64, 8192, 1000, step=64)

    st.divider()
    st.caption("Steps:  1 Dataset  →  2 Variables  →  3 Optimization  →  4 Results")

lm_ready = bool(api_key)


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(
    ["📁  Dataset", "🎯  Variables", "⚡  Optimization", "📊  Results"]
)


# ── Tab 1: Dataset ─────────────────────────────────────────────────────────────
# ── Dataset guardrail constants ───────────────────────────────────────────────
MAX_DISPLAY_ROWS  = 100    # rows shown in the dataframe preview
WARN_UPLOAD_ROWS  = 500    # warn the user above this many rows
MAX_TRAIN_DEFAULT = 200    # default training cap sent to optimizers
MAX_EVAL_DEFAULT  = 150    # default evaluation cap


def _estimate_llm_calls(opt_name: str, n_train: int, n_val: int,
                         n_eval: int, params: dict) -> str:
    """Return a rough human-readable LLM call estimate for one optimizer run."""
    try:
        if opt_name == "LabeledFewShot":
            total = n_eval
        elif opt_name == "BootstrapFewShot":
            boots = int(params.get("max_bootstrapped_demos", 4))
            total = n_train * boots + n_eval
        elif opt_name == "BootstrapFewShotWithRandomSearch":
            boots = int(params.get("max_bootstrapped_demos", 4))
            n_cand = int(params.get("num_candidate_programs", 10))
            total = n_train * boots + n_cand * n_val + n_eval
        elif opt_name == "MIPROv2":
            auto  = params.get("auto", "light")
            trials = {"light": 7, "medium": 20, "heavy": 50}.get(auto, int(params.get("num_trials", 10)))
            n_cand = int(params.get("num_candidates", 5))
            total = n_cand * n_train + trials * min(n_val, 35) + n_eval
        elif opt_name == "COPRO":
            depth   = int(params.get("depth", 3))
            breadth = int(params.get("breadth", 10))
            total = depth * breadth * n_train + n_eval
        elif opt_name == "BootstrapFewShotWithOptuna":
            boots  = int(params.get("max_bootstrapped_demos", 4))
            trials = int(params.get("num_candidate_programs", 16))
            total  = n_train * boots + trials * n_val + n_eval
        elif opt_name == "GEPA":
            auto  = params.get("auto", "light")
            # rough: each full eval = forward pass on a fraction of trainset;
            # auto presets map to ~10 / 25 / 60 full evals respectively
            full_evals = {"light": 10, "medium": 25, "heavy": 60}.get(auto, 15)
            total = full_evals * n_train + n_eval
        else:
            total = n_train + n_eval
        if total < 50:
            label = "low"
        elif total < 300:
            label = "moderate"
        elif total < 1000:
            label = "high"
        else:
            label = "very high"
        return f"~{total:,} calls ({label})"
    except Exception:
        return "unknown"


def _df_uploader(key_prefix: str, label: str, example_df=None, example_label="") -> pd.DataFrame | None:
    """Reusable dataset uploader widget. Returns a DataFrame or None."""
    method = st.radio(
        "Input method",
        ["Upload CSV", "Paste CSV text"] + (["Use example dataset"] if example_df is not None else []),
        horizontal=True,
        key=f"{key_prefix}_method",
    )
    df_out = None
    if method == "Upload CSV":
        f = st.file_uploader(f"Upload {label} CSV", type="csv", key=f"{key_prefix}_upload")
        if f:
            df_out = pd.read_csv(f)
    elif method == "Paste CSV text":
        raw = st.text_area("Paste CSV content", height=180,
                           placeholder="question,context,answer\n...",
                           key=f"{key_prefix}_paste")
        if raw.strip():
            try:
                df_out = pd.read_csv(StringIO(raw))
            except Exception as e:
                st.error(f"Could not parse: {e}")
    elif example_df is not None:
        df_out = example_df
        st.info(example_label)

    # ── Guardrail: large file warning + sampling control ─────────────────────
    if df_out is not None and len(df_out) > WARN_UPLOAD_ROWS:
        st.warning(
            f"Large dataset detected: **{len(df_out):,} rows**. "
            f"Using all rows during optimization means potentially thousands of LLM calls. "
            f"Set a row cap below."
        )
        cap = st.number_input(
            f"Row cap for {label} set",
            min_value=50,
            max_value=len(df_out),
            value=min(500, len(df_out)),
            step=50,
            key=f"{key_prefix}_cap",
            help="Rows are sampled randomly (reproducible seed). Originals not modified.",
        )
        if cap < len(df_out):
            df_out = df_out.sample(n=cap, random_state=42).reset_index(drop=True)
            st.info(f"Sampled **{cap}** rows from {len(df_out) + (len(df_out) - cap):,}-row file.")

    return df_out


def _df_stats(df: pd.DataFrame):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing", int(df.isnull().sum().sum()))


with tab1:
    st.header("1.  Provide Datasets")
    st.caption(
        "Upload a **separate train and test set** for honest evaluation. "
        "If you only have one file, use the auto-split option in the Optimization tab."
    )

    EXAMPLE_TRAIN = pd.DataFrame({
        "question": [
            "What is the capital of France?", "What is 2 + 2?",
            "Who wrote Romeo and Juliet?", "What is the boiling point of water in Celsius?",
            "What color is the sky on a clear day?", "How many continents are on Earth?",
            "What is the largest planet in the solar system?", "Who painted the Mona Lisa?",
            "What is the chemical symbol for gold?", "In what year did World War II end?",
        ],
        "context": [
            "European geography", "Basic arithmetic", "English literature",
            "Physics / chemistry", "Natural phenomena", "World geography",
            "Astronomy", "Renaissance art", "Chemistry / periodic table",
            "Modern world history",
        ],
        "answer": [
            "Paris", "4", "William Shakespeare", "100", "Blue",
            "7", "Jupiter", "Leonardo da Vinci", "Au", "1945",
        ],
    })

    EXAMPLE_TEST = pd.DataFrame({
        "question": [
            "What is the speed of light in km/s (approx)?",
            "Who developed the theory of relativity?",
            "What is the smallest prime number?",
            "Which ocean is the largest?",
            "What gas do plants absorb from the atmosphere?",
        ],
        "context": ["Physics", "Physics", "Mathematics", "Geography", "Biology"],
        "answer":  ["300000", "Albert Einstein", "2", "Pacific", "Carbon dioxide"],
    })

    left, right = st.columns(2)

    with left:
        st.subheader("Training Data  (used for optimization)")
        df_train_loaded = _df_uploader(
            "train", "training",
            example_df=EXAMPLE_TRAIN,
            example_label="Using built-in example training set (10 rows).",
        )
        if df_train_loaded is not None:
            st.session_state["df"] = df_train_loaded
            st.success(f"Training set ready — **{len(df_train_loaded)} rows**")
            st.dataframe(df_train_loaded, use_container_width=True, height=220)
            _df_stats(df_train_loaded)

    with right:
        st.subheader("Test / Evaluation Data  (held-out scoring only)")
        df_test_loaded = _df_uploader(
            "test", "test",
            example_df=EXAMPLE_TEST,
            example_label="Using built-in example test set (5 rows).",
        )
        if df_test_loaded is not None:
            st.session_state["df_test"] = df_test_loaded
            st.success(f"Test set ready — **{len(df_test_loaded)} rows**")
            st.dataframe(df_test_loaded, use_container_width=True, height=220)
            _df_stats(df_test_loaded)
        else:
            st.info("No separate test file — the Optimization tab will auto-split the training data.")


# ── Tab 2: Variables ───────────────────────────────────────────────────────────
with tab2:
    st.header("2.  Define Input Features (X) and Target (Y)")

    if st.session_state["df"] is None:
        st.warning("Upload a dataset on the **Dataset** tab first.")
    else:
        df_v    = st.session_state["df"]
        columns = df_v.columns.tolist()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Input Features  (X)")
            x_cols = st.multiselect(
                "Select input columns",
                columns,
                default=columns[:-1] if len(columns) > 1 else columns,
            )
        with col_b:
            st.subheader("Target Variable  (Y)")
            available  = [c for c in columns if c not in x_cols] or columns
            target_col = st.selectbox("Select target column", available)

        if x_cols and target_col:
            st.code("DSPy Signature:   " + ", ".join(x_cols) + " -> " + target_col,
                    language="text")

            task_desc = st.text_area(
                "Task description / instructions (optional)",
                placeholder="e.g. 'Given the question and context, answer concisely and accurately.'",
                height=90,
            )

            st.subheader("Evaluation Metric")
            metric_type = st.selectbox(
                "Metric used during optimization",
                [
                    "Exact Match (case-insensitive)",
                    "Contains Answer",
                    "F1 Token Overlap",
                    "Always True (unsupervised / instruction tuning only)",
                ],
                help=(
                    "This metric scores each prediction during optimization. "
                    "Choose what best matches your task."
                ),
            )

            st.session_state["x_cols"]      = x_cols
            st.session_state["target_col"]  = target_col
            st.session_state["task_desc"]   = task_desc
            st.session_state["metric_type"] = metric_type


# ── Tab 3: Optimization ────────────────────────────────────────────────────────
with tab3:
    st.header("3.  Select DSPy Optimizer & Run")

    # Optimizer catalog
    OPTIMIZERS: dict = {
        "BootstrapFewShot": {
            "desc": (
                "Teacher-student bootstrapping. Runs the training examples through a teacher "
                "program, keeps successful traces as few-shot demonstrations, and uses them "
                "to guide the student. Fast and effective for small datasets."
            ),
            "params": {
                "max_bootstrapped_demos": dict(type="int", default=4, min=1, max=20,
                                               help="Max teacher-generated demos per predictor."),
                "max_labeled_demos":      dict(type="int", default=16, min=0, max=50,
                                               help="Max labeled demos drawn directly from the training set."),
            },
            "needs_valset": False,
        },
        "BootstrapFewShotWithRandomSearch": {
            "desc": (
                "Extends BootstrapFewShot with a random search over many candidate programs. "
                "Tries `num_candidate_programs` configurations and returns the one that scores "
                "highest on the validation set."
            ),
            "params": {
                "max_bootstrapped_demos": dict(type="int", default=4,  min=1, max=20,
                                               help="Max bootstrapped demos."),
                "max_labeled_demos":      dict(type="int", default=16, min=0, max=50,
                                               help="Max labeled demos."),
                "num_candidate_programs": dict(type="int", default=10, min=1, max=50,
                                               help="Number of candidate programs to evaluate."),
            },
            "needs_valset": True,
        },
        "MIPROv2": {
            "desc": (
                "Multi-prompt Instruction Proposal and Optimization v2. State-of-the-art optimizer "
                "that uses an LM to propose candidate instructions, then searches over them via "
                "Bayesian optimization. Jointly optimizes instructions and few-shot examples."
            ),
            "params": {
                "auto":                   dict(type="select", default="light",
                                               options=["light", "medium", "heavy", "None (manual)"],
                                               help="Budget preset sets num_candidates & num_trials automatically. "
                                                    "Choose 'None (manual)' to set them yourself."),
                "num_candidates":         dict(type="int", default=5,  min=2,  max=30,
                                               help="[Only used when auto=None] Candidate instructions to propose."),
                "max_bootstrapped_demos": dict(type="int", default=4,  min=0,  max=20,
                                               help="Max bootstrapped demos per predictor."),
                "max_labeled_demos":      dict(type="int", default=4,  min=0,  max=20,
                                               help="Max labeled demos per predictor."),
                "num_trials":             dict(type="int", default=10, min=1,  max=100,
                                               help="[Only used when auto=None] Number of Bayesian optimization trials."),
            },
            "needs_valset": True,
        },
        "COPRO": {
            "desc": (
                "Cooperative Prompt Optimization. Uses coordinate ascent: the LM proposes "
                "candidate instructions, evaluates them on the training set, and iteratively "
                "refines. Does not add few-shot demos — optimizes instructions only."
            ),
            "params": {
                "depth":            dict(type="int",   default=3,   min=1,   max=10,
                                         help="Number of optimization iterations."),
                "breadth":          dict(type="int",   default=10,  min=2,   max=30,
                                         help="Candidate instructions proposed per iteration."),
                "init_temperature": dict(type="float", default=1.4, min=0.1, max=2.0,
                                         help="Sampling temperature for instruction proposals."),
            },
            "needs_valset": False,
        },
        "BootstrapFewShotWithOptuna": {
            "desc": (
                "Uses Optuna's Bayesian hyperparameter search to find the best combination of "
                "bootstrapped few-shot demonstrations over many trials."
            ),
            "params": {
                "max_bootstrapped_demos": dict(type="int", default=4,  min=1, max=20,
                                               help="Max bootstrapped demos (also used as max_demos in compile)."),
                "max_labeled_demos":      dict(type="int", default=16, min=0, max=50,
                                               help="Max labeled demos."),
                "num_candidate_programs": dict(type="int", default=16, min=4, max=100,
                                               help="Number of Optuna trials."),
            },
            "needs_valset": True,
        },
        "LabeledFewShot": {
            "desc": (
                "Simple baseline — selects k labeled examples directly from the training set "
                "as few-shot demonstrations. No bootstrapping, no instruction search. "
                "Useful as a sanity-check baseline."
            ),
            "params": {
                "k": dict(type="int", default=4, min=1, max=20,
                          help="Number of labeled examples to include as demonstrations."),
            },
            "needs_valset": False,
        },
        "GEPA": {
            "desc": (
                "[Experimental] Generative Evolution of Prompt Artefacts — an evolutionary optimizer "
                "that uses LM-driven reflection to iteratively improve instructions. "
                "State-of-the-art for complex tasks; more expensive than bootstrap-based methods."
            ),
            "params": {
                "auto": dict(
                    type="select", default="light",
                    options=["light", "medium", "heavy"],
                    help="Budget preset controlling number of evolutionary evaluations.",
                ),
                "reflection_minibatch_size": dict(
                    type="int", default=3, min=1, max=20,
                    help="Number of examples used per reflection/feedback step.",
                ),
            },
            "needs_valset": True,
        },
    }

    if st.session_state["x_cols"] is None:
        st.warning("Define variables on the **Variables** tab first.")
    else:
        all_opt_names = list(OPTIMIZERS.keys())

        # ── Mode selector ─────────────────────────────────────────────────────
        mode = st.radio(
            "Mode",
            ["Single optimizer  (custom parameters)", "Compare multiple optimizers  (default parameters)"],
            horizontal=True,
        )
        compare_mode = mode.startswith("Compare")

        st.divider()

        # ════════════════════════════════════════════════════════════════════
        # SINGLE MODE
        # ════════════════════════════════════════════════════════════════════
        if not compare_mode:
            selected = st.selectbox("Optimizer", all_opt_names)
            opt_cfg  = OPTIMIZERS[selected]
            st.info(f"**{selected}** — {opt_cfg['desc']}")

            st.subheader("Parameters")
            params: dict = {}
            pcols = st.columns(2)
            for idx, (pname, pmeta) in enumerate(opt_cfg["params"].items()):
                with pcols[idx % 2]:
                    if pmeta["type"] == "int":
                        params[pname] = st.number_input(
                            pname, min_value=pmeta["min"], max_value=pmeta["max"],
                            value=pmeta["default"], help=pmeta.get("help", ""),
                        )
                    elif pmeta["type"] == "float":
                        params[pname] = st.slider(
                            pname, min_value=float(pmeta["min"]),
                            max_value=float(pmeta["max"]), value=float(pmeta["default"]),
                            step=0.1, help=pmeta.get("help", ""),
                        )
                    elif pmeta["type"] == "select":
                        params[pname] = st.selectbox(
                            pname, pmeta["options"],
                            index=pmeta["options"].index(pmeta["default"]),
                            help=pmeta.get("help", ""),
                        )

            optimizers_to_run   = [selected]
            params_per_optimizer = {selected: params}

        # ════════════════════════════════════════════════════════════════════
        # COMPARE MODE
        # ════════════════════════════════════════════════════════════════════
        else:
            st.markdown("Select which optimizers to run. All will use their **default parameters**.")

            select_all = st.checkbox("Select all 6 optimizers", value=False)
            chosen = st.multiselect(
                "Optimizers to compare",
                all_opt_names,
                default=all_opt_names if select_all else ["LabeledFewShot", "BootstrapFewShot"],
                help="Each optimizer runs in sequence on the same train/val split.",
            )
            if not chosen:
                st.warning("Select at least one optimizer.")

            # Show default params as a read-only summary
            if chosen:
                with st.expander("Default parameters that will be used", expanded=False):
                    for oname in chosen:
                        st.markdown(f"**{oname}**")
                        defaults_md = "  ".join(
                            f"`{k}={v['default']}`"
                            for k, v in OPTIMIZERS[oname]["params"].items()
                        )
                        st.markdown(defaults_md or "*(no configurable params)*")

            optimizers_to_run    = chosen
            params_per_optimizer = {
                oname: {pname: pmeta["default"]
                        for pname, pmeta in OPTIMIZERS[oname]["params"].items()}
                for oname in chosen
            }

        # ── Shared: data split ────────────────────────────────────────────────
        st.subheader("Data Split / Evaluation")
        df_sz      = st.session_state["df"]
        df_test_sz = st.session_state.get("df_test")

        if df_test_sz is not None:
            train_ratio = 1.0
            st.success(
                f"Using separate test set — "
                f"Train: **{len(df_sz)}** rows  |  Test: **{len(df_test_sz)}** rows"
            )
            st.caption("Scores evaluated on the held-out test file only.")
        else:
            train_ratio = st.slider("Training set ratio (auto-split)", 0.5, 0.95, 0.8, 0.05)
            n_tr  = int(len(df_sz) * train_ratio)
            n_val = len(df_sz) - n_tr
            st.caption(f"Train+Val: **{n_tr}** rows  |  Held-out eval: **{n_val}** rows")
            if n_val < 20:
                st.warning(
                    f"Only **{n_val}** eval examples — scores may not be meaningful. "
                    "Upload a separate test CSV in the Dataset tab for honest evaluation."
                )

        # ── Guardrails: training + eval caps ─────────────────────────────────
        st.subheader("Size Guardrails")
        total_train_avail = len(df_sz)
        total_eval_avail  = len(df_test_sz) if df_test_sz is not None else max(1, int(total_train_avail * (1 - train_ratio)))

        ga, gb = st.columns(2)
        with ga:
            _train_min = min(10, total_train_avail)
            max_train = st.number_input(
                "Max training rows sent to optimizer",
                min_value=_train_min,
                max_value=total_train_avail,
                value=min(MAX_TRAIN_DEFAULT, total_train_avail),
                step=10,
                help=(
                    "DSPy optimizers make 1–50 LLM calls per training example. "
                    "Keep this ≤ 300 to avoid runaway costs. "
                    "Rows are sampled randomly (seed=42)."
                ),
            )
        with gb:
            _eval_min = min(10, total_eval_avail)
            max_eval = st.number_input(
                "Max eval rows for scoring",
                min_value=_eval_min,
                max_value=total_eval_avail,
                value=min(MAX_EVAL_DEFAULT, total_eval_avail),
                step=10,
                help=(
                    "Each eval row = 1 LLM call per optimizer run. "
                    "50–200 rows gives reliable scores without excessive cost."
                ),
            )

        if total_train_avail > MAX_TRAIN_DEFAULT or total_eval_avail > MAX_EVAL_DEFAULT:
            st.info(
                f"Your dataset has **{total_train_avail:,}** training rows and "
                f"**{total_eval_avail:,}** eval rows. "
                f"Capped at **{int(max_train)}** train / **{int(max_eval)}** eval "
                f"(edit the caps above)."
            )

        # ── Cost estimate ─────────────────────────────────────────────────────
        if optimizers_to_run:
            st.subheader("Estimated LLM Calls")
            est_rows = []
            for oname in optimizers_to_run:
                p_est = params_per_optimizer.get(oname, {})
                est   = _estimate_llm_calls(oname, int(max_train), int(max_train * 0.2),
                                            int(max_eval), p_est)
                est_rows.append({"Optimizer": oname, "Estimated calls": est})
            st.dataframe(pd.DataFrame(est_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Estimates are approximate. Actual calls depend on caching, retries, "
                "and optimizer internals. Each call uses your API quota."
            )

        st.divider()

        # ── Run button ────────────────────────────────────────────────────────
        n_sel  = len(optimizers_to_run)
        btn_label = (
            f"🚀 Run Optimization"
            if not compare_mode else
            f"🚀 Run Comparison  ({n_sel} optimizer{'s' if n_sel != 1 else ''})"
        )

        if not lm_ready:
            st.error("Enter an API key in the sidebar before running.")
        elif not optimizers_to_run:
            st.warning("Select at least one optimizer above.")
        else:
            if st.button(btn_label, type="primary", use_container_width=True):
                # ── Build LM once ────────────────────────────────────────
                lm_kwargs: dict = dict(
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    api_key=api_key,
                )
                if base_url_val:
                    lm_kwargs["api_base"] = base_url_val
                lm = dspy.LM(lm_model_id, **lm_kwargs)
                st.session_state["lm_instance"] = lm

                # ── Build shared examples once ───────────────────────────
                df_run      = st.session_state["df"]
                df_test_run = st.session_state.get("df_test")
                x_cols_r    = st.session_state["x_cols"]
                target_r    = st.session_state["target_col"]
                task_d      = st.session_state["task_desc"] or ""
                metric_t    = st.session_state["metric_type"]
                sig_str     = ", ".join(x_cols_r) + " -> " + target_r
                cap_train   = int(max_train)
                cap_eval    = int(max_eval)

                def _to_examples(df_src, cap=None):
                    src = df_src
                    if cap and len(src) > cap:
                        src = src.sample(n=cap, random_state=42).reset_index(drop=True)
                    exs = []
                    for _, row in src.iterrows():
                        d = {c: str(row[c]) for c in src.columns if pd.notna(row[c])}
                        exs.append(dspy.Example(**d).with_inputs(*x_cols_r))
                    return exs

                all_train = _to_examples(df_run, cap=cap_train)

                if df_test_run is not None:
                    n_val_from_train = max(1, int(len(all_train) * 0.2))
                    trainset = all_train[:-n_val_from_train]
                    valset   = all_train[-n_val_from_train:]
                    evalset  = _to_examples(df_test_run, cap=cap_eval)
                else:
                    n = len(all_train)
                    n_tr_r  = max(1, int(n * train_ratio * 0.75))
                    n_val_r = max(1, int(n * train_ratio * 0.25))
                    trainset = all_train[:n_tr_r]
                    valset   = all_train[n_tr_r : n_tr_r + n_val_r]
                    evalset  = all_train[n_tr_r + n_val_r:]
                    # apply eval cap after splitting
                    if len(evalset) > cap_eval:
                        import random; rng = random.Random(42)
                        evalset = rng.sample(evalset, cap_eval)
                    if not evalset:
                        evalset = valset

                st.info(
                    f"Running with: **{len(trainset)}** train  |  "
                    f"**{len(valset)}** val (optimizer selection)  |  "
                    f"**{len(evalset)}** held-out eval"
                )
                metric = create_metric_fn(metric_t, target_r)

                # ── Score the unoptimized baseline ONCE before any optimizer runs ──
                with dspy.context(lm=lm):
                    base_score = evaluate_compiled(
                        make_dspy_module(sig_str, task_d), evalset, metric
                    )
                st.session_state["baseline_score"] = base_score
                st.info(
                    f"Baseline (no optimization): **{base_score if base_score is not None else 'N/A'}** "
                    f"on {len(evalset)} eval examples"
                )

                # ── Run each optimizer in sequence ───────────────────────
                status_box      = st.empty()
                results_so_far: list[dict] = []

                for run_idx, opt_name in enumerate(optimizers_to_run):
                    prog = st.progress(0, text=f"[{run_idx+1}/{n_sel}] Starting {opt_name}…")
                    try:
                        prog.progress(15, text=f"[{run_idx+1}/{n_sel}] {opt_name} — compiling…")
                        opt_params = dict(params_per_optimizer[opt_name])

                        # Reset token tracking before this optimizer run
                        if hasattr(dspy.settings, "litellm_usage_logs"):
                            dspy.settings.litellm_usage_logs = []

                        with dspy.context(lm=lm):
                            # valset is used by optimizers for candidate selection;
                            # evalset is the separate held-out set used only for scoring.
                            compiled = dispatch_optimizer(
                                opt_name, opt_params,
                                make_dspy_module(sig_str, task_d),
                                trainset, valset, metric,
                            )
                            prog.progress(80, text=f"[{run_idx+1}/{n_sel}] {opt_name} — scoring on held-out eval…")
                            eval_score = evaluate_compiled(compiled, evalset, metric)

                        # Collect diagnostics: demo count and whether instructions changed
                        n_demos    = 0
                        instr_changed = False
                        base_instr    = dspy.Signature(sig_str).instructions if not task_d else task_d
                        try:
                            for _, pred in compiled.named_predictors():
                                n_demos += len(pred.demos or [])
                                if pred.signature.instructions != base_instr:
                                    instr_changed = True
                        except Exception:
                            pass

                        prog.progress(95, text=f"[{run_idx+1}/{n_sel}] {opt_name} — saving…")
                        used_params = params_per_optimizer[opt_name]
                        rj = record_run(opt_name, compiled, eval_score,
                                        trainset, evalset, metric_t, sig_str, task_d, used_params)
                        rj["meta"]["n_demos"]       = n_demos
                        rj["meta"]["instr_changed"] = instr_changed
                        rj["meta"]["baseline_score"] = base_score
                        
                        # Capture token usage and calculate cost
                        token_usage = get_token_usage()
                        pricing = get_model_pricing(lm_model_id)
                        cost = calculate_cost(
                            token_usage["input_tokens"],
                            token_usage["output_tokens"],
                            pricing
                        )
                        
                        # Store token usage and cost in both session state and result JSON
                        rj["meta"]["token_usage"] = token_usage
                        rj["meta"]["cost_usd"] = cost
                        
                        st.session_state["optimization_runs"][-1]["token_usage"] = token_usage
                        st.session_state["optimization_runs"][-1]["cost_usd"] = cost
                        
                        # update the stored copy
                        st.session_state["optimization_runs"][-1]["result_json"] = rj
                        st.session_state["optimization_runs"][-1]["n_demos"]     = n_demos
                        st.session_state["optimization_runs"][-1]["instr_changed"] = instr_changed

                        delta = (
                            round(eval_score - base_score, 2)
                            if eval_score is not None and base_score is not None
                            else None
                        )
                        results_so_far.append({
                            "optimizer": opt_name, "score": eval_score, "delta": delta,
                            "tokens": token_usage["total_tokens"], "cost": cost
                        })
                        delta_str = f"  Δ baseline: {delta:+.2f}" if delta is not None else ""
                        cost_str = f"  Cost: ${cost:.4f}" if cost > 0 else ""
                        prog.progress(100, text=(
                            f"[{run_idx+1}/{n_sel}] {opt_name} — done  "
                            f"score: {eval_score}{delta_str}{cost_str}"
                        ))

                    except Exception:
                        prog.empty()
                        st.error(f"{opt_name} failed:")
                        st.code(traceback.format_exc())

                # ── Final summary ────────────────────────────────────────
                if results_so_far:
                    def _fmt_delta(d):
                        if d is None:
                            return "N/A"
                        return "±0" if d == 0 else f"{d:+.2f}"

                    summary = "  |  ".join(
                        f"**{r['optimizer']}**: {r['score']} ({_fmt_delta(r['delta'])}) | {r['tokens']} tokens | ${r['cost']:.4f}"
                        for r in results_so_far
                    )
                    status_box.success(f"✅ Done — {summary} — see **Results** tab.")


# ── Tab 4: Results ─────────────────────────────────────────────────────────────
with tab4:
    st.header("4.  Optimization Results")

    runs: list = st.session_state.get("optimization_runs", [])

    if not runs:
        st.info("Run optimization on the **Optimization** tab to see results here.")
    else:
        # ── Header controls ────────────────────────────────────────────────
        hcol1, hcol2 = st.columns([6, 1])
        with hcol1:
            st.caption(f"{len(runs)} optimization run(s) recorded.")
        with hcol2:
            if st.button("🗑  Clear all", help="Remove all recorded runs"):
                st.session_state["optimization_runs"] = []
                st.session_state["optimization_done"]  = None
                st.rerun()

        # ══════════════════════════════════════════════════════════════════
        # SECTION A – Comparison table & chart
        # ══════════════════════════════════════════════════════════════════
        st.subheader("Score Comparison")

        baseline_score = st.session_state.get("baseline_score")

        # Build comparison dataframe
        compare_rows = []
        total_cost = 0.0
        for idx, r in enumerate(runs):
            score_val = r["score"]
            delta_val = (
                round(score_val - baseline_score, 2)
                if score_val is not None and baseline_score is not None
                else None
            )
            token_usage = r.get("token_usage", {})
            cost = r.get("cost_usd", 0.0)
            total_cost += cost
            compare_rows.append({
                "Run #":           idx + 1,
                "Optimizer":       r["optimizer"],
                "Score":           score_val if score_val is not None else "—",
                "Δ Baseline":      (f"{delta_val:+.2f}" if delta_val is not None else "—"),
                "Input Tokens":    token_usage.get("input_tokens", 0),
                "Output Tokens":   token_usage.get("output_tokens", 0),
                "Total Tokens":    token_usage.get("total_tokens", 0),
                "Cost (USD)":      f"${cost:.4f}" if cost > 0 else "$0.00",
                "Demos added":     r.get("n_demos", "—"),
                "Instr. changed":  ("Yes" if r.get("instr_changed") else "No"),
                "Eval rows":       r["n_eval"],
                "Metric":          r["metric"],
                "Time (UTC)":      r["timestamp"],
            })
        df_compare = pd.DataFrame(compare_rows)

        # Baseline row
        if baseline_score is not None:
            st.metric("Baseline (no optimization)", f"{baseline_score}", help="Unoptimized module score on the same eval set")

        # Cost summary
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Cost (USD)", f"${total_cost:.4f}", help="Sum of all optimization runs")
        with c2:
            avg_cost = total_cost / len(runs) if runs else 0
            st.metric("Average Cost per Run", f"${avg_cost:.4f}")

        # Highlight best and flag identical scores
        numeric_scores = [r["score"] for r in runs if r["score"] is not None]
        best_score     = max(numeric_scores) if numeric_scores else None
        all_same       = len(set(numeric_scores)) == 1 and len(numeric_scores) > 1

        if all_same:
            st.warning(
                "**All optimizers returned the same score.** Possible causes:\n"
                "- **Eval set too small** — with < 20 examples a 1-example change = 5%+ jump; "
                "optimizers may all get the same subset right by chance.\n"
                "- **LLM already at ceiling** — the base model answers these examples correctly "
                "without any optimization. Try a harder task or stricter metric (Exact Match → F1).\n"
                "- **Valset = evalset leakage** — if the same data is used for optimizer candidate "
                "selection AND final scoring, all optimizers converge to the same score. "
                "Upload a **separate test CSV** in the Dataset tab to fix this.\n"
                "- **Too few training examples** — bootstrapping needs enough successes to find "
                "good few-shot demos. Try at least 50 training examples.\n"
                "- **Metric too coarse** — Exact Match on short answers gives 0 or 1 per example. "
                "Try **F1 Token Overlap** for more signal."
            )

        def _highlight_best(row):
            score = row["Score"]
            if best_score is not None and score == best_score and not all_same:
                return ["background-color: #d4edda; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_compare.style.apply(_highlight_best, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart with baseline reference
        chart_data = []
        if baseline_score is not None:
            chart_data.append({"label": "Baseline", "score": baseline_score})
        chart_data += [
            {"label": f"#{r_idx+1} {r['optimizer']}", "score": r["score"]}
            for r_idx, r in enumerate(runs) if r["score"] is not None
        ]
        if chart_data:
            df_chart = pd.DataFrame(chart_data).set_index("label")
            st.bar_chart(df_chart, y="score", y_label="Score", x_label="Run")
        else:
            st.caption("No numeric scores to chart (metric 'Always True' gives no signal).")

        st.divider()

        # ══════════════════════════════════════════════════════════════════
        # SECTION B – Per-run detailed view (user picks which run to inspect)
        # ══════════════════════════════════════════════════════════════════
        st.subheader("Inspect a Run")

        run_labels = [
            f"Run #{i+1} — {r['optimizer']}  (score: {r['score'] if r['score'] is not None else '—'})"
            for i, r in enumerate(runs)
        ]
        selected_run_idx = st.selectbox(
            "Select run to inspect",
            range(len(runs)),
            format_func=lambda i: run_labels[i],
            index=len(runs) - 1,   # default to most recent
        )
        run      = runs[selected_run_idx]
        rj       = run["result_json"]
        compiled = run["compiled"]

        # Score banner
        sc1, sc2, sc3, sc4 = st.columns(4)
        score_display = f"{run['score']:.2f}" if run["score"] is not None else "N/A"
        sc1.metric("Score",     score_display)
        sc2.metric("Eval rows", run["n_eval"])
        sc3.metric("Optimizer", run["optimizer"])
        sc4.metric("Metric",    run["metric"])

        # Token usage and cost banner
        token_usage = run.get("token_usage", {})
        cost_usd = run.get("cost_usd", 0.0)
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Input Tokens", f"{token_usage.get('input_tokens', 0):,}")
        tc2.metric("Output Tokens", f"{token_usage.get('output_tokens', 0):,}")
        tc3.metric("Total Tokens", f"{token_usage.get('total_tokens', 0):,}")
        tc4.metric("Cost (USD)", f"${cost_usd:.4f}")

        # ── Per-predictor details ──────────────────────────────────────────
        st.subheader("Optimized Predictors")
        for pname, pdata in rj["predictors"].items():
            with st.expander(f"🔵  Predictor: `{pname}`", expanded=True):
                instructions = pdata["signature"].get("instructions", "")
                st.markdown("**Optimized Instructions**")
                if instructions:
                    st.info(instructions)
                else:
                    st.caption("*(no custom instructions generated by this optimizer)*")

                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**Input Fields**")
                    for fname, fmeta in pdata["signature"]["fields"].items():
                        if fmeta["role"] == "input":
                            prefix = fmeta.get("prefix", fname + ":")
                            desc   = fmeta.get("description") or "*(auto)*"
                            st.markdown(f"- **`{fname}`** ({prefix}) — {desc}")
                with cb:
                    st.markdown("**Output Fields**")
                    for fname, fmeta in pdata["signature"]["fields"].items():
                        if fmeta["role"] == "output":
                            prefix = fmeta.get("prefix", fname + ":")
                            desc   = fmeta.get("description") or "*(auto)*"
                            st.markdown(f"- **`{fname}`** ({prefix}) — {desc}")

                demos = pdata.get("few_shot_examples", [])
                if demos:
                    st.markdown(f"**Few-Shot Examples  ({len(demos)})**")
                    for i, demo in enumerate(demos, 1):
                        with st.expander(f"Example {i}", expanded=False):
                            st.json(demo)
                else:
                    st.caption("No few-shot examples selected for this predictor.")

        # ── JSON export ────────────────────────────────────────────────────
        st.subheader("Final Prompt JSON")
        json_str = json.dumps(rj, indent=2, default=str)
        safe_name = run["optimizer"].replace(" ", "_").replace("(", "").replace(")", "")
        st.download_button(
            label=f"⬇️  Download  {safe_name}_prompt.json",
            data=json_str,
            file_name=f"run{selected_run_idx+1}_{safe_name}_prompt.json",
            mime="application/json",
            use_container_width=True,
        )
        st.json(rj, expanded=False)

        st.divider()

        # ── Live inference ─────────────────────────────────────────────────
        st.subheader("Test This Run")
        x_inf  = st.session_state["x_cols"]
        tgt_if = st.session_state["target_col"]

        test_vals: dict = {}
        for col in x_inf:
            test_vals[col] = st.text_input(f"Input — `{col}`", key=f"infer_{col}")

        if st.button("▶  Run Inference", use_container_width=True):
            if all(v.strip() for v in test_vals.values()):
                with st.spinner("Calling LLM…"):
                    try:
                        lm_inst = st.session_state.get("lm_instance")
                        with dspy.context(lm=lm_inst):
                            result = compiled(**test_vals)
                        output = getattr(result, tgt_if, str(result))
                        st.success(f"**{tgt_if}:** {output}")
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        st.code(traceback.format_exc())
            else:
                st.warning("Fill in all input fields before running inference.")
