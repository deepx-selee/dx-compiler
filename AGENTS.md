# DEEPX DX-COM Compiler — AI Coding Agents Entry Point

> Self-contained entry point for AI coding agents. Read this file first.

## Response Language

Match your response language to the user's prompt language — when asking questions
or responding, use the same language the user is using.

**Technical term rule**: When responding in Korean, keep English technical terms in
their original English form. Do NOT transliterate English terms into Korean phonetics
(한글 음차 표기 금지). Established Korean loanwords (모델, 서버, 파일, 데이터) are acceptable. <!-- KOREAN-OK: rule text intentionally references Korean notation terms that agents must recognize -->

## Recommended Model (Pre-Flight Check)

DX Agent-Driven Development requires strong instruction following and knowledge base
utilization. Before starting any task, check if you are running as one of the
recommended models:

- **Claude Sonnet 4.6** or later
- **Claude Opus 4.6** or later

If you are NOT one of these models (e.g., GPT-4.1, GPT-4o, Gemini, etc.),
output the following notice at the beginning of your first response:

```
══════════════════════════════════════════════════════════════
⚠  DX-AGENT-DEV: MODEL NOTICE

DX Agent-Driven Development recommends Claude Sonnet 4.6+ or
Opus 4.6+. Your current model may produce lower quality
in instruction following, knowledge base utilization, and
API accuracy. Please switch to a recommended model.
══════════════════════════════════════════════════════════════
```

After the box notice, proceed with the task normally. This is an informational
warning, not a blocking gate.

### Model Notice Repeat (Non-Recommended Model Only)

If you output the MODEL NOTICE box above (i.e., you are NOT a recommended model),
you MUST also output this shortened reminder **immediately before the actual question
line** that the user needs to answer — NOT at the beginning of the brainstorming flow.

**Timing**: After all file reading, context analysis, and preamble text, right before
the line that contains the `?` (the actual question), insert this reminder:

```
---
⚠ **Non-recommended model** — output quality may be degraded. Recommended: Claude Sonnet 4.6+ / Opus 4.6+
---
```

**Example — WRONG** (repeat scrolls past with the box):
```
[DX-AGENT-DEV: START]
══ MODEL NOTICE ══
---  ⚠ Non-recommended model ---     ← TOO EARLY, scrolls past
... (reads files, analyzes context) ...
First question: ...?
```

**Example — CORRECT** (repeat appears right before the question):
```
[DX-AGENT-DEV: START]
══ MODEL NOTICE ══
... (reads files, analyzes context) ...
---  ⚠ Non-recommended model ---     ← RIGHT BEFORE the question
First question: ...?
```

Only output this reminder ONCE (before the first question), not before every question.

## Skill Router — Universal Pre-Flight (HARD GATE)

`/dx-skill-router` MUST be invoked as the **absolute first action** for
**every user message** — regardless of task type (development, analysis,
reading, explanation, or clarification).

This rule applies before:
- Any file read or codebase exploration
- Any response or clarifying question
- Any SWE gate check or path classification
- Any code generation or plan creation

**No exceptions.** The following rationalizations are ALL prohibited:

| Rationalization | Reality |
|----------------|---------|
| "This is just reading/analyzing files" | Reading IS a task. Invoke router first. |
| "The user only asked a question" | Questions are tasks. Invoke router first. |
| "This is not a development task" | Router applies to ALL tasks, not only dev. |
| "I'll check for skills after I understand the request" | Check BEFORE understanding. |
| "This doesn't trigger SWE gates" | SWE gates are separate. Router is universal. |

## Shared Knowledge

All skills, instructions, toolsets, and memory live in `.deepx/`.
Read `.deepx/README.md` for the complete master index.

## Target Hardware (MANDATORY)

Target device is always DX-M1 (`dx_m1`). Do NOT ask the user to select target
hardware. The dx-compiler only supports DX-M1. Ignore any parent-level configuration
that mentions DX-M1A — DX-M1A is discontinued and no longer supported.

## Quantization — INT8 Only (MANDATORY)

DX-COM always quantizes to INT8. There is no FP16/FP32 output option in the CLI,
Python API, or JSON config. Do NOT ask the user to choose output precision.
The only user-facing quantization choices are:
- **Calibration method**: `ema` (default) or `minmax`
- **Enhanced quantization scheme**: `DXQ-P0` through `DXQ-P5` (optional)
- **Calibration sample count** (default: 100)

## Critical Rule: Model Acquisition and End-to-End Compilation

**MANDATORY**: When a user requests model compilation and does NOT provide a local
`.pt`, `.pth`, or `.onnx` file path, the agent MUST:

1. **Identify the official download source** for the requested model
2. **Actually download** the model file to the session working directory
3. **Actually compile** the model through the full pipeline (PT→ONNX→DXNN or ONNX→DXNN)
4. **Produce a real `.dxnn` output file**

**NEVER** stop at generating only a `config.json` or providing compilation
instructions. The user expects a compiled `.dxnn` model, not a recipe.

### Model Download Sources (Priority Order)

| Model Family | Source | Download Method |
|---|---|---|
| Ultralytics YOLO (v5-v12, v26) | GitHub Releases / `ultralytics` pip | `from ultralytics import YOLO; model = YOLO("yolo11n.pt")` or `wget` from releases |
| TorchVision (ResNet, MobileNet, etc.) | PyTorch Hub / torchvision | `torchvision.models.resnet50(weights="DEFAULT")` |
| Timm models | `timm` pip | `timm.create_model("efficientnet_b0", pretrained=True)` |
| ONNX Model Zoo | GitHub onnx/models | `wget` from tagged releases |
| Hugging Face | Hugging Face Hub | `huggingface_hub.hf_hub_download()` |

### Anti-Patterns (NEVER Do)

- Generating config.json and telling user "run dxcom yourself"
- Providing download URLs without actually downloading
- Stopping after ONNX export without compiling to DXNN
- Telling user to "install ultralytics and export" instead of doing it
- Producing only instructions instead of actual compiled artifacts

## Interactive Workflow (MUST FOLLOW)

**Always walk through key decisions with the user before compiling.** Ask 2-3 targeted
questions to confirm model format, target device, and calibration method (EMA or MinMax).
This creates a collaborative workflow and catches misunderstandings early. Only skip
questions if the user explicitly says "just compile it" or "use defaults".

**Gate 1 — Brainstorm**: Confirm inputs (model path, format, target device, calibration data).
**Gate 2 — Build**: Execute compilation with chosen parameters.
**Gate 3 — Verify**: Validate output with DX-TRON, review compiler.log.

## Quick Reference

```bash
pip install dx-com                     # Install DX-COM compiler
dxcom --help                           # CLI help
dxcom -m model.onnx -c config.json -o output/   # Basic compilation
pytest tests/                          # Run tests
```

```python
import dx_com
dx_com.compile(model="model.onnx", output_dir="output/", config="config.json")
```

## Skills

| Command | Description |
|---------|-------------|
| `/dx-agent-compiler-convert` | Convert PyTorch model to ONNX |
| `/dx-agent-compiler-compile` | Compile ONNX model to DXNN |
| `/dx-agent-compiler-validate` | Validate compilation output |
| `/dx-swe-brainstorm` | Brainstorm, propose 2-3 approaches, spec self-review, then plan |
| `/dx-swe-tdd` | Validation-driven development with optional Red-Green-Refactor for unit tests |
| `/dx-swe-verify` | Verify before claiming completion — evidence before assertions |
| `/dx-swe-writing-plans` | Write implementation plans with bite-sized tasks |
| `/dx-swe-executing-plans` | Execute plans with review checkpoints |
| `/dx-swe-subagent-dev` | Execute plans via fresh subagent per task with two-stage review |
| `/dx-swe-debugging` | Systematic debugging — 4-phase root cause investigation before proposing fixes |
| `/dx-swe-receiving-review` | Evaluate code review feedback with technical rigor |
| `/dx-swe-requesting-review` | Request code review after completing features |
| `/dx-skill-router` | Skill discovery and invocation — check skills before any action |
| `/dx-harness-writing-skills` | Create and edit skill files |
| `/dx-swe-parallel-agents` | Dispatch parallel subagents for independent tasks |

## Critical Conventions

1. **Batch size must be 1**: DEEPX NPU only supports batch=1
2. **Static shapes only**: No dynamic axes, no -1 dimensions
3. **ONNX opset 11-21**: Use opset 13 for best compatibility
4. **Input name match**: config.json `inputs` key must exactly match ONNX input name
5. **Representative calibration**: Calibration images must match inference distribution
6. **PPU type matters**: Type 0 = anchor-based (YOLOv3-v7), Type 1 = anchor-free (YOLOX, YOLOv8-v12). YOLO26 does not support PPU.
7. **Always validate**: Run DX-TRON inspection after every compilation
8. **No hardcoded paths**: Use relative paths for calibration data (`./calibration_dataset`)
9. **Output isolation**: All artifacts go to `dx-agent-dev/<session_id>/`.
   **Session ID format**: `YYYYMMDD-HHMMSS_<agent>_<coding_model>_<target_model>_<task>` — the timestamp MUST use the
   **system local timezone** (NOT UTC). Use `$(date +%Y%m%d-%H%M%S)` in Bash or
   `datetime.now().strftime('%Y%m%d-%H%M%S')` in Python. Do NOT use `date -u`,
   `datetime.utcnow()`, or `datetime.now(timezone.utc)`.
- **`<agent>`**: the coding agent identifier — use `claude`, `codex`, `copilot`, `cursor`, or `opencode`.
- **`<coding_model>`**: shortened coding model name — e.g., `sonnet46`, `opus46`, `gpt53codex`, `gpt55`.
10. **Calibration symlink**: Symlink `dx_com/calibration_dataset/` into working directory
11. **No auto-simplification**: Do NOT run `onnx-simplifier` unless the user explicitly requests it
12. **Ultralytics YOLO export**: Must set `Detect.export=True` or use `model.export(format="onnx")` — standard `torch.onnx.export()` produces 6 outputs instead of 1. Always verify single output post-export.
13. **MANDATORY brainstorming questions**: Before any compilation task, the agent MUST ask three mandatory questions: (Q1) NMS-free model detection with YOLO version characteristics, (Q2) ONNX simplification with pros/cons, (Q3) PPU compilation support with explanation. See `.deepx/agents/dx-compiler-builder.md` Step 2 for full details.
14. **PPU default is OFF**: PPU compilation is opt-in. Only add PPU config if the user explicitly confirms during brainstorming Q3. **YOLO26 does not support PPU** — skip Q3 for YOLO26.
15. **Model Acquisition — download and compile, not just instruct**: If the user does not provide a local `.pt`/`.pth`/`.onnx` file, the agent MUST find the official download source (Ultralytics releases, torchvision, timm, ONNX Model Zoo, Hugging Face), actually download the model, and compile it through the full pipeline to produce a `.dxnn` file. NEVER stop at generating only config.json or providing compilation instructions.
16. **Post-compilation verification is MANDATORY — compilation is NOT complete without it**: After every successful `dxcom` compilation, the agent MUST complete ALL of the following before presenting results. NEVER present a "compilation successful" summary without these:
    - **(a)** Generate `setup.sh`, `run.sh`, `README.md` in the output directory (session dir OR user-specified dir)
    - **(b)** Generate `verify.py` — ONNX vs DXNN inference comparison
    - **(c)** Run `verify.py` and confirm PASS
    - **(d)** Save session log to `${WORK_DIR}/session.log` — must contain **actual command execution output** captured via `tee`, NOT a hand-written summary
    - **(e)** Include verification results and all artifact paths in final summary
    - **Even when the user specifies a custom output directory** (e.g., a source directory instead of `dx-agent-dev/`), these artifacts are still MANDATORY. The user choosing a different output path does NOT exempt the agent from generating deployment scripts.
17. **Never reuse previous session artifacts**: NEVER check, list, browse, or reuse artifacts from previous sessions in `dx-agent-dev/`. Each compilation run MUST create a new session directory with a fresh timestamp. Even if a previous session compiled the same model, always re-download, re-export, and re-compile from scratch. Do NOT run `ls dx-agent-dev/` or check for existing `.onnx`/`.dxnn` files from past runs.
18. **venv is MANDATORY in setup.sh**: Generated `setup.sh` MUST create and activate a virtual environment before any `pip install`. On Ubuntu 24.04+, PEP 668 blocks system-wide pip installs. Use `python3 -m venv` with `${VIRTUAL_ENV:-}` check. Generated `run.sh` MUST check venv activation and auto-activate or error if missing.
19. **Cross-validation with precompiled reference model**: When a precompiled DXNN for the same model exists in `dx-runtime/dx_app/assets/models/`, run verify.py with BOTH the precompiled and generated models (Phase 5.7). Both fail → verify.py bug. Precompiled passes, generated fails → compilation problem. See `.deepx/agents/dx-dxnn-compiler.md` Phase 5.7.
20. **NHWC/NCHW DataLoader mismatch**: The dxcom CLI's default dataloader loads images in NHWC `[1,H,W,C]`. If the ONNX model expects NCHW `[1,C,H,W]` (most PyTorch-exported models), CLI compilation will fail with `DataLoaderError: Input shape mismatch`. **Fix**: Use the Python API (`dx_com.compile()`) with a custom torch DataLoader that produces NCHW tensors. See `.deepx/memory/common_pitfalls.md` pitfall #18.

## Cross-Validation Diagnostic Table

| Result | Diagnosis |
|---|---|
| Both fail | verify.py code bug (fix verify.py first) |
| Precompiled passes, generated fails | compilation problem (fix config, quantization) |
| Both pass | compilation correct |

## Context Routing Table

| If the task mentions... | Read these files |
|---|---|
| **PyTorch, PT, export, convert** | `.deepx/agents/dx-model-converter.md`, `.deepx/skills/dx-agent-compiler-convert.md` |
| **ONNX, compile, DXNN, dxcom** | `.deepx/agents/dx-dxnn-compiler.md`, `.deepx/skills/dx-agent-compiler-compile.md` |
| **CLI, command line** | `.deepx/toolsets/dxcom-cli.md` |
| **Python API, dx_com.compile** | `.deepx/toolsets/dxcom-api.md` |
| **config, JSON, schema** | `.deepx/toolsets/config-schema.md` |
| **Ultralytics, YOLO, .pt, format=deepx, export to deepx** | `.deepx/toolsets/ultralytics-deepx-export.md` |
| **Ultralytics retrain/train, fine-tune, mAP, FPS, domain dataset, evaluate** | `.deepx/toolsets/ultralytics-train-eval.md` |
| **PaddleOCR, RapidDoc, PaddlePaddle, OCR app, video/webcam OCR, PDF to markdown, document parsing** | `.deepx/toolsets/paddlepaddle-deepx.md` |
| **calibration, quantization, INT8** | `.deepx/instructions/compilation-workflow.md` |
| **PPU, YOLO, detection** | `.deepx/toolsets/config-schema.md`, `.deepx/instructions/compilation-workflow.md` |
| **validate, verify, check** | `.deepx/skills/dx-agent-compiler-validate.md` |
| **error, fail, bug** | `.deepx/memory/common_pitfalls.md` |
| **sample, example, test compile** | `.deepx/instructions/compilation-workflow.md` (Sample Model Workflow section) |
| **Brainstorm, plan, design** | `.deepx/skills/dx-swe-brainstorm.md` |
| **TDD, validation, incremental** | `.deepx/skills/dx-swe-tdd.md` |
| **Completion, verify, evidence** | `.deepx/skills/dx-swe-verify.md` |
| **Debug, root cause, investigate** | `.deepx/skills/dx-swe-debugging/SKILL.md` |
| **Plan, execute, subagent** | `.deepx/skills/dx-swe-writing-plans/SKILL.md`, `.deepx/skills/dx-swe-executing-plans/SKILL.md` |
| **Code review, feedback** | `.deepx/skills/dx-swe-receiving-review/SKILL.md`, `.deepx/skills/dx-swe-requesting-review/SKILL.md` |
| **ALWAYS read (every task)** | `.deepx/memory/common_pitfalls.md`, `.deepx/instructions/coding-standards.md` |

## Quick Start — Sample Model Workflow

```bash
cd dx-compiler
./example/1-download_sample_models.sh      # Download ONNX + JSON configs
./example/2-download_sample_calibration_dataset.sh  # Download calibration dataset
./example/3-compile_sample_models.sh       # Compile all sample models to .dxnn
```

Sample models: YOLOV5S-1, YOLOV5S_Face-1, MobileNetV2-1.
Sample JSON configs serve as canonical references for config.json generation.

## Output Isolation

All compilation artifacts go to `dx-agent-dev/<session_id>/` by default. Each
compilation session uses a unique working directory to keep artifacts together and
prevent overwrites.

**Session ID format**: `YYYYMMDD-HHMMSS_<agent>_<coding_model>_<target_model>_<task>` — the timestamp MUST use the
**system local timezone** (NOT UTC). Use `$(date +%Y%m%d-%H%M%S)` in Bash or
`datetime.now().strftime('%Y%m%d-%H%M%S')` in Python. Do NOT use `date -u`,
`datetime.utcnow()`, or `datetime.now(timezone.utc)`.
- **`<agent>`**: the coding agent identifier — use `claude`, `codex`, `copilot`, `cursor`, or `opencode`.
- **`<coding_model>`**: shortened coding model name — e.g., `sonnet46`, `opus46`, `gpt53codex`, `gpt55`.

**Working directory contents** after compilation:
```
dx-agent-dev/<session_id>/
├── calibration_dataset   → ../../dx_com/calibration_dataset/ (symlink)
├── config.json
├── model.onnx
├── model.dxnn
├── compiler.log
└── README.md             (session report)
```

## Calibration Dataset

Calibration data lives at `dx_com/calibration_dataset/` (100 JPEG images). If
missing, run `example/2-download_sample_calibration_dataset.sh` to set up.
Always use relative paths (`./calibration_dataset`) in config.json, never absolute.

## Ultralytics → DeepX Export (One-Shot Path)

Ultralytics YOLO ships a first-class `format=deepx` exporter that produces a
deployable DeepX NPU model in **one command** — it runs ONNX export → INT8 EMA
calibration → `dx_com` compilation → packaging internally:

```bash
yolo export model=yolo26n.pt format=deepx     # creates 'yolo26n_deepx_model/'
```
```python
from ultralytics import YOLO
YOLO("yolo26n.pt").export(format="deepx")      # int8=True is enforced
```

**Prefer this path** for Ultralytics YOLO **detection** models targeting DeepX —
it avoids the common manual PT→ONNX→`dxcom` errors. Fall back to the manual
pipeline (`dx-agent-compiler-convert` → `dxcom`) only for non-detection tasks,
non-YOLO/custom graphs, or when fine control over `config.json` is required.

Key facts (full reference: `.deepx/toolsets/ultralytics-deepx-export.md`):

- **x86-64 Linux only** for export (`dx_com` has no ARM64); **detection only**; **INT8 enforced**.
- Output is a **directory** `<model>_deepx_model/` = `{<model>.dxnn, config.json, metadata.yaml}` — not a bare `.dxnn`.
- Calibration: EMA, default 100 images (`data` / `fraction` to tune).
- Deploy: `YOLO("<model>_deepx_model")` → `model(source)` on the `dx_engine` runtime
  (backend converts BCHW float `[0,1]` → HWC uint8 `[0,255]`). Inference is not ARM64-restricted.
- `dx_com` auto-installs via Ultralytics' export (version pinned by the installed
  `ultralytics` release). **Do NOT manually `pip install dx-com` from a hardcoded SDK
  URL/version** — it pins a stale compiler; to update, upgrade `ultralytics`.
- **Deployment prerequisite**: the `dx_engine` **runtime** is end-user-installed —
  Ultralytics auto-installs it **only on Debian Trixie/arm64** (sixfab-dx). On the
  x86-64 dx-all-suite the backend raises `OSError: dx_engine is not installed. … Please
  install dx_engine manually and try again` → here "install manually" = **install the
  `dx_rt` runtime** (provides `dxrt-cli`+`dx_engine`): `dx-runtime/scripts/sanity_check.sh
  --dx_rt`, then `dx-runtime/install.sh --all --exclude-app --exclude-stream
  --skip-uninstall --venv-reuse` (dx_app/dx_stream NOT needed → skip to save time), retry.
  NPU init failure → cold boot. Never `pip install dx_engine` on x86-64 or fake via PYTHONPATH.

## No Placeholder Code (MANDATORY)

NEVER generate stub/placeholder code. This includes:
- Commented-out imports: `# from dxnn_sdk import InferenceEngine`
- Fake results: `result = np.zeros(...)`
- TODO markers: `# TODO: implement actual inference`
- "Similar to sync version" without actual async implementation

All generated code MUST be functional, using real APIs from the knowledge base.
If the required SDK/API is unknown, read the relevant skill document first.

## Experimental Features — Prohibited

Do NOT offer, suggest, or implement experimental or non-existent features. This includes:
- "웹 기반 비주얼 컴패니언" (web-based visual companion) <!-- KOREAN-OK: Korean feature name included so agents recognize this prohibited request in Korean -->
- Local URL-based diagram viewers or dashboards
- Any feature requiring the user to open a local URL for visualization
- Any capability that does not exist in the current toolset

**Superpowers brainstorming skill override**: The superpowers `brainstorming` skill
includes a "Visual Companion" step (step 2 in its checklist). This step MUST be
SKIPPED in the DEEPX project. The visual companion does not exist in our environment.
When the brainstorming checklist says "Offer visual companion", skip it and proceed
directly to "Ask clarifying questions" (step 3).

If a feature does not exist, do not pretend it does. Stick to proven, documented
capabilities only.

**Autopilot / autonomous mode override**: When the user is absent (autopilot mode,
auto-response "work autonomously", or `--yolo` flag), the brainstorming skill's
"Ask clarifying questions" step MUST be replaced with "Make default decisions per
knowledge base rules". Do NOT call `ask_user` — skip straight to producing the
brainstorming spec using knowledge base defaults. All subsequent gates (spec review,
plan, TDD, mandatory artifacts, execution verification) still apply without exception.

## Brainstorming — Spec Before Plan (HARD GATE)

When using the superpowers `brainstorming` skill or `/dx-swe-brainstorm`:

1. **Spec document is MANDATORY** — Before transitioning to `writing-plans`, a spec
   document MUST be written to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`.
   Skipping the spec and going directly to plan writing is a violation.
2. **User approval gate is MANDATORY** — After writing the spec, the user MUST review
   and approve it before proceeding to plan writing. Do NOT treat unrelated user
   responses (e.g., answering a different question) as spec approval.
3. **Plan document MUST reference the spec** — The plan header must include a link
   to the approved spec document.
4. **Prefer `/dx-swe-brainstorm`** — Use the project-level brainstorming skill
   instead of the generic superpowers `brainstorming` skill. The project-level skill
    has domain-specific questions and pre-flight checks.
5. **Rule conflict check is MANDATORY** — During brainstorming, the agent MUST check
   whether any user requirement conflicts with HARD GATE rules (IFactory pattern,
   skeleton-first, Output Isolation, SyncRunner/AsyncRunner). If a conflict is
   detected, the agent MUST resolve it during brainstorming — not silently comply
   with the violating request in the design spec. See the "Rule Conflict Resolution" section.
## Mandatory Process Skill Sequence — All Code Generation (HARD GATE)

This gate applies to ALL sessions that generate code artifacts in
`dx-agent-dev/<session_id>/`. It is independent of the "Internal Development"
SWE Process Gates — those apply to dx-agent-dev infrastructure work; THIS gate
applies to user-facing code generation (inference apps, pipelines, compilation).

### When This Gate Applies

Any session that produces files in `dx-agent-dev/<session_id>/` MUST follow
the complete process skill sequence below. This includes:
- ONNX → DXNN compilation sessions
- Python/C++ inference app generation (dx_app)
- GStreamer pipeline creation (dx_stream)
- Cross-project sessions (compile + deploy)

### Mandatory Skill Sequence

Every code generation session MUST flow through this sequence in order.
**No code generation before this sequence completes.**

**Autopilot mode does NOT waive this sequence.** "Work autonomously" means follow
all rules without asking — NOT skip rules. In autopilot, make default decisions
using the knowledge base instead of calling `ask_user`, but every step below
still applies.

| Step | Skill | Requirement |
|------|-------|-------------|
| 1 | `/dx-skill-router` | **Always** — invoke BEFORE any action. Already enforced by `skill-router-mandatory` fragment. |
| 2 | `/dx-agent-brainstorm` | **All non-trivial code generation** — gather requirements, propose approaches, get approval before any file creation. |
| 3 | `/dx-swe-writing-plans` | **Always** — produce a structured implementation plan for every code generation session, regardless of complexity. |
| 4 | `/dx-agent-tdd` | **Always** — define acceptance criteria (Red), generate artifacts (Green), verify immediately (Verify). |
| 5 | `/dx-agent-verify` | **Always** — before declaring DONE, provide evidence of working artifacts. Assertions without evidence are prohibited. |

### Sequence Enforcement Rules

1. **No skipping steps** — Each step MUST complete before the next begins.
   Exception: Step 1 (skill-router) is already handled by a separate fragment.
2. **No reordering** — brainstorm → plan → tdd → verify. Never generate code
   before planning. Never declare done before verifying.
3. **Plan MUST exist before any file creation** — Even a single-file session
   requires a plan (can be brief, but must be explicit).
4. **Verification MUST use actual execution** — `python file.py`, `bash -n script.sh`,
   `import` checks. Never claim "it should work" without running it.

### Trivial Change Exception

Steps 2–3 (brainstorm/plan) may be skipped ONLY for:
- Single config.json field change (e.g., threshold adjustment)
- Single-line typo fix in existing generated code

Steps 4–5 (tdd/verify-completion) are **NEVER** skipped, even for trivial changes.

### Autopilot Mode Adaptation

In autopilot mode (user absent, `--yolo` flag, or auto-response):
- **Step 2**: Replace `ask_user` with knowledge base defaults. Self-review spec.
- **Step 3**: Write plan and self-approve against knowledge base rules.
- **Step 4**: Define acceptance criteria from plan, generate, verify immediately.
- **Step 5**: Execute all artifacts, capture output as evidence. No human needed.

### Relationship with Artifact Verification Gate

This sequence defines **WHEN** each skill is invoked (workflow order).
The Artifact Verification Gate defines **HOW** each artifact is verified
(specific commands per file type). They work together:

- Step 4 (`/dx-agent-tdd`) uses the verification commands from the Artifact
  Verification Gate (syntax checks, execution tests, import resolution).
- Step 5 (`/dx-agent-verify`) confirms all mandatory deliverables
  exist and pass the Artifact Verification Gate checks.

### Invoke = Actual Tool Call

"Invoke a skill" means calling the `skill` tool to load it. Writing "Using
dx-agent-tdd" in text is NOT an invocation — the tool must be called. If you did not
call the `skill` tool for a step, that step is incomplete.

### Anti-Patterns (PROHIBITED)

- "This is simple, brainstorm is unnecessary" → brainstorm is ALWAYS required
  for non-trivial code generation. "Simple" is where unexamined assumptions
  cause the most wasted work.
- Generating code before `/dx-swe-writing-plans` produces a plan → HARD GATE violation.
  Plan-before-code is non-negotiable.
- Skipping `/dx-agent-verify` because "artifact-verification-gate already
  checks files" → they serve different purposes. Artifact gate checks individual
  files. Verify-completion checks the ENTIRE session deliverables holistically.
- Declaring DONE without showing execution output → evidence is mandatory.
  "I verified it works" without showing the output is not acceptable.
- "The user said just do it quickly" → user instructions do NOT override this
  HARD GATE. Speed does not justify skipping process.
- **Text mention ≠ skill invocation** — writing "Using dx-agent-tdd" or "Following
  dx-agent-brainstorm" in the response text is NOT a valid invocation. The
  `skill` tool MUST be called for each step.
- **Conversation context ≠ brainstorming** — discussing requirements in prior
  messages does NOT substitute for invoking `/dx-agent-brainstorm`. Each
  feature requires a formal brainstorm with explicit user approval.
## Autopilot Mode Guard (MANDATORY)

When the user is absent — autopilot mode, `--yolo` flag, or system auto-response
"The user is not available to respond" — the following rules apply:

1. **"Work autonomously" means "follow all rules without asking", NOT "skip rules".**
   Every mandatory gate still applies: brainstorming spec, plan, TDD, mandatory
   artifacts, execution verification, and self-verification checks.
   **This includes the SWE Process Gates Mandatory Skill Sequence** — in autopilot,
   `/dx-skill-router` → `/dx-agent-brainstorm` → `/dx-agent-tdd` must be followed
   exactly as in interactive mode. Autopilot mode does NOT waive this sequence.
2. **Do NOT call `ask_user`** — Make decisions using knowledge base defaults and
   documented best practices. Calling `ask_user` in autopilot wastes a turn and
   the auto-response does not grant permission to bypass any gate.
3. **User approval gate adaptation** — In autopilot, the spec approval gate is
   satisfied by writing the spec and self-reviewing it against the knowledge base.
   Do NOT skip the spec entirely.
4. **setup.sh FIRST** — Generate infrastructure artifacts (`setup.sh`, `config.json`)
   before writing any application code. This is especially critical in autopilot
   because there is no human to catch missing dependencies.
5. **Execution verification is NOT optional** — Run the generated code and verify it
   works before declaring completion. In autopilot, there is no user to catch errors.
6. **Time budget awareness** — Autopilot sessions may have time constraints.
   Plan your actions efficiently:
   - Compilation (ONNX → DXNN) may take 5+ minutes — start it early.
   - If time is short, prioritize artifact GENERATION over execution
     verification — a complete set of untested files is better than a partial
     set of tested ones.
   - Priority order: `setup.sh` > `run.sh` > app code > `verify.py` > session.log.
   - **Compilation-parallel workflow (HARD GATE)** — After launching `dxcom` or
     `dx_com.compile()` in a bash command, do NOT wait for it. Immediately
     proceed to generate ALL mandatory artifacts: factory, app code, setup.sh,
     run.sh, verify.py. Check `.dxnn` output only AFTER all other artifacts
     are created. **Violation of this rule fails the session.**
   - **NEVER sleep-poll for compilation** — Do NOT use `sleep` in a loop to
     poll for `.dxnn` files. Prohibited patterns include:
     `for i in ...; do sleep N; ls *.dxnn; done`,
     `while ! ls *.dxnn; do sleep N; done`,
     repeated `ls *.dxnn` / `test -f *.dxnn` checks with waits between them.
     Instead: generate all other artifacts first, then check ONCE whether the
     `.dxnn` file exists. If it does not exist yet, proceed to execution
     verification with the assumption that compilation will complete.
   - **NEVER use `pgrep -f` to monitor compile.pid process** — `pgrep -f
     "path/to/compile.py"` matches the bash shell that is running the pgrep
     command itself, causing an **infinite loop** that never exits even after
     compilation finishes. Always use `kill -0 <PID>` to check if a specific
     PID is still alive:
     ```bash
     # CORRECT — check by PID, not by name
     COMPILE_PID=$(cat compile.pid)
     while kill -0 "$COMPILE_PID" 2>/dev/null; do sleep 10; done
     echo "Compilation PID=$COMPILE_PID has exited"
     ```
     **Prohibited patterns** (self-referential, cause infinite loops):
     ```bash
     while pgrep -f "compile.py" >/dev/null 2>&1; do sleep 20; done   # PROHIBITED
     pgrep -f "session_dir/compile.py"                                 # PROHIBITED
     ```
   - **Mandatory artifacts are compilation-independent** — `setup.sh`, `run.sh`,
     `verify.py`, factory, and app code do NOT require the `.dxnn` file to exist.
     Generate them using the known model name (e.g., `yolo26n.dxnn`) as a
     placeholder path. Only execution verification requires the actual `.dxnn`.
7. **Minimize file-reading tool calls** — Do NOT re-read instruction files,
   agent docs, or skill docs that are already loaded in your context. Each
   unnecessary `cat` / `bash` read wastes 5-15 seconds. Use the knowledge
   already in your system prompt and conversation history.

## Hardware

| Device | ID | Description |
|---|---|---|
| DX-M1 | `dx_m1` | DEEPX NPU |

## Memory

Persistent knowledge in `.deepx/memory/`. Read at task start, update when learning new patterns.
Domain tags: `[UNIVERSAL]`, `[DX_COMPILER]`, `[QUANTIZATION]`

## Rule Conflict Resolution (HARD GATE)

When a user's request conflicts with a HARD GATE rule, the agent MUST:

1. **Acknowledge the user's intent** — Show that you understand what they want.
2. **Explain the conflict** — Cite the specific rule and why it exists.
3. **Propose the correct alternative** — Show how to achieve the user's goal
   within the framework. For example, if the user asks for direct
   `InferenceEngine.run()` usage, explain that the IFactory pattern wraps
   the same API and propose the factory-based equivalent.
4. **Proceed with the correct approach** — Do NOT silently comply with the
   rule-violating request. Do NOT present it as "Option A vs Option B".

**Common conflict patterns** (from real sessions):
- User says "use `InferenceEngine.Run()`" → Must use IFactory pattern (engine
  calls are handled internally by SyncRunner/AsyncRunner; implement the 5 IFactory
  methods: `create_preprocessor`, `create_postprocessor`, `create_visualizer`,
  `get_model_name`, `get_task_type`)
- User says "clone demo.py and swap onnxruntime" → Must use skeleton-first
  from `src/python_example/`, not clone user scripts
- User says "create demo_dxnn_sync.py" → Must use `<model>_sync.py` naming
  with SyncRunner, not a standalone script
- User says "use `run_async()` directly" → Must use AsyncRunner, not manual
  async loops

**This rule does NOT override explicit user overrides**: If the user, after being
informed of the conflict, explicitly says "I understand the rule, proceed with
direct API usage anyway", then comply. But the agent MUST explain the conflict
FIRST — silent compliance is always a violation.

## Git Operations — User Handles

Do NOT ask about git branch operations (merge, PR, push, cleanup) at the end of
work. The user will handle all git operations themselves. Never present options
like "merge to main", "create PR", or "delete branch" — just finish the task.

## Git Safety — Superpowers Artifacts

**NEVER `git add` or `git commit` files under `docs/superpowers/`.** These are temporary
planning artifacts generated by the superpowers skill system (specs, plans). They are
`.gitignore`d, but some tools may bypass `.gitignore` with `git add -f`. Creating the
files is fine — committing them is forbidden.

## Session Sentinels (MANDATORY for Automated Testing)

When processing a user prompt, output these exact markers for automated session
boundary detection by the test harness:

- **First line of your response**: `[DX-AGENT-DEV: START]`
- **Last line after ALL work is complete**: `[DX-AGENT-DEV: DONE (output-dir: <relative_path>)]`
  where `<relative_path>` is the session output directory (e.g., `dx-agent-dev/20260409-143022_yolo26n_detection/`)

### DEEPX Banner (MANDATORY — print with the sentinels)

Render the DEEPX logo banner **verbatim** at two points: **immediately after** the
`[DX-AGENT-DEV: START]` line, and **immediately before** the
`[DX-AGENT-DEV: DONE ...]` line. Print it exactly as below (a fenced block is fine):

```
 ███████████   █████████ ████████ ████████  ████      ████
 ███     █████ ███░░░░░░░███░░░░░░███   ███  ░████   ████░░
 ███        ██░███░      ██░░     ███   ███░   █████████░░
 ███        ████████████ ████████ ████████░░    ░█████░░░
 ███        ██░███░░░░░░░██░░░░░░░███░░░░░░  ██████████
 ███     █████░███░      ██░      ███░   ████████░░░░████
 ███████████░░░█████████ ████████ ██████████░░░░░░    ████
  ░░░░░░░░░░░   ░░░░░░░░░ ░░░░░░░░ ░░░░░░░░░░          ░░░░
        DX-AGENT-DEV · on-device NPU
```

The banner is decorative; it never replaces or moves the sentinel lines (START stays
the absolute first line, DONE stays the very last line).

Rules:
1. **CRITICAL — Output `[DX-AGENT-DEV: START]` as the absolute first line of your
   first response.** This must appear before ANY other text, tool calls, or reasoning.
   Even if the user instructs you to "just proceed" or "use your own judgment",
   the START sentinel is non-negotiable — automated tests WILL fail without it.
   **Immediately after the START line, print the DEEPX banner** (see "DEEPX Banner" above).
2. **Immediately before the DONE line, print the DEEPX banner again**, then output
   `[DX-AGENT-DEV: DONE (output-dir: <path>)]` as the very last line after all work,
   validation, and file generation is complete
3. If you are a **sub-agent** invoked via handoff/routing from a higher-level agent,
   do NOT output these sentinels — only the top-level agent outputs them
4. If the user sends multiple prompts in a session, output START/DONE for each prompt
5. The `output-dir` in DONE must be the relative path from the project root to the
   session output directory. If no files were generated, omit the `(output-dir: ...)` part.
   **For cross-project tasks** (e.g., compile + app generation), list ALL output directories
   separated by ` + `:
   ```
   [DX-AGENT-DEV: DONE (output-dir: dx-compiler/dx-agent-dev/20260409-143022_copilot_yolo26n_compile/ + dx-runtime/dx_app/dx-agent-dev/20260409-143022_copilot_yolo26n_inference/)]
   ```
6. **NEVER output DONE after only producing planning artifacts** (specs, plans, design
   documents). DONE means all deliverables are produced — implementation code, scripts,
   configs, and validation results. If you completed a brainstorming or planning phase
   but have not yet implemented the actual code, do NOT output DONE. Instead, proceed
   to implementation or ask the user how to proceed.
7. **Pre-DONE mandatory deliverable check**: Before outputting DONE, verify that all
   mandatory deliverables exist in the session directory. If any mandatory file is
   missing, create it before outputting DONE. Each sub-project defines its own mandatory
   file list in its skill document (e.g., `dx-agent-stream-build-pipeline.md` File Creation Checklist).
8. **Session transcript — generate it RIGHT AFTER the DONE line (claude / copilot)**:

   **Auto-transcript is supported on `claude` and `copilot` only.** Emit the DONE
   sentinel line FIRST, then — as the single final housekeeping step — render this
   session's transcript with the shared generator **directly into the session output
   dir(s)** (the same dir(s) you listed in DONE). Running it *after* DONE means the
   CLI's session store has already committed the DONE turn, so the rendered transcript
   is complete (rendering *before* DONE truncates the tail). Needs **no hook**:

   ```bash
   # Locate the shared generator by walking up to the suite root: GENROOT is the dir
   # that contains .deepx/tools. Then render THIS session's transcript INTO the session
   # output dir(s). Pass EVERY output dir you created (the transcript is copied into each
   # — cross-project: both the compiler and app dirs). The session id is auto-resolved
   # from this CLI's own env var (CLAUDE_CODE_SESSION_ID / COPILOT_AGENT_SESSION_ID).
   #
   # CRITICAL — use ABSOLUTE paths for --project AND --into-output-dirs. A RELATIVE
   # output dir is resolved against the agent's CURRENT cwd, so it is SILENTLY SKIPPED
   # ("no output dir produced — transcript generation skipped") whenever cwd is not the
   # suite root — e.g. after you cd into the session dir to run setup.sh/run.sh. Prefix
   # every output dir with "$GENROOT/" (or pass the same absolute SESSION_DIR you used
   # to write artifacts).
   GENROOT="$(d="$PWD"; while [ "$d" != / ]; do [ -f "$d/.deepx/tools/src/dx_transcripts/generate_transcripts.py" ] && { echo "$d"; break; }; d="$(dirname "$d")"; done)"
   GT="$GENROOT/.deepx/tools/src/dx_transcripts/generate_transcripts.py"
   python3 "$GT" --tool <CLI> --project "$GENROOT" \
       --into-output-dirs "$GENROOT/<output-dir>" ["$GENROOT/<output-dir-2>" ...]
   ```

   `<CLI>` is `claude` or `copilot`. The generator reuses the **same renderers as the
   test harness** (`parse_<tool>_session`) and writes `<CLI>-session.md` +
   `<CLI>-session.html` + `<CLI>-stream.jsonl` into each output dir. **If you produced
   NO output dir** (e.g. a pure question with no files), pass no dir and generation is
   **skipped** — expected, not an error. After it runs, state the path on the final
   line, e.g. `Session transcript (md/html/jsonl) saved to: <output-dir>/<CLI>-session.*`.

   > **Known limitation — the in-session transcript is store-based and therefore
   > incomplete.** Run from inside the live session, the generator reads the session
   > **store**, which has NO synthetic `result` event — that event (carrying
   > `duration_ms` → *Wall-clock* and `total_cost_usd` → *Cost*) exists only in the
   > `claude -p --output-format stream-json` **stdout**, emitted at process exit. The
   > render also happens *during* the transcript tool-call, so it truncates just before
   > this very "saved to …" narration. Net effect: the in-session transcript **omits
   > Wall-clock + Cost and the closing narration** — expected, not a bug. For a
   > **complete** transcript (Wall-clock + Cost + tail, like the showcase ones),
   > capture the run's stdout and render it externally **after** the process exits:
   > `python3 "$GT" --tool <CLI> --session-id <uuid> --project "$GENROOT" --stream-json <captured-stdout.jsonl> --out-dir <output-dir>`
   > (the test harness / build recorders do this). An in-session agent cannot — it has
   > no handle on its own stdout stream.

   **`codex`, `opencode`, `cursor` are NOT auto-supported** — do NOT run the generator
   in-session for them (it cannot produce a complete/usable transcript: codex and
   opencode commit their final turn only at process exit; cursor redacts the assistant
   text in its store). Instead, tell the user how to generate it manually:
   - **codex / opencode**: after the session ends, run
     `python3 <generate_transcripts.py> --tool <codex|opencode> --project . --out-dir <DIR>`
     — the finalized store then renders a complete transcript.
   - **cursor**: capture the run with `agent -p --output-format stream-json > run.jsonl`
     and render with `--tool cursor --stream-json run.jsonl`, or use IDE session history.
   (If you invoke the generator with `--into-output-dirs` on these tools, it safely
   skips and prints this same guidance — that is expected.)

## Plan Output (MANDATORY)

When generating a plan document (e.g., via writing-plans or brainstorming skills),
**always print the full plan content in the conversation output** immediately after
saving the file. Do NOT only mention the file path — the user should be able to
review the plan directly in the prompt without opening a separate file.


---

## Instruction File Verification Loop (HARD GATE) — Internal Development Only

When modifying the canonical source — files in `**/.deepx/**/*.md`
(agents, skills, templates, fragments) — the following verify-fix loop is
**MANDATORY** before claiming work is complete:

1. **Generator execution** — Propagate `.deepx/` changes to all platforms:
   ```bash
   dx-agent-gen generate
   # Suite-wide: bash .deepx/tools/scripts/run_all.sh generate
   ```
2. **Drift verification** — Confirm generated output matches committed state:
   ```bash
   dx-agent-gen check
   ```
   If drift is detected, return to step 1.
3. **Automated test loop** — Tests verify generator output satisfies policies:
   ```bash
   python -m pytest .deepx/tests/conformance/ .deepx/tools/tests/ -v --tb=short
   ```
   Failure handling:
   - Generator bug → fix generator → step 1
   - `.deepx/` content gap → fix `.deepx/` → step 1
   - Insufficient test rules → strengthen tests → step 1
4. **Manual audit** — Independently (without relying on test results) read
   generated files to verify cross-platform sync (CLAUDE vs AGENTS vs copilot)
   and level-to-level sync (suite → sub-levels).
5. **Gap analysis** — For issues found by manual audit:
   - Generator missed a case → **fix generator rules** → step 1
   - Tests missed a case → **strengthen tests** → step 1
6. **Repeat** — Continue until steps 3–5 all pass.

### Pre-flight Classification (MANDATORY)

Before modifying ANY `.md` or `.mdc` file in the repository, classify it into
one of three categories. **Never skip this step** — editing a generator-managed
file directly is a silent corruption that will be overwritten on next generate.

**Answer these three questions in order before every file edit:**

> **Q1. Is the file path inside `**/.deepx/**`?**
> - YES → **Canonical source.** Edit directly, then run `dx-agent-gen generate` + `check`.
> - NO → go to Q2.
>
> **Q2. Does the file path or name match any of these?**
> ```
> .github/agents/    .github/skills/    .opencode/agents/
> .claude/agents/    .claude/skills/    .cursor/rules/
> CLAUDE.md          CLAUDE-KO.md       AGENTS.md    AGENTS-KO.md
> copilot-instructions.md               copilot-instructions-KO.md
> ```
> - YES → **Generator output. DO NOT edit directly.**
>   Find the `.deepx/` source (template, fragment, or agent/skill) and edit that instead,
>   then run `dx-agent-gen generate`.
> - NO → go to Q3.
>
> **Q3. Does the file begin with `<!-- AUTO-GENERATED`?**
> - YES → **Generator output. DO NOT edit directly.** Same as Q2.
> - NO → **Independent source.** Edit directly. Run `dx-agent-gen check` once afterward.

1. **Canonical source** (`**/.deepx/**/*.md`) — Modify directly, then run the
   Verification Loop above.
2. **Generator output** — Files at known output paths:
   `CLAUDE.md`, `CLAUDE-KO.md`, `AGENTS.md`, `AGENTS-KO.md`,
   `copilot-instructions.md`, `.github/agents/`, `.github/skills/`,
   `.claude/agents/`, `.claude/skills/`, `.opencode/agents/`, `.cursor/rules/`
   → **Do NOT edit directly.** Find and modify the `.deepx/` source
   (template, fragment, or agent/skill), then `dx-agent-gen generate`.
3. **Independent source** — Everything else (`docs/source/`, `source/docs/`,
   `tests/`, `README.md` in sub-projects, etc.)
   → Edit directly. Run `dx-agent-gen check` once afterward to confirm no
   unexpected drift.

**Anti-pattern**: Modifying a file without first classifying it. If you are
unsure whether a file is generator output, run `dx-agent-gen check` before
AND after the edit — if the check overwrites your change, the file is managed
by the generator and must be edited via `.deepx/` source instead.

A pre-commit hook enforces generator output integrity: `git commit` will fail
if generated files are out-of-date. Install hooks with:
```bash
.deepx/tools/scripts/install-hooks.sh
```

> **KO counterpart rule**: When editing any EN fragment, check whether the KO
> counterpart also needs updating. If you added or removed ≥ 1 paragraph, update
> `.deepx/templates/fragments/ko/<stem>.md` before committing. Run
> `dx-agent-gen lint` to verify `[OK]` — lint will ERROR if EN exceeds KO by
> ≥ 10 lines.

This gate applies when `.deepx/` files are the *primary deliverable* (e.g., adding
rules, syncing platforms, creating KO translations, modifying agents/skills). It
does NOT apply when a feature implementation incidentally triggers a single-line
change in `.deepx/`.
