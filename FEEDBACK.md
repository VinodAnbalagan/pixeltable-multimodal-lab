# Pixeltable Beta Feedback Log

Running log of what worked, what was confusing, and what broke while testing Pixeltable (local + Cloud beta). Written for the Pixeltable team.

**Setup:** macOS (Apple Silicon, 16 GB), Python via `uv`, `pixeltable[serve]` 0.7.12, repo `apreshill/pxt-champs`.

Legend: ✅ worked well · ⚠️ friction / confusing · ❌ broken

---

## Local quickstart (pxt-champs `guides/quickstart.md`)

| # | Step | Status | Notes |
|---|---|---|---|
| 1 | `uv sync` + `pxt --version` | ✅ | Clean install, printed `pxt 0.7.12` as documented. |
| 2 | `pxt init` | ✅ | Printed `already configured by pyproject.toml` (the guide says it prints `wrote pyproject.toml`). Minor wording mismatch. |
| 3 | `pxt service example --out app.py` | ✅ | Wrote the simple example app. Replaces the repo's autocropper `app.py` by design (it returns in `build-an-app.md`). |
| 7 | `curl` POST to `/docs` | ✅ | Returned `id`, `title_upper`, `summary` exactly as the guide shows. |
| 8 | `pxt dashboard` | ✅ | Opened instantly. The table card (inputs vs computed columns, with UDF badges) and the "Column Data Flow" view make the pipeline easy to understand at a glance. |
| 4 | `pxt schema diff` / `update` | ✅ | Plan output (`+ hello/docs create`, "0 destructive") is clear and reassuring, similar to a Terraform plan. |
| 5 | `pxt service update` | ✅ | Clear plan showing each route being added. Confirmation prompt is a nice safety net. |
| 6 | `pxt service list` | ✅ | Shows routes with inputs and outputs, which is a very helpful summary of the API. |

---

## Cloud beta

_(to be filled in during / after the setup session)_

---

## Project-based testing

_(Hallucination Spotter and Silent Film Foley: performance, video handling, local models via LM Studio, Hugging Face models on Cloud)_

---

## Docs

_(gaps, unclear wording, missing examples)_

---

## Things I liked

- The plan / apply style of `diff` then `update` makes changes feel safe.
- One `app.py` holding both the tables and the API is easy to reason about.
