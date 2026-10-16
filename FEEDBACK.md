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

| # | Step | Status | Notes |
|---|---|---|---|
| 1 | `pxt login` / `pxt whoami` | ✅ | Browser code flow worked. `whoami` clearly said "No organization yet" and gave the exact next command. |
| 2 | `pxt org create` + `pxt db status` | ✅ | `main` went PROVISIONING → AVAILABLE. Status output (cpu / memory / disk / pods) is very readable. |
| 3 | `pxt db update` (first upload) | ❌ | Failed after ~3s: `SSL: CERTIFICATE_VERIFY_FAILED` in `_put_artifact` (`urllib.request.urlopen`). Cause: python.org macOS Python 3.12 has no CA certificates until "Install Certificates.command" is run. Login and status worked, so only the artifact upload path hits it. Suggest: use `certifi` for the upload's SSL context (like `requests` does), or catch this error and print the macOS fix. Common setup for Mac users. **Resolved:** linked Python 3.12's OpenSSL cafile to `certifi`, then had to `kill` the background daemon on :22089 (it kept the old SSL settings). After that, `pxt db update` → `applied AVAILABLE`. Note: the daemon restart step isn't obvious; a `pxt daemon restart` command (or a hint in the error) would help. |
| 4 | `pxt schema update` + `pxt service update` on Cloud | ✅ | Public URL returned immediately (`https://vinod-lab-main.svc.pxt.run/ingest`). Same commands as local, only the target changes, which is very nice. |
| 5 | `insert.py` against `pxt://vinod-lab:main/docs` | ✅ / ⚠️ | Worked (`Hello` → `HELLO`). Minor confusion: it first prints `Connected to Pixeltable database at: postgresql+psycopg://...pgdata` (the *local* DB) even though the insert goes to Cloud. The docs mention this, but a line like "Using Cloud table pxt://vinod-lab:main/docs" would remove the doubt. |
| 6 | `pxt key create` + `curl` with `X-api-key` | ✅ | Hosted route returned `id`, `title_upper`, `summary` as documented. |

---

## Project-based testing

| # | Area | Status | Notes |
|---|---|---|---|
| 1 | `@pxt.udf` in a standalone script | ✅ | Defining a UDF at the top level of a script raises a clear error explaining to move it into an importable module. Fixed in one step. |
| 2 | Frame iterator + YOLOX on a 19 s video | ✅ | `frame_iterator(fps=1)` produced 19 frames instantly; `yolox_m` on all frames took ~46 s on an M-series Mac. Stored computed columns mean re-running the script skips finished work, which is great for iterating. |
| 3 | `av` vs `cv2` dylib warning | ⚠️ | With `pixeltable` + `pixeltable-yolox` installed together, macOS prints `objc: Class AVFFrameReceiver is implemented in both .../av/.dylibs/libavdevice... and .../cv2/.dylibs/libavdevice...`. Harmless so far, but noisy and alarming ("mysterious crashes"). Possibly `opencv-python-headless` would avoid the duplicate. |
| 4 | Dashboard "Add cloud catalog…" | ❌ | Entering the Cloud database spun for a while, then failed with `gaierror: [Errno 8] nodename nor servname provided, or not known`. Unclear which value the dialog expects (`pxt://vinod-lab:main`? a URL?). A placeholder/example in the dialog would help. |

---

## Docs

_(gaps, unclear wording, missing examples)_

---

## Things I liked

- The plan / apply style of `diff` then `update` makes changes feel safe.
- One `app.py` holding both the tables and the API is easy to reason about.
