# Pixeltable Beta Feedback Log

Running log of what worked, what was confusing, and what broke while testing Pixeltable (local + Cloud beta). Written for the Pixeltable team.

**Setup:** macOS (Apple Silicon, 16 GB), Python 3.12 via `uv`. Quickstart on `pixeltable[serve]` 0.7.12 (repo `apreshill/pxt-champs`); projects on 0.7.18. Cloud org on the Community plan.

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
| 3 | `pxt db update` (first upload) | ❌ | Failed after ~3s: `SSL: CERTIFICATE_VERIFY_FAILED` in `_put_artifact` (`urllib.request.urlopen`). Cause: python.org macOS Python 3.12 ships without CA certificates until "Install Certificates.command" is run. `pxt login` and `pxt db status` worked on the same setup; only the artifact upload failed. **Resolved:** linked Python 3.12's OpenSSL cafile to `certifi`, then stopped the background daemon on :22089 (it was still running with the old SSL state). After that, `pxt db update` → `applied AVAILABLE`. |
| 4 | `pxt schema update` + `pxt service update` on Cloud | ✅ | Public URL returned immediately (`https://vinod-lab-main.svc.pxt.run/ingest`). Same commands as local, only the target changes, which is very nice. |
| 5 | `insert.py` against `pxt://vinod-lab:main/docs` | ✅ / ⚠️ | Worked (`Hello` → `HELLO`). Minor confusion: it first prints `Connected to Pixeltable database at: postgresql+psycopg://...pgdata` (the *local* DB) even though the insert goes to Cloud. The docs mention this, but a line like "Using Cloud table pxt://vinod-lab:main/docs" would remove the doubt. |
| 6 | `pxt key create` + `curl` with `X-api-key` | ✅ | Hosted route returned `id`, `title_upper`, `summary` as documented. |
| 7 | Local 0.7.12 vs Cloud | ✅ | `pxt ls` against Cloud returned a clear message: `Unsupported proxy protocol version: 5 (server expects 7) ... update to the latest Pixeltable`. Upgrading to 0.7.18 fixed it. |
| 8 | Resource request above plan limit | ✅ | Asking for 2 vCPU / 6144 MB returned a clear 403 explaining the Community limit (1 vCPU / 2048 MB) and exactly what to set. |
| 9 | Image build with torch + transformers | ✅ | `pxt db update` with BLIP/CLAP dependencies (torch, transformers, librosa, ffmpeg) completed (`generation 2 done`). It took ~30+ minutes; build progress isn't visible beyond "waiting for image build". |
| 10 | BLIP + CLAP on Cloud at 1 vCPU / 2048 MB | ✅ | Inserted 200 sounds (CLAP audio embedding index) and 3 clips → 46 scenes (BLIP captions + CLAP text search via a `@pxt.query` computed column). All values computed; results matched the local run. |
| 11 | Inserts into Cloud tables | ⚠️ | `insert()` into Cloud tables printed nothing, whereas local inserts print "Inserted N rows with 0 errors". Had to check with `pxt ls --counts` / `pxt rows`. |

---

## Project-based testing

| # | Area | Status | Notes |
|---|---|---|---|
| 1 | `@pxt.udf` in a standalone script | ✅ | Defining a UDF at the top level of a script raises a clear error explaining to move it into an importable module. Fixed in one step. |
| 2 | Frame iterator + YOLOX on a 19 s video | ✅ | `frame_iterator(fps=1)` produced 19 frames instantly; `yolox_m` on all frames took ~46 s on an M-series Mac. Stored computed columns mean re-running the script skips finished work, which is great for iterating. |
| 3 | `av` vs `cv2` dylib warning | ⚠️ | With `pixeltable` + `pixeltable-yolox` installed together, macOS prints `objc: Class AVFFrameReceiver is implemented in both .../av/.dylibs/libavdevice... and .../cv2/.dylibs/libavdevice...`. No functional problems observed; only the warning. |
| 4 | Stored stills as an audit trail | ✅ | `frame_iterator(fps=4)` plus a stored computed image column gave a screenshot every 0.25 s. Reading them as contact sheets let us verify exactly what a vision model was given, and settle timing questions (e.g. when the cannon fires) that pausing a video player could not. Frames from the iterator are unstored by default, so this needs the extra stored column; a short docs note on "keep the frames your model saw" would help. |
| 5 | Retrying only failed rows | ✅ | A second computed column that keeps good results and re-calls the model only for failed rows recomputed 3 of 9 shots in 173 s instead of re-running all 9 (~35 min). Incremental computation made this cheap. |
| 6 | Comparing variants as columns | ✅ | Several versions of the same step (`timeline`, `timeline_final`, `timeline_v2`) lived side by side on one view, so outputs could be compared without copying data. |
| 7 | `video_splitter` with computed split points | ✅ | `segment_times=` accepted a computed column (cut times derived from `scene_detect_content`), giving one row per shot. |
| 8 | `scene_detect_content` defaults on old film | ⚠️ | On a 1926 black-and-white clip it missed one hard cut (~24.4 s) at default settings. Expected behaviour for a content detector, but worth knowing when using shots as units. |
| 9 | Script-style audio embedding index | ✅ | `add_embedding_index('audio', audio_embed=clap_audio, string_embed=clap_text)` with custom CLAP UDFs worked, and `similarity(string=...)` searched audio by text. |
| 10 | Ordered, stateful processing | ⚠️ | A step that depends on the previous row (a running "story so far" across consecutive video windows) does not fit computed columns, which are computed per row independently. It worked as a script loop that inserts one row per window, but a pattern for ordered or windowed computations over an iterator view would be useful. |
| 11 | Long-running UDF calls | ✅ | UDFs calling a local model through an OpenAI-compatible client ran for several minutes per row without timeouts or errors from Pixeltable. |
| 12 | Generated audio as a computed column | ✅ | A UDF returning `pxt.Audio` (MusicGen text-to-music) worked as a computed column. Each piece is generated once and stored; changing only mix settings with `update()` left the audio untouched, so iterating on the mix took seconds instead of regenerating. |
| 13 | Audio library index at scale (local) | ✅ | A CLAP embedding index over a filtered FSD50K subset (CC0 / CC-BY clips) built locally with the same `add_embedding_index` call as the 200-clip ESC-50 library. |
| 14 | `overlay_text` with apostrophes | ❌ | Text containing a straight apostrophe followed by a comma fails, e.g. `Et la raison est qu'au début d'une conversation, les gens…` → `No such filter: 'les gens mettent encore leur:fontsize=20…'`. The apostrophe ends the quoted `drawtext` text, so the comma is read as a filter separator. Common in French and in English contractions. Workaround: replace `'` with `’`. |
| 15 | `overlay_text` needs `drawtext` | ⚠️ | Homebrew's default `ffmpeg` on macOS is built without libfreetype, so `overlay_text` fails with `No such filter: 'drawtext'`. Installing `ffmpeg-full` fixed it. A note in the `overlay_text` docs would help. |
| 16 | `clip()` on a file with extra streams | ⚠️ | On an archive.org MP4, `clip()` (which maps all streams, `-map 0`) failed with `Could not find tag for codec h264 in stream #3`. A stream copy keeping only `0:v:0` and `0:a:0` fixed it. An option to keep only the main video/audio streams would help. |
| 17 | Whisper + translation pipeline | ✅ | `extract_audio` → `whisper.transcribe` on a 1-hour lecture, a sentence-embedding index over 705 segments, and `translation()` with four opus-mt models as computed columns all worked first time. Whisper `base.en` scored 5.7% WER against MIT's official captions. |

---

## Things I liked

- **Incremental computation.** Adding a column computed only that column; re-running a script skipped finished work. Iterating on a pipeline never meant re-calling the model.
- **Media as first-class types.** Video, image and audio columns, with frame extraction as a one-line view.
- **Embedding index on audio.** Text-to-audio search with CLAP worked by declaring an index; no separate vector database.
- **Same file, local or Cloud.** Only the target changes (`/foley` vs `pxt://org:main`), and results matched exactly across both.
- **Plan before apply.** `diff` then `update`, with "0 destructive" spelled out, makes changes feel safe.
- **One file for tables and API.** `app.py` holding both the schema and the routes is easy to reason about.
- **Clear error messages.** The UDF-in-script error, the version mismatch and the plan-limit 403 each said exactly what to do. That was very helpful. 
- **Dashboard.** Inputs vs computed columns, UDF badges and column lineage make a pipeline readable at a glance.
- **Built-in video tools.** `make_video` and `with_audio` meant the final renders stayed inside Pixeltable.
