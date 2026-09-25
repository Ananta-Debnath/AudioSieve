# PIVOT STAGE 06 — Post-merge integration + demo readiness

Branch: `wasek` (the `ananta` branch has already been merged in cleanly).
Code freeze: Sun 2026-09-27. Live demo on Mon 2026-09-28 from a laptop in the lab.
**The lab's internet may be missing or slow, so the app must run fully offline.**

## 0. Ground rules

- Do not change any DSP behaviour in `effects/` or in the separation code. Only wiring, fixes,
  packaging, and tests.
- `stft.py` and `utils.py` are shared by both tracks. Touch them only if a merged import is broken,
  and keep the fix minimal.
- Commit each numbered section separately, so any single step can be reverted.

## 1. Verify the merge

1. Run `python -m pytest tests -m "not slow"`, then the slow tests once. Everything that passed
   before the merge must still pass. Fix any breakage caused by the merge.
2. Fix `bass_mask.py`: it imports `drum_mask`, but that module is now `drum_mask_gen.py`. Import
   it correctly. Make sure every module on the separation path imports cleanly:
   `python -c "import <module>"` for each one.
3. Check that `.gitignore` covers both tracks' needs (uploads, results, caches, virtual
   environments, generated audio) and does **not** ignore anything the app needs to run. In
   particular, `static/` (including the demo track and landing assets) and the vendored files
   from section 3 must stay tracked.

## 2. Wire in the real separation

1. Find the separation entry point on the merged code. If there isn't a clean one, create a thin
   wrapper `effects/separation.py` with this shape:

       separate(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]

   - It returns stem name → audio, e.g. `{"drums": ..., "bass": ..., "rest": ...}`.
   - The wrapper only calls the partner's functions. **Do not rewrite their algorithm.**
   - Handle mono and stereo input the same way the other effects do.
2. `/process/separate` must use the real `separate()` and return the stage-04 contract:

       {"run_id": "...",
        "stems": [{"name": "drums", "url": "...", "download_url": "..."}, ...]}

   - Register separation runs in the run registry, like the other tools, so Backstage can
     analyse them.
   - Keep `SEPARATION_MOCK=1` working as a fallback, but it must not be the default.
3. **Measure the separation runtime** on the demo track and on a full 6-minute stereo track. Print
   both timings in the commit message.
   - If a 6-minute track takes more than about 60 s, add a separation-only length cap
     `SEPARATION_MAX_SECONDS` (default 60). The route analyses only the first N seconds.
   - The UI must show a clear note in that case:
     `Separation analyses the first 60 s of the track.`
   - Never let a request run for minutes with no feedback.
4. Check that the Backstage separation panels work for a real separation run (stems listed,
   mask heatmaps if implemented) with no errors.
5. Re-run `scripts/make_landing_assets.py` so the separation card on the landing page gets its
   real preview clip, replacing "coming soon". Commit the regenerated assets.
6. Add tests:
   - `separate()` returns the expected stem names, each stem has the same length as the input,
     and every value is finite.
   - The route returns the contract JSON on a short synthetic clip.

   Keep the test input short so the fast suite stays fast. Mark anything slow
   `@pytest.mark.slow`.

## 3. Offline-proof the frontend

1. **Vendor every CDN dependency** into `static/vendor/`:
   - WaveSurfer.js (the same pinned version)
   - Chart.js (the same pinned version)
   - any Chart.js adapters/plugins in use

   Replace every CDN `<script>` tag with the local file.
2. **Self-host the fonts.** Download Big Shoulders Display and Martian Mono (only the weights in
   use) as `.woff2` into `static/fonts/`. Declare them with `@font-face` and remove the Google
   Fonts `<link>`. Both fonts are under the SIL Open Font License, so include their `OFL.txt`
   license files alongside them.
3. **Verify:** with Wi-Fi turned OFF, load `/`, `/lab`, and every tab. Run each tool once and open
   Backstage. The DevTools Network tab must show **zero** external requests and zero failures.
   Check the fonts render correctly (not a fallback font).

## 4. Reproducible setup on any laptop

1. `requirements.txt`: list every Python dependency actually imported (check the imports across
   the repo), each with a pinned version (`pip freeze` of the working environment, trimmed to
   direct dependencies).
2. **System dependency:** MP3 support needs **ffmpeg**.
   - Detect it at startup. If it's missing, log a clear warning.
   - MP3 uploads then return a clear error saying ffmpeg is needed. WAV and FLAC must still work.
3. Add `run.sh` (and `run.bat` for Windows). Each one:
   - creates or activates a virtual environment
   - installs the requirements
   - runs the app on `http://127.0.0.1:5000`
   - prints that URL
4. **Fresh-clone test:** clone the repo into a new folder, run the script, and complete one full
   run of every tool. Fix whatever breaks: missing files, paths that only worked on the dev
   machine, hard-coded absolute paths, missing directories (create the upload and result folders
   at startup).
5. Make sure debug mode is **off** in the default run configuration, and that the Flask secret
   key (if used) isn't hard-coded to anything sensitive.

## 5. Demo safety

1. **Warm-up script** `scripts/warmup.py`: runs every tool once on the demo track with the demo
   presets, through the same code path as the routes, so imports and caches are warm and any
   error shows up before the presentation.
   - Showcase settings: EQ telephone; reverb RT60 2.5, wet 0.5; echo feedback, 300 ms, g 0.5;
     flanger defaults; separation.
   - Print PASS/FAIL per tool with timings.
2. **Clear old runs:** add an optional `--clean` flag to the run scripts that clears old uploads
   and results, so Backstage's run list starts tidy for the demo.

## 6. Bring main up to date

When sections 1–5 pass, merge `wasek` into `main` (fast-forward if possible) and push. `main` is
what the evaluators will clone. Check that `main` passes the fast test suite after the merge.

## Acceptance checklist

- [ ] All tests pass (fast suite, plus the slow suite once).
- [ ] Real separation works end to end: Studio stem mixer, downloads, Backstage, landing preview.
- [ ] Separation runtime is measured and, if needed, capped with a visible UI note.
- [ ] With Wi-Fi off: zero external requests, the fonts render, every tool works.
- [ ] A fresh clone plus `run.sh` / `run.bat` gives a working app, with no manual fixes.
- [ ] Missing ffmpeg gives a clear message, not a crash.
- [ ] `scripts/warmup.py` prints PASS for all five tools.
- [ ] `main` is up to date, pushed, and green.
