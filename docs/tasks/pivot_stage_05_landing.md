# PIVOT STAGE 05 — SPECTRA landing page

Branch: `wasek` · **Timebox: about 3 hours.** Code freeze Sun 2026-09-27.
Depends on: Stage 04 (Studio + Backstage), which is finished.

## 0. Ground rules

- **Reuse the Stage 04 design system exactly.** Same CSS tokens, the dark/light theme with the
  same `localStorage` key, Big Shoulders Display + Martian Mono, electric cyan, 2px borders, no
  radius, subtle motion only.
  - Import the existing token stylesheet; don't copy it.
- **The landing page makes no backend calls at runtime.** All preview audio and plots are
  pre-generated static files (see section 4), so the page loads instantly and cannot fail during
  the presentation.
- **No DSP in the browser.** Audio elements are for playback only.
- **Do not change the Studio/Backstage behavior.**

## 1. Routing

- `/` → the new landing page (`templates/landing.html`).
- **The app moves to `/lab`.**
- Deep links: `/lab#eq`, `#reverb`, `#echo`, `#flanger`, `#separation`, and `#backstage` open that
  tab. Add hash handling to the existing tab code: on load and on `hashchange`, and update the
  hash when switching tabs.
- Update every internal link and redirect that assumed the app lives at `/`. API routes are unchanged.
- Add tests: `/` returns 200 and contains "SPECTRA"; `/lab` returns 200.

## 2. Page structure (one short scroll)

**Top bar.** Slim and sticky: small `SPECTRA` at the left; at the right, the `SOUND` toggle, the
theme toggle, and `ENTER THE LAB →`.

**Sections, in order:**
1. **Hero** (full viewport height).
2. **Tools reel.**
3. **How it's built.**
4. **Footer.**

**Config.** All editable text lives in one `CONFIG` object at the top of `static/js/landing.js`:

    const CONFIG = {
      appName: "SPECTRA",
      tagline: "Interactive audio signal processing.",
      teamName: "TEAM NAME",                          // placeholder, Wasik will fill
      githubUrl: "https://github.com/OWNER/REPO",     // placeholder, Wasik will fill
      slideMs: 6000,
    };

## 3. Sections

### 3.1 Hero

- **Giant wordmark** `SPECTRA`: Big Shoulders Display 800, `font-size: clamp(6rem, 22vw, 20rem)`,
  line-height 0.85, color `--fg`. Nothing else is decorative. The wordmark *is* the design.
- **Tagline** below it in Martian Mono, around 1.1–1.25rem, color `--fg-dim`:
  `Interactive audio signal processing.`
- **Buttons:**
  - Primary CTA `ENTER THE LAB →` (cyan fill) links to `/lab`.
  - Secondary text link `SEE THE TOOLS ↓` smooth-scrolls to the reel.
- **Load animation:** one subtle fade/slide-up of the wordmark and tagline, about 400ms.
  Disabled under `prefers-reduced-motion`.

### 3.2 Tools reel (stories-style slideshow)

**Layout**
- One large card, about 16:9, with max-width around 960px, centered.
- Above the card: **5 progress segments**, one per tool, like Instagram stories. The active
  segment fills with cyan over `slideMs`, completed ones stay filled, and upcoming ones stay empty.

**Card contents**
- Number and tool name in the heading font, e.g. `02 REVERB`.
- A 1–2 sentence description in Martian Mono, readable size, max about 60 characters per line.
  Use the text in 3.2.1.
- The A/B label (see Audio below).
- `OPEN TOOL →` button, linking to `/lab#<tool>`.
- **Background:** the tool's response plot, drawn large and faint (about 18% opacity) behind the
  text. It uses the static SVGs from section 4, drawn with `currentColor` so they follow the theme.
  Separation uses its faint magma stem spectrogram PNG instead.

**Auto-advance**
- Each slide lasts `CONFIG.slideMs` (6 s). After slide 5 the reel loops back to slide 1.

**Hold to pause**
- Press and hold on the card (pointerdown held ≥ 200ms, mouse or touch) to:
  - pause the progress bar and auto-advance
  - **loop that slide's clip** while held (only if sound is ON)
- On release: stop the audio, and resume progress from where it paused.

**Tap to navigate**
- A quick tap (released under 200ms) on the **left 30%** of the card goes to the previous slide;
  on the rest of the card, it goes to the next slide.
- Taps on `OPEN TOOL →` only follow the link.

**Keyboard**
- ← and → move between slides.
- Space toggles pause/resume, as the keyboard equivalent of hold.
- The reel region is focusable with a visible focus outline.

**Accessibility**
- The region has `aria-roledescription="carousel"`.
- The slide title is announced through an `aria-live="polite"` element.
- Under `prefers-reduced-motion`, **there is no auto-advance.** Slides change only with taps,
  arrows, or the progress segments, and the progress bars just show position.

**Sound**
- The `SOUND` toggle in the top bar **defaults to OFF**. Browsers block audio until the user
  interacts, and this also serves as the mute.
- When sound is ON, **each slide's clip plays once when the slide appears.**
- Toggling OFF stops any playing audio immediately. The preference persists in `localStorage`
  (wrapped in try/catch).

**Audio: the A/B label**
- Each clip is the *same phrase twice*: original first, then processed.
- A label on the card switches **in sync with the audio**:
  - `ORIGINAL` in `--fg` while the original half plays
  - `PROCESSED` in `--accent-text` while the processed half plays (for separation: `STEM: DRUMS`)
- The switch time comes from the clip's metadata JSON (`switch_at_s`); read it with
  `audio.currentTime` in a `timeupdate` / `requestAnimationFrame` loop.
- When sound is off, the label still cycles on the same timing, so the color code is taught visually.

**Robustness**
- **Pause the whole reel, audio included,** when it scrolls out of view (IntersectionObserver
  below 40% visible) or when the browser tab is hidden (`visibilitychange`). Resume when it
  becomes visible again.
- Preload the next slide's clip.
- Prevent the long-press side effects: `user-select: none`, `-webkit-touch-callout: none`, and
  `preventDefault` on `contextmenu` for the card.

#### 3.2.1 Card text

- **01 EQ + FILTER:** Reshape the spectrum: Butterworth filters remove bands, a 5-band EQ
  rebalances them.
- **02 REVERB:** Place the sound in a room by convolving it with an impulse response, via the FFT.
- **03 ECHO + DELAY:** Repeats built from difference equations: one echo (FIR) or decaying
  echoes (IIR).
- **04 FLANGER:** A comb filter whose notches sweep, driven by a slowly varying delay.
- **05 SEPARATION:** Split a mix into stems with non-negative matrix factorization and spectral masks.

### 3.3 How it's built

**Diagram.** An inline SVG (theme-aware via CSS variables; lines in `--fg-dim`, labels in Martian
Mono). It shows **three paths**, not one chain:

                       ┌─ STFT → gain / mask H[k] → ISTFT ──┐  EQ + Filter, Separation
    x[n] ──────────────┼─ zero-pad → FFT → ×H → IFFT ───────┼──► y[n]   Reverb
                       └─ difference equation (time domain) ┘  Echo, Flanger

- Each path is labeled with the tools that use it.
- Include a `<title>`/`<desc>` and a visually hidden text version for screen readers.
- Under 900px wide, stack the paths vertically.

Caption under the diagram:
`Each tool works in the domain that suits it.`

**Four fact blocks** in a 2×2 grid (1 column on mobile). Each has a heading-font title and one line:

1. **NO BLACK BOXES:** Only numpy FFTs and array math. No scipy.signal, no Web Audio effects, no ML.
2. **TESTED AGAINST REFERENCES:** 160+ tests compare every effect with textbook implementations.
3. **EVERY SYSTEM IS VISIBLE:** Each tool plots its frequency or impulse response live.
4. **BACKSTAGE ANALYSIS:** Spectrograms, difference maps and level stats for every run.

End the section with a second `ENTER THE LAB →` CTA.

### 3.4 Footer

- The team name (`CONFIG.teamName`) and a GitHub link (`CONFIG.githubUrl`) with the label
  `SOURCE ON GITHUB ↗`. Nothing else.
- A 2px top border; small but readable text (14px or larger).

## 4. Asset generation script: `scripts/make_landing_assets.py`

A one-off script, run manually. Its outputs are committed under `static/landing/`. It uses the
**real effect functions** from `effects/`, so the previews are genuinely our DSP.

### 4.1 Source

- Input is `static/demo/demo.wav`.
- The phrase start and length are CLI args, with defaults of 1.6 s starting at the loudest region.
  Find that region with a simple RMS scan.

### 4.2 Per-tool showcase settings (chosen to be clearly audible)

| Tool | Settings |
|---|---|
| EQ | `telephone` preset |
| Reverb | rt60 2.5 s, pre-delay 20 ms, wet 0.5, linear |
| Echo | feedback mode, 300 ms delay, g 0.5, mix 0.6 |
| Flanger | its default params with depth near max |
| Separation | the `drums` stem via the merged separation entry point |

**Separation fallback:** if the separation import fails (not merged yet), skip it and print a
warning. The card then shows `PREVIEW COMING SOON`. Re-run the script after the merge.

### 4.3 Processing the phrase

- **Process with context.** Apply each effect to the phrase plus 1 s of pre-roll, then cut the
  pre-roll off. This avoids onset artifacts, especially for flanger and separation.
- **Processed half length:**
  - Reverb and echo: phrase length + 0.6 s, so the tail is heard, with a 150 ms fade-out.
  - Everything else: the phrase length, with a 30 ms fade at each end.
- **Loudness match.** Scale the processed half to the original half's RMS. Otherwise "processed"
  just sounds "quieter", especially the telephone preset.

### 4.4 Output files

**Clip:** original half + 120 ms silence + processed half, saved as
`static/landing/clips/<tool>.mp3` at 128 kbps using pydub (already used in `trim.py`).

**Metadata:** `static/landing/clips/<tool>.json`:

    {"tool": "reverb", "switch_at_s": 1.72, "duration_s": 3.9,
     "label_b": "PROCESSED", "params": {...}}

`label_b` is `"STEM: DRUMS"` for separation.

**Response SVGs:** `static/landing/responses/<tool>.svg`, produced with the same functions as the
`/response/*` routes and the showcase params:
- a single `<polyline>` with `stroke="currentColor"`, fill none, and `viewBox="0 0 1000 400"`
- normalized to fill the box, with no axes and no text

**Separation background:** a faint magma spectrogram PNG of the stem, about 800px wide.

**Size budget:** keep all landing assets under about 1.5 MB total. Print the sizes at the end of
the run.

## 5. Acceptance checklist

- [ ] `/` shows the landing page; `/lab` shows the app; `/lab#reverb` etc. open the right tab,
      including Backstage.
- [ ] The hero wordmark is huge and crisp at 1366×768, at 1920×1080, and on mobile width.
- [ ] The reel auto-advances every 6 s and loops after slide 5. Progress segments are correct.
- [ ] Hold pauses the reel and loops the clip; release resumes. Short taps go back and forward.
      Arrows and space work.
- [ ] Sound defaults OFF. When ON, each slide's clip plays once on appear, and the
      ORIGINAL/PROCESSED label switches in sync with the audio.
- [ ] The reel pauses when scrolled away or when the tab is hidden, with no audio playing off-screen.
- [ ] Reduced motion: no auto-advance, and no animations.
- [ ] Both themes look right, including the faint SVG backgrounds and the diagram.
- [ ] No runtime backend calls from `/` (verify in the DevTools network tab). No console errors.
- [ ] All tests pass, including the new route tests.
- [ ] Commit the script plus the generated assets:
      `pivot stage 05: SPECTRA landing page + preview assets`.

## 6. Out of scope

- Analytics
- A team or supervisor section
- The pivot story on the page (it goes in the README)
- Any new effects
