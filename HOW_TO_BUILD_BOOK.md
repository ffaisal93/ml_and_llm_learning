# How to Build This Repo as an mdBook

This repo is now configured to render as an mdBook — a self-contained, searchable, themed website built from the existing markdown files. Two files were added:

- **`book.toml`** — mdBook configuration. Sets title, theme, search, math support, GitHub link, etc.
- **`SUMMARY.md`** — the table of contents. Auto-generated from the `NN_*/` folder structure (one chapter per folder, README.md as the chapter intro, DEEP_DIVE/INTERVIEW_GRILL/PLAYBOOK/SOLUTIONS as sub-pages in priority order).

Existing files (READMEs, deep dives, grills, solution files, etc.) are *not* moved — mdBook reads them in place via `src = "."` in `book.toml`.

---

## 1. Install mdBook

Pick one of three ways.

### Option A — Pre-compiled binary (fastest, no Rust)

1. Go to https://github.com/rust-lang/mdBook/releases
2. Download the binary for your OS (macOS: `mdbook-vX.Y.Z-x86_64-apple-darwin.tar.gz` or `aarch64-apple-darwin` on Apple Silicon).
3. Extract and put `mdbook` somewhere on your `PATH`:

```bash
# macOS Apple Silicon example
cd ~/Downloads
tar xzf mdbook-v*-aarch64-apple-darwin.tar.gz
mv mdbook /usr/local/bin/
chmod +x /usr/local/bin/mdbook
mdbook --version    # sanity check
```

### Option B — Via Homebrew (macOS)

```bash
brew install mdbook
```

### Option C — Via Cargo (if you have Rust ≥ 1.88)

```bash
# Install Rust first if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Then install mdBook
cargo install mdbook
```

---

## 2. (Recommended) Install the KaTeX preprocessor for math

The repo has hundreds of `$...$` and `$$...$$` math blocks. mdBook's default behavior doesn't render LaTeX — you need a preprocessor.

The `book.toml` is preconfigured to support `mdbook-katex` (commented out by default; uncomment after install).

```bash
cargo install mdbook-katex
```

Then edit `book.toml` and **uncomment** the `[preprocessor.katex]` section at the bottom:

```toml
[preprocessor.katex]
after = ["links"]
macros = ""
leqno = false
fleqn = false
throw-on-error = false
error-color = "#cc0000"
min-rule-thickness = -1
max-size = "infinity"
max-expand = 1000
trust = false
```

Alternative: leave the default MathJax rendering on (already enabled via `mathjax-support = true` in `book.toml`). MathJax is slower but works without an extra binary.

---

## 3. Build & view the book

From the repo root:

```bash
cd /Users/faisal/Projects/ml_and_llm_learning

# Build static HTML output into ./book/
mdbook build

# Or — better for development — start a local server and auto-rebuild on changes:
mdbook serve --open
```

`mdbook serve --open` opens your browser to `http://localhost:3000` automatically. Edit any `.md` file and the page reloads.

Output of `mdbook build` lives in `./book/` (already in `.gitignore` thanks to your existing rules covering `build/`; if not, add it).

---

## 4. (Optional) Publish to GitHub Pages

The cleanest option is to commit the source and have GitHub Actions build the book on every push.

### Workflow file — `.github/workflows/mdbook.yml`

```yaml
name: Deploy mdBook

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    env:
      MDBOOK_VERSION: 0.4.40
    steps:
      - uses: actions/checkout@v4

      - name: Install mdBook
        run: |
          curl -sSL "https://github.com/rust-lang/mdBook/releases/download/v${MDBOOK_VERSION}/mdbook-v${MDBOOK_VERSION}-x86_64-unknown-linux-gnu.tar.gz" \
            | tar -xz -C /usr/local/bin/

      - name: Install mdbook-katex (optional — uncomment if using KaTeX)
        run: cargo install mdbook-katex || true

      - name: Build
        run: mdbook build

      - uses: actions/upload-pages-artifact@v3
        with:
          path: ./book

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

Then in your GitHub repo settings → Pages → set source to "GitHub Actions". Every push to `main` rebuilds and publishes.

---

## 5. Updating the Table of Contents

`SUMMARY.md` was auto-generated. When you add a new file or folder, you have two options:

**Option A: Re-run the generator** (the script you used was inline in your shell, so save this as `tools/gen_summary.py` for reuse):

```python
import os
from pathlib import Path

ROOT = Path('.')
OUT = []
OUT.append('# Summary\n')
OUT.append('[Introduction](README.md)\n')

folders = sorted(
    [p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit() and '_' in p.name],
    key=lambda p: (int(p.name.split('_')[0]), p.name)
)

for folder in folders:
    md_files = sorted(folder.glob('*.md'))
    if not md_files: continue
    readme = folder / 'README.md'
    if readme.exists():
        title_file = readme
        other_files = [f for f in md_files if f.name != 'README.md']
    else:
        title_file = md_files[0]
        other_files = md_files[1:]

    chapter_title = folder.name.replace('_', ' ').title()
    parts = chapter_title.split()
    if parts and parts[0].isdigit():
        parts[0] = parts[0].lstrip('0') or '0'
    chapter_title = ' '.join(parts)

    OUT.append(f'\n# {chapter_title}\n')
    OUT.append(f'- [{folder.name}]({title_file.as_posix()})')

    def sort_key(p):
        name = p.name.upper()
        if 'DEEP_DIVE' in name: return (0, name)
        if 'INTERVIEW_GRILL' in name or 'GRILL' in name: return (1, name)
        if 'PLAYBOOK' in name: return (2, name)
        if 'SOLUTIONS' in name: return (3, name)
        return (9, name)

    for f in sorted(other_files, key=sort_key):
        sub = f.stem.replace('_', ' ').title()
        OUT.append(f'  - [{sub}]({f.as_posix()})')

Path('SUMMARY.md').write_text('\n'.join(OUT))
print("SUMMARY.md regenerated")
```

Run with `python3 tools/gen_summary.py`.

**Option B: Edit `SUMMARY.md` by hand** for fine-grained control over chapter ordering and naming. Format:

```md
# Chapter Title
- [Display name](path/to/file.md)
  - [Sub-page](path/to/file.md)
```

---

## 6. Common gotchas

- **Math doesn't render.** You forgot to install/enable mdbook-katex, or you're relying on default MathJax which can be flaky. Confirm by viewing one of the math-heavy files (e.g. [`04_transformers/MODERN_LLM_ARCHITECTURE_CHOICES.md`](04_transformers/MODERN_LLM_ARCHITECTURE_CHOICES.md)).
- **404 on a chapter.** The path in `SUMMARY.md` doesn't match the actual file location. Double-check.
- **Chapter not appearing.** Check that the file is listed in `SUMMARY.md`. If not, mdBook silently skips it. (You can find orphaned files via `mdbook test` warnings.)
- **`book.toml: src = "."` weirdness.** mdBook will treat *every* `.md` file at the source root as potentially renderable. Files not listed in `SUMMARY.md` get a "draft" warning but don't appear in the sidebar — usually fine.
- **Custom anchors / cross-file links.** Markdown links to other repo files (`[link](07_llm_problems/LLM_EVALUATION_DEEP_DIVE.md)`) work in mdBook automatically.

---

## 7. What you get

- Searchable, single-source view of all 280+ markdown files.
- Navigation sidebar grouped by topic folder.
- Light/dark/Rust/Coal/Navy/Ayu themes.
- Print-to-PDF (each chapter or the whole book).
- Copy-button on code blocks.
- Optional GitHub-Pages hosting via the workflow above.

---

## TL;DR

```bash
brew install mdbook                    # or cargo install mdbook
cargo install mdbook-katex             # optional but recommended for math
# (then uncomment [preprocessor.katex] in book.toml)

cd /Users/faisal/Projects/ml_and_llm_learning
mdbook serve --open
```

That's it. The whole repo is now a browseable, searchable book.
