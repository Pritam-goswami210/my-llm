# Hardware report — Step 1 environment audit

**Machine:** local Windows desktop (`C:\Users\Pritam\Desktop\my-LLM`)
**Audit date:** 2026-10-22
**All values below were actually detected on this machine — nothing guessed.**

---

## 1. Detected specifications

| Component | Detected value | Notes |
|---|---|---|
| OS | Windows 11 Pro, 10.0.26200, 64-bit | PowerShell 5.1 available and used for all checks |
| CPU | Intel Core i5-4570T @ 2.90 GHz | 2 physical cores / 4 logical threads. Haswell generation (2013), low-power 35 W desktop chip. Supports AVX2, which PyTorch uses. |
| PyTorch thread usage | `torch.get_num_threads()` = 2 | PyTorch currently uses 2 threads; we can raise this to 4 later and benchmark |
| RAM | 15.9 GB total / 8.8 GB free | Plenty for tiny models (they will use well under 2 GB) |
| Disk C: | 227.5 GB total / **54.3 GB free** | Where this project lives. More than enough for our whole roadmap. |
| Disk D: | 10 GB total / 9.5 GB free | Small partition/drive; not needed |
| GPU (graphics) | Intel HD Graphics 4600 (integrated), driver 20.19.15.4531 | 2013-era integrated graphics |
| Dedicated GPU VRAM | **Not reliably detectable** | Windows reported "2.03 GB", but that API reports *shared* system memory for integrated graphics and is capped at 4 GB — honest answer: there is no dedicated VRAM because it is an iGPU |
| `nvidia-smi` | Not found | Confirms **no NVIDIA GPU / driver installed** |
| PyTorch CUDA | `torch.cuda.is_available()` = **False**, device count = 0 | Software stack confirms no NVIDIA acceleration |
| Alt acceleration paths | CUDA ✗ (NVIDIA-only) · XPU ✗ (needs a modern Intel Arc-class iGPU/discrete card) · DirectML ✗ (needs a DirectX-12-class GPU; Intel HD 4600/iGPU driver 20.19 predates proper DX12 support) | Conclusion: **CPU-only training** |
| Python | **3.12.10** at `C:\Users\Pritam\AppData\Local\Programs\Python\Python312\python.exe` (default `python`) | We will standardize on 3.12.10 — it enjoys the broadest library wheel support |
| Python (secondary) | 3.14.6 available via the `py` launcher | Exists, but we will not use it — plain 3.12 avoids compatibility surprises |
| pip | 26.2.1 | Working |
| Git | 2.56.0.windows.1 | Working. Note: this folder is **not** a git repository yet (we can `git init` if/when you approve) |
| conda | Not found | Not needed; we will use plain Python/venv or `uv` if you approve |
| uv | 0.13.1 installed | Optional modern package tool; not using it yet |
| **PyTorch** | **2.14.1+cpu — already installed** | Great news: we can start writing code with zero installs |
| numpy / tqdm | 1.26.4 / 4.70.1 — already installed | The three packages in `requirements.txt` are all present |

---

## 2. What this machine can honestly do

**Bottleneck = CPU: only 2 effective cores.** Training speed is limited by raw compute,
not by memory. Memory (RAM and disk) is plentiful for everything in our roadmap.

### Recommended models (character-level tokenizer keeps the vocabulary tiny)

| Config | Params (≈) | Settings | Purpose |
|---|---|---|---|
| **V1 "starter" (recommended first run)** | ~0.8 M | d_model 128 · 4 heads · 4 layers · context 128 · batch 32 · dropout 0.1 · AdamW lr 3e-4 | Trains in **roughly 20–60 min** on ~1 MB of text; produces readable char-level output. This is the Stage-3 target. |
| V2 "stretch" (later) | ~3 M | d_model 256 · 4 heads · 4 layers · context 256 · batch 16–32 | Noticeably better text, roughly 5× slower than V1; a Stage-5 upgrade |
| V3 "overnight" (much later) | ~11–12 M | the config found in `import os.py` (d_model 384 · 6 heads · 6 layers · context 256 · batch 64) | At least a full **day or several** of continuous CPU time. Not practical until we have real benchmarks; plausible only as an overnight experiment after Stage 5 |

Time estimates above are order-of-magnitude engineering estimates from parameter/FLOP math
on a 2-core CPU — we will **measure actual speed with a benchmark script** before committing
to any long run.

### Sizing checks (why these are safe)

- **RAM:** V1 needs well under 1 GB; V2 under ~2 GB. Free RAM is 8.8 GB → fine.
- **Disk:** checkpoint for V1 ≈ 3–4 MB, V2 ≈ 13 MB fp32, V3 ≈ 45 MB. 54 GB free → we could
  keep thousands of checkpoints; everything fits comfortably on a 32 GB USB drive later
  (code + data + checkpoints will be far below 1 GB even for V3).
- **USB portability (Stage 7):** even V3 at ~45 MB per checkpoint + code + a small corpus
  leaves >30 GB of the 32 GB drive free. The portability goal is realistic by design —
  that is exactly why "small but ours" is the right architecture target.

---

## 3. Pre-existing file inspection (done — NOT modified)

The folder already contained one file, which I inspected but left completely untouched:

- **`import os.py`** — a complete, working character-level GPT training script
  (Karpathy-tutorial style) with a custom architecture and a "~12.5 M parameters"
  configuration comment. Two practical issues if it were ever run as-is:
  1. **It works outside this project folder**: it expects `..\data\input.txt` and writes
     `..\checkpoints\` — both are *parent-directory* relative paths, which conflicts with
     your "stay inside the folder" rule.
  2. **Its config is ~15× the recommended V1 size**; at 2 CPU cores, 5000 iterations would
     take on the order of several days of continuous compute.

  Recommendation: keep it as a reference/teaching artifact (it is a fine learning source),
  and build our own cleanly-organized `src/` version in Stage 2–3 rather than editing it.
  Renaming/moving it is your call, not mine.

---

## 4. If the hardware improves someday (for the record)

- An NVIDIA GPU (even a used 4–6 GB card) → CUDA wheels are ~3 GB to install and deliver
  typically **50–100× speedup**; all our code stays the same (PyTorch abstracts the device).
- A modern Intel Arc GPU → XPU path becomes possible.
- Nothing needs to be done now: V1/V2 are designed for exactly this CPU.

---

**Bottom line:** CPU-only training on a modest 2-core CPU, with generous RAM and disk space.
Perfectly adequate for a tiny, genuinely-from-scratch model — starting with the V1
configuration above.
