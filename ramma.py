import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
import shutil
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import threading
import time
import os
import json
import sys
import types
import re
from scipy.signal import sosfilt, sosfilt_zi, butter, iirpeak, iirnotch, resample_poly

# ----------------------------
# LIBROSA / NUMBA GUARD
# librosa.filters is imported by mel_band_roformer.py (lucidrains bs-roformer).
# librosa in turn imports numba, which is broken in many Windows venvs.
# We inject a pure-NumPy implementation of the one function that
# mel_band_roformer actually needs (librosa.filters.mel) before any
# bs_roformer import runs, so numba is never touched.
# If real librosa is already importable we leave it alone.
# ----------------------------
def _ramma_mel_filterbank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None,
                           norm=None, dtype=np.float32):
    if fmax is None:
        fmax = float(sr) / 2.0
    def _hz2mel(f): return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)
    def _mel2hz(m): return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)
    n_freqs = 1 + n_fft // 2
    freqs   = np.linspace(0.0, float(sr) / 2.0, n_freqs, dtype=np.float64)
    mel_pts = np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_mels + 2)
    hz_pts  = _mel2hz(mel_pts)
    W       = np.zeros((n_mels, n_freqs), dtype=dtype)
    for _i in range(n_mels):
        lo, ctr, hi = hz_pts[_i], hz_pts[_i + 1], hz_pts[_i + 2]
        up   = (freqs >= lo)  & (freqs <= ctr)
        down = (freqs >= ctr) & (freqs <= hi)
        if ctr > lo:  W[_i, up]   = (freqs[up]   - lo)  / (ctr - lo)
        if hi  > ctr: W[_i, down] = (hi - freqs[down]) / (hi  - ctr)
    if norm == 1:
        W *= (2.0 / (hz_pts[2:n_mels + 2] - hz_pts[:n_mels]))[:, np.newaxis]
    return W

_librosa_ok = False
try:
    import librosa as _librosa_probe
    _librosa_probe.filters.mel   # confirm the attribute chain works
    _librosa_ok = True
except Exception:
    pass

if not _librosa_ok:
    _lm          = types.ModuleType("librosa")
    _lf          = types.ModuleType("librosa.filters")
    _lf.mel      = _ramma_mel_filterbank
    _lm.filters  = _lf
    sys.modules["librosa"]         = _lm
    sys.modules["librosa.filters"] = _lf
    print("[RAMMA] librosa/numba unavailable — using built-in mel filterbank "
          "(fine for BS-RoFormer models; Mel-Band models such as the karaoke "
          "one need the real librosa: pip install librosa)")

def _load_state_forgiving(m, sd, label="model"):
    """Load weights, tolerating harmless differences.

    A checkpoint and its config often disagree in small ways — a key renamed
    between versions of the architecture, an extra buffer, a head the config
    does not mention. Those load fine with strict=False and the model works.
    Only a wholesale mismatch means the two files do not belong together, so
    that is the only case reported as incompatible.
    """
    try:
        m.load_state_dict(sd)
        return "exact"
    except Exception as strict_err:
        own = m.state_dict()
        usable = {k: v for k, v in sd.items()
                  if k in own and tuple(own[k].shape) == tuple(v.shape)}
        total = max(1, len(own))
        share = len(usable) / total
        if share < 0.5:
            raise RuntimeError(
                f"{label}: only {len(usable)} of {total} weights "
                f"({share * 100:.0f}%) fit this config — the .ckpt and .yaml "
                f"do not describe the same model.\n{strict_err}")
        missing = total - len(usable)
        m.load_state_dict(usable, strict=False)
        print(f"[Models] {label}: loaded {len(usable)} of {total} weights "
              f"({share * 100:.0f}%); {missing} left at their initial values. "
              f"The config and checkpoint differ slightly but are close "
              f"enough to run.")
        return "partial"


def _ensure_librosa():
    """Try the real librosa again, now, and drop our stand-in if it works.

    The probe above runs once while the program starts. If librosa was slow
    to import, or numba was warming up, or anything else went wrong that
    moment, the stand-in was installed for the rest of the session and every
    Mel-Band model was refused even though librosa is installed. This is
    asked again whenever a model actually needs it.
    """
    global _librosa_ok
    if _librosa_ok:
        return True
    shim = sys.modules.get("librosa")
    # Our stand-in is a bare ModuleType with no __file__; a real install has one.
    if shim is not None and getattr(shim, "__file__", None) is None:
        sys.modules.pop("librosa", None)
        sys.modules.pop("librosa.filters", None)
    try:
        import librosa as _probe
        _probe.filters.mel
        _librosa_ok = True
        print("[RAMMA] librosa found — Mel-Band models can use it")
        return True
    except Exception as e:
        # Put the stand-in back so imports inside the model code still work.
        if shim is not None:
            sys.modules["librosa"] = shim
            sys.modules["librosa.filters"] = getattr(shim, "filters", None)
        print(f"[RAMMA] librosa still unavailable ({e}) — falling back to the "
              f"built-in mel filterbank, which usually works")
        return False


# BS-RoFormer-SW via bs-roformer-infer (openmirlab)
# Install: pip install bs-roformer-infer
# Model weights (~400 MB) are downloaded automatically on first run.
# bs-roformer-infer uses PyTorch directly — no onnxruntime, no diffq-fixed.

# ----------------------------
# THEME — INDUSTRIAL EVIL
# ----------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG          = "#0a0a0a"       # near-black forge floor
PANEL       = "#111111"       # recessed steel panel
BORDER      = "#2a2a2a"       # cold iron border
RED         = "#8B0000"       # deep blood red
BRIGHT_RED  = "#cc0000"       # hot-forge red
GLOW_RED    = "#ff2200"       # arc-weld flare
STEEL       = "#3a3a3a"       # brushed steel mid-tone
STEEL_LIGHT = "#555555"       # highlight on steel
TEXT_DIM    = "#666666"       # etched label
TEXT_MAIN   = "#c8c8c8"       # cold white stencil
BRIGHT_GREEN = "#00ff66"      # high-visibility green for action buttons

# Mute / Solo buttons — sized so they stay legible at any window size.
MUTE_ON       = "#cc4400"     # M button when muted; also hover when idle
SOLO_ON       = "#00aa44"     # S button when soloed; also hover when idle
MUTE_ON_HOVER = "#7a2600"     # darker red, hovering an already-muted button
SOLO_ON_HOVER = "#00662a"     # darker green, hovering an already-soloed one

MS_BTN_W    = 26
MS_BTN_H    = 20
FONT_MS_BTN = ("Courier New", 12, "bold")

FONT_TITLE  = ("Courier New", 15, "bold")
FONT_LABEL  = ("Courier New", 15, "bold")   # stem cell names
FONT_SMALL  = ("Courier New",  11)


# ----------------------------
# LOCKED SLIDER — prevents ghost-drag when mouse button is not held
# CTkSlider's internal canvas binds <Motion> globally, which can fire
# even without a button press (e.g. after a fast release).  We fix this
# by tracking the true button state and suppressing value changes when
# the mouse is up.
# ----------------------------
class LockedSlider(ctk.CTkSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mouse_down = False
        # Bind on the slider's internal canvas (created by CTkSlider)
        self.bind("<ButtonPress-1>",   self._on_press,   add="+")
        self.bind("<ButtonRelease-1>", self._on_release, add="+")

    def _on_press(self, event):
        self._mouse_down = True

    def _on_release(self, event):
        self._mouse_down = False

    # Override the internal drag handler so value only changes on real drags
    def _clicked(self, event):
        if self._mouse_down:
            super()._clicked(event)

    def _on_leave(self, event):
        # Also release lock if pointer leaves while button is nominally up
        if event.state & 0x100 == 0:   # Button-1 bit not set in state mask
            self._mouse_down = False
        if hasattr(super(), '_on_leave'):
            super()._on_leave(event)


app = ctk.CTk()
app.title("R·A·M·M·A - STEM ENGINE v0.667.1")
app.configure(fg_color=BG)
app.resizable(True, True)

# ── Minimum window size ────────────────────────────────────────────────────
# 1680x800 is the smallest size the layout is designed and tested for: at
# that size every label fits (see TEXT FITTING). The window can be made as
# large as you like.
#
# On a screen that is itself 1680x800, the title bar and taskbar take some of
# those 800 pixels, so a hard 800px minimum would force the window partly off
# screen. The minimum is therefore capped to the space the screen actually
# has — which on any larger display is the full 1680x800.
MIN_WIN_W, MIN_WIN_H = 1680, 800


def _apply_min_size():
    try:
        sw, sh = app.winfo_screenwidth(), app.winfo_screenheight()
    except Exception:
        sw, sh = MIN_WIN_W, MIN_WIN_H
    # Leave room for the title bar and a taskbar when the screen is small.
    usable_h = sh - 80 if sh <= MIN_WIN_H + 80 else sh
    app.minsize(min(MIN_WIN_W, sw), min(MIN_WIN_H, usable_h))


_apply_min_size()

# Hide the main window until the splash is done
app.withdraw()

# ── SPLASH SCREEN ────────────────────────────────────────────────────────────
# A full-screen-ish Toplevel that shows while the model loads.
# Destroyed and replaced by the main window once init is complete.
# ---------------------------------------------------------------------------
_splash = tk.Toplevel(app)
_splash.overrideredirect(True)   # no title bar / borders

_SP_W, _SP_H = 840, 380
sw = app.winfo_screenwidth()
sh = app.winfo_screenheight()
sx = (sw - _SP_W) // 2
sy = (sh - _SP_H) // 2
_splash.geometry(f"{_SP_W}x{_SP_H}+{sx}+{sy}")
_splash.configure(bg=BG)
_splash.lift()
_splash.attributes("-topmost", True)

# Outer border frame
_sp_border = tk.Frame(_splash, bg=BRIGHT_RED, bd=0)
_sp_border.place(x=0, y=0, width=_SP_W, height=_SP_H)

_sp_inner = tk.Frame(_sp_border, bg=BG, bd=0)
_sp_inner.place(x=2, y=2, width=_SP_W-4, height=_SP_H-4)

# Scanline texture — horizontal lines every 3px for the CRT look
for _y in range(0, _SP_H, 3):
    tk.Frame(_sp_inner, bg="#0d0d0d", height=1).place(x=0, y=_y, width=_SP_W-4)

# ── Keeping text inside the window ─────────────────────────────────────────
# Labels are placed with anchor="center", so anything wider than the splash
# is cropped at both edges. Rather than guessing font sizes that happen to
# fit, measure each label and shrink it until it does; if it still will not
# fit at the smallest readable size, let it wrap onto more lines.
import tkinter.font as _tkfont


def _sp_fit(label, max_w=None, min_size=7, allow_wrap=True):
    """Shrink (and if needed wrap) *label* so it fits the splash width."""
    max_w = max_w or (_SP_W - 24)
    try:
        spec = label.cget("font")
        f    = _tkfont.Font(font=spec)
        fam  = f.actual("family")
        size = abs(int(f.actual("size")))
        wgt  = f.actual("weight")
        text = label.cget("text")

        while size > min_size:
            if _tkfont.Font(family=fam, size=size, weight=wgt).measure(text) <= max_w:
                break
            size -= 1
        label.configure(font=(fam, size, wgt) if wgt == "bold" else (fam, size))

        # Still too wide at the floor: wrap instead of cropping.
        if allow_wrap and _tkfont.Font(family=fam, size=size,
                                       weight=wgt).measure(text) > max_w:
            label.configure(wraplength=max_w, justify="center")
    except Exception:
        pass   # a splash that looks slightly off beats one that crashes
    return label


# Title block
_sp_title = _sp_fit(tk.Label(_sp_inner, text="R·A·M·M·A",
                             font=("Courier New", 42, "bold"),
                             fg=GLOW_RED, bg=BG), allow_wrap=False)
_sp_title.place(relx=0.5, y=52, anchor="center")

# Decorative divider
tk.Frame(_sp_inner, bg=RED, height=1).place(x=40, y=126, width=_SP_W-84)

# Glyph / flavour text
_sp_fit(tk.Label(_sp_inner, text="⚙  INITIALISING BS-ROFORMER ENGINE  ⚙",
                 font=("Courier New", 12),
                 fg=RED, bg=BG), allow_wrap=False
        ).place(relx=0.5, y=152, anchor="center")

# Status label
_sp_status = tk.Label(_sp_inner, text="LOADING…",
                       font=("Courier New", 10, "bold"),
                       fg=TEXT_DIM, bg=BG)
_sp_status.place(relx=0.5, y=224, anchor="center")
_SP_STATUS_FONT = ("Courier New", 10, "bold")

# Progress bar (hand-drawn with a tk.Frame fill)
_SP_BAR_W = _SP_W - 120
_sp_bar_bg = tk.Frame(_sp_inner, bg=PANEL,
                       highlightthickness=1, highlightbackground=STEEL)
_sp_bar_bg.place(relx=0.5, y=248, anchor="center",
                  width=_SP_BAR_W, height=14)

_sp_bar_fill = tk.Frame(_sp_bar_bg, bg=RED)
_sp_bar_fill.place(x=1, y=1, width=0, height=12)

# Animated byline — cycles red → purple → gold
_sp_byline = tk.Label(_sp_inner, text="Program made by: Sai & Eidii with help from megy, Denchik Games, Maggot, deton24 and trexmus - Models by: becruily, gabox, jarredou & unwa",
                       font=("Courier New", 11, "bold"),
                       fg=GLOW_RED, bg=BG)
# Long credits: prefer wrapping onto a second line over shrinking the type
# to the point of being unreadable, so the floor here is higher.
_sp_fit(_sp_byline, max_w=_SP_W - 40, min_size=9)
_sp_byline.place(relx=0.5, y=_SP_H - 14, anchor="s")

_sp_byline.update()

# Colour cycle: glow red → purple → gold, loop forever until splash dies
_SP_COLOURS = ["#e6c000"]
_sp_colour_idx = 0

def _cycle_byline_colour():
    global _sp_colour_idx
    try:
        _sp_byline.configure(fg=_SP_COLOURS[_sp_colour_idx % len(_SP_COLOURS)])
        _sp_colour_idx += 1
        app.after(120, _cycle_byline_colour)
    except Exception:
        pass   # splash destroyed — stop silently

app.after(120, _cycle_byline_colour)

_splash.update()

# Progress stages — (label, fraction_of_bar)
_SP_STAGES = [
    ("IMPORTS READY",          0.15),
    ("BUILDING UI…",           0.35),
    ("LOADING BS-ROFORMER MODEL…",  0.60),
    ("MODEL READY",            0.85),
    ("FINALISING…",            0.95),
]
_sp_stage_idx = 0

def _splash_set(frac, label):
    """Update the splash bar and status label."""
    w = max(0, int((_SP_BAR_W - 2) * frac))
    _sp_bar_fill.place_configure(width=w)
    # Status messages vary in length ("LOADING BS-ROFORMER MODEL…" is the
    # longest), so refit each one rather than letting it run off the edge.
    _sp_status.configure(text=label, font=_SP_STATUS_FONT)
    _sp_fit(_sp_status, allow_wrap=False)
    _splash.update_idletasks()

def _splash_advance(stage_name, frac):
    app.after(0, lambda: _splash_set(frac, stage_name))

# Mark imports done
_splash_set(0.15, "IMPORTS READY")

def _finish_splash():
    """Destroy the splash and reveal the maximised main window."""
    _splash_set(1.0, "READY  ▶")
    _splash.update()
    app.after(320, _do_show_main)

def _do_show_main():
    global _startup
    _splash.destroy()
    # Maximise on all platforms
    try:
        app.state("zoomed")            # Windows / some Linux WMs
    except Exception:
        app.geometry(f"{sw}x{sh}+0+0")  # fallback
    app.deiconify()
    app.lift()
    app.focus_force()
    # Scroll to top so LOAD/PLAY/STOP are immediately visible
    _startup = False
    app.after(100, lambda: _scroll_canvas.yview_moveto(0.0))

def _poll_model_ready():
    """Poll until the model thread is done, then close the splash."""
    if model_ready:
        _finish_splash()
    else:
        _splash_set(0.35 + 0.50 * min(1.0, (_poll_model_ready._ticks * 0.015)),
                    "LOADING BS-ROFORMER MODEL…")
        _poll_model_ready._ticks += 1
        app.after(120, _poll_model_ready)
_poll_model_ready._ticks = 0

# ── END SPLASH SETUP ─────────────────────────────────────────────────────────

running = True

def on_close():
    global running
    running = False
    stop()
    _save_dirs()
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_close)

# ----------------------------
# DEVICE / MODEL  — BS-RoFormer-SW via bs-roformer-infer
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The 6 stems the SW model produces — must match model output order exactly.
_STEM_KEYS  = ["vocals", "drums", "bass", "guitar", "piano", "other"]
# bs-roformer-infer model slug for the SW model by jarredou
_BSR_SLUG   = "roformer-model-bs-roformer-sw-by-jarredou"
# Local cache directory for downloaded weights
_BSR_CACHE  = os.path.join(os.path.expanduser("~"), ".cache", "bs-roformer-infer")

# ----------------------------
# INFERENCE PERFORMANCE
# The model is small enough that one 8-second chunk can't keep a modern GPU
# busy: most of the time is spent waiting on kernel launches and on copying
# each result back to the CPU. Feeding several chunks per forward pass, and
# running them in half precision, keeps the card saturated instead.
# ----------------------------
# Set _SAFE_MODE = True to turn every GPU optimisation off in one go
# (one chunk at a time, full precision, no fused attention) — the behaviour
# this app had before the speed-ups. Useful for ruling the GPU path out.
_SAFE_MODE   = False
_INFER_FP16  = True   # half precision on CUDA — roughly 2x faster

# Chunking. Every chunk overlaps its neighbours by _OVERLAP_SECONDS at each
# end, and the overlap is computed twice — so short chunks with wide overlap
# waste real time: 8 s chunks with 0.5 s overlap process 1.14x the track,
# 12 s chunks with 0.25 s overlap only 1.04x.
# Follow each model's own config for chunking. A RoFormer is trained at one
# chunk length (audio.chunk_size) and evaluated with a set number of
# overlapping passes (inference.num_overlap); feeding it shorter chunks with
# a token crossfade costs separation quality, most audibly at chunk seams.
# Set _USE_MODEL_CHUNKING = False to go back to the fixed values below.
_USE_MODEL_CHUNKING = True
_MAX_MODEL_CHUNK_S  = 20.0   # ceiling, so a huge chunk_size cannot exhaust VRAM
_MAX_NUM_OVERLAP    = 8      # ceiling: each extra pass costs proportional time.
                             # The karaoke model asks for 8 and is small and
                             # quick, so it is allowed; the big separators
                             # ask for 2 anyway. Lower this to trade a little
                             # quality for speed.

_CHUNK_SECONDS   = 12.0
_OVERLAP_SECONDS = 0.25

# How much audio to push through the model in one forward pass. The batch is
# derived from this so changing the chunk length doesn't change VRAM use.
_BATCH_SECONDS = 36.0

# The instrumental model returns one stem instead of six, so its activations
# and its output are far smaller — it can take a bigger bite per pass.
_INST_BATCH_FACTOR = 2.0

# Chunks quieter than this are silence (lead-ins, gaps, fade-outs); they are
# written as zeros instead of being pushed through the model.
_SILENCE_PEAK = 1e-4

# Run the instrumental model at the same time as the stems, or after them.
# Sharing one GPU between two models doesn't make the pair finish sooner, it
# just makes the song you're waiting for take about twice as long to appear.
# Sequential means the song is playable in roughly half the time and the
# instrumental fills in shortly after.
_INST_CONCURRENT = False

# Separate the instrumental at all. Turning this off halves the GPU work per
# song; the INSTRUM cell then stays empty.
_INST_AUTO = True

# Run the instrumental pass BEFORE the stems. The backing track is then ready
# first and is playable on its own while the six stems are still separating —
# useful when the instrumental is what you're really after. The stems arrive
# second, so the mixer cells fill in later than usual.
_INST_FIRST = False

# Instrumental only: skip the six-stem model entirely and separate nothing but
# the backing track. Half the GPU work of a normal load, and the fastest way
# to get an instrumental — the six mixer cells stay empty, since those stems
# are never produced.
_INST_ONLY = False

if _SAFE_MODE:
    _INFER_FP16      = False
    _CHUNK_SECONDS   = 8.0
    _OVERLAP_SECONDS = 0.5
    _BATCH_SECONDS   = 8.0

# TF32 matmuls and cuDNN autotuning: free speed on Ampere and newer.
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

# ── Keeping the interface responsive while a model runs ─────────────────────
# Separation competes with the window for the CPU. Even on CUDA, a RoFormer's
# forward pass does a lot of CPU-side work (STFT set-up, reshapes, the
# Python between kernels), and PyTorch spreads that over every core by
# default — leaving none for Tk, which is what makes windows such as the
# LOADLIST stutter while a track separates. Three measures:
#
# 1. Leave one core for the interface.
try:
    _cores = os.cpu_count() or 2
    torch.set_num_threads(max(1, _cores - 1))
except Exception:
    pass

# 2. Hand the interpreter lock over more often. The UI thread has to hold it
#    to redraw anything; the default 5 ms turn means a busy worker can make
#    each redraw wait several turns in a row.
try:
    sys.setswitchinterval(0.002)
except Exception:
    pass


# 3. Run separation threads at a lower OS priority (Windows), so that when the
#    CPU is contended the scheduler serves the window first. Separation loses
#    almost nothing: the interface needs very little time, just promptly.
def _run_low_priority(fn, *args, **kwargs):
    """Thread target: lower this thread's priority, then run *fn*."""
    _lower_thread_priority()
    return fn(*args, **kwargs)


def _lower_thread_priority():
    if os.name != "nt":
        return
    try:
        import ctypes
        THREAD_PRIORITY_BELOW_NORMAL = -1
        k32 = ctypes.windll.kernel32
        k32.SetThreadPriority(k32.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL)
    except Exception:
        pass


_fade_cache: dict = {}


def _fade_tensor(fade, length, device, dtype):
    """The overlap-add window as a tensor shaped to broadcast over a batch.

    Cached: the same window is used for every batch of a separation, so it is
    built and uploaded once rather than per batch.
    """
    key = (id(fade), int(length), str(device), str(dtype))
    t = _fade_cache.get(key)
    if t is None:
        w = np.asarray(fade[:length], dtype=np.float32)
        if len(w) < length:          # last chunk: pad with zeros
            w = np.pad(w, (0, length - len(w)))
        t = torch.from_numpy(w)
        t = t.to(device=device, dtype=dtype) if dtype is not None else t.to(device)
        t = t.view(1, 1, 1, -1)      # (B, stems, ch, T)
        if len(_fade_cache) > 8:
            _fade_cache.clear()
        _fade_cache[key] = t
    return t


def _infer_ctx():
    """inference_mode where available; no_grad on older PyTorch."""
    try:
        return torch.inference_mode()
    except AttributeError:
        return torch.no_grad()


def _amp_ctx(run_device, use_fp16=None):
    """Half-precision autocast on CUDA, a no-op everywhere else."""
    if use_fp16 is None:
        use_fp16 = _INFER_FP16
    if use_fp16 and run_device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    import contextlib
    return contextlib.nullcontext()


def _free_vram_gb():
    """Free VRAM in GB, or None when that can't be determined."""
    try:
        if device.type != "cuda":
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:
        return None


def _chunking_from_config(cfg, sr, fallback_chunk=None):
    """(chunk seconds, overlap seconds per side) for a model's own config.

    num_overlap is how many passes cover each sample: 2 means the hop is half
    a chunk, so the overlap on each side is a quarter of it.
    """
    chunk_s = fallback_chunk or _CHUNK_SECONDS
    overlap_s = _OVERLAP_SECONDS
    if not _USE_MODEL_CHUNKING or cfg is None:
        return chunk_s, overlap_s
    try:
        raw = int(cfg.audio.chunk_size) / float(sr or 44100)
        if raw > 0:
            chunk_s = max(4.0, min(_MAX_MODEL_CHUNK_S, raw))
    except Exception:
        pass

    # Snap the chunk down to a whole number of STFT frames. The model turns
    # each chunk into frames of stft_hop_length samples; a chunk that is not
    # a multiple of that leaves a ragged part-frame at the end of every
    # chunk, which lands right where chunks are stitched together and takes
    # the edges — the weakest part of the output — with it. Note this is
    # stft_hop_length, not audio.hop_length: the two differ in some configs
    # and it is the former the model actually uses.
    try:
        hop = int(cfg.model.get("stft_hop_length", 0) or 0)
    except Exception:
        hop = 0
    if not hop:
        try:
            hop = int(cfg.audio.hop_length)
        except Exception:
            hop = 0
    if hop > 0:
        samples = int(round(chunk_s * float(sr or 44100)))
        snapped = (samples // hop) * hop
        if snapped >= hop and snapped != samples:
            print(f"[Chunking] {samples} samples is {samples / hop:.3f} STFT "
                  f"frames of {hop}; using {snapped} ({snapped // hop} whole "
                  f"frames) so chunks line up with the model's own grid")
        if snapped >= hop:
            chunk_s = snapped / float(sr or 44100)
    try:
        num_overlap = int(cfg.inference.num_overlap)
    except Exception:
        num_overlap = 1
    num_overlap = max(1, min(_MAX_NUM_OVERLAP, num_overlap))
    if num_overlap > 1:
        # hop = chunk / num_overlap, so each side overlaps by half the gap
        overlap_s = chunk_s * (1.0 - 1.0 / num_overlap) / 2.0
    return chunk_s, overlap_s


def _batch_size_for(run_device, chunk_seconds=None):
    """Chunks per forward pass, scaled to the VRAM actually free.

    Three models can be resident at once (six-stem, instrumental, karaoke),
    so a fixed batch that fits on an empty card will not fit on a busy one.
    The budget shrinks with the free memory rather than waiting for an OOM
    and backing off afterwards.
    """
    if run_device.type != "cuda":
        return 1
    chunk_seconds = chunk_seconds or _CHUNK_SECONDS

    budget = _BATCH_SECONDS
    free = _free_vram_gb()
    if free is not None:
        if free < 1.5:
            budget = min(budget, chunk_seconds)          # one chunk at a time
        elif free < 3.0:
            budget = min(budget, chunk_seconds * 2)
        elif free < 5.0:
            budget = min(budget, chunk_seconds * 3)
        elif free >= 10.0:
            # Plenty spare: larger batches keep the card busier between
            # kernel launches. Still bounded, and an OOM halves it anyway.
            budget = max(budget, chunk_seconds * 6)
        elif free >= 7.0:
            budget = max(budget, chunk_seconds * 4)
    return max(1, int(budget / max(1.0, chunk_seconds)))


# The stems pass and the instrumental pass read the same file. Decoding and
# resampling an mp3 twice is pure waste, so the last decode is kept here.
_decoded_cache      = {}            # path -> (audio ndarray, sr)
_decoded_cache_lock = threading.Lock()


def _read_audio_cached(path):
    """_read_audio_file with a one-entry cache, shared by both passes."""
    with _decoded_cache_lock:
        hit = _decoded_cache.get(path)
    if hit is not None:
        audio, sr = hit
        return audio.copy(), sr
    audio, sr = _read_audio_file(path)
    with _decoded_cache_lock:
        _decoded_cache.clear()       # keep only the most recent file
        _decoded_cache[path] = (np.asarray(audio, dtype=np.float32), int(sr))
    return audio, sr


# Files the six-stem model itself uses. Recorded so the instrumental loader
# can never pick them up: loading the same checkpoint into both slots puts two
# copies of the same model on the GPU and runs both over every song.
_bsr_ckpt_resolved = None
_bsr_cfg_resolved  = None

# The two model loaders run on separate threads and both need pieces of the
# bs_roformer package. Importing its submodules from two threads at once, in
# different orders, trips Python's own import machinery:
#
#   deadlock detected by _ModuleLock('bs_roformer.inference')
#
# So every import of the package goes through here: one lock, done once, and
# called on the main thread before either loader starts.
_bsr_import_lock  = threading.Lock()
_bsr_import_cache = {}


def _import_bs_roformer():
    """Import bs_roformer once, under a lock, and hand back what's needed."""
    with _bsr_import_lock:
        if _bsr_import_cache:
            return _bsr_import_cache

        import importlib
        pkg = importlib.import_module("bs_roformer")

        # Pull the submodule in the same breath, so no other thread can start
        # importing the package while this one is part-way through it.
        yaml_loader = None
        try:
            inference   = importlib.import_module("bs_roformer.inference")
            yaml_loader = getattr(inference, "SafeLoaderWithTuple", None)
        except Exception as e:
            print(f"[BS-RoFormer] bs_roformer.inference unavailable ({e}) — "
                  f"using a plain tuple-aware YAML loader")

        if yaml_loader is None:
            import yaml as _yaml

            class yaml_loader(_yaml.SafeLoader):
                pass

            yaml_loader.add_constructor(
                "tag:yaml.org,2002:python/tuple",
                lambda loader, node: tuple(loader.construct_sequence(node)))

        _bsr_import_cache.update(
            registry=getattr(pkg, "MODEL_REGISTRY", None),
            get_model_from_config=getattr(pkg, "get_model_from_config", None),
            ensure_model_assets=getattr(pkg, "ensure_model_assets", None),
            yaml_loader=yaml_loader,
        )
        return _bsr_import_cache

# ============================================================
# MODEL PICKER
# Finds the .ckpt files in the models folder, pairs each with its .yaml, and
# lets a cell choose which one it uses. Adding a model is therefore a matter
# of dropping the two files in the folder — nothing here needs editing.
# ============================================================
# Filename fragments that say who trained a model, so a cell can credit them.
_MODEL_AUTHORS = [
    ("becruily",     "becruily"),
    ("gabox",        "gabox"),
    ("aufr33",       "aufr33"),
    ("unwa",         "unwa"),
    ("gilliaan",     "gilliaan"),
    ("jarredou",     "jarredou"),
    ("viperx",       "ViperX"),
    ("kimberley",    "Kim"),
    ("mesk",         "MESK"),
    ("sucial",       "Sucial"),
    ("zfturbo",      "ZFTurbo"),
]

# Models whose filename carries no author, matched on the model's own name.
# "BS-Roformer-Resurrection-Inst.ckpt" says nothing about unwa, and the
# bowed-strings checkpoint nothing about gilliaan, so they are named here.
_MODEL_NAME_AUTHORS = [
    ("resurrection",  "unwa"),
    ("bowed",         "gilliaan"),
    ("string",        "gilliaan"),
    ("sw-fixed",      "jarredou"),
    ("sw_fixed",      "jarredou"),
    ("bs-rofo-sw",    "jarredou"),
]

# Which files belong to which cell. A name matching any fragment counts.
_MODEL_ROLES = {
    "karaoke":      ("karaoke", "kara", "lead_back", "leadback"),
    "instrumental": ("inst", "resurrection"),
    "strings":      ("bowed", "string", "violin", "cello"),
    "vocals":       ("vocals", "voc"),
    "main":         ("sw-fixed", "sw_fixed", "bs-rofo-sw", "6stem", "six"),
}


def _model_author(name):
    """Who trained this model, from its filename, or an empty string.

    Most community checkpoints carry the author's name; the ones that do not
    are matched on the model's own name instead.
    """
    low = os.path.basename(name).lower()
    for token, author in _MODEL_AUTHORS:
        if token in low:
            return author
    for token, author in _MODEL_NAME_AUTHORS:
        if token in low:
            return author
    return ""


def _model_title(name):
    """A readable name for a checkpoint file."""
    base = os.path.splitext(os.path.basename(name))[0]
    for token, author in _MODEL_AUTHORS:
        base = re.sub(rf"[-_ ]*{re.escape(token)}[-_ ]*", " ", base,
                      flags=re.IGNORECASE)
    base = base.replace("_", " ").replace("-", " ").strip()
    return " ".join(base.split()).upper() or os.path.basename(name).upper()


def _pair_config_for(ckpt, folder):
    """The .yaml that goes with a checkpoint: the closest name match."""
    base = os.path.splitext(os.path.basename(ckpt))[0].lower()
    yamls = [f for f in os.listdir(folder)
             if f.lower().endswith((".yaml", ".yml"))]
    if not yamls:
        return None
    # Exactly the same name wins, then the longest shared run of characters.
    for y in yamls:
        if os.path.splitext(y)[0].lower() == base:
            return os.path.join(folder, y)

    def score(y):
        yb = os.path.splitext(y)[0].lower().replace("config", "").strip("_- ")
        common = 0
        for token in re.split(r"[-_ .]+", yb):
            if token and token in base:
                common += len(token)
        return common

    best = max(yamls, key=score)
    return os.path.join(folder, best) if score(best) > 2 else None


def scan_models(role):
    """Every (label, ckpt, yaml) pair in the models folder for *role*."""
    folder = _MODELS_DIR
    out = []
    if not os.path.isdir(folder):
        return out
    hints = _MODEL_ROLES.get(role, ())
    # "vocals" would otherwise match the karaoke and instrumental files too.
    excl = {"vocals": _VOC_NAME_EXCLUDE,
            "karaoke": _KARA_NAME_EXCLUDE,
            "instrumental": _INST_NAME_EXCLUDE}.get(role, ())
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".ckpt", ".pth", ".th")):
            continue
        low = name.lower()
        if excl and any(x in low for x in excl):
            continue
        if hints and not any(h in low for h in hints):
            continue
        ckpt = os.path.join(folder, name)
        cfg = _pair_config_for(ckpt, folder)
        if not cfg:
            continue
        author = _model_author(name)
        label = _model_title(name) + (f"  ·  {author}" if author else "")
        out.append((label, ckpt, cfg))
    return out


def _role_model(role):
    """The loaded model object for a role, or None."""
    return {"karaoke": lambda: kara_model,
            "instrumental": lambda: inst_model,
            "strings": lambda: strings_model,
            "vocals": lambda: vocals_model,
            "main": lambda: model}.get(role, lambda: None)()


def _model_paths(role):
    return {"karaoke": (_KARA_CKPT_PATH, _KARA_CFG_PATH),
            "instrumental": (_INST_CKPT_PATH, _INST_CFG_PATH),
            "strings": (_STR_CKPT_PATH, _STR_CFG_PATH),
            "vocals": (_VOC_CKPT_PATH, _VOC_CFG_PATH),
            "main": (_MAIN_CKPT_PATH, _MAIN_CFG_PATH)}.get(role, ("", ""))


def _restore_model_paths(role, paths):
    """Put a role's checkpoint paths back after a failed switch."""
    global _KARA_CKPT_PATH, _KARA_CFG_PATH, kara_model, kara_model_ready
    global _INST_CKPT_PATH, _INST_CFG_PATH, inst_model, inst_model_ready
    global _STR_CKPT_PATH, _STR_CFG_PATH, strings_model, strings_model_ready
    global _VOC_CKPT_PATH, _VOC_CFG_PATH, vocals_model, vocals_model_ready
    global _MAIN_CKPT_PATH, _MAIN_CFG_PATH, model, model_ready
    ck, cf = paths
    if role == "karaoke":
        _KARA_CKPT_PATH, _KARA_CFG_PATH = ck, cf
        kara_model, kara_model_ready = None, False
    elif role == "instrumental":
        _INST_CKPT_PATH, _INST_CFG_PATH = ck, cf
        inst_model, inst_model_ready = None, False
    elif role == "strings":
        _STR_CKPT_PATH, _STR_CFG_PATH = ck, cf
        strings_model, strings_model_ready = None, False
    elif role == "vocals":
        _VOC_CKPT_PATH, _VOC_CFG_PATH = ck, cf
        vocals_model, vocals_model_ready = None, False
    elif role == "main":
        _MAIN_CKPT_PATH, _MAIN_CFG_PATH = ck, cf
        model, model_ready = None, False


def _model_error_box(role, ckpt, cfg):
    """Tell the user the files did not fit this slot, and why that happens."""
    try:
        messagebox.showerror(
            "RAMMA — model not compatible",
            f"{os.path.basename(ckpt)}\n\ncould not be loaded as the "
            f"{role} model.\n\nThe usual causes are:\n"
            f"  •  the .yaml does not belong to this .ckpt\n"
            f"  •  the checkpoint is a different architecture than the "
            f"config describes\n"
            f"  •  it is a Mel-Band model and librosa is not installed\n\n"
            f"Config used:\n{os.path.basename(cfg) if cfg else '(none)'}\n\n"
            f"The previous model has been put back. The console shows the "
            f"error the loader reported.")
    except Exception:
        pass


def _apply_model_choice(role, ckpt, cfg):
    """Point a role at a different checkpoint and load it again."""
    prev = _model_paths(role)
    global _KARA_CKPT_PATH, _KARA_CFG_PATH, kara_model, kara_model_ready
    global _INST_CKPT_PATH, _INST_CFG_PATH, inst_model, inst_model_ready
    global _STR_CKPT_PATH, _STR_CFG_PATH, strings_model, strings_model_ready
    global _VOC_CKPT_PATH, _VOC_CFG_PATH, vocals_model, vocals_model_ready
    global _MAIN_CKPT_PATH, _MAIN_CFG_PATH, model, model_ready

    title, author = _model_title(ckpt), _model_author(ckpt)
    print(f"[Models] {role}: switching to {os.path.basename(ckpt)}"
          + (f" (by {author})" if author else ""))

    if role == "karaoke":
        _KARA_CKPT_PATH, _KARA_CFG_PATH = ckpt, cfg
        kara_model, kara_model_ready = None, False
        loader = load_kara_model
    elif role == "instrumental":
        _INST_CKPT_PATH, _INST_CFG_PATH = ckpt, cfg
        inst_model, inst_model_ready = None, False
        loader = load_inst_model
    elif role == "strings":
        _STR_CKPT_PATH, _STR_CFG_PATH = ckpt, cfg
        strings_model, strings_model_ready = None, False
        loader = load_strings_model
    elif role == "vocals":
        _VOC_CKPT_PATH, _VOC_CFG_PATH = ckpt, cfg
        vocals_model, vocals_model_ready = None, False
        loader = load_vocals_model
    elif role == "main":
        _MAIN_CKPT_PATH, _MAIN_CFG_PATH = ckpt, cfg
        model, model_ready = None, False
        loader = load_model
    else:
        return

    _set_model_credit(role, title, author)

    def _load_and_check():
        loader()
        if _role_model(role) is not None:
            print(f"[Models] {role}: {os.path.basename(ckpt)} loaded")
            return
        # The loader prints why; put the previous model back so the cell is
        # not left with nothing, and say plainly that the file did not fit.
        print(f"[Models] {role}: {os.path.basename(ckpt)} could not be used — "
              f"restoring the previous model")
        _restore_model_paths(role, prev)
        if prev[0]:
            loader()
        if running:
            app.after(0, lambda: _model_error_box(role, ckpt, cfg))
        app.after(0, sync_model_credits) if running else None

    threading.Thread(target=_load_and_check, daemon=True).start()


# Cells show the name and author of whichever model they are using.
_model_credit_labels = {}     # role -> [(title label, author label), ...]


def _register_model_credit(role, title_lbl, author_lbl):
    _model_credit_labels.setdefault(role, []).append((title_lbl, author_lbl))


def _set_model_credit(role, title, author):
    """Show a model's name and author on every cell that uses that role.

    An unknown author leaves the existing credit alone rather than replacing
    it with a dash: not recognising a filename is no reason to strip the
    name of whoever made the model.
    """
    for title_lbl, author_lbl in _model_credit_labels.get(role, []):
        try:
            title_lbl.configure(text=title)
            if author:
                author_lbl.configure(text=author)
        except Exception:
            pass


def _found_ckpt_for(role):
    """The checkpoint a role's own search would pick, or an empty string."""
    try:
        if role == "karaoke":
            return _kara_find_files()[0] or ""
        if role == "instrumental":
            return _inst_find_files()[0] or ""
        if role == "strings":
            return _str_find_files()[0] or ""
        if role == "vocals":
            return _voc_find_files()[0] or ""
    except Exception:
        pass
    return ""


def sync_model_credits():
    """Name the models actually in use, rather than the built-in defaults.

    With several karaoke models installed the cells should credit the one
    being loaded, not whichever was hard-coded when the cell was written.
    """
    for role in ("karaoke", "instrumental", "strings", "vocals"):
        ckpt = _found_ckpt_for(role)
        if not ckpt:
            continue
        _set_model_credit(role, _model_title(ckpt), _model_author(ckpt))


_BROWSE_LABEL = "⋯  BROWSE FOR A MODEL…"


def browse_for_model(role):
    """Pick a .ckpt from anywhere, and the .yaml that goes with it."""
    ckpt = filedialog.askopenfilename(
        title=f"Choose the {role} checkpoint (.ckpt)",
        initialdir=_MODELS_DIR if os.path.isdir(_MODELS_DIR) else None,
        filetypes=[("Model checkpoints", "*.ckpt *.pth *.th"),
                   ("All files", "*.*")])
    if not ckpt:
        return
    # Always ask for the config as well. A checkpoint and a config that do
    # not belong together fail in confusing ways, so the pairing is the
    # user's to make rather than something guessed from filenames. A likely
    # match is offered as the starting selection.
    folder = os.path.dirname(ckpt)
    guess = _pair_config_for(ckpt, folder)
    cfg = filedialog.askopenfilename(
        title=f"Choose the .yaml config for {os.path.basename(ckpt)}",
        initialdir=folder,
        initialfile=os.path.basename(guess) if guess else "",
        filetypes=[("Model configs", "*.yaml *.yml"),
                   ("All files", "*.*")])
    if not cfg:
        print("[Models] No config chosen — a checkpoint cannot be loaded "
              "without one.")
        if running:
            try:
                messagebox.showerror(
                    "RAMMA — no config chosen",
                    f"{os.path.basename(ckpt)} needs its .yaml config file "
                    f"as well.\n\nNothing has been changed.")
            except Exception:
                pass
        return
    print(f"[Models] Chosen: {os.path.basename(ckpt)} + "
          f"{os.path.basename(cfg)}")
    _apply_model_choice(role, ckpt, cfg)


def add_model_picker(cell, role, after=None):
    """A collapsible "MODEL" row: click it to choose a different checkpoint.

    Folded by default, because most of the time the choice never changes.
    """
    wrap = ctk.CTkFrame(cell, fg_color="transparent")
    if after is not None:
        wrap.pack(fill="x", after=after)
    else:
        wrap.pack(fill="x")

    open_state = [False]
    menu_holder = ctk.CTkFrame(wrap, fg_color="transparent")

    header = ctk.CTkButton(wrap, text="▸ MODEL",
                           fg_color=STEEL, hover_color=STEEL_LIGHT,
                           text_color="#e6c000",     # gold: legible on steel
                           font=("Courier New", 11, "bold"),
                           corner_radius=0, border_width=1,
                           border_color=BORDER, height=18)
    header.pack(fill="x", padx=6, pady=(1, 0))

    menu = ctk.CTkOptionMenu(
        menu_holder, values=["(no models found)"],
        font=("Courier New", 10), dropdown_font=("Courier New", 11),
        fg_color=STEEL, button_color=RED, button_hover_color=BRIGHT_RED,
        dropdown_fg_color=PANEL, dropdown_hover_color=STEEL,
        text_color=TEXT_MAIN, corner_radius=0, height=22)
    menu.pack(fill="x", padx=6, pady=(1, 2))

    def _refresh():
        pairs = scan_models(role)
        if not pairs:
            menu.configure(values=[_BROWSE_LABEL, "(none in the models folder)"])
            menu.set("(none in the models folder)")
            return []
        menu.configure(values=[p[0] for p in pairs] + [_BROWSE_LABEL])
        # Show the one actually in use: the explicit path when set, else
        # whatever the loader's own search found.
        current = {"karaoke": _KARA_CKPT_PATH, "instrumental": _INST_CKPT_PATH,
                   "strings": _STR_CKPT_PATH, "vocals": _VOC_CKPT_PATH,
                   "main": _MAIN_CKPT_PATH}.get(role, "")
        if not current:
            current = _found_ckpt_for(role)
        for label, ckpt, _cfg in pairs:
            if current and os.path.abspath(ckpt) == os.path.abspath(current):
                menu.set(label)
                break
        else:
            menu.set(pairs[0][0])
        return pairs

    def _chosen(label):
        if label == _BROWSE_LABEL:
            browse_for_model(role)
            _refresh()
            return
        for lab, ckpt, cfg in scan_models(role):
            if lab == label:
                _apply_model_choice(role, ckpt, cfg)
                return

    menu.configure(command=_chosen)

    def _toggle():
        open_state[0] = not open_state[0]
        if open_state[0]:
            _refresh()
            menu_holder.pack(fill="x", after=header)
            header.configure(text="▾ MODEL")
        else:
            menu_holder.pack_forget()
            header.configure(text="▸ MODEL")

    header.configure(command=_toggle)
    return wrap


# ============================================================
# TIPS
# A read-only window of tips, shown from the TIPS button. Editing needs the
# editor passphrase. Only its SHA-256 hash is kept, on the line below, and it
# is written there the first time you set one — so publishing ramma.py
# publishes the lock, never the passphrase. The text lives in tips.txt
# beside this file; list tips.txt in _UPDATE_FILES and your edits reach
# everyone with the next update.
# ============================================================
_TIPS_EDITOR_HASH = "4b22cc2644f3e1c4889cef71fba9fcf2c7fe95ac5d827bff0f4556191a29c9c2"
_TIPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tips.txt")
_TIPS_DEFAULT = ("No tips yet.\n\nThe author of this copy of RAMMA can add "
                 "some with the EDIT button.")


def _read_tips():
    try:
        with open(_TIPS_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        return text if text.strip() else _TIPS_DEFAULT
    except FileNotFoundError:
        return _TIPS_DEFAULT
    except Exception as e:
        return f"The tips could not be read: {e}"


def _write_tips(text):
    with open(_TIPS_PATH, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def _passphrase_hash(phrase):
    import hashlib
    return hashlib.sha256(("RAMMA-tips:" + phrase).encode("utf-8")).hexdigest()


def _store_editor_hash(h):
    """Write the hash into this file, so it travels with ramma.py."""
    global _TIPS_EDITOR_HASH
    me = os.path.abspath(__file__)
    with open(me, "rb") as f:
        data = f.read()
    new = re.sub(rb'^_TIPS_EDITOR_HASH = "[0-9a-f]*"',
                 b'_TIPS_EDITOR_HASH = "' + h.encode() + b'"',
                 data, count=1, flags=re.MULTILINE)
    if new == data:
        raise RuntimeError("the _TIPS_EDITOR_HASH line was not found")
    with open(me, "wb") as f:
        f.write(new)
    _TIPS_EDITOR_HASH = h


# ============================================================
# AUTO-UPDATE
# Checks the GitHub repository for a newer ramma.py and offers to install it.
# The new file is compiled before it replaces anything, and the running one
# is kept as a backup, so a bad or truncated download cannot leave you
# without a working program.
# ============================================================
_UPDATE_REPO   = "MizakiTheCondemned/RAMMA"            # "user/repo" — set this to switch updates on
_UPDATE_BRANCH = "main"
_UPDATE_FILE   = "ramma.py"    # the file in the repo to track
# Every file to keep up to date. Each entry is a path inside the repository;
# it is installed at the same path beside ramma.py. When the repository
# layout differs from yours, give the pair instead:
#     ("src/ramma.py", "ramma.py")        repo path -> local path
# Files that do not exist locally yet are simply added.
_UPDATE_FILES  = [
    "ramma.py",
    # "requirements.txt",
    # "RAMMA.bat",
    "models.json",
    "tips.txt",     # uncomment once tips.txt is in your repository
]
_UPDATE_CHECK  = True          # look for updates at start-up
_UPDATE_ASK    = True          # ask before installing; False installs quietly
_UPDATE_STATE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "ramma_update.json")


def _update_state():
    try:
        with open(_UPDATE_STATE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_update_state(data):
    try:
        with open(_UPDATE_STATE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
    except Exception as e:
        print("[Update] Could not save update state:", e)


def _github_commit_info():
    """(date, message) of the newest commit, for the prompt. Optional.

    Purely cosmetic: GitHub's API is rate-limited for anonymous callers and
    returns 403 often enough that the update must not depend on it. The
    check itself compares file contents instead.
    """
    import urllib.request
    try:
        url = (f"https://api.github.com/repos/{_UPDATE_REPO}/commits"
               f"?sha={_UPDATE_BRANCH}&path={_UPDATE_FILE}&per_page=1")
        req = urllib.request.Request(url, headers={
            "User-Agent": "RAMMA",
            "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
        if data:
            top = data[0]
            return (top.get("commit", {}).get("committer", {}).get("date", ""),
                    (top.get("commit", {}).get("message", "") or "").split("\n")[0])
    except Exception:
        pass
    return "", ""


def _download_update(repo_path=None):
    """Fetch one file from the branch; returns its text."""
    return _download_bytes(repo_path or _UPDATE_FILE).decode("utf-8")


def _download_bytes(repo_path):
    """Fetch one file from the branch, as raw bytes."""
    import urllib.request
    url = (f"https://raw.githubusercontent.com/{_UPDATE_REPO}/"
           f"{_UPDATE_BRANCH}/{repo_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "RAMMA"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _update_targets():
    """[(repo path, absolute local path), ...] for everything tracked."""
    here = os.path.dirname(os.path.abspath(__file__))
    entries = list(_UPDATE_FILES) or [_UPDATE_FILE]
    out = []
    for e in entries:
        if isinstance(e, (tuple, list)) and len(e) == 2:
            repo_path, local = e
        else:
            repo_path = local = str(e)
        if os.path.basename(str(local)) == os.path.basename(__file__) and \
                os.path.dirname(str(local)) in ("", "."):
            local_abs = os.path.abspath(__file__)     # the running file
        else:
            local_abs = os.path.join(here, str(local))
        out.append((str(repo_path), local_abs))
    return out


def _bytes_hash(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _text_hash(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def install_files(changed):
    """Install several downloaded files together, or none of them.

    Every file is checked before anything is written — Python files must
    compile, and ramma.py itself must not be suspiciously small — so a bad
    download cannot leave you with half an update. Each file that is
    replaced keeps its previous version as <name>.bak.
    """
    me = os.path.abspath(__file__)
    for repo_path, local, data in changed:
        if local.lower().endswith(".py"):
            try:
                compile(data.decode("utf-8"), local, "exec")
            except (SyntaxError, UnicodeDecodeError) as e:
                print(f"[Update] {repo_path} does not compile ({e}) — "
                      f"nothing has been installed.")
                return False
        if os.path.abspath(local) == me and len(data) < 10000:
            print(f"[Update] {repo_path} is only {len(data)} bytes — looks "
                  f"truncated; nothing has been installed.")
            return False

    written = []
    try:
        for repo_path, local, data in changed:
            os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
            if os.path.exists(local):
                shutil.copy2(local, local + ".bak")
            with open(local, "wb") as f:
                f.write(data)
            written.append(repo_path)
    except Exception as e:
        print(f"[Update] Stopped after {len(written)} file(s): {e}. The "
              f"previous versions are in the .bak files beside them.")
        return False

    _save_update_state({"files": {rp: _bytes_hash(d) for rp, _l, d in changed},
                        "installed": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"[Update] Installed {len(written)} file(s): {', '.join(written)}. "
          f"Previous versions kept as .bak — restart RAMMA to use them.")
    return True


def install_update(new_hash, text):
    """Replace this file with *text*, keeping the current one as a backup."""
    me = os.path.abspath(__file__)
    # Refuse anything that will not compile or is obviously truncated.
    if len(text) < 10000:
        print(f"[Update] The download is only {len(text)} bytes — ignoring it.")
        return False
    try:
        compile(text, me, "exec")
    except SyntaxError as e:
        print(f"[Update] The downloaded file does not compile ({e}); "
              f"keeping the current one.")
        return False

    backup = me + ".bak"
    try:
        shutil.copy2(me, backup)
        with open(me, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print("[Update] Could not write the update:", e)
        return False

    _save_update_state({"hash": new_hash,
                        "installed": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"[Update] Installed. The previous version is at "
          f"{os.path.basename(backup)} — restart RAMMA to use the new one.")
    return True


def _update_notice(kind, title, text):
    """A dialog on the UI thread, for a check the user asked for."""
    if not running:
        return
    def _show():
        try:
            {"info": messagebox.showinfo,
             "warn": messagebox.showwarning,
             "error": messagebox.showerror}[kind](title, text)
        except Exception:
            pass
    app.after(0, _show)


def check_for_update(quiet=True, manual=False):
    """Compare every tracked file with the repository and offer the update.

    Contents are compared rather than commit ids: it needs only
    raw.githubusercontent.com, which has no rate limit for this, and it
    answers the question that actually matters — is what is published
    different from what is here?
    """
    if not _UPDATE_REPO:
        if not quiet:
            print("[Update] No repository set (_UPDATE_REPO near the top).")
        if manual:
            _update_notice("warn", "RAMMA — updates are off",
                           "No GitHub repository is set.\n\nPut yours in "
                           "_UPDATE_REPO near the top of ramma.py, e.g.\n"
                           "    _UPDATE_REPO = \"yourname/ramma\"")
        return None

    changed, missing = [], []
    for repo_path, local in _update_targets():
        try:
            data = _download_bytes(repo_path)
        except Exception as e:
            if getattr(e, "code", None) == 404:
                missing.append(repo_path)
            else:
                print(f"[Update] Could not reach GitHub: {e}")
                if manual:
                    _update_notice("error", "RAMMA — could not check",
                                   f"GitHub could not be reached:\n\n{e}\n\n"
                                   f"Check your internet connection and try "
                                   f"again.")
                return None
            continue
        try:
            with open(local, "rb") as f:
                mine = f.read()
        except FileNotFoundError:
            mine = None
        # Compare ignoring line-ending differences, so a checkout with
        # Windows line endings is not seen as a different version.
        norm = lambda b: b.replace(b"\r\n", b"\n") if b is not None else None
        if norm(data) != norm(mine):
            changed.append((repo_path, local, data))

    if missing:
        print(f"[Update] GitHub has no file at "
              f"{', '.join(f'{_UPDATE_REPO}/{_UPDATE_BRANCH}/{m}' for m in missing)}. "
              f"Check the repo name, branch and file paths, and that the "
              f"repository is public — private ones cannot be read without "
              f"signing in.")
    if missing and manual and not changed:
        _update_notice("error", "RAMMA — could not check",
                       "GitHub has no file at:\n  " +
                       "\n  ".join(f"{_UPDATE_REPO}/{_UPDATE_BRANCH}/{m}"
                                    for m in missing) +
                       "\n\nCheck the repository name, branch and file "
                       "paths, and that the repository is public.")
        return None
    if not changed:
        if not missing and not quiet:
            print("[Update] Already up to date.")
        if manual:
            _update_notice("info", "RAMMA — up to date",
                           "You have the latest version.")
        return False

    # Declining an update is not remembered: every start-up asks again
    # while a newer version is waiting, so it cannot be missed for good.
    names = [rp for rp, _l, _d in changed]
    date, message = _github_commit_info()
    print(f"[Update] {len(changed)} file(s) differ from GitHub: "
          f"{', '.join(names)}" + (f" — {message} ({date})" if message else ""))

    def _proceed():
        if install_files(changed) and running:
            try:
                messagebox.showinfo(
                    "RAMMA updated",
                    "Updated:\n  " + "\n  ".join(names) +
                    "\n\nRestart RAMMA to use the new version.\n"
                    "Previous versions were kept as .bak files.")
            except Exception:
                pass

    if not _UPDATE_ASK:
        _proceed()
        return True
    if running:
        def _ask():
            try:
                if messagebox.askyesno(
                        "RAMMA — update available",
                        "Newer versions of these files are on GitHub:\n  "
                        + "\n  ".join(names) + "\n\n"
                        + (f"{message}\n({date})\n\n" if message else "")
                        + "Install them now?\nYour current files are kept "
                          "as .bak."):
                    threading.Thread(target=_proceed, daemon=True).start()
                else:
                    print("[Update] Not installed — you will be asked again "
                          "next time RAMMA starts.")
            except Exception:
                pass
        app.after(0, _ask)
    return True


# ============================================================
# MODEL DOWNLOADER
# The checkpoints are far too large for a git repository, so the program
# fetches them on first run from the links in models.json (written beside
# this file the first time it runs, then yours to edit). Nothing is
# downloaded if the file is already there.
# ============================================================
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "models.json")

# Written out when models.json is missing. Fill in your own links: a Hugging
# Face file link works as-is — the "/blob/" form is rewritten to "/resolve/".
_DEFAULT_MANIFEST = {
    "_comment": [
        "Files RAMMA downloads on first run. Edit the urls to point at your",
        "own copies. sha256 is optional: when present the file is verified",
        "after downloading and a corrupt download is discarded.",
        "required=false means the program runs without it.",
    ],
    "files": [
        {"name": "BS-Rofo-SW-Fixed.ckpt",
         "url": "https://huggingface.co/<user>/<repo>/resolve/main/BS-Rofo-SW-Fixed.ckpt",
         "sha256": "", "required": True,
         "note": "six-stem separator"},
        {"name": "BS-Rofo-SW-Fixed.yaml",
         "url": "https://huggingface.co/<user>/<repo>/resolve/main/BS-Rofo-SW-Fixed.yaml",
         "sha256": "", "required": True,
         "note": "its config"},
        {"name": "BS-Roformer-Resurrection-Inst.ckpt",
         "url": "https://huggingface.co/<user>/<repo>/resolve/main/BS-Roformer-Resurrection-Inst.ckpt",
         "sha256": "", "required": False,
         "note": "instrumental model"},
        {"name": "BS-Roformer-Resurrection-Inst.yaml",
         "url": "https://huggingface.co/<user>/<repo>/resolve/main/BS-Roformer-Resurrection-Inst.yaml",
         "sha256": "", "required": False,
         "note": "its config"},
        {"name": "bowed_strings.ckpt",
         "url": "https://huggingface.co/<user>/<repo>/resolve/main/bowed_strings.ckpt",
         "sha256": "", "required": False,
         "note": "gilliaan's bowed strings model"},
        {"name": "bowed_strings.yaml",
         "url": "https://huggingface.co/<user>/<repo>/resolve/main/bowed_strings.yaml",
         "sha256": "", "required": False,
         "note": "its config"},
        {"name": "mel_band_roformer_karaoke_becruily.ckpt",
         "url": "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt",
         "sha256": "", "required": False,
         "note": "lead / backing vocal split"},
        {"name": "config_karaoke_becruily.yaml",
         "url": "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/config_karaoke_becruily.yaml",
         "sha256": "", "required": False,
         "note": "its config"},
    ],
}


def _hf_direct(url):
    """A Hugging Face page link turned into a direct download link."""
    if "huggingface.co" in url and "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")
    if "huggingface.co" in url and "download=true" not in url:
        url += ("&" if "?" in url else "?") + "download=true"
    return url


def _read_manifest():
    """The download list, creating a template beside the script if missing."""
    try:
        if not os.path.exists(_MANIFEST_PATH):
            with open(_MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(_DEFAULT_MANIFEST, f, indent=2)
            print(f"[Models] Wrote a template to {_MANIFEST_PATH} — put your "
                  f"own links in it.")
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("files", [])
    except Exception as e:
        print("[Models] Could not read models.json:", e)
        return []


def _sha256_of(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_file(url, dest, expect_sha="", progress=None):
    """Fetch *url* to *dest*, resuming a part-file and verifying if asked.

    Downloads to dest.part first, so an interrupted run never leaves a
    half-written checkpoint that looks complete.
    """
    import urllib.request
    url = _hf_direct(url)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0

    req = urllib.request.Request(url, headers={"User-Agent": "RAMMA"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except Exception as e:
        if have:                      # the server may refuse to resume
            os.remove(part)
            return _download_file(url, dest, expect_sha, progress)
        raise RuntimeError(f"could not start the download: {e}")

    total = int(resp.headers.get("Content-Length", 0) or 0)
    if resp.status == 206:
        total += have
    elif have:
        have = 0                      # not a resume after all: start over

    mode = "ab" if (have and resp.status == 206) else "wb"
    done = have if mode == "ab" else 0
    last = -1
    with open(part, mode) as f:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total and progress:
                pct = int(done * 100 / total)
                if pct != last:
                    last = pct
                    progress(pct, done, total)

    if expect_sha:
        got = _sha256_of(part)
        if got.lower() != expect_sha.lower():
            os.remove(part)
            raise RuntimeError(f"checksum mismatch (expected {expect_sha[:12]}…, "
                               f"got {got[:12]}…) — the file was discarded")
    os.replace(part, dest)
    return dest


def missing_model_files():
    """Entries from the manifest whose file is not on disk yet."""
    out = []
    for entry in _read_manifest():
        name = entry.get("name", "")
        if not name:
            continue
        if not os.path.exists(os.path.join(_MODELS_DIR, name)):
            out.append(entry)
    return out


def download_models(only_required=False, status=None):
    """Fetch whatever is missing. Returns (fetched, failed)."""
    os.makedirs(_MODELS_DIR, exist_ok=True)
    missing = [e for e in missing_model_files()
               if e.get("required", True) or not only_required]
    if not missing:
        print("[Models] Everything is already in the models folder")
        return 0, 0

    fetched = failed = 0
    for entry in missing:
        name = entry["name"]
        url = entry.get("url", "")
        if not url or "<user>" in url:
            print(f"[Models] {name}: no link set in models.json — skipped")
            failed += 1
            continue
        dest = os.path.join(_MODELS_DIR, name)
        note = entry.get("note", "")
        print(f"[Models] Downloading {name} ({note})…")

        def _prog(pct, done, total, _n=name):
            msg = f"{_n}  {pct}%  ({done / 1e6:.0f} / {total / 1e6:.0f} MB)"
            if status:
                status(msg)
            if pct % 10 == 0:
                print(f"[Models]   {msg}")

        try:
            _download_file(url, dest, entry.get("sha256", ""), _prog)
            print(f"[Models] {name} ready")
            fetched += 1
        except Exception as e:
            print(f"[Models] {name} failed: {e}")
            failed += 1
    return fetched, failed


# A fresh copy of RAMMA has no model files. Fetch whatever models.json lists
# and the models folder lacks, before anything tries to load them.
try:
    _missing = missing_model_files()
    if _missing:
        _names = ", ".join(e["name"] for e in _missing)
        print(f"[Models] Missing: {_names}")
        _splash_set(0.16, "DOWNLOADING MODELS…")
        download_models(status=lambda msg: _splash_set(0.18, msg[:42].upper()))
except Exception as _e:
    print("[Models] Download step skipped:", _e)


# The six-stem model normally downloads itself into _BSR_CACHE, but a local
# copy is used when there is one — so both models can live side by side in the
# models folder:
#
#     models/BS-Rofo-SW-Fixed.ckpt              -> the six stems
#     models/BS-Rofo-SW-Fixed.yaml
#     models/BS-Roformer-Resurrection-Inst.ckpt -> the instrumental
#     models/BS-Roformer-Resurrection-Inst.yaml
#
# Set these to absolute paths to name the files directly; leave them empty to
# search the folders below.
_MAIN_CKPT_PATH = ""
_MAIN_CFG_PATH  = ""

model           = None   # the loaded BSRoformer nn.Module
model_ready     = False
_bsr_config     = None   # ml_collections ConfigDict for the model
_bsr_stem_order = None   # list of stem names in model output order, read from YAML

# ----------------------------
# BS-ROFORMER RESURRECTION INST — unwa
# Used for background instrumental separation (runs alongside the main
# separation).  Unlike the main model this one is NOT downloaded: point the
# two paths below at the .ckpt and .yaml you downloaded yourself, or drop
# both files into one of the folders listed in _INST_SEARCH_DIRS and they
# will be found automatically.
# ----------------------------
_INST_TITLE  = "BS-ROFORMER RESURRECTION"
_INST_CREDIT = "by unwa"

# Optional: absolute paths to your files.  Leave as "" to auto-detect.
# e.g. _INST_CKPT_PATH = r"C:\Users\Robin\VocalSeparator\models\inst.ckpt"
_INST_CKPT_PATH = ""
_INST_CFG_PATH  = ""

# Folders searched (in order) when the paths above are empty.
_INST_SEARCH_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(os.path.expanduser("~"), ".cache", "ramma",
                 "bs-roformer-resurrection-inst"),
    os.getcwd(),
]
# A file whose name contains any of these is preferred when a folder holds
# several checkpoints.
_INST_NAME_HINTS = ("resurrection", "unwa", "inst")
# "inst" also matches instvoc-style vocal models, and karaoke files are not
# instrumental models either.
_INST_NAME_EXCLUDE = ("karaoke", "kara", "instvoc", "inst_voc", "instvocal",
                      "vocals", "bowed", "string")

# ── gilliaan's bowed strings model ─────────────────────────────────────────
# Same arrangement as the instrumental model: point these at the files, or
# drop them in the models folder and let the name hints find them.
_STR_CKPT_PATH = ""
_STR_CFG_PATH  = ""
_STR_SEARCH_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    os.path.dirname(os.path.abspath(__file__)),
    os.getcwd(),
]
_STR_NAME_HINTS = ("bowed", "string", "gilliaan")

# ── Vocals model (becruily's, by default) ─────────────────────────────────
# The six-stem model's vocals are good enough to start mixing with, but a
# dedicated vocals model does the job far better. It runs on the MAIN TRACK
# — not on the six-stem vocals — and its result replaces the VOCALS cell.
# The karaoke split then runs on that, so the lead and backing come from the
# better vocal.
_VOC_CKPT_PATH = ""
_VOC_CFG_PATH  = ""
_VOC_SEARCH_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    os.path.dirname(os.path.abspath(__file__)),
    os.getcwd(),
]
# A vocals model, not the karaoke one and not an instrumental one.
_VOC_NAME_HINTS   = ("vocals", "voc")
# Only the things a vocals model definitely is not. "inst" is deliberately
# absent: plenty of vocal models are named instvoc / inst_voc, and a real
# instrumental model has no "voc" in its name to match on in the first place.
_VOC_NAME_EXCLUDE = ("karaoke", "kara", "bowed", "string",
                     "sw-fixed", "sw_fixed", "rofo-sw")
_VOC_TITLE  = "VOCALS MODEL"
_VOC_CREDIT = "by becruily"
_VOC_AUTO   = True       # refine the vocals after every separation

vocals_model       = None
vocals_model_ready = False
_vocals_refining   = False
_voc_load_lock  = threading.Lock()
_voc_sr         = 44100
_voc_stem_idx   = 0
_voc_chunk_s    = 8.0
_voc_overlap_s  = _OVERLAP_SECONDS


def _voc_find_files():
    """Locate a dedicated vocals checkpoint and its config."""
    ck = _VOC_CKPT_PATH if _VOC_CKPT_PATH and os.path.isfile(_VOC_CKPT_PATH) else None
    cf = _VOC_CFG_PATH  if _VOC_CFG_PATH  and os.path.isfile(_VOC_CFG_PATH)  else None
    if ck and cf:
        return ck, cf
    for folder in _VOC_SEARCH_DIRS:
        if not os.path.isdir(folder):
            continue
        cks, cfs = [], []
        for name in os.listdir(folder):
            low = name.lower()
            if any(x in low for x in _VOC_NAME_EXCLUDE):
                continue
            if not any(h in low for h in _VOC_NAME_HINTS):
                continue
            full = os.path.join(folder, name)
            if low.endswith((".ckpt", ".pth", ".th")):
                cks.append(full)
            elif low.endswith((".yaml", ".yml")):
                cfs.append(full)
        if cks and cfs:
            return sorted(cks)[0], sorted(cfs)[0]
    return ck, cf
_STR_AUTO   = True        # run the strings model after every track
_STR_FIRST  = False       # run it before the six-stem pass, so the strings
                          # are playable sooner (the stems then follow)
_STR_QUICK  = True        # fill the cell from OTHER until the model has run,
                          # exactly as the instrumental cell does. No switch:
                          # it costs nothing and is replaced by the model.
_STR_TITLE  = "BOWED STRINGS"
_STR_CREDIT = "by gilliaan"

strings_model       = None
strings_model_ready = False
_strings_separating = False
_str_load_lock   = threading.Lock()
_str_sr          = 44100
_str_stem_idx    = 0
_str_chunk_s     = 8.0
_str_overlap_s   = _OVERLAP_SECONDS


def _str_find_files():
    """Locate the bowed-strings checkpoint and config."""
    ck = _STR_CKPT_PATH if _STR_CKPT_PATH and os.path.isfile(_STR_CKPT_PATH) else None
    cf = _STR_CFG_PATH  if _STR_CFG_PATH  and os.path.isfile(_STR_CFG_PATH)  else None
    if ck and cf:
        return ck, cf
    for folder in _STR_SEARCH_DIRS:
        if not os.path.isdir(folder):
            continue
        cks, cfs = [], []
        for name in os.listdir(folder):
            low = name.lower()
            full = os.path.join(folder, name)
            if not any(h in low for h in _STR_NAME_HINTS):
                continue
            if low.endswith((".ckpt", ".pth", ".th")):
                cks.append(full)
            elif low.endswith((".yaml", ".yml")):
                cfs.append(full)
        if cks and cfs:
            return sorted(cks)[0], sorted(cfs)[0]
    return ck, cf

# Names belonging to the six-stem model. A file matching one of these is never
# taken as the instrumental model, however it is named on disk.
_MAIN_MODEL_MARKERS = ("bs-rofo-sw", "bs_rofo_sw", "sw-fixed", "sw_fixed",
                       "bs-roformer-sw", "roformer-model-bs-roformer-sw")


def _is_main_model_file(path):
    """True when *path* belongs to the six-stem model rather than a
    dedicated instrumental one."""
    if not path:
        return False
    ap = os.path.normcase(os.path.abspath(path))
    for other in (_bsr_ckpt_resolved, _bsr_cfg_resolved):
        if other and os.path.normcase(os.path.abspath(other)) == ap:
            return True
    # Anything inside the main model's download cache
    try:
        if ap.startswith(os.path.normcase(os.path.abspath(_BSR_CACHE))):
            return True
    except Exception:
        pass
    # .lower() explicitly: os.path.normcase only folds case on Windows.
    name = os.path.basename(ap).lower()
    return any(marker in name for marker in _MAIN_MODEL_MARKERS)

# ----------------------------
# MEL-BAND ROFORMER KARAOKE — becruily
# Splits the separated vocals into lead and backing vocals, so the BG VOX
# cell is produced by a model instead of being imported from a file.
# Same folders as the other two models; files are matched on these hints.
# ----------------------------
_KARA_TITLE  = "MEL-BAND KARAOKE"
_KARA_CREDIT = "by becruily"
_KARA_CKPT_PATH = ""      # set to name the files directly
_KARA_CFG_PATH  = ""
# Must actually say "karaoke": matching on an author's name alone grabbed
# that author's other models — becruily's vocals model was being loaded as
# the karaoke model, which is why the lead/backing split came out wrong.
_KARA_NAME_HINTS   = ("karaoke", "kara")
_KARA_NAME_EXCLUDE = ("instvoc", "inst_voc", "instrum", "resurrection",
                      "bowed", "string", "sw-fixed", "sw_fixed", "rofo-sw")

kara_model        = None            # loaded nn.Module
kara_model_ready  = False
_kara_load_lock   = threading.Lock()
_kara_separating  = False
_kara_sr          = 44100
_kara_chunk_s     = 8.0
_kara_overlap_s   = _OVERLAP_SECONDS   # replaced from the model's config
_kara_back_idx    = None            # output index holding the backing vocals
_kara_lead_idx    = None

inst_model       = None   # loaded BSRoformer nn.Module (instrumental)
inst_model_ready = False  # set True once weights are loaded (or failed gracefully)
_inst_separating = False  # True while the background instrumental job is running
_inst_load_lock  = threading.Lock()  # prevents duplicate loads
_inst_sr         = 44100  # model sample rate, overwritten from the YAML
_inst_stem_idx   = 0      # index of the instrumental stem in the model output
_inst_chunk_s    = 8.0    # inference chunk length in seconds
_inst_overlap_s  = _OVERLAP_SECONDS   # replaced from the model's config

def set_gpu_mode():
    global device
    if state.separating:
        gpu_var.set(device.type == "cuda")
        return
    device = torch.device("cuda" if gpu_var.get() and torch.cuda.is_available() else "cpu")
    if model is not None:
        model.to(device)
    if inst_model is not None:
        inst_model.to(device)

def _find_main_model_files():
    """Look for a local six-stem checkpoint + config.

    Returns (ckpt, cfg), either of which may be None. Files belonging to the
    instrumental model are skipped, so the two never adopt each other's
    weights; a pair is only accepted when both parts are found together.
    """
    ck = _MAIN_CKPT_PATH if _MAIN_CKPT_PATH and os.path.isfile(_MAIN_CKPT_PATH) else None
    cf = _MAIN_CFG_PATH  if _MAIN_CFG_PATH  and os.path.isfile(_MAIN_CFG_PATH)  else None
    if ck and cf:
        return ck, cf

    def _is_inst_name(path):
        name = os.path.basename(path).lower()
        return any(h in name for h in _INST_NAME_HINTS + _KARA_NAME_HINTS)

    def _pick(files):
        # A name that looks like the six-stem model wins; failing that, the
        # only candidate in the folder.
        marked = [f for f in files
                  if any(m in os.path.basename(f).lower()
                         for m in _MAIN_MODEL_MARKERS)]
        if marked:
            return sorted(marked)[0]
        return sorted(files)[0] if len(files) == 1 else None

    for d in _INST_SEARCH_DIRS:          # same folders as the instrumental
        try:
            if not os.path.isdir(d):
                continue
            entries = [os.path.join(d, f) for f in os.listdir(d)
                       if not _is_inst_name(os.path.join(d, f))]
        except OSError:
            continue
        c1 = ck or _pick([f for f in entries
                          if f.lower().endswith((".ckpt", ".pth", ".bin"))])
        c2 = cf or _pick([f for f in entries
                          if f.lower().endswith((".yaml", ".yml"))])
        if c1 and c2:
            return c1, c2
    return None, None


def _bsr_download_weights(entry, model_dir):
    """Download checkpoint + config for *entry* into _BSR_CACHE/<slug>/."""
    os.makedirs(model_dir, exist_ok=True)
    # Preferred: call the package's own downloader directly (no subprocess).
    ensure_model_assets = _import_bs_roformer().get("ensure_model_assets")
    if ensure_model_assets is not None:
        try:
            ck, cf = ensure_model_assets(entry, _BSR_CACHE)
            return str(ck), str(cf)
        except Exception as e:
            print(f"[BS-RoFormer] ensure_model_assets failed ({e}) — "
                  f"falling back to the CLI")
    # Fallback: the CLI that ships with bs-roformer-infer.
    import subprocess
    for cmd in (
        [sys.executable, '-m', 'bs_roformer.download',
         '--model', entry.slug, '--output-dir', _BSR_CACHE],
        ['bs-roformer-download',
         '--model', entry.slug, '--output-dir', _BSR_CACHE],
    ):
        try:
            r = subprocess.run(cmd, check=False)
            if r.returncode == 0:
                return None
        except FileNotFoundError:
            continue
    raise RuntimeError('Model download failed — is bs-roformer-infer installed '
                       '(pip install bs-roformer-infer) and are you online?')


def load_model():
    global model, model_ready, _bsr_config
    try:
        _bsr = _import_bs_roformer()
        MODEL_REGISTRY        = _bsr["registry"]
        get_model_from_config = _bsr["get_model_from_config"]
        import yaml
        from ml_collections import ConfigDict

        entry     = MODEL_REGISTRY.get(_BSR_SLUG)
        model_dir = os.path.join(_BSR_CACHE, entry.slug)
        ckpt_path = os.path.join(model_dir, entry.checkpoint)
        cfg_path  = os.path.join(model_dir, entry.config)

        # A local copy in the models folder is used in preference to the
        # download, so you can supply your own six-stem checkpoint.
        _local_ck, _local_cf = _find_main_model_files()
        if _local_ck and _local_cf:
            ckpt_path, cfg_path = _local_ck, _local_cf
            print(f'[BS-RoFormer] Using local model files from '
                  f'{os.path.dirname(ckpt_path)!r}')

        # Download if either file is missing
        if not os.path.isfile(ckpt_path) or not os.path.isfile(cfg_path):
            print(f'[BS-RoFormer] Downloading model to {model_dir} ...')
            found = _bsr_download_weights(entry, model_dir)
            if found:
                ckpt_path, cfg_path = found
        if not os.path.isfile(ckpt_path) or not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f'Model files missing after download: {ckpt_path} / {cfg_path}')

        global _bsr_ckpt_resolved, _bsr_cfg_resolved
        _bsr_ckpt_resolved, _bsr_cfg_resolved = ckpt_path, cfg_path

        # The SW config uses !!python/tuple tags, so the loader has to know
        # about them; _import_bs_roformer() supplies one either way.
        _YLoader = _bsr["yaml_loader"]
        with open(cfg_path, 'r', encoding='utf-8') as f:
            _bsr_config = ConfigDict(yaml.load(f, Loader=_YLoader))

        # Read the stem order the model was trained with.
        # The YAML config stores it under training.instruments.
        # We lower-case and strip whitespace to match our mixer keys.
        global _bsr_stem_order
        try:
            raw_order = list(_bsr_config.training.instruments)
            _bsr_stem_order = [s.strip().lower() for s in raw_order]
        except Exception:
            _bsr_stem_order = None
        print(f'[BS-RoFormer] Stem order from config: {_bsr_stem_order}')

        # flash_attn=True routes attention through PyTorch's fused
        # scaled_dot_product_attention — no extra package, much faster on GPU.
        m = None
        _flash_was = None   # defined up front: the except below reads it
        if torch.cuda.is_available() and not _SAFE_MODE:
            try:
                _flash_was = _bsr_config.model.get('flash_attn', None)
                _bsr_config.model.flash_attn = True
                m = get_model_from_config('bs_roformer', _bsr_config)
            except Exception as fe:
                print(f'[BS-RoFormer] Fused attention unavailable ({fe}) — '
                      f'using the config default')
                try:
                    if _flash_was is not None:
                        _bsr_config.model.flash_attn = _flash_was
                except Exception:
                    pass
                m = None
        if m is None:
            m = get_model_from_config('bs_roformer', _bsr_config)
        if m is None:
            raise RuntimeError('get_model_from_config returned None')

        # weights_only=False for older checkpoints serialised with pickle
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        m.load_state_dict(state_dict)
        m.to(device)
        m.eval()
        model = m   # only expose the model once it is fully loaded
        print(f'[BS-RoFormer] Model ready on {device}')
    except Exception as e:
        print(f'[BS-RoFormer] Failed to load model: {e}')
        import traceback; traceback.print_exc()
    model_ready = True

# Import the package here, on the main thread, before any loader thread runs.
# Belt and braces alongside the lock in _import_bs_roformer(): with the import
# already finished, the two threads can never race over it at all.
try:
    _import_bs_roformer()
except Exception as _e:
    print(f"[BS-RoFormer] Could not import bs_roformer: {_e}")

threading.Thread(target=load_model, daemon=True).start()


# ----------------------------
# BS-ROFORMER RESURRECTION INST — model loader + inference
# ----------------------------
def _inst_find_files():
    """Locate the instrumental checkpoint + config.

    Returns (ckpt_path, cfg_path); either may be None if nothing was found.
    Explicit _INST_CKPT_PATH / _INST_CFG_PATH win; otherwise every folder in
    _INST_SEARCH_DIRS is scanned for a .ckpt/.pth and a .yaml/.yml, preferring
    names that contain one of _INST_NAME_HINTS.
    """
    ck = _INST_CKPT_PATH if _INST_CKPT_PATH and os.path.isfile(_INST_CKPT_PATH) else None
    cf = _INST_CFG_PATH  if _INST_CFG_PATH  and os.path.isfile(_INST_CFG_PATH)  else None
    if ck and cf:
        return ck, cf

    def _pick(files):
        """Prefer a hinted name, else the single candidate, else None."""
        files = [f for f in files
                 if not any(x in os.path.basename(f).lower()
                            for x in _INST_NAME_EXCLUDE)]
        hinted = [f for f in files
                  if any(h in os.path.basename(f).lower() for h in _INST_NAME_HINTS)]
        if hinted:
            return sorted(hinted)[0]
        return sorted(files)[0] if len(files) == 1 else None

    for d in _INST_SEARCH_DIRS:
        try:
            if not os.path.isdir(d):
                continue
            entries = [os.path.join(d, f) for f in os.listdir(d)]
        except OSError:
            continue
        # Never adopt the six-stem model's files, nor the karaoke model's.
        entries = [f for f in entries
                   if not _is_main_model_file(f)
                   and not any(h in os.path.basename(f).lower()
                               for h in _KARA_NAME_HINTS)]
        if ck is None:
            ck = _pick([f for f in entries
                        if f.lower().endswith((".ckpt", ".pth", ".bin"))])
        if cf is None:
            cf = _pick([f for f in entries
                        if f.lower().endswith((".yaml", ".yml"))])
        if ck and cf:
            break
    return ck, cf


def load_inst_model():
    """Load unwa's BS-RoFormer Resurrection Inst from local files.

    Called from a daemon thread at startup so the main model and UI are never
    blocked.  Sets inst_model_ready=True whether loading succeeds or fails,
    so separate_inst() can always check the flag without hanging.
    """
    global inst_model, inst_model_ready, _inst_sr, _inst_stem_idx
    global _inst_chunk_s, _inst_overlap_s
    with _inst_load_lock:
        if inst_model_ready:
            return
        try:
            ckpt_path, cfg_path = _inst_find_files()
            if not ckpt_path or not cfg_path:
                raise FileNotFoundError(
                    "Instrumental model files not found. Put the "
                    "BS-Roformer-Resurrection-Inst .ckpt and .yaml in "
                    f"{_INST_SEARCH_DIRS[0]!r}, or set _INST_CKPT_PATH / "
                    "_INST_CFG_PATH near the top of this file.")
            if _is_main_model_file(ckpt_path) or _is_main_model_file(cfg_path):
                raise RuntimeError(
                    f"{os.path.basename(ckpt_path)} is the six-stem model's "
                    "own checkpoint. Loading it as the instrumental model "
                    "would run that model twice over every song. Put a "
                    "dedicated instrumental model in the folder instead.")
            print(f"[Inst] Checkpoint: {ckpt_path}")
            print(f"[Inst] Config:     {cfg_path}")

            import yaml
            from ml_collections import ConfigDict
            _YL = _import_bs_roformer()["yaml_loader"]
            with open(cfg_path, "r", encoding="utf-8") as f:
                inst_cfg = ConfigDict(yaml.load(f, Loader=_YL))

            # Sample rate and chunk length straight from the config.
            try:
                _inst_sr = int(inst_cfg.audio.sample_rate)
            except Exception:
                _inst_sr = 44100
            try:
                # Follow the model's own config: it was trained and
                # evaluated at this chunk length and overlap.
                _inst_chunk_s, _inst_overlap_s = _chunking_from_config(
                    inst_cfg, _inst_sr)
                print(f"[Inst] Chunking from config: {_inst_chunk_s:.2f}s "
                      f"chunks, {_inst_overlap_s:.2f}s overlap per side")
            except Exception:
                _inst_chunk_s, _inst_overlap_s = 8.0, _OVERLAP_SECONDS

            # Resurrection Inst is a BS-RoFormer, so the same builder the main
            # model uses works here — no Mel-Band RoFormer needed.
            get_model_from_config = _import_bs_roformer()["get_model_from_config"]
            if get_model_from_config is None:
                raise RuntimeError("bs_roformer.get_model_from_config missing "
                                   "— is bs-roformer-infer installed?")
            m = None
            if torch.cuda.is_available() and not _SAFE_MODE:
                try:
                    inst_cfg.model.flash_attn = True
                    m = get_model_from_config("bs_roformer", inst_cfg)
                except Exception as fe:
                    print(f"[Inst] Fused attention unavailable ({fe}) — "
                          f"using the config default")
                    m = None
            if m is None:
                m = get_model_from_config("bs_roformer", inst_cfg)
            if m is None:
                raise RuntimeError(
                    "get_model_from_config returned None — is this a "
                    "bs_roformer config?")

            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if isinstance(state_dict, dict):
                # Checkpoints saved from DDP carry a "module." prefix.
                state_dict = {k[7:] if k.startswith("module.") else k: v
                              for k, v in state_dict.items()}
            m.load_state_dict(state_dict)

            # Which output index holds the instrumental?
            _inst_stem_idx = 0
            try:
                stems = [str(x).lower() for x in inst_cfg.training.instruments]
                print(f"[Inst] Stems in config: {stems}")
                has_inst = any("instrum" in n or n in ("inst", "music", "karaoke")
                               for n in stems)
                if len(stems) >= 3 and not has_inst:
                    # vocals/drums/bass/guitar/piano/other — this is the
                    # six-stem separator, not an instrumental model. Loading
                    # it here would put a second copy of that model on the GPU
                    # and run it over every song alongside the first.
                    raise RuntimeError(
                        f"{os.path.basename(cfg_path)} describes a "
                        f"{len(stems)}-stem separator ({', '.join(stems)}), "
                        "not an instrumental model. Point _INST_CKPT_PATH / "
                        "_INST_CFG_PATH at a dedicated instrumental model, or "
                        "leave the folder empty to use the free quick mix.")
                if len(stems) > 1:
                    for i, name in enumerate(stems):
                        if "instrum" in name or name in ("inst", "music", "other"):
                            _inst_stem_idx = i
                            break
            except RuntimeError:
                raise
            except Exception:
                pass

            m.to(device)
            m.eval()
            inst_model = m
            print(f"[Inst] Model ready on {device} "
                  f"({_inst_sr} Hz, {_inst_chunk_s:.0f}s chunks, "
                  f"stem index {_inst_stem_idx})")
        except Exception as e:
            import traceback
            print(f"[Inst] Failed to load model: {e}")
            traceback.print_exc()
        inst_model_ready = True


# The INSTRUM cell is roughly drums + bass + guitar + piano + other. With it
# and those five all audible the backing track plays twice over, so bringing
# INSTRUM in mutes them, and taking it out again restores them.
_INST_OVERLAP     = ("drums", "bass", "guitar", "piano", "other")
_inst_muted_by_us = set()   # stems muted on INSTRUM's behalf, to restore later


def _sync_instrumental_overlap():
    """Mute the five backing stems while INSTRUM is audible, restore after.

    Only stems muted here are un-muted later: a stem you had muted yourself
    before bringing INSTRUM in stays muted when INSTRUM goes out again.
    """
    inst_on = (state.instrumental is not None
               and not state.stem_mute.get("instrumental", False))
    if inst_on:
        for _k in _INST_OVERLAP:
            if not state.stem_mute.get(_k, False):
                state.stem_mute[_k] = True
                _inst_muted_by_us.add(_k)
    else:
        for _k in list(_inst_muted_by_us):
            state.stem_mute[_k] = False
        _inst_muted_by_us.clear()
    for _k in _INST_OVERLAP:
        try:
            _paint_ms(_k)
        except (NameError, KeyError):
            pass


def _split_vocals_active():
    """True when FRT VOX or BG VOX holds audio and is not muted.

    Those two cells are the vocals stem in two halves. While either is live,
    playing the VOCALS stem as well doubles the vocal, so every place that
    would un-mute it has to ask this first.
    """
    for _key, _data in (("front_vocals", state.fv_data),
                        ("bg_vocals",    state.bg_vocals_data)):
        if _data is not None and not state.stem_mute.get(_key, False):
            return True
    return False


# Halves muted because VOCALS was switched on, to bring back when it goes off.
_halves_muted_by_vocals = set()


def _split_halves_exist():
    return state.fv_data is not None or state.bg_vocals_data is not None


# STRINGS and OTHER are independent cells: both play unless you mute one.
# They do carry some of the same material, so the two together are louder
# than either alone — the Ø PHASE button on each is there for judging how
# much they share, and muting is yours to decide.


def _vocals_toggled_with_split():
    """VOCALS and the two split halves are the same vocal: play one or the other.

    Switching VOCALS on mutes whichever halves are playing; switching it off
    again brings exactly those halves back. A half you had muted yourself is
    left muted either way.
    """
    if not state.stem_mute.get("vocals", False):
        for _k, _d in (("front_vocals", state.fv_data),
                       ("bg_vocals",    state.bg_vocals_data)):
            if _d is not None and not state.stem_mute.get(_k, False):
                state.stem_mute[_k] = True
                _halves_muted_by_vocals.add(_k)
    else:
        for _k in list(_halves_muted_by_vocals):
            state.stem_mute[_k] = False
        _halves_muted_by_vocals.clear()
    for _k in ("vocals", "front_vocals", "bg_vocals"):
        try:
            _paint_ms(_k)
        except (NameError, KeyError):
            pass


def _restore_vocals_mute():
    """Put the VOCALS stem's mute back to what it should be right now.

    Once VOCALS is a VCA it carries no audio, and its mute means "silence
    both halves". Muting it automatically — which is what the old rule did
    whenever a half was active — therefore silenced the very cells it was
    meant to be making way for.
    """
    if getattr(state, "vocals_is_vca", False):
        state.stem_mute["vocals"] = False
        try:
            _paint_ms("vocals")
        except (NameError, KeyError):
            pass
        return
    state.stem_mute["vocals"] = _split_vocals_active()
    try:
        _paint_ms("vocals")
    except (NameError, KeyError):
        pass


def _split_replaces_vocals():
    """Hand the vocal over to FRT VOX + BG VOX.

    The VOCALS cell stays in the mixer but stops carrying audio: from here
    it is a VCA over the two halves, so its fader, M and S move both at
    once. Its own mute is cleared — muting it would now mean silencing the
    halves, which is not what was asked for by the split landing.
    """
    # Un-mute the two halves. They were muted when the stems landed, back
    # when they held nothing; without this they stay silent and the split
    # looks as though it produced nothing at all.
    _swap_to_split_vocals()

    # VOCALS carries no audio from here, so its own mute must be clear: as
    # a VCA, muting it means silencing both halves, which is not what the
    # split landing should do.
    state.vocals_is_vca = True
    state.stem_mute["vocals"] = False
    for _k in ("vocals", "front_vocals", "bg_vocals"):
        try:
            _paint_ms(_k)
        except Exception:
            pass
    if running:
        try:
            _update_vocals_status()
        except Exception:
            pass


def _swap_to_split_vocals():
    """Hand the vocal over to FRT VOX and BG VOX once the split exists.

    The two halves sum back to the vocals stem, so leaving all three audible
    plays the vocal twice — which is why it sounded twice as loud. The
    original stem is muted and the halves are un-muted, together.
    """
    if state.fv_data is None and state.bg_vocals_data is None:
        return
    state.stem_mute["vocals"] = True
    _halves_muted_by_vocals.clear()
    if state.fv_data is not None:
        state.stem_mute["front_vocals"] = False
    if state.bg_vocals_data is not None:
        state.stem_mute["bg_vocals"] = False
    for _k in ("vocals", "front_vocals", "bg_vocals"):
        _paint_ms(_k)


def _mute_vocal_cells():
    """Mute FRT VOX and BG VOX while they are still empty.

    Called when the stems land. At that point the split has usually not run,
    so the vocal lives only in the VOCALS stem and these two cells must stay
    quiet. A cell that already holds a half is left alone: this runs on a
    short timer after the stems pass, and a quick split can beat it, in
    which case re-muting would undo the hand-over.
    """
    for _key, _data in (("front_vocals", state.fv_data),
                        ("bg_vocals",    state.bg_vocals_data)):
        if _data is not None:
            continue
        state.stem_mute[_key] = True
        try:
            _paint_ms(_key)
        except (NameError, KeyError):
            pass


def _resample_to(audio, from_sr, to_sr):
    """Resample an (N, 2) float32 array between two sample rates."""
    if from_sr == to_sr:
        return np.asarray(audio, dtype=np.float32)
    from scipy.signal import resample_poly as _rp
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    g    = _gcd(int(from_sr), int(to_sr))
    up   = int(to_sr) // g
    down = int(from_sr) // g
    return np.stack([
        _rp(audio[:, 0], up, down).astype(np.float32),
        _rp(audio[:, 1], up, down).astype(np.float32),
    ], axis=1)


def _quick_strings(stems):
    """A stand-in strings part, taken from the six-stem split.

    Bowed strings mostly land in OTHER (with some in GUITAR), so that is
    what fills the cell the moment the stems arrive — something to hear and
    mix with straight away. gilliaan's model replaces it when its pass
    finishes, exactly as unwa's model replaces the quick instrumental.
    """
    if not stems:
        return None
    part = stems.get("other")
    if part is None:
        return None
    return part.copy()


def _quick_instrumental(stems):
    """Sum every non-vocal stem — an instrumental for free.

    The main separation has already split the track, so mixing back
    everything except the vocals gives a usable instrumental instantly.
    It isn't as clean as unwa's dedicated model (vocal bleed that the
    6-stem model left behind stays in), so the model pass still runs and
    replaces this as soon as it's done.
    """
    if not stems:
        return None
    parts = [v for k, v in stems.items()
             if "vocal" not in k.lower() and v is not None]
    if not parts:
        return None
    out = np.zeros_like(parts[0])
    for part in parts:
        n = min(len(out), len(part))
        out[:n] += part[:n]
    return np.clip(out, -1.0, 1.0)


# ----------------------------
# KARAOKE MODEL — loader + inference
# ----------------------------
def _kara_find_files():
    """Locate the karaoke checkpoint + config.

    Unlike the other two models this one is only ever matched by name
    (_KARA_NAME_HINTS): picking "the only pair in the folder" would risk
    grabbing one of the other models.
    """
    ck = _KARA_CKPT_PATH if _KARA_CKPT_PATH and os.path.isfile(_KARA_CKPT_PATH) else None
    cf = _KARA_CFG_PATH  if _KARA_CFG_PATH  and os.path.isfile(_KARA_CFG_PATH)  else None
    if ck and cf:
        return ck, cf

    def _hinted(files):
        hits = []
        for f in files:
            low = os.path.basename(f).lower()
            if any(x in low for x in _KARA_NAME_EXCLUDE):
                continue
            if any(h in low for h in _KARA_NAME_HINTS):
                hits.append(f)
        return sorted(hits)[0] if hits else None

    for d in _INST_SEARCH_DIRS:
        try:
            if not os.path.isdir(d):
                continue
            entries = [os.path.join(d, f) for f in os.listdir(d)]
        except OSError:
            continue
        if ck is None:
            ck = _hinted([f for f in entries
                          if f.lower().endswith((".ckpt", ".pth", ".bin"))])
        if cf is None:
            cf = _hinted([f for f in entries
                          if f.lower().endswith((".yaml", ".yml"))])
        if ck and cf:
            break
    return ck, cf


def load_kara_model():
    """Load becruily's Mel-Band RoFormer karaoke model, if it is present.

    This one is a Mel-Band RoFormer rather than a BS-RoFormer, and
    bs-roformer-infer does not ship that architecture — so it is imported
    separately and the failure is reported plainly rather than silently
    leaving the BG VOX cell empty.
    """
    global kara_model, kara_model_ready, _kara_sr, _kara_chunk_s, _kara_overlap_s
    global _kara_back_idx, _kara_lead_idx, _KARA_CKPT_PATH, _KARA_CFG_PATH
    with _kara_load_lock:
        if kara_model_ready:
            return
        try:
            ckpt_path, cfg_path = _kara_find_files()
            if not ckpt_path or not cfg_path:
                print("[Karaoke] No karaoke model found — BG VOX stays empty. "
                      "Put mel_band_roformer_karaoke_becruily.ckpt and its .yaml "
                      f"in {_INST_SEARCH_DIRS[0]!r}.")
                kara_model_ready = True
                return
            _KARA_CKPT_PATH, _KARA_CFG_PATH = ckpt_path, cfg_path
            print(f"[Karaoke] Checkpoint: {ckpt_path}")
            print(f"[Karaoke] Config:     {cfg_path}")

            import yaml
            from ml_collections import ConfigDict
            _YL = _import_bs_roformer()["yaml_loader"]
            with open(cfg_path, "r", encoding="utf-8") as f:
                kara_cfg = ConfigDict(yaml.load(f, Loader=_YL))

            try:
                _kara_sr = int(kara_cfg.audio.sample_rate)
            except Exception:
                _kara_sr = 44100
            try:
                _kara_chunk_s = max(4.0, min(12.0,
                                             int(kara_cfg.audio.chunk_size) / _kara_sr))
                _kara_chunk_s, _kara_overlap_s = _chunking_from_config(
                    kara_cfg, _kara_sr, fallback_chunk=_kara_chunk_s)
                print(f"[Karaoke] Chunking from config: {_kara_chunk_s:.2f}s "
                      f"chunks, {_kara_overlap_s:.2f}s overlap per side")
            except Exception:
                _kara_chunk_s = 8.0

            try:
                print("[Karaoke] Stems in config: "
                      f"{[str(x).lower() for x in kara_cfg.training.instruments]}")
            except Exception:
                pass

            name_hint = (os.path.basename(ckpt_path) + os.path.basename(cfg_path)).lower()
            is_mel = ("mel_band" in name_hint or "mel-band" in name_hint
                      or "num_bands" in dict(kara_cfg.model))

            if is_mel and not _ensure_librosa():
                raise RuntimeError(
                    "A Mel-Band RoFormer needs the real librosa. This model "
                    "asks librosa.filters.mel how many FFT bins belong to each "
                    "band, and those counts set the width of every mask-"
                    "estimator layer. RAMMA's built-in filterbank is close but "
                    "not identical, so the layers come out a different size "
                    "from the checkpoint and loading fails with hundreds of "
                    "'size mismatch' errors. Install it into this venv:\n"
                    "    python -m pip install librosa")

            if is_mel:
                # ZFTurbo's mel_band_roformer.py is written for the MSST repo
                # layout and imports "models.bs_roformer.attend". Rather than
                # making you edit the file in site-packages, point that name
                # at the installed package: models -> a stub whose
                # bs_roformer attribute is the real package.
                try:
                    import bs_roformer as _bsr_pkg
                    if "models" not in sys.modules:
                        _shim = types.ModuleType("models")
                        _shim.__path__ = []
                        sys.modules["models"] = _shim
                    sys.modules.setdefault("models.bs_roformer", _bsr_pkg)
                    setattr(sys.modules["models"], "bs_roformer", _bsr_pkg)
                    for _sub in ("attend", "bs_roformer"):
                        try:
                            import importlib
                            sys.modules.setdefault(
                                f"models.bs_roformer.{_sub}",
                                importlib.import_module(f"bs_roformer.{_sub}"))
                        except Exception:
                            pass
                except Exception as _se:
                    print(f"[Karaoke] Could not alias the MSST import path: {_se}")

                try:
                    from bs_roformer.mel_band_roformer import MelBandRoformer
                except ImportError as ie:
                    raise RuntimeError(
                        "bs_roformer.mel_band_roformer is missing. bs-roformer-infer "
                        "ships only the BS-RoFormer architecture, so a Mel-Band model "
                        "cannot be built. Copy mel_band_roformer.py (and attend.py if "
                        "needed) into the installed bs_roformer folder — do NOT "
                        "pip install BS-RoFormer over it, that replaces the package "
                        f"the other two models use. Original error: {ie}") from ie

                import inspect
                accepted = set(inspect.signature(MelBandRoformer.__init__).parameters)
                kw = {}
                for k, v in dict(kara_cfg.model).items():
                    if k in accepted:
                        kw[k] = tuple(v) if isinstance(v, list) else v
                if "sample_rate" in accepted:
                    kw.setdefault("sample_rate", _kara_sr)
                if "flash_attn" in accepted:
                    kw["flash_attn"] = bool(torch.cuda.is_available()) and not _SAFE_MODE
                m = MelBandRoformer(**kw)
            else:
                get_model_from_config = _import_bs_roformer()["get_model_from_config"]
                m = get_model_from_config("bs_roformer", kara_cfg)
            if m is None:
                raise RuntimeError("Could not build the karaoke model from its config")

            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict):
                sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
            _load_state_forgiving(m, sd, os.path.basename(ckpt_path))

            # Which output is the backing vocals, and which the lead?
            _kara_back_idx, _kara_lead_idx = None, None
            try:
                stems = [str(x).lower() for x in kara_cfg.training.instruments]
                print(f"[Karaoke] Stems in config: {stems}")
                for i, nm in enumerate(stems):
                    if any(t in nm for t in ("back", "bg", "choir", "harmon", "other")):
                        _kara_back_idx = i
                    elif any(t in nm for t in ("lead", "main", "vocal")):
                        if _kara_lead_idx is None:
                            _kara_lead_idx = i
                if _kara_back_idx is None and len(stems) == 2:
                    # Two stems, one of them the lead: the other is the backing.
                    _kara_back_idx = 1 - (_kara_lead_idx or 0)
            except Exception:
                pass
            if _kara_back_idx is None:
                _kara_back_idx = 1   # karaoke models put the backing second

            m.to(device)
            m.eval()
            kara_model = m
            print(f"[Karaoke] Model ready on {device} ({_kara_sr} Hz, "
                  f"backing = output {_kara_back_idx})")
        except Exception as e:
            import traceback
            print(f"[Karaoke] Failed to load model: {e}")
            traceback.print_exc()
        kara_model_ready = True


# How the two karaoke outputs are assigned to the cells.
#   "auto"   — follow the config, but swap if the audio says otherwise
#   "config" — trust the config's stem names
#   "swap"   — always the other way round
_KARA_ORIENT = "auto"
# Take BG VOX as "the vocals stem minus the lead" instead of using whatever
# the model calls its second output. The input to this pass is the separated
# VOCALS stem, so the residue is by definition the rest of the singing —
# nothing instrumental can appear in it, whatever a given karaoke model
# names its outputs or which order it emits them in. Set to False to use the
# model's own second output.
_KARA_BACKING_FROM_RESIDUE = True
_KARA_SWAP_MARGIN_DB = 2.5     # how much louder before "auto" believes it


def _rms_db(x):
    if x is None or len(x) == 0:
        return -120.0
    r = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float32) ** 2)))
    return 20.0 * np.log10(max(r, 1e-9))


def toggle_backing_source():
    """Switch BG VOX between the model's own output and the residue.

    Which sounds better depends on the karaoke model: the residue can never
    contain anything that was not in the vocals stem, but it also carries
    whatever the lead half left behind, artefacts included.
    """
    global _KARA_BACKING_FROM_RESIDUE
    _KARA_BACKING_FROM_RESIDUE = not _KARA_BACKING_FROM_RESIDUE
    src = "the vocals-stem residue" if _KARA_BACKING_FROM_RESIDUE else \
          "the model's own second output"
    print(f"[Karaoke] BG VOX will come from {src} — press ⟳ SPLIT VOX to "
          f"hear the difference")
    btn = globals().get("_backing_src_btn")
    if btn is not None:
        try:
            btn.configure(text="BG: RESIDUE" if _KARA_BACKING_FROM_RESIDUE
                          else "BG: MODEL")
        except Exception:
            pass


def swap_vocal_halves():
    """Exchange FRT VOX and BG VOX, for when the automatic choice is wrong."""
    with audio_lock:
        state.fv_data, state.bg_vocals_data = (state.bg_vocals_data,
                                               state.fv_data)
        state.fv_sr, state.bg_vocals_sr = state.bg_vocals_sr, state.fv_sr
    print("[Karaoke] FRT VOX and BG VOX swapped by hand")
    if running:
        for fn in ("_update_bgv_button", "_update_fv_button"):
            f = globals().get(fn)
            if f is not None:
                try:
                    f()
                except Exception:
                    pass


def _orient_halves(lead, back):
    """Return (lead, back) the right way round, whatever the model's order.

    Config stem names are not reliable across karaoke models: some list the
    lead first, some the backing, and some name them in ways that match
    neither. The lead vocal is reliably the louder and more continuous of
    the two, so when the halves look swapped by that measure, they are put
    back. Set _KARA_ORIENT to "config" to trust the names instead.
    """
    if lead is None or back is None:
        return lead, back
    if _KARA_ORIENT == "config":
        return lead, back
    if _KARA_ORIENT == "swap":
        print("[Karaoke] Halves swapped (_KARA_ORIENT = 'swap')")
        return back, lead

    lead_db, back_db = _rms_db(lead), _rms_db(back)
    # "Activity": how much of the time each half is actually sounding. The
    # lead sings through most of a song; backing vocals come and go.
    def _active(x):
        mono = np.abs(np.mean(np.asarray(x, dtype=np.float32), axis=1))
        step = max(1, len(mono) // 2000)
        frames = mono[:len(mono) // step * step].reshape(-1, step).max(axis=1)
        peak = float(frames.max()) if len(frames) else 0.0
        if peak <= 1e-6:
            return 0.0
        return float(np.mean(frames > peak * 0.05))

    lead_act, back_act = _active(lead), _active(back)
    print(f"[Karaoke] FRT candidate {lead_db:.1f} dB, {lead_act * 100:.0f}% "
          f"active | BG candidate {back_db:.1f} dB, {back_act * 100:.0f}% active")
    if (back_db - lead_db) > _KARA_SWAP_MARGIN_DB and back_act >= lead_act:
        print("[Karaoke] The backing half is the louder and busier of the "
              "two — this model lists its outputs the other way round, so "
              "the halves have been swapped.")
        return back, lead
    return lead, back


def separate_bg_vocals(into=None, cancel=None):
    """Split the separated vocals into lead and backing, keeping both.

    Runs on state.stems["vocals"] — the karaoke model only has to look at the
    vocal, so this is far cheaper than another pass over the whole mix. One
    pass yields both halves: the backing goes to state.bg_vocals_data and the
    lead to state.fv_data, the places the old IMPORT buttons filled, so both
    cells behave exactly as before.
    """
    global _kara_separating
    bg = into is not None
    if not bg and _kara_separating:
        print("[Karaoke] A split is already running — not starting another")
        return
    if not bg:
        _kara_separating = True

    if not kara_model_ready:
        load_kara_model()
    if kara_model is None:
        print("[Karaoke] No karaoke model loaded — FRT VOX and BG VOX stay "
              "empty. (Muting a cell never stops the split; this is the "
              "model itself failing to load — see the [Karaoke] lines at "
              "start-up.)")
        if not bg:
            _kara_separating = False
        return

    src = None
    if bg:
        src = into.get("_vocals_src")
    elif state.stems:
        src = state.stems.get("vocals")
    if src is None:
        print("[Karaoke] No VOCALS stem to split — FRT VOX and BG VOX stay "
              "empty. (The six-stem pass has to finish first.)")
        if not bg:
            _kara_separating = False
        return

    if not bg:
        _fg_enter("karaoke split")
    else:
        _bg_wait_for_foreground(cancel)
        if cancel is not None and cancel.is_set():
            return
    try:
        print("[Karaoke] Splitting lead / backing vocals ...")
        t0 = time.perf_counter()
        audio = np.asarray(src, dtype=np.float32)
        in_sr = int(state.sr) if state.sr else 44100
        if in_sr != _kara_sr:
            audio = _resample_to(audio, in_sr, _kara_sr)

        run_device = device
        sr_i       = int(_kara_sr)
        n_samples  = len(audio)
        chunk_n    = int(round(_kara_chunk_s * sr_i))
        overlap_n  = int(round(globals().get("_kara_overlap_s", _OVERLAP_SECONDS) * sr_i))
        step_n     = chunk_n - 2 * overlap_n

        out_back = np.zeros((n_samples, 2), dtype=np.float32)
        out_lead = np.zeros((n_samples, 2), dtype=np.float32)
        wgt      = np.zeros(n_samples, dtype=np.float32)
        fade     = np.ones(chunk_n, dtype=np.float32)
        fade[:overlap_n]  = np.linspace(0, 1, overlap_n)
        fade[-overlap_n:] = np.linspace(1, 0, overlap_n)

        starts   = list(range(0, n_samples, step_n))
        n_chunks = len(starts)
        batch_n  = _batch_size_for(run_device, chunk_n / sr_i)
        use_fp16 = _INFER_FP16

        def _split_upd(v):
            if not bg and running:
                app.after(0, lambda: _set_split_progress(v))

        _split_upd(0.02)
        kara_model.eval()
        ci = 0
        with _infer_ctx():
            while ci < n_chunks:
                if not running:
                    break
                if cancel is not None and cancel.is_set():
                    print("[Karaoke] Cancelled")
                    return
                batch_starts = starts[ci:ci + batch_n]
                chunks, kept = [], []
                for start in batch_starts:
                    end   = min(start + chunk_n, n_samples)
                    chunk = audio[start:end]
                    pad   = chunk_n - len(chunk)
                    if pad:
                        chunk = np.pad(chunk, ((0, pad), (0, 0)))
                    if float(np.max(np.abs(chunk))) < _SILENCE_PEAK:
                        actual = min(end - start, chunk_n)
                        wgt[start:start + actual] += fade[:actual]
                        continue
                    chunks.append(chunk.T)
                    kept.append(start)
                if not chunks:
                    ci += len(batch_starts)
                    continue
                try:
                    x = torch.from_numpy(_stack_padded(chunks, batch_n)).to(run_device)
                    with _amp_ctx(run_device, use_fp16):
                        out_t = kara_model(x)
                    if out_t.dim() == 4:
                        n_out = out_t.shape[1]
                        # The config may not have named its stems, leaving
                        # these unset; fall back to "the second output is
                        # the backing" rather than failing the whole pass.
                        b_idx = _kara_back_idx
                        if b_idx is None:
                            b_idx = 1 if n_out > 1 else 0
                        b_idx = min(b_idx, n_out - 1)
                        l_idx = _kara_lead_idx
                        if l_idx is None or l_idx >= n_out or l_idx == b_idx:
                            l_idx = 1 - b_idx if n_out > 1 else None
                        stem_t = out_t[:, b_idx]
                        lead_t = out_t[:, l_idx] if l_idx is not None else None
                    else:
                        stem_t, lead_t = out_t, None
                    stem_np = stem_t.float().cpu().numpy()
                    lead_np = (lead_t.float().cpu().numpy()
                               if lead_t is not None else None)
                except RuntimeError as e:
                    if run_device.type == "cuda":
                        torch.cuda.empty_cache()
                    if batch_n > 1:
                        batch_n = max(1, batch_n // 2)
                        print(f"[Karaoke] {e}\n[Karaoke] Retrying with batch {batch_n}")
                        continue
                    if _is_oom(e) and run_device.type == "cuda":
                        # Full precision would need twice the memory; finish
                        # this track on the CPU instead.
                        run_device = torch.device("cpu")
                        kara_model.to(run_device)
                        use_fp16 = False
                        print("[Karaoke] Out of GPU memory even one chunk at a time — finishing this track on the CPU")
                        continue
                    if use_fp16:
                        use_fp16 = False
                        print(f"[Karaoke] {e}\n[Karaoke] Retrying in full precision")
                        continue
                    raise

                t_out = stem_np.shape[-1]
                for bi, start in enumerate(kept):
                    end    = min(start + chunk_n, n_samples)
                    actual = min(end - start, t_out)
                    w      = fade[:actual]
                    out_back[start:start + actual] += stem_np[bi, :, :actual].T * w[:, None]
                    if lead_np is not None:
                        out_lead[start:start + actual] += \
                            lead_np[bi, :, :actual].T * w[:, None]
                    wgt[start:start + actual]      += w
                ci += len(batch_starts)
                _split_upd(min(0.99, ci / max(1, n_chunks)))

        wgt = np.maximum(wgt, 1e-8)
        if run_device.type != device.type:
            kara_model.to(device)
        out_back /= wgt[:, None]
        out_lead /= wgt[:, None]
        have_lead = bool(np.any(out_lead))

        if not have_lead:
            # Only one output came back — either the model emits a single
            # stem (a 3-D result), or its config never said which output is
            # the lead. The two halves always sum to the vocal that went in,
            # so the missing one is simply the rest of it. Without this the
            # cell that did not get a stem stays empty and the split looks
            # like it failed.
            ref = np.asarray(src, dtype=np.float32)
            n = min(len(ref), len(out_back))
            derived = np.zeros_like(out_back)
            derived[:n] = ref[:n] - out_back[:n]
            if np.any(derived):
                out_lead = derived
                have_lead = True
                print("[Karaoke] The model returned one stem; the other half "
                      "was taken as what it leaves behind in the vocal.")
            else:
                print("[Karaoke] The model returned one stem and it accounts "
                      "for the whole vocal — nothing left for the other cell.")
        if sr_i != in_sr:
            out_back = _resample_to(out_back, sr_i, in_sr)
            if have_lead:
                out_lead = _resample_to(out_lead, sr_i, in_sr)

        el = time.perf_counter() - t0
        print(f"[Karaoke] Lead + backing vocals in {el:.1f}s "
              f"({(n_samples / sr_i) / max(el, 1e-6):.1f}x realtime)")

        if have_lead:
            out_lead, out_back = _orient_halves(out_lead, out_back)

            if _KARA_BACKING_FROM_RESIDUE:
                # BG VOX = what the lead leaves behind in the vocals stem.
                ref = np.asarray(src, dtype=np.float32)
                n = min(len(ref), len(out_lead))
                residue = np.zeros_like(out_lead)
                residue[:n] = ref[:n] - out_lead[:n]
                model_db, res_db = _rms_db(out_back), _rms_db(residue)
                print(f"[Karaoke] Backing: model output {model_db:.1f} dB, "
                      f"residue of the vocals stem {res_db:.1f} dB "
                      f"— using the residue")
                out_back = residue

        # Warn when a model plainly did not split anything, rather than
        # filling both cells with the same audio.
        if have_lead:
            ref = np.asarray(src, dtype=np.float32)
            n = min(len(ref), len(out_lead))
            if n:
                a = out_lead[:n].ravel()
                b = ref[:n].ravel()
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                corr = float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0
                if corr > 0.999:
                    print("[Karaoke] This model returned the vocals unchanged "
                          "— it does not appear to split lead from backing. "
                          "FRT VOX holds the whole vocal and BG VOX is empty.")

        if bg:
            into["bg_vocals"] = out_back
            if have_lead:
                into["front_vocals"] = out_lead
            return
        sr_now = int(state.sr) if state.sr else 44100
        state.bg_vocals_data = out_back
        state.bg_vocals_sr   = sr_now
        if have_lead:
            state.fv_data = out_lead
            state.fv_sr   = sr_now
        print(f"[Karaoke] FRT VOX: {'filled' if have_lead else 'empty'} | "
              f"BG VOX: filled — the VOCALS stem is muted in their favour")
        if running:
            app.after(0, _split_replaces_vocals)
            app.after(0, _update_bgv_button)
            app.after(0, _update_fv_button)
    except Exception as e:
        import traceback
        print(f"[Karaoke] Separation error: {e}")
        traceback.print_exc()
    finally:
        if not bg:
            _fg_leave()
            _kara_separating = False
            if running:
                app.after(0, _clear_split_progress)
                app.after(0, _update_bgv_button)
                app.after(0, _update_fv_button)


def load_strings_model():
    """Load gilliaan's bowed-strings model, in the same way as the others."""
    global strings_model, strings_model_ready, _str_sr, _str_stem_idx
    global _str_chunk_s, _str_overlap_s
    with _str_load_lock:
        if strings_model_ready:
            return
        try:
            ckpt_path, cfg_path = _str_find_files()
            if not ckpt_path or not cfg_path:
                raise FileNotFoundError(
                    "Bowed-strings model not found. Put gilliaan's .ckpt and "
                    f".yaml in {_STR_SEARCH_DIRS[0]!r} (models.json can "
                    "download them), or set _STR_CKPT_PATH / _STR_CFG_PATH.")
            if _is_main_model_file(ckpt_path) or _is_main_model_file(cfg_path):
                raise RuntimeError(
                    f"{os.path.basename(ckpt_path)} is the six-stem model — "
                    "not a strings model.")
            print(f"[Strings] Checkpoint: {ckpt_path}")
            print(f"[Strings] Config:     {cfg_path}")

            import yaml
            _bsr = _import_bs_roformer()
            get_model_from_config = _bsr["get_model_from_config"]
            _YL = _bsr["yaml_loader"]
            from ml_collections import ConfigDict
            with open(cfg_path, "r", encoding="utf-8") as f:
                str_cfg = ConfigDict(yaml.load(f, Loader=_YL))

            try:
                _str_sr = int(str_cfg.audio.sample_rate)
            except Exception:
                _str_sr = 44100
            _str_chunk_s, _str_overlap_s = _chunking_from_config(str_cfg, _str_sr)
            print(f"[Strings] Chunking from config: {_str_chunk_s:.2f}s chunks, "
                  f"{_str_overlap_s:.2f}s overlap per side")

            # Mel-band models need the real librosa (see the karaoke loader).
            is_mel = ("mel" in os.path.basename(cfg_path).lower()
                      or "mel" in os.path.basename(ckpt_path).lower()
                      or bool(str_cfg.model.get("num_bands", 0)))
            if is_mel and not _ensure_librosa():
                print("[Models] Mel-Band model without librosa — trying the "
                      "built-in filterbank rather than refusing outright.")
            arch = "mel_band_roformer" if is_mel else "bs_roformer"
            if is_mel:
                # ZFTurbo's mel_band_roformer.py imports models.bs_roformer;
                # point that at the installed package (as the karaoke loader
                # does) so the import resolves.
                try:
                    import bs_roformer as _bsr_pkg
                    if "models" not in sys.modules:
                        _shim = types.ModuleType("models")
                        _shim.__path__ = []
                        sys.modules["models"] = _shim
                    sys.modules.setdefault("models.bs_roformer", _bsr_pkg)
                    setattr(sys.modules["models"], "bs_roformer", _bsr_pkg)
                except Exception:
                    pass

            m = get_model_from_config(arch, str_cfg)
            if m is None:
                raise RuntimeError("get_model_from_config returned None")

            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict):
                sd = {k[7:] if k.startswith("module.") else k: v
                      for k, v in sd.items()}
            _load_state_forgiving(m, sd, os.path.basename(ckpt_path))

            # Which output holds the strings?
            _str_stem_idx = 0
            try:
                stems = [str(x).lower() for x in str_cfg.training.instruments]
                print(f"[Strings] Stems in config: {stems}")
                for i, name in enumerate(stems):
                    if any(t in name for t in ("string", "bowed", "violin",
                                               "cello", "orchestr")):
                        _str_stem_idx = i
                        break
            except Exception:
                pass

            m.to(device)
            m.eval()
            strings_model = m
            print(f"[Strings] Model ready on {device} ({_str_sr} Hz, "
                  f"stem index {_str_stem_idx})")
        except Exception as e:
            print(f"[Strings] Failed to load model: {e}")
        strings_model_ready = True


def load_vocals_model():
    """Load the dedicated vocals model."""
    global vocals_model, vocals_model_ready, _voc_sr, _voc_stem_idx
    global _voc_chunk_s, _voc_overlap_s
    with _voc_load_lock:
        if vocals_model_ready:
            return
        try:
            ckpt_path, cfg_path = _voc_find_files()
            if not ckpt_path or not cfg_path:
                raise FileNotFoundError(
                    "No dedicated vocals model found — the six-stem vocals "
                    "will be used as they are.")
            print(f"[Vocals] Checkpoint: {ckpt_path}")
            print(f"[Vocals] Config:     {cfg_path}")

            import yaml
            _bsr = _import_bs_roformer()
            get_model_from_config = _bsr["get_model_from_config"]
            _YL = _bsr["yaml_loader"]
            from ml_collections import ConfigDict
            with open(cfg_path, "r", encoding="utf-8") as f:
                voc_cfg = ConfigDict(yaml.load(f, Loader=_YL))

            try:
                _voc_sr = int(voc_cfg.audio.sample_rate)
            except Exception:
                _voc_sr = 44100
            _voc_chunk_s, _voc_overlap_s = _chunking_from_config(voc_cfg, _voc_sr)
            print(f"[Vocals] Chunking from config: {_voc_chunk_s:.2f}s chunks, "
                  f"{_voc_overlap_s:.2f}s overlap per side")

            name_hint = (os.path.basename(ckpt_path) +
                         os.path.basename(cfg_path)).lower()
            is_mel = ("mel_band" in name_hint or "mel-band" in name_hint
                      or bool(voc_cfg.model.get("num_bands", 0)))
            if is_mel and not _ensure_librosa():
                print("[Models] Mel-Band model without librosa — trying the "
                      "built-in filterbank rather than refusing outright.")
            if is_mel:
                try:
                    import bs_roformer as _bsr_pkg
                    if "models" not in sys.modules:
                        _shim = types.ModuleType("models")
                        _shim.__path__ = []
                        sys.modules["models"] = _shim
                    sys.modules.setdefault("models.bs_roformer", _bsr_pkg)
                    setattr(sys.modules["models"], "bs_roformer", _bsr_pkg)
                except Exception:
                    pass
            arch = "mel_band_roformer" if is_mel else "bs_roformer"

            m = get_model_from_config(arch, voc_cfg)
            if m is None:
                raise RuntimeError("get_model_from_config returned None")

            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict):
                sd = {k[7:] if k.startswith("module.") else k: v
                      for k, v in sd.items()}
            _load_state_forgiving(m, sd, os.path.basename(ckpt_path))

            # Which output holds the vocals?
            _voc_stem_idx = 0
            try:
                stems = [str(x).lower() for x in voc_cfg.training.instruments]
                print(f"[Vocals] Stems in config: {stems}")
                for i, name in enumerate(stems):
                    if "vocal" in name and "back" not in name:
                        _voc_stem_idx = i
                        break
            except Exception:
                pass

            m.to(device)
            m.eval()
            vocals_model = m
            print(f"[Vocals] Model ready on {device} ({_voc_sr} Hz, "
                  f"vocals = output {_voc_stem_idx})")
        except Exception as e:
            print(f"[Vocals] {e}")
        vocals_model_ready = True


def refine_vocals(path, cancel=None):
    """Run the vocals model over the MAIN TRACK and replace the VOCALS stem.

    Deliberately takes the original file rather than the six-stem vocals:
    feeding it an already-separated stem would pile one model's mistakes on
    top of another's.
    """
    global _vocals_refining
    if _vocals_refining:
        return False
    if not vocals_model_ready:
        load_vocals_model()
    if vocals_model is None:
        return False

    _vocals_refining = True
    _fg_enter("vocals model")
    if running:
        app.after(0, _update_vocals_status)
    try:
        audio, file_sr = _read_audio_file(path)
        audio = _resample_audio(audio, file_sr)
        sr_i = int(state.sr or _voc_sr)
        x_full = audio.T.astype(np.float32)
        n_samples = x_full.shape[1]

        chunk_n   = int(round(_voc_chunk_s * sr_i))
        overlap_n = int(round(_voc_overlap_s * sr_i))
        step_n    = max(1, chunk_n - 2 * overlap_n)
        fade = np.ones(chunk_n, dtype=np.float32)
        if overlap_n > 0:
            fade[:overlap_n]  = np.linspace(0, 1, overlap_n)
            fade[-overlap_n:] = np.linspace(1, 0, overlap_n)

        out = np.zeros((2, n_samples), dtype=np.float32)
        wgt = np.zeros(n_samples, dtype=np.float32)
        starts  = list(range(0, n_samples, step_n))
        batch_n = _batch_size_for(device, chunk_n / sr_i)
        use_fp16 = _INFER_FP16
        run_device = device
        ci = 0
        t0 = time.time()

        with _infer_ctx():
            i = 0
            while i < len(starts):
                if cancel is not None and cancel.is_set():
                    print("[Vocals] Cancelled")
                    return False
                batch_starts = starts[i:i + batch_n]
                chunks, kept = [], []
                for st_i in batch_starts:
                    seg = x_full[:, st_i:st_i + chunk_n]
                    if seg.shape[1] < chunk_n:
                        seg = np.pad(seg, ((0, 0), (0, chunk_n - seg.shape[1])))
                    chunks.append(seg)
                    kept.append(st_i)
                try:
                    xb = torch.from_numpy(_stack_padded(chunks, batch_n))
                    xb = (xb.pin_memory().to(run_device, non_blocking=True)
                          if run_device.type == "cuda" else xb.to(run_device))
                    with _amp_ctx(run_device, use_fp16):
                        pred = vocals_model(xb)
                    arr = pred.float().cpu().numpy()
                except RuntimeError as e:
                    if run_device.type == "cuda":
                        torch.cuda.empty_cache()
                    if batch_n > 1:
                        batch_n = max(1, batch_n // 2)
                        print(f"[Vocals] {e}\n[Vocals] Retrying with batch {batch_n}")
                        continue
                    if _is_oom(e) and run_device.type == "cuda":
                        # Full precision would need twice the memory; finish
                        # this track on the CPU instead.
                        run_device = torch.device("cpu")
                        vocals_model.to(run_device)
                        use_fp16 = False
                        print("[Vocals] Out of GPU memory even one chunk at a time — finishing this track on the CPU")
                        continue
                    if use_fp16:
                        use_fp16 = False
                        print(f"[Vocals] {e}\n[Vocals] Retrying in full precision")
                        continue
                    raise

                if arr.ndim == 4:
                    arr = arr[:, min(_voc_stem_idx, arr.shape[1] - 1)]
                for bi, st_i in enumerate(kept):
                    actual = min(chunk_n, n_samples - st_i, arr.shape[-1])
                    w = fade[:actual]
                    out[:, st_i:st_i + actual] += arr[bi, :, :actual] * w[None, :]
                    wgt[st_i:st_i + actual]    += w
                ci += len(batch_starts)
                i  += len(batch_starts)
                if running:
                    app.after(0, lambda v=min(0.99, ci / max(1, len(starts))):
                              _set_vocals_progress(v))

        np.divide(out, np.maximum(wgt, 1e-8)[None, :], out=out)
        if run_device.type != device.type:
            vocals_model.to(device)
        refined = np.ascontiguousarray(out.T)
        with audio_lock:
            if state.stems:
                n = min(len(refined), state.stems["vocals"].shape[0])
                state.stems["vocals"][:n] = refined[:n]
                if n < state.stems["vocals"].shape[0]:
                    state.stems["vocals"][n:] = 0.0
            state.vocals_refined = True
        el = time.time() - t0
        print(f"[Vocals] Refined vocals in {el:.1f}s "
              f"({n_samples / sr_i / max(el, 1e-6):.1f}x realtime) — the "
              f"VOCALS cell now holds the dedicated model's output")
        return True
    except Exception as e:
        print(f"[Vocals] Separation error: {e}")
        return False
    finally:
        _fg_leave()
        _vocals_refining = False
        if running:
            app.after(0, _clear_vocals_progress)
            app.after(0, _update_vocals_status)


def separate_strings(path=None, cancel=None):
    """Pull the bowed strings out of the OTHER stem.

    The six-stem model has already set the strings aside in OTHER, so that
    is what this works on rather than the whole mix. Afterwards STRINGS
    holds what the model found and OTHER holds the rest: the two are
    complementary, so nothing is heard twice.
    """
    global _strings_separating
    if _strings_separating:
        return
    if not strings_model_ready:
        load_strings_model()
    if strings_model is None:
        print("[Strings] No model loaded — the STRINGS cell stays empty.")
        return
    if not state.stems or state.stems.get("other") is None:
        print("[Strings] No OTHER stem yet — nothing to take the strings from.")
        return

    _strings_separating = True
    _fg_enter("strings")
    try:
        audio = np.asarray(state.stems["other"], dtype=np.float32).copy()
        sr_i = int(state.sr or _str_sr)
        x_full = audio.T.astype(np.float32)          # (2, N)
        n_samples = x_full.shape[1]

        chunk_n   = int(round(_str_chunk_s * sr_i))
        overlap_n = int(round(_str_overlap_s * sr_i))
        step_n    = max(1, chunk_n - 2 * overlap_n)
        fade = np.ones(chunk_n, dtype=np.float32)
        if overlap_n > 0:
            fade[:overlap_n]  = np.linspace(0, 1, overlap_n)
            fade[-overlap_n:] = np.linspace(1, 0, overlap_n)

        out = np.zeros((2, n_samples), dtype=np.float32)
        wgt = np.zeros(n_samples, dtype=np.float32)

        starts   = list(range(0, n_samples, step_n))
        n_chunks = len(starts)
        batch_n  = _batch_size_for(device, chunk_n / sr_i)
        use_fp16 = _INFER_FP16
        run_device = device
        ci = 0
        t0 = time.time()

        with _infer_ctx():
            i = 0
            while i < len(starts):
                if cancel is not None and cancel.is_set():
                    print("[Strings] Cancelled")
                    return
                batch_starts = starts[i:i + batch_n]
                chunks, kept = [], []
                for st_i in batch_starts:
                    seg = x_full[:, st_i:st_i + chunk_n]
                    if seg.shape[1] < chunk_n:
                        seg = np.pad(seg, ((0, 0), (0, chunk_n - seg.shape[1])))
                    chunks.append(seg)
                    kept.append(st_i)
                try:
                    xb = torch.from_numpy(_stack_padded(chunks, batch_n))
                    xb = (xb.pin_memory().to(run_device, non_blocking=True)
                          if run_device.type == "cuda" else xb.to(run_device))
                    with _amp_ctx(run_device, use_fp16):
                        pred = strings_model(xb)
                    arr = pred.float().cpu().numpy()
                except RuntimeError as e:
                    if run_device.type == "cuda":
                        torch.cuda.empty_cache()
                    if batch_n > 1:
                        batch_n = max(1, batch_n // 2)
                        print(f"[Strings] {e}\n[Strings] Retrying with batch {batch_n}")
                        continue
                    if _is_oom(e) and run_device.type == "cuda":
                        # Full precision would need twice the memory; finish
                        # this track on the CPU instead.
                        run_device = torch.device("cpu")
                        strings_model.to(run_device)
                        use_fp16 = False
                        print("[Strings] Out of GPU memory even one chunk at a time — finishing this track on the CPU")
                        continue
                    if use_fp16:
                        use_fp16 = False
                        print(f"[Strings] {e}\n[Strings] Retrying in full precision")
                        continue
                    raise

                # (B, stems, 2, T) or (B, 2, T) when the model has one output
                if arr.ndim == 4:
                    arr = arr[:, min(_str_stem_idx, arr.shape[1] - 1)]
                for bi, st_i in enumerate(kept):
                    actual = min(chunk_n, n_samples - st_i, arr.shape[-1])
                    w = fade[:actual]
                    out[:, st_i:st_i + actual] += arr[bi, :, :actual] * w[None, :]
                    wgt[st_i:st_i + actual]    += w
                ci += len(batch_starts)
                i  += len(batch_starts)
                if running:
                    app.after(0, lambda v=min(0.99, ci / max(1, n_chunks)):
                              _set_strings_progress(v))

        np.divide(out, np.maximum(wgt, 1e-8)[None, :], out=out)
        if run_device.type != device.type:
            strings_model.to(device)
        strings = np.ascontiguousarray(out.T)
        with audio_lock:
            state.strings_data = strings
            state.strings_sr   = sr_i
            state.strings_is_quick = False
            # OTHER becomes what is left once the strings are taken out —
            # the same as summing OTHER with an inverted copy of them.
            if state.stems and state.stems.get("other") is not None:
                other = state.stems["other"]
                n = min(len(other), len(strings))
                other[:n] -= strings[:n]
                print("[Strings] OTHER now holds what is left after the "
                      "strings were removed")
        el = time.time() - t0
        print(f"[Strings] Strings in {el:.1f}s "
              f"({n_samples / sr_i / max(el, 1e-6):.1f}x realtime)")
        if running:
            app.after(0, _update_strings_button)
    except Exception as e:
        print(f"[Strings] Separation error: {e}")
    finally:
        _fg_leave()
        _strings_separating = False
        if running:
            app.after(0, _clear_strings_progress)
            app.after(0, _update_strings_button)


def separate_inst(path, into=None, cancel=None, ui_progress=False):
    """Run BS-RoFormer Resurrection Inst on *path* in a background thread.

    Produces a stereo Instrumental stem and stores it in
    state.instrumental (resampled to 44100 Hz, shape (N,2) float32).

    Called automatically from load_file() / load_file_path() right after the
    main separation thread is launched, so it runs completely in parallel.
    With *into* the result is cached in that dict instead (playlist pre-load)
    and *cancel* aborts the run at the next chunk.
    """
    global _inst_separating
    bg = into is not None
    if not bg:
        if _inst_separating:
            return
        _inst_separating = True

    def _upd(v):
        """Report progress: on this cell always, and on the main bar when
        this pass is the whole job (instrumental-only mode)."""
        if bg:
            return
        if running:
            app.after(0, lambda: _set_inst_progress(v))
            if ui_progress:
                app.after(0, lambda: _set_progress(v))

    def _finish_ui(_gen=_load_generation()):
        """Instrumental-only mode: do what separate() normally does at the end."""
        if not (ui_progress and not bg):
            return
        if not _is_current_load(_gen):
            return   # a newer load owns the window now
        state.separating = False
        if running:
            def _show():
                progress_bar.pack_forget()
                _clear_progress()
                wave_canvas.pack(fill="both", expand=True)
                for b in _all_import_btns():
                    b.configure(state="normal")
                _unlock_transport()
                _playlist_refresh()
                _apply_pending_autosolo()
                _flash_track_name(state.current_audio_name)
            app.after(0, _show)

    # Make sure the model is loaded before we start inference.
    # load_inst_model() is idempotent and locked, so calling it here is safe
    # even if the startup thread already finished.
    if not inst_model_ready:
        load_inst_model()

    if inst_model is None:
        print("[Inst] Model not available — skipping instrumental separation.")
        if bg:
            return
        _inst_separating = False
        if running:
            app.after(0, _clear_inst_progress)
            app.after(0, _update_inst_status_label)
        return

    if bg:
        _bg_wait_for_foreground(cancel)
        if cancel is not None and cancel.is_set():
            print("[Inst] Pre-load cancelled")
            return
    else:
        _fg_enter("instrumental")
    try:
        print(f"[Inst] {'Pre-loading' if bg else 'Starting'} instrumental separation "
              f"of {os.path.basename(path)!r} ...")

        _upd(0.08)
        # 1. Read & resample to the model's rate
        audio, sr_local = _read_audio_cached(path)
        audio = np.asarray(audio, dtype=np.float32)   # (N, 2)
        model_sr = int(_inst_sr)
        if sr_local != model_sr:
            from scipy.signal import resample_poly as _rp
            def _gcd(a, b):
                while b:
                    a, b = b, a % b
                return a
            g    = _gcd(sr_local, model_sr)
            up   = model_sr // g
            down = sr_local  // g
            audio = np.stack([
                _rp(audio[:, 0], up, down).astype(np.float32),
                _rp(audio[:, 1], up, down).astype(np.float32),
            ], axis=1)

        # 2. Chunked overlap-add inference
        n_samples  = len(audio)
        sr_i       = model_sr
        chunk_n    = int(round(_inst_chunk_s * sr_i))
        overlap_n  = int(round(globals().get("_inst_overlap_s", _OVERLAP_SECONDS) * sr_i))
        step_n     = chunk_n - 2 * overlap_n

        out_inst = np.zeros((n_samples, 2), dtype=np.float32)
        wgt      = np.zeros(n_samples,     dtype=np.float32)

        fade              = np.ones(chunk_n, dtype=np.float32)
        fade[:overlap_n]  = np.linspace(0, 1, overlap_n)
        fade[-overlap_n:] = np.linspace(1, 0, overlap_n)

        starts   = list(range(0, n_samples, step_n))
        n_chunks = len(starts)

        run_device = device
        batch_n    = _batch_size_for(run_device,
                                     (chunk_n / sr_i) / _INST_BATCH_FACTOR)
        use_fp16   = _INFER_FP16
        _t_inst    = time.perf_counter()
        inst_model.eval()
        ci = 0
        with _infer_ctx():
            while ci < n_chunks:
                if not running:
                    break
                if cancel is not None and cancel.is_set():
                    print("[Inst] Cancelled — another track was loaded")
                    return
                batch_starts = starts[ci:ci + batch_n]

                chunks   = []
                kept     = []       # starts that actually need the model
                for start in batch_starts:
                    end   = min(start + chunk_n, n_samples)
                    chunk = audio[start:end]
                    pad   = chunk_n - len(chunk)
                    if pad:
                        chunk = np.pad(chunk, ((0, pad), (0, 0)))
                    if float(np.max(np.abs(chunk))) < _SILENCE_PEAK:
                        # Silence in, silence out — just carry the weight so
                        # the overlap-add still normalises correctly.
                        actual = min(end - start, chunk_n)
                        wgt[start:start + actual] += fade[:actual]
                        continue
                    chunks.append(chunk.T)
                    kept.append(start)

                if not chunks:
                    ci += len(batch_starts)
                    continue

                try:
                    # Input: (B, 2, T)
                    x = torch.from_numpy(_stack_padded(chunks, batch_n)).to(run_device)
                    with _amp_ctx(run_device, use_fp16):
                        # Output: (B, num_stems, 2, T) for multi-stem models,
                        #         (B, 2, T) when the model has a single stem.
                        out_t = inst_model(x)
                    if out_t.dim() == 4:
                        stem_t = out_t[:, _inst_stem_idx]
                    else:
                        stem_t = out_t
                    stem_np = stem_t.float().cpu().numpy()      # (B, 2, T_out)
                except RuntimeError as e:
                    if run_device.type == "cuda":
                        torch.cuda.empty_cache()
                    if batch_n > 1:
                        batch_n = max(1, batch_n // 2)
                        print(f"[Inst] {e}\n[Inst] Retrying with batch size {batch_n}")
                        continue
                    if _is_oom(e) and run_device.type == "cuda":
                        # Full precision would need twice the memory. Move to
                        # the CPU for the rest: slower, but it finishes.
                        run_device = torch.device("cpu")
                        inst_model.to(run_device)
                        use_fp16 = False
                        print("[Inst] Out of GPU memory even one chunk at a time "
                              "— finishing this track on the CPU")
                        continue
                    if use_fp16:
                        use_fp16 = False
                        print(f"[Inst] {e}\n[Inst] Retrying in full precision")
                        continue
                    raise

                t_out = stem_np.shape[-1]
                for bi, start in enumerate(kept):
                    end    = min(start + chunk_n, n_samples)
                    actual = min(end - start, t_out)
                    w      = fade[:actual]
                    seg    = stem_np[bi, :, :actual].T          # (T, 2)
                    out_inst[start:start + actual] += seg * w[:, None]
                    wgt[start:start + actual]      += w

                ci += len(batch_starts)
                _upd(min(0.99, 0.15 + 0.8 * ci / n_chunks))
                pct = int(100 * ci / n_chunks)
                if pct % 10 == 0 and not bg:
                    print(f"[Inst] {pct}% complete")

        wgt = np.maximum(wgt, 1e-8)
        out_inst /= wgt[:, None]
        if run_device.type != device.type:
            inst_model.to(device)       # ready on the GPU for the next track

        # 3. Back to the app's playback rate if the model used another one
        if sr_i != 44100:
            from scipy.signal import resample_poly as _rp
            def _gcd2(a, b):
                while b:
                    a, b = b, a % b
                return a
            g    = _gcd2(sr_i, 44100)
            up   = 44100 // g
            down = sr_i  // g
            out_inst = np.stack([
                _rp(out_inst[:, 0], up, down).astype(np.float32),
                _rp(out_inst[:, 1], up, down).astype(np.float32),
            ], axis=1)

        _el = time.perf_counter() - _t_inst
        print(f"[Inst] Instrumental in {_el:.1f}s "
              f"({(n_samples / sr_i) / max(_el, 1e-6):.1f}x realtime)")
        if bg:
            into["instrumental"] = out_inst
            print(f"[Inst] Pre-loaded instrumental — {len(out_inst) / 44100:.1f} s")
            return
        state.instrumental = out_inst
        state.instrumental_is_quick = False
        print(f"[Inst] Instrumental ready — {len(out_inst) / 44100:.1f} s")
        if ui_progress:
            state.loaded_audio_name = state.current_audio_name
            # This pass is the entire load, so it owns the waveform too.
            mono = np.mean(out_inst, axis=1)
            state.waveform_data = mono[::max(1, len(mono) // 2000)]
            with audio_lock:
                state.sr       = 44100
                state.position = 0
                _eq_zi_state.clear()
            state.loop_start = state.loop_end = None
            _upd(1.0)
            _finish_ui()
        elif state.stems is None and running:
            # Arrived before the stems: it can be played on its own already.
            app.after(0, _unlock_transport)
            print("[Inst] Playable now — stems still separating")

    except Exception as e:
        import traceback
        print(f"[Inst] Separation error: {e}")
        traceback.print_exc()
        if bg:
            return
        state.instrumental = None
        _finish_ui()
    finally:
        if not bg:
            _fg_leave()
            # Always take the cell's progress bar down. It used to be
            # cleared only on success (or in INSTRUMENTAL-ONLY mode), so an
            # error part-way through left it frozen at whatever it last
            # showed — which looked exactly like the pass hanging.
            if running:
                app.after(0, _clear_inst_progress)
                app.after(0, _update_inst_status_label)
            _inst_separating = False
            if running:
                app.after(0, _clear_inst_progress)
                app.after(0, _update_inst_status_label)
                app.after(0, _apply_pending_autosolo)


# Start loading the instrumental model in parallel with the main model.
# Must be placed here — after load_inst_model() is defined.
threading.Thread(target=load_inst_model, daemon=True).start()
threading.Thread(target=load_kara_model, daemon=True).start()
threading.Thread(target=load_strings_model, daemon=True).start()
threading.Thread(target=load_vocals_model, daemon=True).start()


# ----------------------------
# STATE
# All mutable audio/mixer state lives in one place so functions receive it
# explicitly (via the module-level `state` singleton) rather than scattering
# `global` declarations across hundreds of call sites.
# ----------------------------
import dataclasses
from typing import Optional

@dataclasses.dataclass
class AppState:
    # ── playback ──────────────────────────────────────────────
    stems:         Optional[dict]      = None   # name -> (N,2) float32 ndarray
    stem_volumes:  dict = dataclasses.field(default_factory=dict)
    sr:            Optional[int]       = None
    position:      int                 = 0
    stream:        object              = None   # sd.OutputStream | None
    waveform_data: Optional[object]    = None   # downsampled mono array

    # ── per-stem peak meters (written by audio thread, read by UI) ──
    # Float reads/writes are atomic in CPython — no lock needed.
    _meter_levels: dict = dataclasses.field(default_factory=dict)

    # ── master bus ────────────────────────────────────────────
    volume_master: float = 0.7
    stereo_width:  float = 1.0
    reverb_master: float = 0.0   # master reverb wet mix
    air_master:    float = 0.0   # master air boost/cut (0.0 = flat)

    # ── per-stem controls ─────────────────────────────────────
    stem_widths:  dict = dataclasses.field(default_factory=dict)  # key -> 0.0–2.0
    stem_reverbs: dict = dataclasses.field(default_factory=dict)  # key -> 0.0–1.0
    stem_air:     dict = dataclasses.field(default_factory=dict)  # key -> -1.0–1.0
    # Cells whose polarity is flipped (Ø). Inverting one of a pair of cells
    # that share material cancels what they have in common, which is how you
    # hear what is only in one of them.
    stem_invert: dict = dataclasses.field(default_factory=dict)
    stem_pan:     dict = dataclasses.field(default_factory=dict)  # key -> -1.0–1.0
    stem_mute:    dict = dataclasses.field(default_factory=dict)  # key -> bool
    stem_solo:    dict = dataclasses.field(default_factory=dict)  # key -> bool
    stem_nudge:   dict = dataclasses.field(default_factory=dict)  # key -> int samples

    # ── compressor per stem ───────────────────────────────────
    stem_comp_enabled: dict = dataclasses.field(default_factory=dict)
    stem_comp_thresh:  dict = dataclasses.field(default_factory=dict)
    stem_comp_ratio:   dict = dataclasses.field(default_factory=dict)
    stem_comp_attack:  dict = dataclasses.field(default_factory=dict)
    stem_comp_release: dict = dataclasses.field(default_factory=dict)

    # ── gate per stem ─────────────────────────────────────────
    stem_gate_enabled:  dict = dataclasses.field(default_factory=dict)
    stem_gate_thresh:   dict = dataclasses.field(default_factory=dict)
    stem_gate_attack:   dict = dataclasses.field(default_factory=dict)
    stem_gate_release:  dict = dataclasses.field(default_factory=dict)

    # ── leveller per stem ─────────────────────────────────────
    stem_lvl_enabled:   dict = dataclasses.field(default_factory=dict)
    stem_lvl_threshold: dict = dataclasses.field(default_factory=dict)
    stem_lvl_amount:    dict = dataclasses.field(default_factory=dict)

    # ── limiter per stem ──────────────────────────────────────
    stem_lim_enabled:   dict = dataclasses.field(default_factory=dict)
    stem_lim_threshold: dict = dataclasses.field(default_factory=dict)  # dBFS, e.g. -6.0
    stem_lim_ceiling:   dict = dataclasses.field(default_factory=dict)  # dBFS, e.g. -0.1

    # ── EQ ────────────────────────────────────────────────────
    eq_bands: dict = dataclasses.field(default_factory=lambda: {
        k: [0] * 5 for k in [
            "vocals", "drums", "bass", "guitar", "piano", "other",
            "any", "any+", "any++", "atmos_fl", "atmos_fr", "atmos_c", "atmos_lfe", "atmos_bl", "atmos_br",
            "front_vocals", "bg_vocals", "hidden_layer",
            "instrumental",
        ]
    })

    # ── de-bleed ──────────────────────────────────────────────
    stem_debleed: dict = dataclasses.field(default_factory=dict)  # target -> {src: amount}

    # ── user-imported stems ───────────────────────────────────
    fv_data:          Optional[object] = None
    fv_sr:            Optional[int]    = None
    fv_volume:        float            = 1.0
    fv_vff_enabled:   bool             = False
    fv_vff_lead_cut:  float            = 0.5
    fv_vff_body_cut:  float            = 0.5
    fv_vff_presence:  float            = 0.5
    fv_vff_bkg_vol:   float            = 1.0

    bg_vocals_data:    Optional[object] = None
    bg_vocals_sr:      Optional[int]    = None
    bg_vocals_volume:  float            = 1.0
    bgv_vff_enabled:   bool             = False
    bgv_vff_lead_cut:  float            = 0.5
    bgv_vff_body_cut:  float            = 0.5
    bgv_vff_presence:  float            = 0.5
    bgv_vff_bkg_vol:   float            = 1.0

    hl_data:          Optional[object] = None
    hl_sr:            Optional[int]    = None
    hl_volume:        float            = 1.0
    hl_vff_enabled:   bool             = False
    hl_vff_lead_cut:  float            = 0.5
    hl_vff_body_cut:  float            = 0.5
    hl_vff_presence:  float            = 0.5
    hl_vff_bkg_vol:   float            = 1.0


    # ── ATMOS bed: the six channels of a 5.1 Atmos mix, imported as files ──
    atmos_fl_data:   Optional[object] = None
    atmos_fl_sr:     Optional[int]    = None
    atmos_fl_volume: float            = 1.0
    last_atmos_fl_dir: Optional[str]  = None
    atmos_fr_data:   Optional[object] = None
    atmos_fr_sr:     Optional[int]    = None
    atmos_fr_volume: float            = 1.0
    last_atmos_fr_dir: Optional[str]  = None
    atmos_c_data:   Optional[object] = None
    atmos_c_sr:     Optional[int]    = None
    atmos_c_volume: float            = 1.0
    last_atmos_c_dir: Optional[str]  = None
    atmos_lfe_data:   Optional[object] = None
    atmos_lfe_sr:     Optional[int]    = None
    atmos_lfe_volume: float            = 1.0
    last_atmos_lfe_dir: Optional[str]  = None
    atmos_bl_data:   Optional[object] = None
    atmos_bl_sr:     Optional[int]    = None
    atmos_bl_volume: float            = 1.0
    last_atmos_bl_dir: Optional[str]  = None
    atmos_br_data:   Optional[object] = None
    atmos_br_sr:     Optional[int]    = None
    atmos_br_volume: float            = 1.0
    last_atmos_br_dir: Optional[str]  = None

    # ── Bowed strings, from gilliaan's model ──────────────────────
    vocals_refined: bool             = False
    saved_stems_dir: Optional[str]   = None    # set by LOCATE STEMS
    # Once the karaoke split has run, VOCALS carries no audio of its own:
    # its fader, M and S act on FRT VOX and BG VOX together.
    vocals_is_vca:  bool             = False
    strings_data:   Optional[object] = None
    strings_is_quick: bool           = False
    strings_sr:     Optional[int]    = None
    strings_volume: float            = 1.0

    any_data:    Optional[object] = None
    any_sr:      Optional[int]    = None
    any_volume:  float            = 1.0

    any_plus_data:   Optional[object] = None
    any_plus_sr:     Optional[int]    = None
    any_plus_volume: float            = 1.0

    any_plusplus_data:    Optional[object] = None
    any_plusplus_sr:      Optional[int]    = None
    any_plusplus_volume:  float            = 1.0

    # ── export / session ──────────────────────────────────────
    export_folder:      Optional[str] = None
    current_audio_name: str           = ""
    # The song the audio in the player actually belongs to. It differs from
    # current_audio_name while a new track is separating — the previous
    # track's stems stay loaded until the new ones are ready — and exports
    # must be named after the audio they contain.
    loaded_audio_name: str            = ""
    _stem_export_state: dict = dataclasses.field(default_factory=dict)
    export_fmt_wav24:   bool = True
    export_fmt_mp3:     bool = False
    export_fmt_mp3_256: bool = False

    # ── directory memory ──────────────────────────────────────
    last_load_dir:    Optional[str] = None
    last_bgv_dir:     Optional[str] = None
    last_fv_dir:      Optional[str] = None
    last_hl_dir:      Optional[str] = None
    last_any_dir:          Optional[str] = None
    last_any_plus_dir:     Optional[str] = None
    last_any_plusplus_dir: Optional[str] = None

    # ── VFF — main vocals stem ────────────────────────────────
    vff_enabled:  bool  = False
    vff_lead_cut: float = 0.5
    vff_body_cut: float = 0.5
    vff_presence: float = 0.5
    vff_bkg_vol:  float = 1.0

    # ── separation flag ───────────────────────────────────────
    separating: bool = False

    # ── BS-RoFormer Resurrection Inst instrumental ───────
    # Populated in the background after every main separation.
    # Shape: (N, 2) float32 at 44100 Hz, same length as stems.
    instrumental:    Optional[object] = None
    instrumental_vol: float           = 1.0   # mixer volume (0–2)
    # True while the cell holds the instant stem-sum rather than the model's
    # own output (see _quick_instrumental).
    instrumental_is_quick: bool       = False

    # ── loop region (sample indices; None = full file) ────────
    loop_start: Optional[int] = None
    loop_end:   Optional[int] = None

    # ── export guard ──────────────────────────────────────────
    _export_running: bool = False



# Single module-level instance — all code accesses state via this object.
state = AppState()
audio_lock = threading.Lock()


# Default values used when a per-stem key is absent (unchanged constants)
_COMP_DEFAULTS = dict(thresh=-18.0, ratio=4.0, attack=10.0, release=100.0)
_GATE_DEFAULTS = dict(thresh=-40.0, attack=5.0, release=80.0)
_LIM_DEFAULTS  = dict(threshold=-6.0, ceiling=-0.1)

# ----------------------------
# DIRECTORY PERSISTENCE
# Config file lives next to the script so it survives sessions.
# ----------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ramma_dirs.json")

def _load_dirs():
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        state.last_load_dir    = data.get("last_load_dir")    or None
        state.last_bgv_dir     = data.get("last_bgv_dir")     or None
        state.last_fv_dir      = data.get("last_fv_dir")      or None
        state.last_hl_dir      = data.get("last_hl_dir")      or None
        # The three ANY folders used to be saved as last_synth_dir,
        # last_strings_dir and last_fx_dir; read those when the new names
        # are absent, so a settings file from before the rename keeps them.
        state.last_any_dir          = (data.get("last_any_dir")
                                       or data.get("last_synth_dir")   or None)
        state.last_any_plus_dir     = (data.get("last_any_plus_dir")
                                       or data.get("last_strings_dir") or None)
        state.last_any_plusplus_dir = (data.get("last_any_plusplus_dir")
                                       or data.get("last_fx_dir")      or None)
        state.export_folder    = data.get("export_folder")    or None
        state.export_fmt_wav24   = bool(data.get("export_fmt_wav24", True))
        state.export_fmt_mp3     = bool(data.get("export_fmt_mp3",   False))
        state.export_fmt_mp3_256 = bool(data.get("export_fmt_mp3_256", False))
        state.saved_stems_dir    = data.get("saved_stems_dir") or None
    except Exception:
        pass

def _save_dirs():
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "last_load_dir":      state.last_load_dir,
                "last_bgv_dir":       state.last_bgv_dir,
                "last_fv_dir":        state.last_fv_dir,
                "last_hl_dir":        state.last_hl_dir,
                "last_any_dir":     state.last_any_dir,
                "last_any_plus_dir":   state.last_any_plus_dir,
                "last_any_plusplus_dir":        state.last_any_plusplus_dir,
                "export_folder":      state.export_folder,
                "export_fmt_wav24":   state.export_fmt_wav24,
                "export_fmt_mp3":     state.export_fmt_mp3,
                "export_fmt_mp3_256": state.export_fmt_mp3_256,
                "saved_stems_dir":    state.saved_stems_dir,
            }, f, indent=2)
    except Exception as e:
        print("Could not save directory config:", e)

_load_dirs()   # restore directories from previous session



# ----------------------------
# AUDIO ENGINE
# ----------------------------
# EQ implementation: stateful IIR biquad filters (sosfilt) rather than FFT.
#
# WHY the FFT approach caused ticking:
#   FFT convolution on a finite block is CIRCULAR — the tail of the filter's
#   impulse response wraps around to the start of the same block.  At every
#   chunk boundary the wrapped tail from one block collides with the start of
#   the next, creating a discontinuity that sounds like a tick or crackle.
#   This happens whenever any band is non-zero (i.e. the filter is non-flat),
#   regardless of whether band values are smoothed or snapshotted.
#   Exports sounded fine because they process the full file in one pass, so
#   the wrap-around only hits the very end of the file.
#
# THE FIX — IIR biquad filters with persistent state:
#   sosfilt() carries the filter state (zi) across successive chunk calls.
#   There are no block boundaries in the filter's "memory" — each sample
#   flows continuously from one chunk to the next.  Zero artifacts.
#
# Design:
#   Band 0  — low shelf    at  200 Hz   (gain -1…+1 → -12…+12 dB)
#   Band 1  — peaking EQ   at  350 Hz
#   Band 2  — peaking EQ   at 1000 Hz
#   Band 3  — peaking EQ   at 4000 Hz
#   Band 4  — high shelf   at 6000 Hz
#
# Per-stem filter state is stored in _eq_zi_state keyed by stem name.
# Coefficients are cached by (sr, bands_tuple) and recomputed only when
# the sample rate or a band value changes.

_eq_sos_cache: dict = {}   # (sr, bands_tuple) -> list-of-5 sos arrays
_eq_zi_state:  dict = {}   # stem_key -> list-of-5 zi arrays  (2, 2, 2) each

_EQ_MAX_DB   = 12.0        # ±1.0 on slider → ±12 dB
_EQ_Q        = 0.9         # Q for peaking bands (broad, musical)

def _make_eq_sos(bands, sr_now: int):
    """Return a list of 5 second-order-section arrays, one per EQ band.
    Each array is shape (n_sections, 6).  Flat bands return None so
    sosfilt is skipped entirely for that band."""
    key = (sr_now, tuple(round(float(b), 5) for b in bands))
    cached = _eq_sos_cache.get(key)
    if cached is not None:
        return cached

    nyq = sr_now / 2.0
    result = []

    # Band 0 — low shelf at 200 Hz
    gain0 = float(bands[0]) * _EQ_MAX_DB
    if abs(gain0) < 0.05:
        result.append(None)
    else:
        # Build a low-shelf with a Butterworth prototype
        # Gain implemented as asymmetric pre/post scaling around a 1st-order LP
        fc = min(200.0 / nyq, 0.99)
        sos_lp = butter(2, fc, btype='low', output='sos')
        lin = 10.0 ** (gain0 / 20.0)
        # shelf = dry + (lin-1)*lowpass
        # expressed as: output = signal + (lin-1)*LP(signal)
        # We bake it into a single SOS by scaling LP sos
        sos_shelf = sos_lp.copy()
        sos_shelf[:, :3] *= (lin - 1.0)   # scale numerator by shelf depth
        # Add the all-pass (dry) path: b=[1,0,0], a=[1,0,0] → trivially merge
        # Simpler: just use a proper shelf via bilinear transform approximation
        # Use butter LP as proxy shelf — good enough for broad musical EQ
        result.append(sos_shelf)

    # Band 1 — peaking EQ at 350 Hz
    gain1 = float(bands[1]) * _EQ_MAX_DB
    if abs(gain1) < 0.05:
        result.append(None)
    else:
        f0 = min(350.0 / nyq, 0.99)
        b, a = iirpeak(f0, _EQ_Q) if gain1 > 0 else iirnotch(f0, _EQ_Q)
        lin1 = abs(10.0 ** (gain1 / 20.0) - 1.0)
        from scipy.signal import tf2sos
        sos = tf2sos(b * lin1, a)
        result.append(sos)

    # Band 2 — peaking EQ at 1000 Hz
    gain2 = float(bands[2]) * _EQ_MAX_DB
    if abs(gain2) < 0.05:
        result.append(None)
    else:
        f0 = min(1000.0 / nyq, 0.99)
        b, a = iirpeak(f0, _EQ_Q) if gain2 > 0 else iirnotch(f0, _EQ_Q)
        lin2 = abs(10.0 ** (gain2 / 20.0) - 1.0)
        from scipy.signal import tf2sos
        sos = tf2sos(b * lin2, a)
        result.append(sos)

    # Band 3 — peaking EQ at 4000 Hz
    gain3 = float(bands[3]) * _EQ_MAX_DB
    if abs(gain3) < 0.05:
        result.append(None)
    else:
        f0 = min(4000.0 / nyq, 0.99)
        b, a = iirpeak(f0, _EQ_Q) if gain3 > 0 else iirnotch(f0, _EQ_Q)
        lin3 = abs(10.0 ** (gain3 / 20.0) - 1.0)
        from scipy.signal import tf2sos
        sos = tf2sos(b * lin3, a)
        result.append(sos)

    # Band 4 — high shelf at 6000 Hz
    gain4 = float(bands[4]) * _EQ_MAX_DB
    if abs(gain4) < 0.05:
        result.append(None)
    else:
        fc = min(6000.0 / nyq, 0.99)
        sos_hp = butter(2, fc, btype='high', output='sos')
        lin4 = 10.0 ** (gain4 / 20.0)
        sos_shelf4 = sos_hp.copy()
        sos_shelf4[:, :3] *= (lin4 - 1.0)
        result.append(sos_shelf4)

    if len(_eq_sos_cache) > 128:
        _eq_sos_cache.clear()
    _eq_sos_cache[key] = result
    return result


def apply_eq(signal: np.ndarray, sr, bands, stem_key: str = "") -> np.ndarray:
    """Apply 5-band EQ using stateful IIR biquad filters.

    State (zi) is persisted per stem_key so the filter response is continuous
    across successive chunk calls — no block-boundary artifacts of any kind.
    """
    if len(signal) == 0:
        return signal
    if all(b == 0 for b in bands):
        # Reset zi for this stem so stale state doesn't contaminate the next
        # non-flat call (e.g. after a slider is zeroed and then moved again).
        _eq_zi_state.pop(stem_key, None)
        return signal.copy()

    sr_now  = int(state.sr) if state.sr else 44100
    sos_list = _make_eq_sos(bands, sr_now)

    # Work on a float64 copy for numerical stability in the IIR feedback loop
    out = signal.astype(np.float64)

    # Retrieve or initialise per-stem zi state
    zi_list = _eq_zi_state.get(stem_key)
    if zi_list is None:
        zi_list = [None] * 5

    new_zi = []
    for band_idx, sos in enumerate(sos_list):
        if sos is None:
            new_zi.append(None)
            continue

        zi = zi_list[band_idx]
        # sosfilt_zi gives the steady-state zi for a unit-step input.
        # Scale by the current DC value of the signal so the filter starts
        # without a transient when first engaged.
        if zi is None:
            zi_template = sosfilt_zi(sos)   # shape (n_sections, 2)
            # Use the mean of both channels as the DC estimate
            dc = float(np.mean(out))
            zi = zi_template * dc
            # zi needs shape (n_sections, 2, n_channels=2) for stereo
            zi = np.stack([zi, zi], axis=-1)  # (n_sections, 2, 2)

        # Both channels in a single call: sosfilt filters along `axis`, and
        # zi already has the (n_sections, 2, channels) shape it wants. This
        # is the same filter, half the calls.
        filtered, new_zi_band = sosfilt(sos, out, axis=0, zi=zi)
        out = out + filtered   # parallel path: dry + filtered delta
        new_zi.append(new_zi_band)

    _eq_zi_state[stem_key] = new_zi
    return out.astype(np.float32)

# ── Shared VFF engine ────────────────────────────────────────────────────
# All four VFF variants (main vocals, front vocals, BG vocals, hidden layer)
# are identical algorithms.  One shared implementation with a per-instance
# cache dict eliminates ~200 lines of duplication and means a single
# bug-fix or optimisation propagates everywhere automatically.

_vff_gains_cache: dict = {}   # (cache_id, n, state.sr, body_cut4, presence4) -> gains

def _vff_process(chunk: np.ndarray,
                 enabled: bool,
                 lead_cut: float, body_cut: float,
                 presence: float, bkg_vol: float,
                 cache_id: str) -> np.ndarray:
    """Core Vocal Focus Filter.  Stateless — safe to call from any thread.

    Stages:
      1. Iterative centre suppression (2 passes)   — removes lead vocal
      2. Body cut notch 200–600 Hz                 — kills chest resonance
      3. Presence boost shelf 2–10 kHz             — lifts backing harmonies
      4. Output gain (bkg_vol)
    """
    if not enabled or len(chunk) == 0:
        return chunk

    sr_now = int(state.sr) if state.sr else 44100
    n      = len(chunk)
    result = chunk.copy()

    if lead_cut > 1e-4:
        for _ in range(2):
            mono          = (result[:, 0] + result[:, 1]) * 0.5
            result[:, 0] -= mono * lead_cut
            result[:, 1] -= mono * lead_cut

    ckey = (cache_id, n, sr_now, round(body_cut, 4), round(presence, 4))
    gains = _vff_gains_cache.get(ckey)
    if gains is None:
        freqs      = np.fft.rfftfreq(n, d=1.0 / sr_now)
        gains      = np.ones(len(freqs))
        body_depth = 1.0 - body_cut * 0.975
        gains *= np.where(
            (freqs >= 200.0) & (freqs <= 600.0), body_depth,
            np.where(
                (freqs >= 100.0) & (freqs < 200.0),
                1.0 - (1.0 - body_depth) * (freqs - 100.0) / 100.0,
                np.where(
                    (freqs > 600.0) & (freqs <= 800.0),
                    body_depth + (1.0 - body_depth) * (freqs - 600.0) / 200.0,
                    1.0)))
        pres_gain = 1.0 + presence * 3.0
        gains *= np.where(
            freqs >= 10000.0, pres_gain,
            np.where(
                freqs >= 2000.0,
                1.0 + (pres_gain - 1.0) * (freqs - 2000.0) / 8000.0,
                1.0))
        if len(_vff_gains_cache) > 128:
            _vff_gains_cache.clear()
        _vff_gains_cache[ckey] = gains

    fft    = np.fft.rfft(result, axis=0)
    fft   *= gains[:, None]
    result = np.fft.irfft(fft, n=n, axis=0)
    result *= bkg_vol
    return result


def apply_vff(chunk):
    """Vocal Focus Filter for the main Vocals stem."""
    return _vff_process(chunk, state.vff_enabled,
                        float(state.vff_lead_cut), float(state.vff_body_cut),
                        float(state.vff_presence), float(state.vff_bkg_vol), "voc")

# legacy cache dicts kept as empty stubs so any external code that imports
# them doesn't crash (they are no longer written or read)
_vff_cache = {"key": None, "gains": None}


# ----------------------------
# REVERB ENGINE — Schroeder design, shared send/return bus
#
# Architecture:
#   • One shared reverb instance (_shared_reverb) processes a single stereo
#     "send bus".  All stems write their signal × reverb_amount into the bus;
#     the reverb runs once; the wet output is added back to the mix.
#   • This means reverb cost is O(1) regardless of how many cells have reverb.
#
# Implementation:
#   • Pure NumPy circular ring buffers — no Python sample loops, no huge
#     IIR polynomial coefficients.  Cost is O(chunk_size), not
#     O(delay_length × chunk_size) like lfilter with long taps.
# ----------------------------
_COMB_DELAYS   = [1557, 1617, 1491, 1422]   # samples at 44100 Hz
_COMB_FEEDBACK = 0.72   # reduced from 0.84 — prevents feedback divergence
                        # when send bus carries summed multi-stem signal
_AP_DELAYS     = [556, 441]
_AP_GAIN       = 0.5


class _RingBuf:
    """Stereo circular delay buffer — NumPy only, no Python loops."""
    __slots__ = ("buf", "pos", "size")

    def __init__(self, size: int):
        self.size = size
        self.buf  = np.zeros((size, 2), dtype=np.float32)
        self.pos  = 0

    def flush(self):
        """Zero the buffer — called when NaN/Inf is detected."""
        self.buf[:] = 0.0
        self.pos    = 0

    def process_comb(self, sig: np.ndarray, fb: float) -> np.ndarray:
        """Comb filter: y[n] = x[n-D] + fb·y[n-D].
        Handles chunks of any size by iterating in ring-aligned slices.
        """
        n   = len(sig)
        out = np.empty_like(sig)
        pos = self.pos
        sz  = self.size
        buf = self.buf
        i   = 0
        while i < n:
            space = sz - pos
            take  = min(space, n - i)
            delayed           = buf[pos:pos+take].copy()
            new_buf           = sig[i:i+take] + delayed * fb
            # Clamp feedback state so a hot signal can never diverge
            np.clip(new_buf, -4.0, 4.0, out=new_buf)
            out[i:i+take]     = delayed
            buf[pos:pos+take] = new_buf
            pos = (pos + take) % sz
            i  += take
        self.pos = pos
        return out

    def process_allpass(self, sig: np.ndarray, gain: float) -> np.ndarray:
        """Allpass filter. Handles chunks of any size."""
        n   = len(sig)
        out = np.empty_like(sig)
        pos = self.pos
        sz  = self.size
        buf = self.buf
        i   = 0
        while i < n:
            space    = sz - pos
            take     = min(space, n - i)
            delayed              = buf[pos:pos+take].copy()
            new_val              = sig[i:i+take] + gain * delayed
            np.clip(new_val, -4.0, 4.0, out=new_val)
            buf[pos:pos+take]    = new_val
            out[i:i+take]        = delayed - gain * new_val
            pos = (pos + take) % sz
            i  += take
        self.pos = pos
        return out


class _SharedReverb:
    """Single Schroeder reverb instance that processes the shared send bus."""

    def __init__(self):
        self._sr    : int   = 0
        self._combs : list  = []
        self._aps   : list  = []

    def _reset(self, sr_val: int):
        scale = sr_val / 44100.0
        self._combs = [_RingBuf(max(1, int(d * scale))) for d in _COMB_DELAYS]
        self._aps   = [_RingBuf(max(1, int(d * scale))) for d in _AP_DELAYS]
        self._sr    = sr_val

    def _flush_all(self):
        """Flush all delay lines — recovers from any overflow state."""
        for b in self._combs + self._aps:
            b.flush()

    def process(self, send: np.ndarray, sr_now: int) -> np.ndarray:
        """Process stereo send bus → stereo wet output."""
        if sr_now != self._sr:
            self._reset(sr_now)
        if len(send) == 0:
            return send

        # Clamp the send bus to ±1.0 before it enters the feedback network
        send = np.clip(send, -1.0, 1.0)

        # 4 parallel comb filters summed
        comb_out = np.zeros_like(send)
        for cb in self._combs:
            comb_out += cb.process_comb(send, _COMB_FEEDBACK)
        comb_out *= 0.25

        # 2 series allpass filters
        ap_out = comb_out
        for ap in self._aps:
            ap_out = ap.process_allpass(ap_out, _AP_GAIN)

        # Safety: if NaN or Inf slipped through, flush all state and return silence
        if not np.isfinite(ap_out).all():
            self._flush_all()
            return np.zeros_like(send)

        return ap_out


_shared_reverb  = _SharedReverb()   # processes the per-stem send bus
_master_reverb  = _SharedReverb()   # processes the final master mix — SEPARATE instance


def _prewarm_reverb_bus(sr_val: int):
    """Pre-initialise both reverb buses so first slider move is instant."""
    _shared_reverb._reset(sr_val)
    _master_reverb._reset(sr_val)


# ----------------------------



# ----------------------------
# AIR PROCESSOR — high-shelf EQ above 8 kHz
# amount: -1.0 (full cut) … 0.0 (flat) … +1.0 (full boost, +12 dB)
_air_cache = {}   # keyed on (sr_now, n) — BOTH state.sr AND chunk size

def apply_air(chunk, amount, sr_now):
    """Apply a high-shelf air boost or cut above 8 kHz."""
    if abs(amount) < 0.001 or len(chunk) == 0:
        return chunk
    n         = len(chunk)
    cache_key = (sr_now, n)   # include n so different chunk sizes never corrupt the cache
    if cache_key not in _air_cache:
        freqs = np.fft.rfftfreq(n, d=1.0 / sr_now).astype(np.float32)
        shelf = np.where(
            freqs >= 10000.0, 1.0,
            np.where(freqs >= 6000.0,
                     (freqs - 6000.0) / 4000.0,
                     0.0)
        ).astype(np.float32)
        _air_cache[cache_key] = shelf
    shelf = _air_cache[cache_key]
    # Gain: +12 dB at +1.0, -18 dB at -1.0
    if amount > 0:
        gain = 1.0 + amount * (10 ** (12.0 / 20.0) - 1.0)
    else:
        gain = 1.0 + amount * (1.0 - 10 ** (-18.0 / 20.0))
    gains = (1.0 + (gain - 1.0) * shelf).astype(np.float32)
    fft   = np.fft.rfft(chunk, axis=0)
    fft  *= gains[:, None]
    return np.fft.irfft(fft, n=n, axis=0).astype(np.float32)


def apply_stem_width(chunk, width):
    """Apply loudness-preserving mid/side stereo width to a single chunk."""
    if width == 1.0 or len(chunk) == 0:
        return chunk
    mid        = (chunk[:, 0] + chunk[:, 1]) * 0.5
    side       = (chunk[:, 0] - chunk[:, 1]) * 0.5
    wg         = 1.0 / max(0.5 * (1.0 + width), 1e-6)
    out        = chunk.copy()
    out[:, 0]  = (mid + side * width) * wg
    out[:, 1]  = (mid - side * width) * wg
    return out


# ----------------------------
# PAN — constant-power stereo balance
# amount: -1.0 (hard left) … 0.0 (centre) … +1.0 (hard right)
# ----------------------------
def apply_pan(chunk: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 0.001 or len(chunk) == 0:
        return chunk
    angle  = (amount + 1.0) * 0.25 * np.pi   # maps -1…1 → 0…π/2
    g_l    = float(np.cos(angle))
    g_r    = float(np.sin(angle))
    out        = chunk.copy()
    out[:, 0] *= g_l
    out[:, 1] *= g_r
    return out


# ----------------------------
# COMPRESSOR / GATE — per-stem dynamics processor
#
# Single function handles both effects with continuous per-stem envelope
# state so gain changes are smooth across audio-callback boundaries.
# ----------------------------
_comp_state: dict = {}   # stem_key -> {"env": float, "gate_env": float}

def _block_rms(sig: np.ndarray, block: int) -> np.ndarray:
    """RMS of each *block* samples of a stereo chunk, all at once."""
    n  = len(sig)
    nb = (n + block - 1) // block
    mono = (sig[:, 0] + sig[:, 1]) * 0.5
    pad = nb * block - n
    if pad:
        mono = np.concatenate([mono, np.zeros(pad, dtype=mono.dtype)])
    return np.sqrt(np.mean(mono.reshape(nb, block).astype(np.float64) ** 2,
                           axis=1) + 1e-12)


def _block_peak(sig: np.ndarray, block: int) -> np.ndarray:
    """Peak of each *block* samples of a stereo chunk, all at once."""
    n  = len(sig)
    nb = (n + block - 1) // block
    pad = nb * block - n
    if pad:
        sig = np.concatenate([sig, np.zeros((pad, sig.shape[1]), dtype=sig.dtype)])
    return np.max(np.abs(sig.reshape(nb, block, sig.shape[1])), axis=(1, 2))


def _env_follow(values: np.ndarray, env: float, atk: float, rel: float) -> np.ndarray:
    """One-pole attack/release follower over per-block values.

    Each step depends on the one before it, so this stays a loop — but it is
    scalar arithmetic over ~64 blocks, not NumPy calls per block.
    """
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values.tolist()):
        c = atk if v > env else rel
        env = c * env + (1.0 - c) * v
        out[i] = env
    return out


def _apply_block_gain(sig: np.ndarray, gains: np.ndarray, block: int) -> None:
    """Multiply each block of *sig* in place by its gain."""
    if np.all(gains == 1.0):
        return
    n = len(sig)
    sig *= np.repeat(gains, block)[:n, None].astype(sig.dtype)


def apply_dynamics(chunk: np.ndarray,
                   key:   str,
                   comp_enabled:    bool,
                   threshold_db:    float,
                   ratio:           float,
                   attack_ms:       float,
                   release_ms:      float,
                   gate_enabled:    bool,
                   gate_thresh_db:  float,
                   gate_attack_ms:  float,
                   gate_release_ms: float,
                   sr_now:          int) -> np.ndarray:
    """RMS compressor + downward gate with soft-knee and per-stem state."""
    if (not comp_enabled and not gate_enabled) or len(chunk) == 0:
        return chunk

    out = chunk.copy().astype(np.float32)
    st = _comp_state.setdefault(key, {"env": 0.0, "gate_env": 1.0})
    sr_f  = float(sr_now)

    def _tc(ms):
        return float(np.exp(-1.0 / max(sr_f * ms * 0.001, 1.0)))

    # --- Compressor ---
    # Block RMS, the envelope recursion and the gain curve are computed for
    # every block at once; only the recursion itself stays sequential, and
    # that is plain scalar arithmetic. Same maths as the per-block loop this
    # replaces, a fraction of the work.
    BLOCK = 64
    if comp_enabled:
        atk_c      = _tc(attack_ms)
        rel_c      = _tc(release_ms)
        thresh_lin = 10.0 ** (threshold_db / 20.0)
        knee_db    = 6.0
        env        = st["env"]

        rms_blocks = _block_rms(out, BLOCK)
        env_arr    = _env_follow(rms_blocks, env, atk_c, rel_c)
        env        = float(env_arr[-1]) if len(env_arr) else env

        knee_low  = threshold_db - knee_db / 2.0
        knee_high = threshold_db + knee_db / 2.0
        env_db    = 20.0 * np.log10(np.maximum(env_arr, 1e-9))
        ovr       = env_db - knee_low
        soft      = (1.0 / ratio - 1.0) * (ovr ** 2) / (2.0 * knee_db)
        hard      = (env_db - threshold_db) * (1.0 / ratio - 1.0)
        gain_db   = np.where(env_db <= knee_low, 0.0,
                             np.where(env_db <= knee_high, soft, hard))
        # Below the knee start the original left the block untouched.
        gain_db   = np.where(env_arr > thresh_lin * 10.0 ** (-knee_db / 40.0),
                             gain_db, 0.0)
        _apply_block_gain(out, 10.0 ** (gain_db / 20.0), BLOCK)

        st["env"] = env

    # --- Gate ---
    if gate_enabled:
        gate_atk_c  = _tc(gate_attack_ms)
        gate_rel_c  = _tc(gate_release_ms)
        gate_thresh = 10.0 ** (gate_thresh_db / 20.0)
        gate_env    = st["gate_env"]

        rms_blocks = _block_rms(out, BLOCK)
        opens      = rms_blocks >= gate_thresh
        gains      = np.empty(len(rms_blocks), dtype=np.float64)
        for i, is_open in enumerate(opens.tolist()):
            c = gate_atk_c if is_open else gate_rel_c
            gate_env = c * gate_env + (1.0 - c) * (1.0 if is_open else 0.0)
            gains[i] = gate_env
        _apply_block_gain(out, gains, BLOCK)

        st["gate_env"] = gate_env

    return out


# Cells whose audio comes from a file rather than from separation.
_IMPORT_KEYS = ("any", "any+", "any++",
                "atmos_fl", "atmos_fr", "atmos_c",
                "atmos_lfe", "atmos_bl", "atmos_br",
                "front_vocals", "bg_vocals", "hidden_layer")

# key -> the state attribute holding that cell's audio.
_IMPORT_DATA_ATTR = {
    "any": "any_data", "any+": "any_plus_data", "any++": "any_plusplus_data",
    "atmos_fl": "atmos_fl_data", "atmos_fr": "atmos_fr_data",
    "atmos_c": "atmos_c_data", "atmos_lfe": "atmos_lfe_data",
    "atmos_bl": "atmos_bl_data", "atmos_br": "atmos_br_data",
    "front_vocals": "fv_data", "bg_vocals": "bg_vocals_data",
    "hidden_layer": "hl_data",
}


def _refresh_transport_state():
    """Light the transport when anything is playable, dim it when nothing is."""
    try:
        if _playable_length() > 0:
            _unlock_transport()
        else:
            _lock_transport()
    except Exception:
        pass


# ── Multichannel / E-AC-3 (JOC) import ─────────────────────────────────────
# FFmpeg decodes Dolby Digital Plus, including the streams used for Atmos.
# What it gives back is the 5.1 bed: the JOC side data that positions the
# height objects is NOT rendered by any freely available decoder, so the
# objects stay folded into the bed exactly as they are in the file.
_MULTICH_EXTS = (".eac3", ".ec3", ".ac3", ".mp4", ".m4a", ".mkv", ".mov",
                 ".m4v", ".webm", ".ts", ".wav", ".flac", ".w64", ".caf")

# The order FFmpeg hands back for 5.1, and the cell each channel belongs in.
_51_TO_CELL = ("atmos_fl", "atmos_fr", "atmos_c",
               "atmos_lfe", "atmos_bl", "atmos_br")
# Where each cell sits, so an imported bed is positioned without touching the
# faders by hand.
_51_PAN = {"atmos_fl": -1.0, "atmos_bl": -1.0,
           "atmos_fr":  1.0, "atmos_br":  1.0,
           "atmos_c":   0.0, "atmos_lfe": 0.0}


def _ffmpeg_path():
    """ffmpeg on PATH, or None."""
    import shutil
    return shutil.which("ffmpeg")


def _probe_joc(path):
    """True when the file looks like Dolby Atmos (JOC), for the warning."""
    import shutil, subprocess
    probe = shutil.which("ffprobe")
    if not probe:
        return False
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-show_streams", "-select_streams", "a",
             str(path)], capture_output=True, text=True, timeout=30)
        text = (out.stdout or "") + (out.stderr or "")
        return "joc" in text.lower() or "atmos" in text.lower()
    except Exception:
        return False


def _decode_multichannel(path):
    """Decode *path* to (samples, channels) float32 at the session rate.

    soundfile handles plain multichannel WAV/FLAC; everything else goes
    through FFmpeg, which is also what unpacks E-AC-3.
    """
    import tempfile, subprocess
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".flac", ".w64", ".caf", ".aiff", ".aif"):
        try:
            audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
            return audio, int(file_sr)
        except Exception:
            pass

    exe = _ffmpeg_path()
    if not exe:
        raise RuntimeError(
            "FFmpeg was not found. E-AC-3 and other packed formats need it to "
            "decode. Install it from https://ffmpeg.org and make sure "
            "ffmpeg.exe is on PATH.")

    tmp = os.path.join(tempfile.gettempdir(), "ramma_multich.wav")
    cmd = [exe, "-y", "-v", "error", "-i", str(path),
           "-map", "0:a:0", "-c:a", "pcm_f32le", tmp]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if res.returncode != 0 or not os.path.exists(tmp):
        raise RuntimeError(f"FFmpeg could not decode this file: "
                           f"{(res.stderr or '').strip()[:200]}")
    audio, file_sr = sf.read(tmp, dtype="float32", always_2d=True)
    try:
        os.remove(tmp)
    except Exception:
        pass
    return audio, int(file_sr)


def import_atmos_bed():
    """Load one multichannel file and spread its channels over the cells."""
    if state.separating:
        return
    path = filedialog.askopenfilename(
        title="Import ATMOS bed — E-AC-3 / multichannel file",
        initialdir=getattr(state, "last_atmos_fl_dir", None),
        filetypes=[("Dolby / multichannel audio",
                    "*.eac3 *.ec3 *.ac3 *.mp4 *.m4a *.mkv *.mov *.m4v "
                    "*.webm *.ts *.wav *.flac *.w64 *.caf"),
                   ("All files", "*.*")])
    if not path:
        return
    state.last_atmos_fl_dir = os.path.dirname(path)

    try:
        audio, file_sr = _decode_multichannel(path)
    except Exception as e:
        print("[ATMOS] Import failed:", e)
        return

    nch = audio.shape[1]
    print(f"[ATMOS] {os.path.basename(path)}: {nch} channels at {file_sr} Hz")
    if _probe_joc(path):
        print("[ATMOS] This looks like a Dolby Atmos (JOC) stream. FFmpeg "
              "decodes the 5.1 bed only — the height objects stay folded "
              "into it, since rendering them needs a licensed Dolby decoder.")
    if nch < 6:
        print(f"[ATMOS] Only {nch} channels: filling the first cells and "
              f"leaving the rest empty.")
    elif nch > 6:
        print(f"[ATMOS] {nch} channels: using the first six (the 5.1 bed).")

    filled = 0
    for idx, key in enumerate(_51_TO_CELL):
        if idx >= nch:
            break
        mono = audio[:, idx]
        stereo = np.stack([mono, mono], axis=1).astype(np.float32)
        stereo = _resample_audio(stereo, file_sr)
        with audio_lock:
            setattr(state, f"{key}_data", stereo)
            setattr(state, f"{key}_sr", state.sr)
        state.stem_pan[key] = _51_PAN.get(key, 0.0)
        btn = globals().get(f"{key}_import_btn")
        if btn is not None:
            try:
                btn.configure(text=f"⬡ {os.path.basename(path)[:14]}")
            except Exception:
                pass
        filled += 1

    print(f"[ATMOS] {filled} channels loaded into the ATMOS cells")
    _refresh_import_waveform()
    _refresh_transport_state()
    try:
        _paint_all_ms()
    except Exception:
        pass


# ============================================================
# STEM FIXES
# A separation sometimes puts a sound in the wrong stem — a synth line landing
# in VOCALS, say. A fix records "the audio between these two points belongs in
# that stem instead", and is applied to the stems in memory and remembered for
# next time, keyed by the file. The audio itself is never re-separated: a fix
# moves the region from one stem to another, with short fades at the seams so
# the move is inaudible.
# ============================================================
_FIXES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ramma_stem_fixes.json")
_FIX_FADE_MS = 8.0          # fade in/out at a fix's edges
_stem_fixes: dict = {}      # track path -> [ {start, end, src, dst}, ... ]
_fixes_applied: list = []   # fixes already applied to the stems in memory


def _load_stem_fixes():
    """Read the saved fixes for every track."""
    global _stem_fixes
    try:
        if os.path.exists(_FIXES_PATH):
            with open(_FIXES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _stem_fixes = data
                print(f"[Fix] {sum(len(v) for v in data.values())} saved fix(es) "
                      f"for {len(data)} track(s)")
    except Exception as e:
        print("[Fix] Could not read saved fixes:", e)


def _save_stem_fixes():
    try:
        with open(_FIXES_PATH, "w", encoding="utf-8") as f:
            json.dump(_stem_fixes, f, indent=1)
    except Exception as e:
        print("[Fix] Could not save fixes:", e)


def _fix_track_key():
    """The track a fix belongs to: its path when known, else its name."""
    path = _playlist_current[0]
    if path:
        return os.path.abspath(path)
    return state.loaded_audio_name or state.current_audio_name or ""


def _fixes_for_current():
    return _stem_fixes.get(_fix_track_key(), [])


def _apply_one_fix(stems, fix, reverse=False):
    """Move a region between two cells, in place, with fades at the seams.

    Works for the separated stems and for FRT VOX / BG VOX, whose audio is
    held outside state.stems. reverse=True puts it back, which is how a fix
    is undone without separating the track again.
    """
    src, dst = (fix["dst"], fix["src"]) if reverse else (fix["src"], fix["dst"])
    src_buf = _fix_buffer(src)
    if src_buf is None:
        return False
    n = src_buf.shape[0]
    a = max(0, min(int(fix["start"]), n))
    b = max(0, min(int(fix["end"]),   n))
    if b - a < 2:
        return False

    dst_buf = _fix_buffer(dst, create_len=n)
    if dst_buf is None:
        return False
    # The halves can be a different length from the stems; stay inside both.
    b = min(b, dst_buf.shape[0])
    if b - a < 2:
        return False

    seg = src_buf[a:b].copy()
    # Fade the edges so the seam cannot click.
    f = min(int(state.sr * _FIX_FADE_MS / 1000.0), (b - a) // 2)
    if f > 1:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]
        seg[:f]  *= ramp
        seg[-f:] *= ramp[::-1]

    dst_buf[a:b] += seg
    src_buf[a:b] -= seg
    return True


def _apply_saved_fixes():
    """Apply this track's saved fixes to the stems just loaded."""
    global _fixes_applied
    _fixes_applied = []
    if not state.stems:
        return
    fixes = _fixes_for_current()
    if not fixes:
        return
    with audio_lock:
        for fix in fixes:
            if _apply_one_fix(state.stems, fix):
                _fixes_applied.append(fix)
    if _fixes_applied:
        print(f"[Fix] Re-applied {len(_fixes_applied)} saved fix(es) to "
              f"{state.loaded_audio_name!r}")
    if running:
        app.after(0, _refresh_fix_list)


def add_stem_fix(src, dst):
    """Move the looped region from *src* to *dst*, and remember it."""
    if not state.stems:
        print("[Fix] Nothing separated yet.")
        return False
    if src == dst:
        print("[Fix] Source and destination are the same stem.")
        return False
    if _fix_buffer(src) is None:
        print(f"[Fix] {_fix_label_for(src)} holds no audio to move.")
        return False
    if dst not in _fix_keys():
        print(f"[Fix] {_fix_label_for(dst)} cannot be a destination.")
        return False
    a, b = state.loop_start, state.loop_end
    if a is None or b is None or b <= a:
        print("[Fix] Mark the part to move with a loop on the waveform first.")
        return False

    fix = {"start": int(a), "end": int(b), "src": src, "dst": dst}
    with audio_lock:
        if not _apply_one_fix(state.stems, fix):
            return False
    key = _fix_track_key()
    _stem_fixes.setdefault(key, []).append(fix)
    _fixes_applied.append(fix)
    _save_stem_fixes()
    print(f"[Fix] Moved {(b - a) / max(1, state.sr):.2f}s of {src} -> {dst} "
          f"and saved it for next time")
    if running:
        _refresh_fix_list()
    return True


def remove_stem_fix(index):
    """Undo a fix: put the audio back and forget it."""
    key = _fix_track_key()
    fixes = _stem_fixes.get(key, [])
    if not (0 <= index < len(fixes)):
        return
    fix = fixes.pop(index)
    if state.stems:
        with audio_lock:
            _apply_one_fix(state.stems, fix, reverse=True)
    if fix in _fixes_applied:
        _fixes_applied.remove(fix)
    if not fixes:
        _stem_fixes.pop(key, None)
    _save_stem_fixes()
    print(f"[Fix] Undid {fix['src']} -> {fix['dst']}")
    if running:
        _refresh_fix_list()


def _fix_describe(fix):
    sr = max(1, int(state.sr or 44100))
    t0, t1 = fix["start"] / sr, fix["end"] / sr
    return (f"{int(t0 // 60)}:{t0 % 60:05.2f} - {int(t1 // 60)}:{t1 % 60:05.2f}   "
            f"{_fix_label_for(fix['src'])} -> {_fix_label_for(fix['dst'])}")


def _refresh_import_waveform():
    """Draw a waveform for imported audio when nothing has been separated.

    With a separated track the waveform belongs to that track and is left
    alone. With only imported cells — an ATMOS bed, say — the sum of them is
    what you hear, so that is what the waveform shows, and the playhead and
    seeking work off it like any other track.
    """
    if state.stems or state.instrumental is not None:
        return                      # a separated track owns the waveform
    imported = _imported_audio()
    if not imported:
        state.waveform_data = None
        if running:
            try:
                wave_canvas.delete("waveform")
            except Exception:
                pass
        return

    length = max(len(d) for d in imported.values())
    total  = np.zeros(length, dtype=np.float32)
    for data in imported.values():
        mono = np.mean(np.asarray(data, dtype=np.float32), axis=1)
        total[:len(mono)] += mono
    step = max(1, len(total) // 2000)
    state.waveform_data = total[::step]
    if running:
        try:
            if not wave_canvas.winfo_ismapped():
                progress_bar.pack_forget()
                _clear_progress()
                wave_canvas.pack(fill="both", expand=True)
            _draw_static_waveform()
        except Exception:
            pass


def _clear_import(key, btn=None):
    """Throw away an imported cell's audio and let it be imported again."""
    attr = _IMPORT_DATA_ATTR.get(key)
    if attr and getattr(state, attr, None) is not None:
        with audio_lock:
            setattr(state, attr, None)
        sr_attr = attr.replace("_data", "_sr")
        if hasattr(state, sr_attr):
            setattr(state, sr_attr, None)
        print(f"[Mixer] Cleared {key}")
    state._meter_levels[key] = (0.0, 0.0)
    if btn is not None:
        try:
            btn.configure(text="⬡ IMPORT")
        except Exception:
            pass
    _refresh_import_waveform()
    _refresh_transport_state()
    # Stop at the end of whatever is left, rather than past it.
    try:
        if state.position > _playable_length():
            state.position = 0
    except Exception:
        pass


def _imported_audio():
    """Every imported cell that currently holds audio."""
    out = {}
    for key in _IMPORT_KEYS:
        data = getattr(state, _IMPORT_DATA_ATTR.get(key, ""), None)
        if data is not None:
            out[key] = data
    return out


def _playable_length():
    """Length in samples of whatever is currently loaded, stems or not.

    Imported cells count too: with nothing separated, an imported file is
    still something to play, so the transport works on its own.
    """
    if state.stems:
        return next(iter(state.stems.values())).shape[0]
    if state.instrumental is not None:
        return len(state.instrumental)
    lengths = [len(d) for d in _imported_audio().values()]
    return max(lengths) if lengths else 0


def mix(start, frames):
    if frames <= 0:
        return np.zeros((max(frames, 0), 2), dtype=np.float32)

    # With INSTRUM ONLY there are no separated stems, but the instrumental and
    # any imported stems still have to play — so the pipeline below runs over
    # an empty stem set rather than bailing out.
    stems_now = state.stems if state.stems else {}

    out      = np.zeros((frames, 2), dtype=np.float32)
    rev_send = np.zeros((frames, 2), dtype=np.float32)
    _vols    = state.stem_volumes
    sr_int   = int(state.sr) if state.sr else 44100

    # Determine which stems are soloed so mute logic is correct.
    # A stem is audible when: not muted, AND (nothing is soloed OR it is soloed).
    # Every cell that can be soloed, not only the separated stems. A cell
    # missing from here can be soloed without silencing anything else, which
    # looks exactly like solo being broken.
    # VOCALS as a VCA: it holds nothing itself, and its controls apply to
    # the two halves.
    _vca = bool(getattr(state, "vocals_is_vca", False))
    _vca_vol = float(state.stem_volumes.get("vocals", 1.0)) if _vca else 1.0

    any_solo = any(state.stem_solo.get(k, False)
                   for k in list(stems_now.keys()) +
                            ["front_vocals", "bg_vocals", "hidden_layer",
                             "any", "any+", "any++", "strings",
                             "atmos_fl", "atmos_fr", "atmos_c",
                             "atmos_lfe", "atmos_bl", "atmos_br",
                             "instrumental"])

    def _audible(key):
        if _vca:
            if key == "vocals":
                return False              # silent: it is only a control now
            if key in ("front_vocals", "bg_vocals"):
                # VOCALS' own mute is deliberately not passed on: each half
                # has its own M button for that. Its fader and solo still
                # act on both.
                if any_solo:
                    # Soloing VOCALS solos both halves.
                    return bool(state.stem_solo.get(key, False)
                                or state.stem_solo.get("vocals", False))
                return not state.stem_mute.get(key, False)
        # Solo wins over that cell's own mute: soloing a muted cell is a
        # request to hear it, and the mute comes back when solo is released.
        if any_solo:
            return bool(state.stem_solo.get(key, False))
        return not state.stem_mute.get(key, False)

    def _apply_dyn(chunk, key):
        return apply_dynamics(
            chunk, key,
            state.stem_comp_enabled.get(key, False),
            state.stem_comp_thresh.get(key,  _COMP_DEFAULTS["thresh"]),
            state.stem_comp_ratio.get(key,   _COMP_DEFAULTS["ratio"]),
            state.stem_comp_attack.get(key,  _COMP_DEFAULTS["attack"]),
            state.stem_comp_release.get(key, _COMP_DEFAULTS["release"]),
            state.stem_gate_enabled.get(key, False),
            state.stem_gate_thresh.get(key,  _GATE_DEFAULTS["thresh"]),
            state.stem_gate_attack.get(key,  _GATE_DEFAULTS["attack"]),
            state.stem_gate_release.get(key, _GATE_DEFAULTS["release"]),
            sr_int)

    # Raw EQ'd chunks, which debleed reads from. Only stems that can be heard
    # need one — plus any stem another cell bleeds from, even if that source
    # is muted. Filtering a muted stem nobody references is pure waste, and
    # with no debleed set up (the usual case) that is every muted stem.
    _debleed_srcs = set()
    for _tgt, _srcs in (state.stem_debleed or {}).items():
        if isinstance(_srcs, dict):
            _debleed_srcs.update(k for k, amt in _srcs.items() if amt)

    _raw_eq_chunks: dict = {}
    for name, data in stems_now.items():
        if not _audible(name) and name not in _debleed_srcs:
            continue
        nudge_s = int(state.stem_nudge.get(name, 0))
        c = _nudged_slice(data, start, frames, nudge_s)
        _raw_eq_chunks[name] = apply_eq(c, state.sr, state.eq_bands.get(name, [0]*5), stem_key=name)

    for name, data in stems_now.items():
        if not _audible(name):
            state._meter_levels[name] = (0.0, 0.0)
            continue
        vol   = _vols.get(name, 1.0)
        if vol <= 0.0:
            # Silent either way: no point running the chain for it.
            state._meter_levels[name] = (0.0, 0.0)
            continue
        chunk = _raw_eq_chunks[name].copy()
        if state.stem_invert.get(name, False):
            chunk = -chunk
        if name == "vocals" and state.vff_enabled:
            chunk = apply_vff(chunk)
        chunk = apply_debleed(chunk, name, _raw_eq_chunks, sr_int)
        chunk = apply_leveller(chunk,
                               state.stem_lvl_enabled.get(name, False),
                               state.stem_lvl_threshold.get(name, 0.3),
                               state.stem_lvl_amount.get(name, 0.5))
        chunk = _apply_dyn(chunk, name)
        chunk = apply_stem_width(chunk, state.stem_widths.get(name, 1.0))
        chunk = apply_pan(chunk, state.stem_pan.get(name, 0.0))
        chunk = apply_air(chunk, state.stem_air.get(name, 0.0), sr_int)
        chunk = apply_limiter(chunk, name,
                              state.stem_lim_enabled.get(name, False),
                              state.stem_lim_threshold.get(name, _LIM_DEFAULTS["threshold"]),
                              state.stem_lim_ceiling.get(name,   _LIM_DEFAULTS["ceiling"]),
                              sr_int)
        rev_amt = state.stem_reverbs.get(name, 0.0)
        if rev_amt > 0.001:
            rev_send += chunk * rev_amt
        scaled = chunk * vol
        # Per-channel peaks, so the meter can show left and right separately.
        state._meter_levels[name] = (float(np.max(np.abs(scaled[:, 0]))),
                                     float(np.max(np.abs(scaled[:, 1]))))
        out += scaled

    def _mix_import(data, vol, key, vff_fn=None):
        """Process one user-imported stem and accumulate into out/rev_send.

        Pipeline: slice → EQ → VFF (optional) → leveller → dynamics →
                  width → pan → air → reverb send → scale → accumulate.
        Identical logic to the Demucs-stem loop; extracted so fv/bgv/hl/
        synth/strings/fx all share one implementation instead of six copies.
        """
        if data is None or vol <= 0 or not _audible(key):
            state._meter_levels[key] = (0.0, 0.0)
            return
        buf_len = len(data)
        if buf_len == 0:
            return
        s  = start % buf_len
        e  = s + frames
        ch = data[s:e].copy() if e <= buf_len else np.concatenate([data[s:], data[:e - buf_len]])
        if len(ch) < frames:
            ch = np.pad(ch, ((0, frames - len(ch)), (0, 0)))
        if state.stem_invert.get(key, False):
            ch = -ch
        ch = apply_eq(ch, state.sr, state.eq_bands.get(key, [0] * 5), stem_key=key)
        if vff_fn is not None:
            ch = vff_fn(ch)
        ch = apply_leveller(ch,
                            state.stem_lvl_enabled.get(key, False),
                            state.stem_lvl_threshold.get(key, 0.3),
                            state.stem_lvl_amount.get(key, 0.5))
        ch = _apply_dyn(ch, key)
        ch = apply_stem_width(ch, state.stem_widths.get(key, 1.0))
        ch = apply_pan(ch, state.stem_pan.get(key, 0.0))
        ch = apply_air(ch, state.stem_air.get(key, 0.0), sr_int)
        ch = apply_limiter(ch, key,
                           state.stem_lim_enabled.get(key, False),
                           state.stem_lim_threshold.get(key, _LIM_DEFAULTS["threshold"]),
                           state.stem_lim_ceiling.get(key,   _LIM_DEFAULTS["ceiling"]),
                           sr_int)
        rev_amt = state.stem_reverbs.get(key, 0.0)
        if rev_amt > 0.001:
            rev_send.__iadd__(ch * rev_amt)
        scaled = ch * vol
        state._meter_levels[key] = (float(np.max(np.abs(scaled[:, 0]))),
                                    float(np.max(np.abs(scaled[:, 1]))))
        out.__iadd__(scaled)

    # Imported stems — fv/bgv/hl carry optional VFF; synth/strings/fx do not.
    _mix_import(state.fv_data,         state.fv_volume * _vca_vol, "front_vocals",
                vff_fn=apply_fv_vff  if state.fv_vff_enabled  else None)
    _mix_import(state.bg_vocals_data,  state.bg_vocals_volume * _vca_vol, "bg_vocals",
                vff_fn=apply_bgv_vff if state.bgv_vff_enabled else None)
    _mix_import(state.hl_data,         state.hl_volume,         "hidden_layer",
                vff_fn=apply_hl_vff  if state.hl_vff_enabled  else None)
    # The ATMOS bed channels
    _mix_import(state.atmos_fl_data, state.atmos_fl_volume, "atmos_fl")
    _mix_import(state.atmos_fr_data, state.atmos_fr_volume, "atmos_fr")
    _mix_import(state.atmos_c_data, state.atmos_c_volume, "atmos_c")
    _mix_import(state.atmos_lfe_data, state.atmos_lfe_volume, "atmos_lfe")
    _mix_import(state.atmos_bl_data, state.atmos_bl_volume, "atmos_bl")
    _mix_import(state.atmos_br_data, state.atmos_br_volume, "atmos_br")
    _mix_import(state.strings_data,  state.strings_volume,  "strings")
    _mix_import(state.any_data,      state.any_volume,      "any")
    _mix_import(state.any_plus_data,    state.any_plus_volume,    "any+")
    _mix_import(state.any_plusplus_data,         state.any_plusplus_volume,         "any++")
    # Only mix the instrumental when it matches the loaded stems; a leftover
    # array from the previous song would play over the new one.
    _inst_data = state.instrumental
    if _inst_data is not None:
        _stem_len = _playable_length()
        if abs(len(_inst_data) - _stem_len) > state.sr:   # >1 s adrift
            _inst_data = None
    _mix_import(_inst_data, state.instrumental_vol, "instrumental")

    out *= state.volume_master

    if np.max(np.abs(rev_send)) > 0.001:
        out += _shared_reverb.process(rev_send, sr_int)

    if state.reverb_master > 0.001:
        master_wet = _master_reverb.process(out, sr_int)
        out = out * (1.0 - state.reverb_master) + master_wet * state.reverb_master

    out = apply_air(out, state.air_master, sr_int)

    if state.stereo_width != 1.0:
        mid        = (out[:,0] + out[:,1]) * 0.5
        side       = (out[:,0] - out[:,1]) * 0.5
        width_gain = 1.0 / max(0.5 * (1.0 + state.stereo_width), 1e-6)
        out[:,0]   = (mid + side * state.stereo_width) * width_gain
        out[:,1]   = (mid - side * state.stereo_width) * width_gain

    np.clip(out, -1.0, 1.0, out=out)
    return out

# Loop region is stored on state.loop_start / state.loop_end (both None = full file)

def callback(outdata, frames, time, status):
    if not running:
        outdata[:] = 0
        return
    with audio_lock:
        length = _playable_length()
        if length == 0:
            outdata[:] = 0
            return

        # Determine effective playback range (full file or loop region)
        ls = state.loop_start if (state.loop_start is not None and state.loop_end is not None
                                  and state.loop_end > state.loop_start) else 0
        le = state.loop_end   if (state.loop_start is not None and state.loop_end is not None
                                  and state.loop_end > state.loop_start) else length
        region = le - ls

        # Keep position within the active region
        if state.position < ls or state.position >= le:
            state.position = ls
        state.position = ls + (state.position - ls) % region

        remaining = le - state.position
        if remaining < frames:
            part1 = mix(state.position, remaining)
            part2 = mix(ls, frames - remaining)
            outdata[:] = np.concatenate([part1, part2])
            state.position = ls + (frames - remaining)
        else:
            outdata[:] = mix(state.position, frames)
            state.position += frames

def play():
    # Imported cells count: an ATMOS bed (or any imported file) plays on its
    # own, with nothing separated.
    if _playable_length() == 0:
        return
    stop()
    state.stream = sd.OutputStream(samplerate=state.sr, channels=2,
                              blocksize=4096, callback=callback)
    state.stream.start()

def stop():
    if state.stream:
        state.stream.stop()
        state.stream.close()
        state.stream = None

# ----------------------------
# LOAD / SEPARATE
# ----------------------------
def load_file():
    if not model_ready:
        return
    path = filedialog.askopenfilename(
        initialdir=state.last_load_dir,
        filetypes=[("Audio files", "*.wav *.flac *.mp3 *.ogg *.aif *.aiff *.m4a *.aac *.mp4 *.m4v *.mov"), ("All files", "*.*")]
    )
    if not path:
        return
    # Track name, list marker and export tracking are set in _start_load.
    _clear_progress()
    # Disable BG Vocals import while separating
    for _b in _all_import_btns():
        app.after(0, lambda b=_b: b.configure(state="disabled"))
    # Hide waveform, show progress bar in the shared slot
    wave_canvas.pack_forget()
    progress_bar.pack(fill="x", padx=4, pady=(WAVE_H // 2 - 5))
    _set_progress(0.0)              # bar and "0%" visible immediately
    if running:
        app.update_idletasks()      # draw it before anything else happens
    state.instrumental = None
    app.after(0, _update_inst_status_label)
    def _load_job(cancel):
        if _INST_ONLY:
            state.stems = None
            state.stem_mute["instrumental"] = False
            state.stem_solo["instrumental"] = False
            if running:
                app.after(0, lambda: _paint_ms("instrumental"))
            separate_inst(path, cancel=cancel, ui_progress=True)
            return
        # Saved stems first: whatever this song already has on disk is
        # imported, and the passes it covers are skipped below.
        import_saved_stems(path)
        # LOAD FIRST: the strings model runs on the track itself, so it can
        # go before the six-stem pass and fill its cell that much sooner.
        if (_STR_AUTO and _STR_FIRST and not cancel.is_set()
                and strings_model is not None and "strings" not in _saved_cover):
            separate_strings(path, cancel=cancel)
        if _INST_AUTO and _INST_CONCURRENT:
            t_inst = threading.Thread(target=_run_low_priority, args=(_inst_or_saved, path),
                                      kwargs={"cancel": cancel}, daemon=True)
            t_inst.start()
            _six_or_saved(path, cancel=cancel)
            t_inst.join()
        elif _INST_AUTO and _INST_FIRST:
            _inst_or_saved(path, cancel=cancel)
            if not cancel.is_set():
                _six_or_saved(path, cancel=cancel)
        else:
            _six_or_saved(path, cancel=cancel)
            if _INST_AUTO and not cancel.is_set():
                _inst_or_saved(path, cancel=cancel)
        # The vocal chain and the strings chain, in parallel.
        _run_post_stem_chains(path, cancel)
    _start_load(path, _load_job)


def report_split_result():
    """Say plainly what the two vocal cells ended up with."""
    def _describe(name, data):
        if data is None:
            return f"{name}: empty"
        return f"{name}: {len(data) / max(1, int(state.sr or 44100)):.1f}s at " \
               f"{_rms_db(data):.1f} dB"
    print(f"[Karaoke] Result — {_describe('FRT VOX', state.fv_data)}, "
          f"{_describe('BG VOX', state.bg_vocals_data)}")


# ============================================================
# SAVED STEMS
# A song that has been separated and exported before need not be separated
# again. Its stems are found by name — <song>_<cell>.wav, exactly what
# EXPORT STEMS writes — and imported, and the passes they cover are skipped.
# ============================================================
_SAVED_STEM_KEYS = ("vocals", "drums", "bass", "guitar", "piano", "other",
                    "instrumental", "front_vocals", "bg_vocals", "strings")
_SIX = ("vocals", "drums", "bass", "guitar", "piano", "other")
_saved_cover = set()     # what the current load took from saved files


def _saved_stem_dirs(path):
    """Folders searched for a song's saved stems, most specific first."""
    song_dir = os.path.dirname(os.path.abspath(path))
    base = os.path.splitext(os.path.basename(path))[0]
    dirs = [state.saved_stems_dir,
            os.path.join(song_dir, f"{base}_stems"),
            state.export_folder,
            song_dir]
    out = []
    for d in dirs:
        if d and os.path.isdir(d) and os.path.abspath(d) not in out:
            out.append(os.path.abspath(d))
    return out


def find_saved_stems(path):
    """{cell key: file} for every saved stem of this song that can be found.

    Loop exports (…_loop.wav) are partial and never used. With several
    copies of one stem, a lossless file beats an MP3, then the newest wins.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    keys = "|".join(sorted(_SAVED_STEM_KEYS, key=len, reverse=True))
    pat = re.compile(rf"^{re.escape(base)}_(?P<key>{keys})"
                     rf"(?P<dup>[ _]\(?\d+\)?)?"
                     rf"(?P<ext>\.wav|\.flac|_256k\.mp3|\.mp3)$",
                     re.IGNORECASE)
    found = {}
    for d in _saved_stem_dirs(path):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            m = pat.match(name)
            if not m:
                continue
            key = m.group("key").lower()
            full = os.path.join(d, name)
            lossless = m.group("ext").lower() in (".wav", ".flac")
            rank = (lossless, os.path.getmtime(full))
            if key not in found or rank > found[key][0]:
                found[key] = (rank, full)
    return {k: v[1] for k, v in found.items()}


def _load_stem_file(f, length=None):
    audio, file_sr = _read_audio_file(f)
    audio = _resample_audio(np.asarray(audio, dtype=np.float32), file_sr)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    if length is not None:
        if len(audio) < length:
            audio = np.pad(audio, ((0, length - len(audio)), (0, 0)))
        elif len(audio) > length:
            audio = audio[:length]
    return np.ascontiguousarray(audio, dtype=np.float32)


def import_saved_stems(path):
    """Import whatever this song has saved, and note which passes it covers.

    Returns the set of groups covered:
      "stems"         all six six-stem cells — the six-stem pass is skipped
      "instrumental"  the INST cell — its pass is skipped
      "split"         both FRT VOX and BG VOX — the vocal chain is skipped
      "strings"       the STRINGS cell — its pass is skipped
    The six-stem model makes all six cells in one pass, so it is skipped
    only when all six are saved; with some missing it runs as usual.
    """
    global _saved_cover
    _saved_cover = set()
    files = find_saved_stems(path)
    if not files:
        return _saved_cover
    print(f"[Stems] Saved stems found for "
          f"{os.path.splitext(os.path.basename(path))[0]!r}: "
          f"{', '.join(sorted(files))}")
    try:
        loaded = {}
        for key, f in files.items():
            try:
                loaded[key] = _load_stem_file(f)
            except Exception as e:
                print(f"[Stems]   {os.path.basename(f)} could not be read ({e}) "
                      f"— that stem will be separated instead")
        if not loaded:
            return _saved_cover
        length = max(len(a) for a in loaded.values())
        for key in loaded:
            loaded[key] = _load_stem_file(files[key], length)
        sr = int(state.sr or 44100)

        if all(k in loaded for k in _SIX):
            new_stems = {k: loaded[k] for k in _STEM_KEYS if k in loaded}
            _commit_saved_stems(new_stems)
            _saved_cover.add("stems")
        if "instrumental" in loaded:
            with audio_lock:
                state.instrumental = loaded["instrumental"]
                state.instrumental_is_quick = False
            _saved_cover.add("instrumental")
        if "front_vocals" in loaded and "bg_vocals" in loaded:
            with audio_lock:
                state.fv_data, state.fv_sr = loaded["front_vocals"], sr
                state.bg_vocals_data, state.bg_vocals_sr = loaded["bg_vocals"], sr
            _saved_cover.add("split")
        if "strings" in loaded:
            with audio_lock:
                state.strings_data, state.strings_sr = loaded["strings"], sr
                state.strings_is_quick = False
            _saved_cover.add("strings")
    except Exception as e:
        print(f"[Stems] Could not use the saved stems ({e}) — separating "
              f"as usual")
        _saved_cover = set()
        return _saved_cover

    skipped = {"stems": "six-stem", "instrumental": "instrumental",
               "split": "vocals + karaoke", "strings": "strings"}
    print(f"[Stems] Skipping: "
          f"{', '.join(skipped[c] for c in ('stems', 'instrumental', 'split', 'strings') if c in _saved_cover) or 'nothing (a set is incomplete)'}")
    if running:
        app.after(0, _after_saved_import)
    return _saved_cover


def _commit_saved_stems(new_stems):
    """Put imported six-stem cells in place, as a finished separation does."""
    mono = np.mean(sum(new_stems.values()), axis=1)
    state.waveform_data = mono[::max(1, len(mono) // 2000)]
    with audio_lock:
        state.stems = new_stems
        state.loaded_audio_name = state.current_audio_name
        state.stem_volumes = {k: state.stem_volumes.get(k, 1.0) for k in new_stems}
        state.sr = 44100
        state.position = 0
        _eq_zi_state.clear()
    _apply_saved_fixes()
    state.separating = False


def _after_saved_import():
    """The UI side of an import: waveform, transport, cell states."""
    try:
        if "stems" in _saved_cover:
            progress_bar.pack_forget()
            _clear_progress()
            wave_canvas.pack(fill="both", expand=True)
            for _b in _all_import_btns():
                _b.configure(state="normal")
            _unlock_transport()
            _draw_static_waveform()
            _flash_track_name(state.current_audio_name)
        if "split" in _saved_cover:
            _split_replaces_vocals()
        for fn in ("_update_inst_status_label", "_update_strings_status_label",
                   "_update_fv_button", "_update_bgv_button",
                   "_update_vocals_status"):
            f = globals().get(fn)
            if f is not None:
                try:
                    f()
                except Exception:
                    pass
        _playlist_refresh()
    except Exception as e:
        print("[Stems] UI update after import:", e)


def _six_or_saved(path, cancel=None):
    """Run the six-stem pass unless all six cells came from saved files."""
    if "stems" in _saved_cover:
        return
    separate(path, cancel=cancel)
    # Saved strings but freshly separated stems: take the strings out of
    # OTHER so the two cells stay complementary, as the model pass would.
    if "strings" in _saved_cover and state.stems and \
            state.stems.get("other") is not None and state.strings_data is not None:
        with audio_lock:
            other = state.stems["other"]
            n = min(len(other), len(state.strings_data))
            other[:n] -= state.strings_data[:n]


def _inst_or_saved(path, cancel=None, **kw):
    """Run the instrumental pass unless the INST cell came from a file."""
    if "instrumental" in _saved_cover:
        return
    return separate_inst(path, cancel=cancel, **kw)


def _run_post_stem_chains(path, cancel):
    """The two chains that follow the six-stem pass, side by side.

    Chain 1  main track -> vocals model -> VOCALS, then karaoke -> FRT + BG
    Chain 2  OTHER stem -> strings model -> STRINGS, OTHER keeps the rest

    They are independent of each other, so they run on two threads; the GPU
    serialises the actual passes, but neither has to wait for the other's
    file reading, resampling or overlap-add.
    """
    def _vocal_chain():
        if "split" in _saved_cover:
            return                      # both halves came from saved files
        # Saved stems mean the VOCALS cell holds what you kept — refining it
        # again from the main track would overwrite that.
        if (_VOC_AUTO and "stems" not in _saved_cover and not cancel.is_set()
                and vocals_model is not None):
            refine_vocals(path, cancel=cancel)
        if not cancel.is_set():
            separate_bg_vocals(cancel=cancel)
            report_split_result()

    def _strings_chain():
        if "strings" in _saved_cover:
            return
        if (_STR_AUTO and not _STR_FIRST and not cancel.is_set()
                and strings_model is not None):
            separate_strings(cancel=cancel)

    threads = [threading.Thread(target=_vocal_chain, daemon=True),
               threading.Thread(target=_strings_chain, daemon=True)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def load_file_path(path):
    """Load and separate *path* directly (used by the loadlist — no dialog).

    A loadlist entry is a shortcut to the LOAD dialog, nothing more: the
    separation starts here and then, the progress bar appears at once, and
    any separation already running is cancelled.
    """
    if not path or not os.path.isfile(path):
        print(f"[Loadlist] File missing: {path!r}")
        return
    if not model_ready:
        print("[Loadlist] Model still loading — try again in a moment")
        return

    # A loadlist entry is just a shortcut to loading that file: cancel
    # whatever is separating and start this one now.
    wave_canvas.pack_forget()
    progress_bar.pack(fill="x", padx=4, pady=(WAVE_H // 2 - 5))
    _set_progress(0.0)          # bar and "0%" visible immediately
    if running:
        _playlist_refresh()
        app.update_idletasks()  # draw it before the passes begin
    # FRT VOX, BG VOX and the instrumental are cleared in _start_load, the
    # same place as for the LOAD button.

    def _load_job_inner(cancel):
        """Every pass takes the cancel flag, so loading another song stops
        this one at its next chunk instead of running it to the end."""
        if _INST_ONLY:
            # Nothing but the backing track — the six-stem model never runs.
            state.stems = None
            # The cell normally starts muted because it duplicates the stems;
            # with no stems to duplicate it has to be audible.
            state.stem_mute["instrumental"] = False
            state.stem_solo["instrumental"] = False
            if running:
                app.after(0, lambda: _paint_ms("instrumental"))
            separate_inst(path, cancel=cancel, ui_progress=True)
            return
        # Saved stems first: whatever this song already has on disk is
        # imported, and the passes it covers are skipped below.
        import_saved_stems(path)
        # LOAD FIRST: the strings model runs on the track itself, so it can
        # go before the six-stem pass and fill its cell that much sooner.
        if (_STR_AUTO and _STR_FIRST and not cancel.is_set()
                and strings_model is not None and "strings" not in _saved_cover):
            separate_strings(path, cancel=cancel)
        if _INST_AUTO and _INST_CONCURRENT:
            t_inst = threading.Thread(target=_run_low_priority, args=(_inst_or_saved, path),
                                      kwargs={"cancel": cancel}, daemon=True)
            t_inst.start()
            _six_or_saved(path, cancel=cancel)
            t_inst.join()
        elif _INST_AUTO and _INST_FIRST:
            # Instrumental first: it's playable on its own while the stems
            # are still being separated.
            _inst_or_saved(path, cancel=cancel)
            if not cancel.is_set():
                _six_or_saved(path, cancel=cancel)
        else:
            # Stems first: the song becomes playable in about half the time,
            # then the instrumental fills in behind it.
            _six_or_saved(path, cancel=cancel)
            if _INST_AUTO and not cancel.is_set():
                _inst_or_saved(path, cancel=cancel)
        # BG VOX comes from the karaoke model, not from a file.
        # The vocal chain and the strings chain, in parallel.
        _run_post_stem_chains(path, cancel)

    _start_load(path, _load_job_inner)



# ----------------------------
# LOADLIST SUPPORT
# Pre-loading has been removed: a song in the loadlist is simply a shortcut
# to loading that file, so double-clicking one starts its separation there
# and then. What remains here is the machinery a load needs.
# ----------------------------
_load_claim_lock = threading.Lock()
_load_running    = [False]

# The stems and instrumental passes of one load may run side by side, so they
# announce themselves here; anything running in the background (none, now
# that pre-loading is gone, but the passes still support it) waits for them.
_fg_cv      = threading.Condition()
_fg_running = 0


# Only one separation pass runs at a time. The chains are still started in
# parallel — they just queue for their turn here instead of fighting over
# the GPU, which made all of them crawl and left three cells sitting at
# "separating" together.
_pass_lock = threading.Lock()


def _pass_acquire(tag="pass"):
    """Wait for whatever is separating now to finish."""
    if _pass_lock.acquire(blocking=False):
        return
    print(f"[Queue] {tag}: waiting for the current separation to finish")
    _pass_lock.acquire()


def _pass_release():
    try:
        _pass_lock.release()
    except RuntimeError:
        pass          # never acquired on this path


def _stack_padded(chunks, batch_n):
    """Stack chunks into a batch, padding with silence up to batch_n.

    Every batch then has the same shape. The last batch of a track is
    usually short — the leftover chunks do not fill it — and with
    cudnn.benchmark on, a new shape makes cuDNN re-run its algorithm search
    for every layer, allocating trial workspace as it goes. On a model this
    size that can take minutes or run out of memory, and it happened at the
    same place in every track: the start of the final batch, which is where
    the instrumental pass sat at 86%. The padding chunks are silence and
    their outputs are never read.
    """
    batch = np.stack(chunks)
    short = batch_n - len(chunks)
    if short > 0:
        batch = np.concatenate(
            [batch, np.zeros((short,) + batch.shape[1:], dtype=batch.dtype)])
    return np.ascontiguousarray(batch)


def _is_oom(err):
    """True for a GPU out-of-memory error."""
    text = str(err).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _fg_enter(tag="pass"):
    global _fg_running
    _pass_acquire(tag)
    with _fg_cv:
        _fg_running += 1


def _release_vram():
    """Return cached blocks to the driver between model passes."""
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def _fg_leave():
    global _fg_running
    with _fg_cv:
        _fg_running = max(0, _fg_running - 1)
        _fg_cv.notify_all()
    _release_vram()
    _pass_release()


def _bg_wait_for_foreground(cancel=None, timeout=600.0):
    """Hold a background pass until the foreground ones are done."""
    deadline = time.monotonic() + timeout
    with _fg_cv:
        while _fg_running:
            if cancel is not None and cancel.is_set():
                return
            if time.monotonic() > deadline:
                return
            _fg_cv.wait(0.5)

# Loading a song cancels whatever is separating and starts at once. Each load
# carries its own cancel flag, so a job being replaced cannot switch off the
# job replacing it.
_fg_cancel   = [threading.Event()]   # the flag the CURRENT load watches
_load_thread = [None]                # the thread running the current load
_load_gen    = [0]                   # bumped per load; stale UI is dropped


def _load_generation():
    return _load_gen[0]


def _is_current_load(gen):
    """False once another load has started, so late UI work can be skipped."""
    return gen == _load_gen[0]


def _cancel_current_load():
    """Ask the separation in flight to stop. Returns its thread, if any."""
    old = _load_thread[0]
    if old is not None and old.is_alive():
        _fg_cancel[0].set()
        print("[Load] Cancelling the separation in progress")
    return old


def _reset_pass_flags():
    """Clear the per-pass "already running" flags when nothing is running.

    Those flags exist so a pass cannot be started twice over. If one were
    ever left set — by a crash, or an exit path that skipped its cleanup —
    every later instrumental or vocal split would decline to start. A load
    beginning with no job alive is proof that nothing is running, so it is a
    safe moment to clear them.
    """
    global _inst_separating, _kara_separating, _strings_separating
    global _vocals_refining
    old = _load_thread[0]
    if old is not None and old.is_alive():
        return
    if (_inst_separating or _kara_separating or _strings_separating
            or _vocals_refining):
        print("[Load] Clearing stale pass flags")
    _inst_separating = False
    _kara_separating = False
    _strings_separating = False
    _vocals_refining = False
    with _fg_cv:
        globals()["_fg_running"] = 0
        _fg_cv.notify_all()
    # And the one-pass-at-a-time lock: if a pass died without releasing it,
    # every later pass would wait on it for ever and the cells would just
    # stay empty with nothing said.
    if _pass_lock.locked():
        print("[Load] Releasing the separation lock left behind by a "
              "previous pass")
        try:
            _pass_lock.release()
        except RuntimeError:
            pass


def _clear_solos():
    """Release every solo when a new track loads.

    A solo left on INSTRUM, or on a cell the new track has not produced yet,
    would silence the whole new track — solo mutes everything else.
    """
    global _inst_autosolo_active
    if any(state.stem_solo.values()):
        print("[Load] Releasing solos from the previous track")
    state.stem_solo.clear()
    _inst_autosolo_active = False
    try:
        _paint_all_ms()
    except (NameError, KeyError):
        pass


def _unload_track_extras():
    """Drop everything derived from the previous track when a new one loads.

    FRT VOX and BG VOX are the previous track's vocal, split in two. The
    LOADLIST path cleared them but the LOAD button did not, so they kept
    playing over the next song. Both paths now come through _start_load, and
    this runs there.

    The VOCALS stem is un-muted at the same time: it was muted only because
    the halves were carrying the vocal, and the new track's split has not
    happened yet.
    """
    state.vocals_refined = False
    state.vocals_is_vca = False
    state.stem_invert.clear()
    if running:
        for _k in list(_invert_btns):
            app.after(0, lambda k=_k: _paint_invert(k))
    if running:
        app.after(0, _show_vocals_cell)
        app.after(0, _update_vocals_status)
    state.fv_data        = None
    state.bg_vocals_data = None
    state.strings_data   = None
    state.strings_is_quick = False
    state.instrumental   = None
    _halves_muted_by_vocals.clear()
    state.instrumental_is_quick = False
    _restore_vocals_mute()          # halves gone -> VOCALS audible again
    if running:
        for _fn in (_clear_split_progress, _update_fv_button,
                    _update_bgv_button, _update_inst_status_label):
            try:
                _fn()
            except Exception:
                pass


def _reset_instrumental_overlap():
    for _k in list(_inst_muted_by_us):
        state.stem_mute[_k] = False
        try:
            _paint_ms(_k)
        except (NameError, KeyError):
            pass
    _inst_muted_by_us.clear()


def _set_current_track(path):
    """Record *path* as the track being loaded — name, list marker, export.

    The name drives the title flash and the export filenames. The LOAD button
    set it; the LOADLIST path relied on the pre-load cache to do so, and when
    pre-loading was removed nothing set it any more — so a LOADLIST track
    showed the previous track's title. Every load now sets it here.
    """
    state.current_audio_name = os.path.splitext(os.path.basename(path))[0]
    state._stem_export_state = {}   # export change tracking is per track
    state.last_load_dir      = os.path.dirname(path)
    # Always the path actually loaded. A file opened with LOAD that is not in
    # the list simply matches no row, so the ▶ marker correctly goes away.
    _playlist_current[0] = path


def _start_load(path, job_inner):
    """Replace whatever is loading with *path*.

    job_inner(cancel_event) does the passes. It runs on a supervisor thread
    that first waits for the outgoing job to notice the cancel, so the two
    never overlap on the GPU, and the UI thread is never blocked meanwhile.
    """
    _reset_pass_flags()
    old_thread = _cancel_current_load()
    _set_current_track(path)
    _clear_solos()
    _reset_instrumental_overlap()
    _unload_track_extras()

    _load_gen[0] += 1
    my_cancel = threading.Event()
    _fg_cancel[0]    = my_cancel
    _load_running[0] = True
    state.separating = True

    def _supervise():
        _lower_thread_priority()
        try:
            if old_thread is not None and old_thread.is_alive():
                # The outgoing pass checks the cancel flag between forward
                # passes, so this can take a few seconds. Say so, rather than
                # leaving a blank bar that looks like nothing happened.
                print("[Load] Waiting for the previous separation to stop...")
                if running:
                    app.after(0, lambda: _set_progress_text("STOPPING PREVIOUS…"))
                old_thread.join(timeout=60)
                # Clear the message, then hold a short beat before the new
                # separation starts reporting, so the two never overlap on
                # screen.
                if running:
                    app.after(0, lambda: _set_progress_text(""))
                time.sleep(_RESTART_GAP_S)
                if my_cancel.is_set():
                    return        # replaced again during the pause
                if running:
                    app.after(0, lambda: _set_progress(0.0))
            if my_cancel.is_set():
                return            # replaced again before we even started
            job_inner(my_cancel)
        finally:
            # Only the newest load may clear the flags; an outgoing job
            # finishing late must not declare the new one finished.
            if _fg_cancel[0] is my_cancel:
                _load_running[0] = False
                state.separating = False
                if running:
                    app.after(0, _playlist_refresh)

    t = threading.Thread(target=_supervise, daemon=True)
    _load_thread[0] = t
    t.start()
    # Show the new track as loading in the list and status line straight away.
    if running:
        try:
            _playlist_refresh()
        except Exception:
            pass


def _claim_load_slot(path):
    """Mark a load as running (used by the instant, cache-free paths)."""
    with _load_claim_lock:
        _load_running[0] = True
        state.separating = True
        return True


def _release_load_slot():
    with _load_claim_lock:
        _load_running[0] = False


def _apply_loaded_result(path, cached):
    """Put an already-separated result into the player straight away."""
    state.current_audio_name = os.path.splitext(os.path.basename(path))[0]
    state.loaded_audio_name  = state.current_audio_name
    state._stem_export_state = {}
    state.waveform_data = cached.get("waveform")
    if state.waveform_data is None and cached.get("instrumental") is not None:
        _mono = np.mean(cached["instrumental"], axis=1)
        state.waveform_data = _mono[::max(1, len(_mono) // 2000)]

    stems_cached = cached.get("stems")
    with audio_lock:
        state.stems = stems_cached
        if stems_cached:
            state.stem_volumes = {k: state.stem_volumes.get(k, 1.0)
                                  for k in stems_cached}
        state.sr           = cached.get("sr", 44100)
        state.position     = 0
        _eq_zi_state.clear()
    state.instrumental = cached.get("instrumental")
    state.bg_vocals_data = cached.get("bg_vocals")
    if state.bg_vocals_data is not None:
        state.bg_vocals_sr = cached.get("sr", 44100)
    state.fv_data = cached.get("front_vocals")
    if state.fv_data is not None:
        state.fv_sr = cached.get("sr", 44100)
    state.instrumental_is_quick = bool(cached.get("instrumental_quick", False))
    state.loop_start = state.loop_end = None

    state.separating = False
    threading.Thread(target=_prewarm_reverb, daemon=True).start()
    print(f"[Preload] Loaded {os.path.basename(path)!r} from cache — no wait")

    if running:
        progress_bar.pack_forget()
        _clear_progress()
        wave_canvas.pack(fill="both", expand=True)
        for b in _all_import_btns():
            b.configure(state="normal")
        _unlock_transport()
        _update_inst_status_label()
        _update_bgv_button()
        _update_fv_button()
        _mute_vocal_cells()
        _playlist_refresh()
        _apply_pending_autosolo()
        _flash_track_name(state.current_audio_name)


def _prewarm_reverb():
    """Pre-initialise the shared reverb bus on a background thread so the
    first reverb slider move is instant and never blocks the audio callback."""
    if state.sr is None:
        return
    _prewarm_reverb_bus(int(state.sr))


def _update_inst_status_label():
    """Refresh the status pill in the instrumental cell.

    Called on the Tk thread after every state change (start / finish / error).
    The label widget is assigned to _inst_status_lbl once the UI is built;
    before that the function is a safe no-op.
    """
    try:
        lbl = _inst_status_lbl   # set by UI builder below
    except NameError:
        return
    if not _INST_AUTO and state.instrumental is None:
        lbl.configure(text="AUTO OFF", text_color=TEXT_DIM)
    elif _inst_separating:
        lbl.configure(text="SEPARATING...", text_color=TEXT_DIM)
    elif state.instrumental is not None:
        if state.instrumental_is_quick:
            lbl.configure(text="QUICK MIX", text_color="#ffaa00")
        else:
            lbl.configure(text="FULL MIX", text_color="#44cc44")
    elif inst_model_ready and inst_model is None:
        lbl.configure(text="MODEL FAILED", text_color=BRIGHT_RED)
    else:
        lbl.configure(text="WAITING", text_color=TEXT_DIM)


def separate(path, into=None, cancel=None):
    """Separate *path* into 6 stems using BS-RoFormer-SW (bs-roformer-infer).

    Runs entirely in-process with PyTorch — no temp files, no onnxruntime.
    The model chunks the audio into overlapping segments and reconstructs each
    stem; chunk size is tuned so an 8 GB GPU (or 16 GB RAM on CPU) never OOMs.

    Normally the result is committed straight to state and the UI follows
    along.  When *into* is a dict the result is stored there instead and the
    UI is left alone — that's how the playlist pre-loads upcoming songs while
    the current one plays.  *cancel* is a threading.Event; when it is set the
    run bails out at the next chunk so a foreground load isn't kept waiting.
    """
    bg = into is not None
    _my_load_gen = _load_generation()

    def update(v):
        if bg:
            return
        if running:
            app.after(0, lambda: _set_progress(v))

    def _abort(msg):
        print(f"[BS-RoFormer] {msg}")
        if bg:
            into["error"] = msg
            return
        state.separating = False
        if running:
            app.after(0, _clear_progress)
            for btn in _all_import_btns():
                app.after(0, lambda b=btn: b.configure(state="normal"))

    if model is None:
        _abort("Model not loaded — cannot separate.")
        return

    update(0.10)

    # ── 1. Read & resample ──────────────────────────────────────────────────
    audio, sr_local = _read_audio_cached(path)
    audio = np.asarray(audio, dtype=np.float32)   # (N, 2)

    _MODEL_SR = 44100
    if sr_local != _MODEL_SR:
        from scipy.signal import resample_poly as _rp
        def _gcd(a, b):
            while b: a, b = b, a % b
            return a
        g    = _gcd(sr_local, _MODEL_SR)
        up   = _MODEL_SR // g
        down = sr_local  // g
        audio = np.stack([
            _rp(audio[:, 0], up, down).astype(np.float32),
            _rp(audio[:, 1], up, down).astype(np.float32),
        ], axis=1)

    update(0.25)

    # ── 2. Chunked inference ────────────────────────────────────────────────
    # Process in overlapping chunks so long tracks fit in VRAM/RAM.
    # overlap=0.1 s on each side; chunks blend with linear cross-fade.
    if bg:
        _bg_wait_for_foreground(cancel)
        if cancel is not None and cancel.is_set():
            print("[BS-RoFormer] Pre-load cancelled")
            return
    else:
        _fg_enter("six-stem")
    try:
        run_device  = device
        sr_i        = _MODEL_SR
        n_samples   = len(audio)
        n_stems     = len(_STEM_KEYS)

        # chunk / overlap in samples — from the model's own config when it
        # has one (see _chunking_from_config), else the values above.
        chunk_s, overlap_s = _chunking_from_config(_bsr_config, _MODEL_SR)
        chunk_n     = int(round(chunk_s   * sr_i))
        overlap_n   = int(round(overlap_s * sr_i))
        step_n      = chunk_n - 2 * overlap_n

        # Output buffer: (n_stems, N, 2)
        out = np.zeros((n_stems, n_samples, 2), dtype=np.float32)
        wgt = np.zeros(n_samples, dtype=np.float32)

        # Linear fade window for overlap-add
        fade = np.ones(chunk_n, dtype=np.float32)
        fade[:overlap_n]  = np.linspace(0, 1, overlap_n)
        fade[-overlap_n:] = np.linspace(1, 0, overlap_n)

        starts = list(range(0, n_samples, step_n))
        n_chunks = len(starts)

        # Several chunks go through the model at once, so the GPU has real
        # work queued instead of stalling between one-chunk passes.
        batch_n  = _batch_size_for(run_device, chunk_s)
        use_fp16 = _INFER_FP16
        _release_vram()
        print(f"[BS-RoFormer] {n_chunks} chunks of {chunk_s:g}s, batch {batch_n}, "
              f"{'fp16' if (use_fp16 and run_device.type == 'cuda') else 'fp32'} "
              f"on {run_device}")
        _t_infer = time.perf_counter()
        model.eval()
        ci = 0
        with _infer_ctx():
            while ci < n_chunks:
                if cancel is not None and cancel.is_set():
                    print("[BS-RoFormer] Separation cancelled — another song "
                          "was loaded" if not bg else "[BS-RoFormer] Pre-load cancelled")
                    return
                batch_starts = starts[ci:ci + batch_n]

                chunks = []
                kept   = []        # starts that actually need the model
                for start in batch_starts:
                    end   = min(start + chunk_n, n_samples)
                    chunk = audio[start:end]                   # (T, 2)
                    pad   = chunk_n - len(chunk)
                    if pad:
                        chunk = np.pad(chunk, ((0, pad), (0, 0)))
                    if float(np.max(np.abs(chunk))) < _SILENCE_PEAK:
                        actual = min(end - start, chunk_n)
                        wgt[start:start + actual] += fade[:actual]
                        continue
                    chunks.append(chunk.T)                     # (2, T)
                    kept.append(start)

                if not chunks:
                    ci += len(batch_starts)
                    update(0.25 + 0.65 * ci / n_chunks)
                    continue

                try:
                    # (B, 2, T). Pinned memory lets the copy to the card run
                    # asynchronously alongside work already queued on it.
                    x = torch.from_numpy(_stack_padded(chunks, batch_n))
                    if run_device.type == "cuda":
                        x = x.pin_memory().to(run_device, non_blocking=True)
                    else:
                        x = x.to(run_device)
                    with _amp_ctx(run_device, use_fp16):
                        # Forward — returns (B, n_stems, 2, T_out).
                        # T_out may differ from chunk_n due to STFT padding.
                        stems_t = model(x)
                    # Apply the overlap-add window here, on the card: it is a
                    # single cheap multiply there, against one over the whole
                    # batch in NumPy afterwards. If the tensor type in use
                    # cannot do it, fall back to windowing in NumPy below.
                    windowed = False
                    try:
                        w_t = _fade_tensor(fade, stems_t.shape[-1],
                                           getattr(stems_t, "device", run_device),
                                           getattr(stems_t, "dtype", None))
                        stems_t = stems_t * w_t
                        windowed = True
                    except Exception:
                        windowed = False
                    # One transfer per batch instead of one per stem per chunk.
                    stems_np = stems_t.float().cpu().numpy()
                except RuntimeError as e:
                    # Step back through the optimisations one at a time rather
                    # than failing the whole separation.
                    if run_device.type == "cuda":
                        torch.cuda.empty_cache()
                    if batch_n > 1:
                        batch_n = max(1, batch_n // 2)
                        print(f"[BS-RoFormer] {e}\n[BS-RoFormer] Retrying with "
                              f"batch size {batch_n}")
                        continue
                    if use_fp16:
                        use_fp16 = False
                        print(f"[BS-RoFormer] {e}\n[BS-RoFormer] Retrying in "
                              f"full precision")
                        continue
                    raise

                t_out = stems_np.shape[-1]
                for bi, start in enumerate(kept):
                    end    = min(start + chunk_n, n_samples)
                    actual = min(end - start, t_out)
                    seg = stems_np[bi, :, :, :actual].transpose(0, 2, 1)
                    if not windowed:
                        seg = seg * fade[:actual][None, :, None]
                    out[:, start:start + actual] += seg
                    wgt[start:start + actual]    += fade[:actual]

                ci += len(batch_starts)
                update(0.25 + 0.65 * ci / n_chunks)

        # Normalise overlap-add
        wgt = np.maximum(wgt, 1e-8)
        out /= wgt[None, :, None]
        _secs = n_samples / sr_i
        _el   = time.perf_counter() - _t_infer
        print(f"[BS-RoFormer] Stems in {_el:.1f}s "
              f"({_secs / max(_el, 1e-6):.1f}x realtime)")

    except Exception as e:
        import traceback; traceback.print_exc()
        _abort(f"Separation error: {e}")
        return
    finally:
        if not bg:
            _fg_leave()

    update(0.92)

    # ── 3. Commit to state ──────────────────────────────────────────────────
    # Use the stem order read from the model YAML config so index 0 always maps
    # to the correct stem name regardless of the order we listed in _STEM_KEYS.
    # Fall back to _STEM_KEYS if the config order wasn't loaded for any reason.
    stem_order = _bsr_stem_order if _bsr_stem_order and len(_bsr_stem_order) == out.shape[0] else _STEM_KEYS
    print(f'[BS-RoFormer] Mapping stems by order: {stem_order}')
    new_stems = {name: out[i] for i, name in enumerate(stem_order)}

    # Ensure every key the mixer expects exists (pad missing ones with silence)
    n_samp = out.shape[1]
    for k in _STEM_KEYS:
        if k not in new_stems:
            print(f'[BS-RoFormer] WARNING: stem {k!r} not in model output — padding silence')
            new_stems[k] = np.zeros((n_samp, 2), dtype=np.float32)

    mono = np.mean(np.sum(out, axis=0), axis=1)
    wave_snapshot = mono[::max(1, len(mono) // 2000)]

    if bg:
        into["stems"]    = new_stems
        into["sr"]       = _MODEL_SR
        into["waveform"] = wave_snapshot
        return

    state.waveform_data = wave_snapshot

    with audio_lock:
        state.stems        = new_stems
        state.loaded_audio_name = state.current_audio_name
        state.stem_volumes = {k: state.stem_volumes.get(k, 1.0) for k in new_stems}
        state.sr           = _MODEL_SR
        state.position     = 0
        _eq_zi_state.clear()   # reset IIR filter state for new track

    # Any fixes saved for this track go on before anything is played, so the
    # audio is in the stems the user corrected it to last time.
    _apply_saved_fixes()

    # No stand-in for STRINGS any more: OTHER holds the strings until the
    # model has run and taken them out, so copying OTHER into STRINGS would
    # play the same material from both cells.

    # The INSTRUM cell can be filled right now from the stems, so it's usable
    # while the dedicated model pass is still running (or instead of it, when
    # AUTO SEPARATE is off).
    if state.instrumental is None:
        quick = _quick_instrumental(new_stems)
        if quick is not None:
            state.instrumental = quick
            state.instrumental_is_quick = True
            if running:
                app.after(0, _update_inst_status_label)

    state.separating = False
    threading.Thread(target=_prewarm_reverb, daemon=True).start()
    update(1.0)

    if running:
        def _show_waveform(_gen=_my_load_gen):
            if not _is_current_load(_gen):
                return   # a newer load owns the window now
            progress_bar.pack_forget()
            _clear_progress()
            wave_canvas.pack(fill="both", expand=True)
            bgv_import_btn.configure(state="normal")
            for _b in _all_import_btns():
                _b.configure(state="normal")
            _unlock_transport()
            _playlist_refresh()
            _apply_pending_autosolo()
            _mute_vocal_cells()
            _flash_track_name(state.current_audio_name)
        app.after(500, _show_waveform)

# ----------------------------
# EXPORT
# ----------------------------
def choose_export_folder():
    folder = filedialog.askdirectory(initialdir=state.export_folder)
    if folder:
        state.export_folder = folder
        _save_dirs()

def _loop_region_samples():
    """Return (start_sample, end_sample) for export.

    If a loop region is set (right-click drag on waveform) returns that range.
    Otherwise returns (0, full_length) so exports cover the whole file.
    The suffix string is also returned so filenames can indicate a region was used.
    """
    if state.stems is None:
        if state.instrumental is None:
            return 0, 0, ""
        length = len(state.instrumental)
    else:
        length = next(iter(state.stems.values())).shape[0]
    ls = state.loop_start
    le = state.loop_end
    if ls is not None and le is not None and le > ls:
        return int(ls), int(min(le, length)), "_loop"
    return 0, length, ""


def export_mix():
    """Render and export the full mix as a single stereo file.
    Only available in RAMMSTEIN theme (± EVILIFY).
    Runs on a background thread with progress feedback.
    """
    if state.stems is None or not state.export_folder:
        return
    if not state.export_fmt_wav24 and not state.export_fmt_mp3 and not state.export_fmt_mp3_256:
        return

    base = state.loaded_audio_name or state.current_audio_name or "mix"
    sr_i              = int(state.sr)
    ex_start, ex_end, loop_suffix = _loop_region_samples()
    length            = ex_end - ex_start   # may be full file or loop region

    formats = []
    if state.export_fmt_wav24:   formats.append(("wav24",   ".wav"))
    if state.export_fmt_mp3:     formats.append(("mp3",     ".mp3"))
    if state.export_fmt_mp3_256: formats.append(("mp3_256", "_256k.mp3"))

    def _do():
        block      = 4096
        n_blocks   = (length + block - 1) // block
        rendered   = np.zeros((length, 2), dtype=np.float32)
        total_jobs = n_blocks + len(formats)

        def _prog(v, lbl=""):
            if running:
                app.after(0, lambda: _export_progress_bar.set(v))
                if lbl:
                    app.after(0, lambda l=lbl: _export_status_lbl.configure(text=l))

        def _show(vis):
            if running:
                def _t():
                    if vis:
                        _export_progress_frame.pack(fill="x", padx=10, pady=(0, 6))
                    else:
                        _export_progress_frame.pack_forget()
                    _write_btn.configure(state="disabled" if vis else "normal")
                    for _emb in (_sai_export_mix_btn,
                                 _eidii_export_mix_btn,
                                 _export_mix_btn):
                        try:
                            _emb.configure(state="disabled" if vis else "normal")
                        except Exception:
                            pass
                app.after(0, _t)

        _show(True)
        _prog(0.0, "RENDERING MIX…")

        # Stop live playback so the audio callback cannot race with the
        # render loop (which would produce sped-up audio during export and
        # corrupt the rendered buffer with interleaved callback reads).
        was_playing = state.stream is not None
        if was_playing:
            app.after(0, stop)
            import time; time.sleep(0.12)   # let the callback drain

        # Flush both reverb buses so stale delay-line state from live
        # playback does not bleed into the first block of the rendered output.
        sr_i_local = int(state.sr)
        _shared_reverb._reset(sr_i_local)
        _master_reverb._reset(sr_i_local)

        for bi in range(n_blocks):
            pos   = bi * block
            frs   = min(block, length - pos)
            chunk = mix(ex_start + pos, frs)   # offset by loop start
            rendered[pos:pos+frs] = chunk[:frs]
            _prog((bi + 1) / total_jobs)

        np.clip(rendered, -1.0, 1.0, out=rendered)

        # Restore clean reverb state so live playback sounds correct after export.
        _shared_reverb._reset(sr_i_local)
        _master_reverb._reset(sr_i_local)

        for fi, (fmt, ext) in enumerate(formats):
            _prog((n_blocks + fi) / total_jobs, f"WRITING MIX ({ext[1:].upper()})…")
            out_name = f"{base}_mix{loop_suffix}{ext}"
            # Numbered if file exists
            out_path = os.path.join(state.export_folder, out_name)
            if os.path.exists(out_path):
                n_suf = 1
                while os.path.exists(os.path.join(state.export_folder, f"{base}_mix{loop_suffix}{n_suf}{ext}")):
                    n_suf += 1
                out_path = os.path.join(state.export_folder, f"{base}_mix{loop_suffix}{n_suf}{ext}")
            try:
                _write_stem_file(rendered, sr_i, out_path, fmt)
            except Exception as e:
                print(f"Export Mix error ({ext}): {e}")

        _prog(1.0, "MIX SAVED ✓")
        if running:
            app.after(1500, lambda: _show(False))
            app.after(1500, lambda: _prog(0.0, ""))

    threading.Thread(target=_do, daemon=True).start()


def _stem_settings_hash(name):
    """Return a hash of all settings that affect the exported sound of a stem."""
    import hashlib, struct
    h = hashlib.md5()
    h.update(str(state.eq_bands.get(name, [0]*5)).encode())
    if name in _EXPORT_EXTRAS:
        vol_now = getattr(state, _EXPORT_EXTRAS[name][1], 1.0)
    else:
        vol_now = state.stem_volumes.get(name, 1.0)
    h.update(struct.pack("f", vol_now))
    h.update(struct.pack("f", state.stem_widths.get(name, 1.0)))
    h.update(struct.pack("f", state.stem_reverbs.get(name, 0.0)))
    h.update(struct.pack("f", state.stem_air.get(name, 0.0)))
    h.update(struct.pack("ff", state.volume_master, state.stereo_width))
    return h.hexdigest()


def render_stem(name):
    """Return the full processed audio for a single Demucs stem as a float32
    array, applying: EQ → VFF (vocals only) → stereo width → air → volume.
    Reverb is applied by processing the signal in chunks through a fresh
    reverb instance so the exported audio matches what you hear during playback.
    """
    raw  = state.stems[name]                      # (N, 2) float32 — unprocessed
    sr_i = int(state.sr)

    # --- Stage 1: EQ → VFF → dynamics → width → pan → air (all stateless/single-pass)
    chunk = apply_eq(raw.copy(), sr_i, state.eq_bands.get(name, [0]*5))
    if name == "vocals" and state.vff_enabled:
        chunk = apply_vff(chunk)
    chunk = apply_dynamics(
        chunk, name + "_export",
        state.stem_comp_enabled.get(name, False),
        state.stem_comp_thresh.get(name,  _COMP_DEFAULTS["thresh"]),
        state.stem_comp_ratio.get(name,   _COMP_DEFAULTS["ratio"]),
        state.stem_comp_attack.get(name,  _COMP_DEFAULTS["attack"]),
        state.stem_comp_release.get(name, _COMP_DEFAULTS["release"]),
        state.stem_gate_enabled.get(name, False),
        state.stem_gate_thresh.get(name,  _GATE_DEFAULTS["thresh"]),
        state.stem_gate_attack.get(name,  _GATE_DEFAULTS["attack"]),
        state.stem_gate_release.get(name, _GATE_DEFAULTS["release"]),
        sr_i)
    chunk = apply_stem_width(chunk, state.stem_widths.get(name, 1.0))
    chunk = apply_pan(chunk, state.stem_pan.get(name, 0.0))
    chunk = apply_air(chunk, state.stem_air.get(name, 0.0), sr_i)

    # --- Stage 2: volume
    vol = state.stem_volumes.get(name, 1.0)
    chunk = chunk * vol

    # --- Stage 3: reverb (stateful — process in 4096-sample blocks with a
    #     fresh reverb instance so we don't disturb live playback state)
    rev_amt = state.stem_reverbs.get(name, 0.0)
    if rev_amt > 0.001:
        export_reverb = _SharedReverb()
        export_reverb._reset(sr_i)
        block   = 4096
        n_samp  = len(chunk)
        out_rev = np.zeros_like(chunk)
        for i in range(0, n_samp, block):
            blk     = chunk[i:i+block]
            send    = np.clip(blk, -1.0, 1.0) * rev_amt
            wet     = export_reverb.process(send, sr_i)
            out_rev[i:i+block] = blk + wet
        chunk = out_rev

    np.clip(chunk, -1.0, 1.0, out=chunk)
    return chunk


def _write_stem_file(data_f32, sr_val, out_path, fmt):
    """Write stem audio to out_path in the requested format.
    fmt: 'wav24'    → WAV PCM 24-bit
         'mp3'      → MP3 320 kbps (requires pydub + ffmpeg)
         'mp3_256'  → MP3 256 kbps (requires pydub + ffmpeg)
    """
    if fmt == "wav24":
        sf.write(out_path, data_f32, sr_val, subtype="PCM_24")
    elif fmt in ("mp3", "mp3_256"):
        bitrate = "320k" if fmt == "mp3" else "256k"
        try:
            from pydub import AudioSegment
            pcm16 = (np.clip(data_f32, -1.0, 1.0) * 32767).astype(np.int16)
            seg = AudioSegment(
                pcm16.tobytes(),
                frame_rate=sr_val,
                sample_width=2,
                channels=data_f32.shape[1] if data_f32.ndim > 1 else 1,
            )
            seg.export(out_path, format="mp3", bitrate=bitrate)
        except Exception as e:
            print(f"MP3 export error (pydub/ffmpeg required): {e}")



# Stems kept outside state.stems: key -> (data attribute, volume attribute).
_EXPORT_EXTRAS = {
    "instrumental": ("instrumental",   "instrumental_vol"),
    "front_vocals": ("fv_data",        "fv_volume"),
    "bg_vocals":    ("bg_vocals_data", "bg_vocals_volume"),
    "strings":      ("strings_data",   "strings_volume"),
}


def export_selected_stems(only=None, default_wav=False):
    """Write the ticked stems to the export folder.

    *only* restricts the job to the given stem names (used by the EXPORT
    button on the INSTRUM cell); when it is None the export-row checkboxes
    decide.  *default_wav* falls back to 24-bit WAV when no format is ticked,
    so a one-click export button always produces a file.

    Stems that live outside state.stems — the instrumental and the two
    karaoke halves — are exportable in exactly the same way; _EXPORT_EXTRAS
    says where each one's audio and volume are kept.
    """
    if not state.export_folder:
        return
    if state._export_running:
        return   # already exporting

    # Collect which stems to export and snapshot settings — all on UI thread
    # before handing off to the background thread.
    available = set(state.stems.keys()) if state.stems else set()
    for _xk, (_data_attr, _vol_attr) in _EXPORT_EXTRAS.items():
        if getattr(state, _data_attr, None) is not None:
            available.add(_xk)

    if only is not None:
        selected = [n for n in only if n in available]
    else:
        selected = [name for name, var in stem_check_vars.items()
                    if var.get() and name in available]
    if not selected:
        return

    # Snapshot loop region and mutable state needed for rendering.
    ex_start, ex_end, loop_suffix = _loop_region_samples()
    with audio_lock:
        # Slice each stem to the export region (full file or loop selection)
        stems_snap = {}
        for n in selected:
            if n in _EXPORT_EXTRAS:
                src = getattr(state, _EXPORT_EXTRAS[n][0])
            else:
                src = state.stems[n]
            stems_snap[n] = np.asarray(src[ex_start:ex_end], dtype=np.float32).copy()
        sr_snap      = int(state.sr) if state.sr else 44100
        eq_snap      = {n: list(state.eq_bands.get(n, [0]*5)) for n in selected}
        vol_snap     = {n: (getattr(state, _EXPORT_EXTRAS[n][1], 1.0)
                            if n in _EXPORT_EXTRAS
                            else state.stem_volumes.get(n, 1.0))
                        for n in selected}
        width_snap   = {n: state.stem_widths.get(n, 1.0)     for n in selected}
        rev_snap     = {n: state.stem_reverbs.get(n, 0.0)    for n in selected}
        air_snap     = {n: state.stem_air.get(n, 0.0)        for n in selected}
        vff_snap     = state.vff_enabled

    formats = []
    if state.export_fmt_wav24:
        formats.append(("wav24",   ".wav"))
    if state.export_fmt_mp3:
        formats.append(("mp3",     ".mp3"))
    if state.export_fmt_mp3_256:
        formats.append(("mp3_256", "_256k.mp3"))
    if not formats:
        if not default_wav:
            return
        formats.append(("wav24", ".wav"))

    base = state.loaded_audio_name or state.current_audio_name or "stem"
    hash_snap     = {n: _stem_settings_hash(n) for n in selected}
    prev_hashes   = dict(state._stem_export_state)
    total_jobs    = len(selected) * len(formats)

    def _do_export():
        completed = 0

        def _set_progress(v, label=""):
            if running:
                app.after(0, lambda: _export_progress_bar.set(v))
                if label:
                    app.after(0, lambda l=label: _export_status_lbl.configure(text=l))

        def _show_export_ui(visible):
            if running:
                def _toggle():
                    if visible:
                        _export_progress_frame.pack(fill="x", padx=10, pady=(0, 6))
                    else:
                        _export_progress_frame.pack_forget()
                    _write_btn.configure(state="disabled" if visible else "normal")
                app.after(0, _toggle)

        _show_export_ui(True)
        _set_progress(0.0, "SAVING STEMS…")

        for name in selected:
            _set_progress(completed / total_jobs, f"RENDERING  {name.upper()}…")
            try:
                # Render stem using snapshotted data (no locks needed)
                raw   = stems_snap[name]
                if raw.size == 0:
                    print(f"Export skipped for {name}: nothing in the selected region")
                    completed += 1
                    continue
                sr_i  = sr_snap
                chunk = apply_eq(raw.copy(), sr_i, eq_snap[name])
                if name == "vocals" and vff_snap:
                    chunk = apply_vff(chunk)
                chunk = apply_stem_width(chunk, width_snap[name])
                chunk = apply_air(chunk, air_snap[name], sr_i)
                chunk = chunk * vol_snap[name]

                rev_amt = rev_snap[name]
                if rev_amt > 0.001:
                    export_reverb = _SharedReverb()
                    export_reverb._reset(sr_i)
                    block  = 4096
                    n_samp = len(chunk)
                    out_rv = np.zeros_like(chunk)
                    for i in range(0, n_samp, block):
                        blk          = chunk[i:i+block]
                        wet          = export_reverb.process(
                                           np.clip(blk, -1.0, 1.0) * rev_amt, sr_i)
                        out_rv[i:i+block] = blk + wet
                    chunk = out_rv

                np.clip(chunk, -1.0, 1.0, out=chunk)

                cur_hash  = hash_snap[name]
                prev_hash = prev_hashes.get(name)
                changed   = (prev_hash is not None and cur_hash != prev_hash)

                for fmt_idx, (fmt, ext) in enumerate(formats):
                    _set_progress(
                        (completed + fmt_idx / len(formats)) / total_jobs,
                        f"WRITING  {name.upper()}  ({ext[1:].upper()})…"
                    )
                    base_path = os.path.join(state.export_folder, f"{base}_{name}{loop_suffix}{ext}")
                    if changed:
                        n_suf = 1
                        while True:
                            candidate = os.path.join(
                                state.export_folder, f"{base}_{name}{loop_suffix}{n_suf}{ext}")
                            if not os.path.exists(candidate):
                                break
                            n_suf += 1
                        out_path = candidate
                    else:
                        out_path = base_path
                    _write_stem_file(chunk, sr_snap, out_path, fmt)

                state._stem_export_state[name] = cur_hash
                completed += 1
                _set_progress(completed / total_jobs)

            except Exception as e:
                print(f"Export error for {name}: {e}")
                completed += 1

        _set_progress(1.0, "DONE ✓")
        # Hide progress bar after a short delay
        if running:
            app.after(1500, lambda: _show_export_ui(False))
            app.after(1500, lambda: _set_progress(0.0, ""))
        state._export_running = False

    state._export_running = True
    threading.Thread(target=_do_export, daemon=True).start()

# ----------------------------
# WAVEFORM / SEEKBAR + PROGRESS BAR (shared slot)
# ----------------------------
WAVE_W, WAVE_H = 1760, 110

# wave_slot, wave_canvas, and progress_bar are created after _sf (the
# scrollable inner frame) is available — see _build_wave_widgets() below.
wave_slot    = None   # replaced after _sf is created
wave_canvas  = None   # replaced after _sf is created
progress_bar     = None   # replaced after _sf is created
progress_pct_lbl = None   # "42%" readout above the bar


def _set_progress(frac):
    """Move the bar and the percentage together."""
    try:
        progress_bar.set(frac)
        pct = max(0, min(100, int(round(frac * 100))))
        progress_pct_lbl.configure(text=f"{pct}%")
        progress_pct_lbl.lift()
    except Exception:
        pass


_RESTART_GAP_S = 0.3   # pause between "STOPPING PREVIOUS…" and the new run


def _set_progress_text(msg):
    """Put a word in place of the percentage, for the brief waiting state."""
    try:
        progress_pct_lbl.configure(text=msg)
        progress_pct_lbl.lift()
    except Exception:
        pass


def _clear_progress():
    try:
        progress_bar.set(0)
        progress_pct_lbl.configure(text="")
        progress_pct_lbl.lower()
    except Exception:
        pass

# Scanline overlay — drawn once when waveform is first shown
_scanlines_drawn = False

def _ensure_scanline():
    """Draw subtle horizontal-line texture across the waveform canvas — once only."""
    global _scanlines_drawn
    if _scanlines_drawn:
        return
    for y in range(0, WAVE_H, 3):
        wave_canvas.create_line(0, y, WAVE_W, y,
                                fill="#0d0d0d", tags="scanlines")
    _scanlines_drawn = True

# ----------------------------
# WAVEFORM ZOOM
# zoom_level: 1.0 = full file visible; 4.0 = quarter of file visible
# zoom_offset: fraction (0.0–1.0) of the file at the left edge of the view
# ----------------------------
_zoom_level  = 1.0   # 1.0 = no zoom
_zoom_offset = 0.0   # left-edge state.position as fraction of total length

def _zoom_frac_to_sample(frac: float) -> int:
    """Convert a 0–1 fraction of the *visible* waveform window to a sample index."""
    length = _playable_length()
    if length == 0:
        return 0
    visible_frac = 1.0 / max(_zoom_level, 1.0)
    sample_frac  = _zoom_offset + frac * visible_frac
    return int(max(0.0, min(1.0, sample_frac)) * length)

# Seek on click or drag
def _seek_from_event(event):
    if _playable_length() == 0:
        return
    frac = max(0.0, min(1.0, event.x / WAVE_W))
    with audio_lock:
        state.position = _zoom_frac_to_sample(frac)

# ----------------------------
# LOOP REGION — right-click drag to set, double-right-click to clear
# ----------------------------
_loop_drag_x0: Optional[float] = None   # pixel x where drag started

def _px_to_sample(px: float) -> int:
    return _zoom_frac_to_sample(max(0.0, min(1.0, px / WAVE_W)))

def _zoom_scroll(event):
    """Ctrl+scroll on the waveform to zoom in/out."""
    global _zoom_level, _zoom_offset
    if _playable_length() == 0:
        return
    factor = 1.15 if (event.delta > 0 or event.num == 4) else 1.0 / 1.15
    pivot  = max(0.0, min(1.0, event.x / WAVE_W))   # zoom around cursor
    old_visible = 1.0 / max(_zoom_level, 1.0)
    _zoom_level  = max(1.0, min(32.0, _zoom_level * factor))
    new_visible = 1.0 / _zoom_level
    # Adjust offset so the pixel under cursor stays fixed
    _zoom_offset = _zoom_offset + pivot * (old_visible - new_visible)
    _zoom_offset = max(0.0, min(1.0 - new_visible, _zoom_offset))
    # Force waveform redraw
    global _last_waveform_id
    _last_waveform_id = None

def _loop_drag_start(event):
    global _loop_drag_x0
    _loop_drag_x0 = float(event.x)

def _loop_drag_move(event):
    global _loop_drag_x0
    if _loop_drag_x0 is None or _playable_length() == 0:
        return
    x0 = _loop_drag_x0
    x1 = float(event.x)
    s  = _px_to_sample(min(x0, x1))
    e  = _px_to_sample(max(x0, x1))
    if e > s:
        state.loop_start = s
        state.loop_end   = e

def _loop_drag_end(event):
    global _loop_drag_x0
    if _loop_drag_x0 is None or _playable_length() == 0:
        _loop_drag_x0 = None
        return
    x0 = _loop_drag_x0
    x1 = float(event.x)
    s  = _px_to_sample(min(x0, x1))
    e  = _px_to_sample(max(x0, x1))
    if e > s + 100:          # ignore accidental micro-drags
        state.loop_start = s
        state.loop_end   = e
    else:
        state.loop_start = None
        state.loop_end   = None
    _loop_drag_x0 = None

def _loop_clear(event=None):
    state.loop_start = None
    state.loop_end   = None

def _draw_loop_overlay():
    """Draw the translucent loop-region overlay on the waveform canvas."""
    wave_canvas.delete("loop_overlay")
    if state.loop_start is None or state.loop_end is None:
        return
    length = _playable_length()
    if length == 0:
        return
    x0 = int(state.loop_start / length * WAVE_W)
    x1 = int(state.loop_end   / length * WAVE_W)
    # Filled semi-transparent rectangle (stipple gives translucency in tk)
    wave_canvas.create_rectangle(x0, 0, x1, WAVE_H,
                                  fill="#003366", stipple="gray25",
                                  outline="", tags="loop_overlay")
    # Bright boundary lines
    wave_canvas.create_line(x0, 0, x0, WAVE_H, fill="#0088ff",
                             width=2, tags="loop_overlay")
    wave_canvas.create_line(x1, 0, x1, WAVE_H, fill="#0088ff",
                             width=2, tags="loop_overlay")

def _build_wave_widgets(parent):
    """Create wave_slot, wave_canvas, and progress_bar inside *parent* (_sf).
    Called once after the scrollable inner frame is ready.
    """
    global wave_slot, wave_canvas, progress_bar, progress_pct_lbl
    wave_slot = tk.Frame(parent, bg=BG, width=WAVE_W, height=WAVE_H)
    wave_slot.pack(pady=(12, 4), padx=20)
    wave_slot.pack_propagate(False)

    wave_canvas = tk.Canvas(wave_slot, width=WAVE_W, height=WAVE_H,
                             bg="#060606", highlightthickness=1,
                             highlightbackground=STEEL)
    # Canvas is hidden initially — shown only after a file loads

    progress_bar = ctk.CTkProgressBar(wave_slot,
                                       progress_color=GLOW_RED,
                                       fg_color=PANEL,
                                       corner_radius=0,
                                       height=10)
    progress_bar.set(0)
    progress_bar.pack(fill="x", padx=4, pady=(WAVE_H // 2 - 5))

    # Separation percentage, centred just above the bar. Placed rather than
    # packed so it floats over the slot without moving anything else.
    progress_pct_lbl = tk.Label(wave_slot, text="", bg=BG,
                                fg=GLOW_RED, font=("Courier New", 17, "bold"))
    progress_pct_lbl.place(relx=0.5, y=WAVE_H // 2 - 22, anchor="center")
    progress_pct_lbl.lower()   # hidden until a separation starts

    wave_canvas.bind("<ButtonPress-1>",   _seek_from_event)
    wave_canvas.bind("<B1-Motion>",       _seek_from_event)
    wave_canvas.bind("<ButtonPress-3>",   _loop_drag_start)
    wave_canvas.bind("<B3-Motion>",       _loop_drag_move)
    wave_canvas.bind("<ButtonRelease-3>", _loop_drag_end)
    wave_canvas.bind("<Double-Button-3>", _loop_clear)
    wave_canvas.bind("<Control-MouseWheel>", _zoom_scroll)
    wave_canvas.bind("<Control-Button-4>",   _zoom_scroll)
    wave_canvas.bind("<Control-Button-5>",   lambda e: _zoom_scroll(
        type('E', (), {'delta': -1, 'num': 5, 'x': e.x})()))

# Cache: track last drawn state so we skip redundant redraws
_last_waveform_id = None   # id of state.waveform_data last drawn (id() changes on new load)
_last_head_x      = -1.0   # last playhead x state.position drawn

def _draw_static_waveform():
    """Draw the static waveform bars — only called when waveform_data changes."""
    global _last_waveform_id
    if state.waveform_data is None or _playable_length() == 0:
        return
    w, h   = WAVE_W, WAVE_H
    mid    = h // 2
    n      = len(state.waveform_data)
    length = _playable_length()
    head_x = (state.position / length) * w if length else 0

    # Map zoom window onto waveform_data indices
    visible_frac = 1.0 / max(_zoom_level, 1.0)
    i_start = int(_zoom_offset * n)
    i_end   = min(n, int((_zoom_offset + visible_frac) * n))
    if i_end <= i_start:
        i_end = i_start + 1

    # Playhead in pixel space within zoom window
    head_frac  = (state.position / length) if length else 0
    head_norm  = (head_frac - _zoom_offset) / visible_frac
    head_px    = head_norm * w

    wave_canvas.delete("waveform")
    played_coords   = []
    unplayed_coords = []
    visible_n = i_end - i_start
    for rel_i, abs_i in enumerate(range(i_start, i_end)):
        x   = rel_i / visible_n * w
        amp = state.waveform_data[abs_i] * 48
        if x <= head_px:
            played_coords.extend([x, mid - amp, x, mid + amp])
        else:
            unplayed_coords.extend([x, mid - amp, x, mid + amp])

    if played_coords:
        wave_canvas.create_line(played_coords,   fill=BRIGHT_RED, tags="waveform")
    if unplayed_coords:
        wave_canvas.create_line(unplayed_coords, fill=RED,        tags="waveform")
    wave_canvas.create_line(0, mid, w, mid, fill=STEEL, width=1, tags="waveform")

    # Time ticks — label them relative to visible window
    for t in range(0, 11):
        tx = t / 10 * w
        wave_canvas.create_line(tx, h-8, tx, h, fill=STEEL_LIGHT, tags="waveform")

    # Zoom indicator in top-right corner
    if _zoom_level > 1.05:
        wave_canvas.create_text(w - 4, 4, anchor="ne",
                                text=f"×{_zoom_level:.1f}",
                                fill=GLOW_RED, font=("Courier New", 9),
                                tags="waveform")

    _last_waveform_id = id(state.waveform_data)
    _ensure_scanline()

# ----------------------------
# TRACK NAME FLASH
# When a track finishes loading, its name fades up over the waveform and back
# out again across one second.
#
# A Tk canvas item has no alpha channel, so the fade is done by interpolating
# the text colour between the canvas background and white — against the dark
# waveform that reads as a fade. The item is recreated on every frame because
# the waveform redraw clears and rebuilds the canvas underneath it.
# ----------------------------
_BANNER_MS      = 1000    # total time on screen
_BANNER_STEP_MS = 40      # frame interval
_banner_job     = [None]  # pending after() id, so a new load cancels the old


def _banner_colour(alpha):
    """Blend the flash colour toward the background as alpha falls to 0."""
    bg = (0x0a, 0x0a, 0x0a)
    fg = (0xff, 0xff, 0xff)
    return "#%02x%02x%02x" % tuple(
        int(b + (f - b) * alpha) for b, f in zip(bg, fg))


def _flash_track_name(name):
    """Fade *name* in and out over the waveform, once."""
    if not running or not name:
        return
    if _banner_job[0] is not None:
        try:
            app.after_cancel(_banner_job[0])
        except Exception:
            pass
        _banner_job[0] = None
    wave_canvas.delete("track_banner")

    steps = max(1, _BANNER_MS // _BANNER_STEP_MS)

    def _frame(i=0):
        _banner_job[0] = None
        wave_canvas.delete("track_banner")
        if not running or i >= steps:
            return
        t = i / steps
        # Up over the first third, hold briefly, then down.
        if t < 0.30:
            alpha = t / 0.30
        elif t < 0.45:
            alpha = 1.0
        else:
            alpha = max(0.0, 1.0 - (t - 0.45) / 0.55)

        colour = _banner_colour(alpha)
        # A dim copy behind the text keeps it legible over bright waveform.
        shadow = _banner_colour(alpha * 0.35)
        for dx, dy, fill in ((2, 2, shadow), (0, 0, colour)):
            wave_canvas.create_text(
                WAVE_W // 2 + dx, WAVE_H // 2 + dy,
                text=name, fill=fill,
                font=("Courier", 39, "bold"),
                tags="track_banner")
        wave_canvas.tag_raise("track_banner")
        _banner_job[0] = app.after(_BANNER_STEP_MS, _frame, i + 1)

    _frame()


def draw_waveform():
    global _last_head_x, _last_waveform_id
    if not running:
        return

    if state.waveform_data is not None and _playable_length() > 0:
        w, h   = WAVE_W, WAVE_H
        length = _playable_length()
        head_x = (state.position / length) * w if length else 0

        # Redraw full static waveform only when a new file was loaded
        # or the playhead has moved more than 1 px (recolours played region).
        if id(state.waveform_data) != _last_waveform_id or abs(head_x - _last_head_x) >= 1.0:
            _draw_static_waveform()
            _last_head_x = head_x
        # Playhead glow — lightweight: just delete+redraw 5 lines
        wave_canvas.delete("playhead")
        for offset, alpha_color in [(-2, "#330000"), (-1, "#660000"),
                                     (0, GLOW_RED),
                                     (1, "#660000"), (2, "#330000")]:
            wave_canvas.create_line(head_x + offset, 0,
                                    head_x + offset, h,
                                    fill=alpha_color,
                                    width=1 if offset != 0 else 2,
                                    tags="playhead")
        # Loop overlay (redraws cheaply since it only touches tagged items)
        _draw_loop_overlay()
        # Keep scanlines on top, and the track-name flash above them
        wave_canvas.tag_raise("scanlines")
        wave_canvas.tag_raise("track_banner")

    if running:
        app.after(80, draw_waveform)  # 80 ms ≈ 12 fps — plenty for a playhead

draw_waveform()

# ----------------------------
# VOLUME METERS
# Each cell registers its Canvas here; _update_meters() redraws them all.
# Design: narrow vertical bar (8px wide), blood-red fill with a glow
# gradient, bright peak-hold tick that decays over ~1 second.
# ----------------------------
METER_W  = 14   # px wide — two channels side by side
_METER_GAP = 2  # px between the left and right bars
METER_H  = 60   # px tall
METER_MS = 80   # redraw interval — matches waveform loop

_meter_canvases = {}   # stem_key -> tk.Canvas
_meter_items    = {}   # stem_key -> canvas item ids, created once
_peak_hold      = {}   # stem_key -> [left, right] peak, decays each tick
_PEAK_DECAY     = 0.85 # multiplier per tick (~1 s to silence at 80 ms)

def make_meter(parent, stem_key):
    """Create and register a stereo volume meter for *stem_key*.

    Two bars side by side, left and right, each with its own peak-hold tick.
    The canvas items are created once here and merely moved on each tick;
    deleting and rebuilding a dozen canvases every 80 ms was a visible part
    of the scrolling lag.
    """
    c = tk.Canvas(parent,
                  width=METER_W, height=METER_H,
                  bg="#060606",
                  highlightthickness=1,
                  highlightbackground=BORDER)

    bar_w = (METER_W - _METER_GAP) // 2
    items = {"bar_w": bar_w}
    for ch in (0, 1):
        x0 = ch * (bar_w + _METER_GAP)
        x1 = x0 + bar_w
        items[f"fill{ch}"] = c.create_rectangle(x0, METER_H, x1, METER_H,
                                                fill="#550000", outline="")
        items[f"cap{ch}"]  = c.create_rectangle(x0, METER_H, x1, METER_H,
                                                fill=GLOW_RED, outline="",
                                                state="hidden")
        items[f"peak{ch}"] = c.create_rectangle(x0, METER_H, x1, METER_H,
                                                fill=GLOW_RED, outline="",
                                                state="hidden")

    _meter_canvases[stem_key] = c
    _meter_items[stem_key]    = items
    _peak_hold[stem_key]      = [0.0, 0.0]
    return c


def _meter_scale(raw):
    """Soft-log scale, so quiet signals are still visible."""
    raw = min(max(float(raw), 0.0), 1.0)
    return float(np.log1p(raw * 9.0) / np.log1p(9.0)) if raw > 0 else 0.0


def _update_meters():
    if not running:
        return
    for key, canvas in _meter_canvases.items():
        items = _meter_items.get(key)
        if items is None:
            continue
        raw = state._meter_levels.get(key, (0.0, 0.0))
        if not isinstance(raw, (tuple, list)):
            raw = (raw, raw)             # tolerate a mono value
        holds = _peak_hold.setdefault(key, [0.0, 0.0])
        if not isinstance(holds, list):
            holds = [float(holds), float(holds)]
            _peak_hold[key] = holds
        bar_w = items["bar_w"]

        for ch in (0, 1):
            level = _meter_scale(raw[ch] if ch < len(raw) else 0.0)
            pk = holds[ch]
            pk = level if level >= pk else pk * _PEAK_DECAY
            holds[ch] = pk

            x0 = ch * (bar_w + _METER_GAP)
            x1 = x0 + bar_w
            fill_h = int(level * METER_H)
            top    = METER_H - fill_h

            colour = (BRIGHT_RED if level > 0.75 else
                      "#8B0000"  if level > 0.40 else "#550000")
            canvas.coords(items[f"fill{ch}"], x0, top, x1, METER_H)
            canvas.itemconfig(items[f"fill{ch}"], fill=colour)

            if fill_h >= 2:
                canvas.coords(items[f"cap{ch}"], x0, top, x1, top + 2)
                canvas.itemconfig(items[f"cap{ch}"], state="normal")
            else:
                canvas.itemconfig(items[f"cap{ch}"], state="hidden")

            if pk > 0.01:
                py = int((1.0 - pk) * METER_H)
                canvas.coords(items[f"peak{ch}"], x0, py, x1, py + 2)
                canvas.itemconfig(items[f"peak{ch}"], state="normal")
            else:
                canvas.itemconfig(items[f"peak{ch}"], state="hidden")

    if running:
        app.after(METER_MS, _update_meters)

_update_meters()




def apply_fv_vff(chunk):
    """Vocal Focus Filter for the Front Vocals stem."""
    return _vff_process(chunk, state.fv_vff_enabled,
                        float(state.fv_vff_lead_cut), float(state.fv_vff_body_cut),
                        float(state.fv_vff_presence), float(state.fv_vff_bkg_vol), "fv")


def _resample_audio(audio, file_sr):
    """Resample stereo float32 array to the project sample rate (44100 Hz).

    Uses the global ``state.sr`` when set (always 44100 after a track is loaded),
    falling back to 44100 directly so import state.stems loaded before the first
    track are still resampled correctly.
    """
    target_sr = int(state.sr) if state.sr is not None else 44100
    if int(file_sr) == target_sr:
        return audio
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    g    = _gcd(target_sr, int(file_sr))
    up   = target_sr     // g
    down = int(file_sr)  // g
    ch0  = resample_poly(audio[:, 0], up, down).astype(np.float32)
    ch1  = resample_poly(audio[:, 1], up, down).astype(np.float32)
    return np.stack([ch0, ch1], axis=1)


_ATMOS_KEYS = ("atmos_fl", "atmos_fr", "atmos_c", "atmos_lfe", "atmos_bl", "atmos_br")


def _all_import_btns():
    """Every import button that exists right now.

    Cells come and go (ANY+ and ANY++ were removed), and their globals stay
    behind as None. Enabling one of those raised inside the load-completion
    block, which is why the transport never lit up after a separation.
    """
    names = ("bgv_import_btn", "fv_import_btn", "hl_import_btn",
             "any_import_btn", "any_plus_import_btn",
             "any_plusplus_import_btn")
    out = [globals().get(n) for n in names]
    out += _atmos_import_btns()
    return [b for b in out if b is not None]


def _atmos_import_btns():
    """The ATMOS import buttons that have been built so far."""
    out = []
    for k in _ATMOS_KEYS:
        b = globals().get(f"{k}_import_btn")
        if b is not None:
            out.append(b)
    return out


def _read_audio_file(path):
    """Read an audio file and return stereo float32 array + original sr.
    Uses soundfile for most formats; falls back to pydub for .m4a and
    other container formats that soundfile cannot decode.
    """
    ext = os.path.splitext(path)[1].lower()
    # Containers soundfile cannot open: pydub hands them to FFmpeg, which
    # pulls the audio track out of a video file just as readily as an m4a.
    if ext in (".m4a", ".aac", ".mp4", ".m4v", ".mov", ".mkv", ".webm"):
        try:
            from pydub import AudioSegment
            seg     = AudioSegment.from_file(path)
            file_sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples /= float(1 << (seg.sample_width * 8 - 1))
            if seg.channels == 1:
                audio = np.stack([samples, samples], axis=1)
            else:
                audio = samples.reshape(-1, seg.channels)[:, :2]
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Could not read {ext}: FFmpeg was not found. {ext} holds its "
                f"audio in a container that needs FFmpeg to unpack. Install it "
                f"from https://ffmpeg.org and make sure ffmpeg.exe is on PATH, "
                f"then try again. ({e})")
        except Exception as e:
            raise RuntimeError(
                f"Could not read {ext} file: {e}. These formats need pydub and "
                f"FFmpeg; a missing or broken FFmpeg is the usual cause.")
    else:
        audio, file_sr = sf.read(path, always_2d=True)
        audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1 or audio.shape[1] == 1:
        audio = np.repeat(audio.reshape(-1, 1), 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio, file_sr


def load_front_vocals():
    """Import Front Vocals from a file.

    No longer wired to a button: the FRT VOX cell is the lead half of the
    karaoke split (see separate_bg_vocals). Kept so presets and any other
    caller still work.
    """
    if _playable_length() == 0 or state.separating:
        return
    path = filedialog.askopenfilename(
        title="Import Front Vocals",
        initialdir=state.last_fv_dir,
        filetypes=[("Audio files", "*.wav *.flac *.mp3 *.ogg *.aif *.aiff *.m4a *.aac *.mp4 *.m4v *.mov"), ("All files", "*.*")]
    )
    if not path:
        return
    state.last_fv_dir = os.path.dirname(path)
    try:
        audio, file_sr = _read_audio_file(path)
        state.fv_data = _resample_audio(audio, file_sr)
        state.fv_sr   = state.sr
        name_no_ext = os.path.splitext(os.path.basename(path))[0]
        fv_import_btn.configure(text=f"⬡ {name_no_ext[:18]}")
    except Exception as e:
        print("Front Vocals load error:", e)




def apply_bgv_vff(chunk):
    """Vocal Focus Filter for the BG Vocals stem."""
    return _vff_process(chunk, state.bgv_vff_enabled,
                        float(state.bgv_vff_lead_cut), float(state.bgv_vff_body_cut),
                        float(state.bgv_vff_presence), float(state.bgv_vff_bkg_vol), "bgv")


def load_bg_vocals():
    """Import BG Vocals from a file.

    No longer wired to a button: the BG VOX cell is produced by the
    karaoke model (see separate_bg_vocals). Kept so presets and any other
    caller still work, and so the old behaviour is one line away.
    """
    if _playable_length() == 0 or state.separating:
        return
    path = filedialog.askopenfilename(
        title="Import BG Vocals",
        initialdir=state.last_bgv_dir,
        filetypes=[("Audio files", "*.wav *.flac *.mp3 *.ogg *.aif *.aiff *.m4a *.aac *.mp4 *.m4v *.mov"), ("All files", "*.*")]
    )
    if not path:
        return
    state.last_bgv_dir = os.path.dirname(path)
    try:
        audio, file_sr = _read_audio_file(path)
        state.bg_vocals_data = _resample_audio(audio, file_sr)
        state.bg_vocals_sr   = state.sr
        name_no_ext = os.path.splitext(os.path.basename(path))[0]
        bgv_import_btn.configure(text=f"⬡ {name_no_ext[:18]}")
    except Exception as e:
        print("BG Vocals load error:", e)




def apply_hl_vff(chunk):
    """Vocal Focus Filter for the Hidden Layer stem."""
    return _vff_process(chunk, state.hl_vff_enabled,
                        float(state.hl_vff_lead_cut), float(state.hl_vff_body_cut),
                        float(state.hl_vff_presence), float(state.hl_vff_bkg_vol), "hl")

# ----------------------------
# LEVELLER — shared per-stem dynamics
# Lifts quiet parts, lowers loud parts using an RMS pivot.
# State stored in dicts keyed by stem name.
# ----------------------------


def apply_leveller(chunk: np.ndarray,
                   enabled: bool,
                   threshold: float,
                   amount: float) -> np.ndarray:
    """Upward/downward leveller.

    Computes a short-term RMS for the chunk, then derives a gain that:
      - lifts samples below the threshold  (quiet parts get louder)
      - attenuates samples above the threshold  (loud parts get quieter)

    THR sets the pivot RMS level (0–1 linear).
    AMT controls how aggressively the gain is applied (0–1).
    At AMT=0 the signal is unchanged; at AMT=1 levels are pulled fully
    towards the threshold.  Gain is soft-clipped to ±12/–18 dB.
    """
    if not enabled or len(chunk) == 0:
        return chunk
    mono = (chunk[:, 0] + chunk[:, 1]) * 0.5
    rms  = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
    if rms <= 1e-9:
        return chunk
    gain = 1.0 + (threshold / rms - 1.0) * float(amount)
    gain = max(0.125, min(gain, 4.0))
    return chunk * np.float32(gain)


def apply_hl_leveller(chunk):
    """Leveller for the Hidden Layer stem (legacy wrapper)."""
    return apply_leveller(chunk,
                          state.stem_lvl_enabled.get("hidden_layer", False),
                          state.stem_lvl_threshold.get("hidden_layer", 0.3),
                          state.stem_lvl_amount.get("hidden_layer", 0.5))


# ----------------------------
# LIMITER — per-stem brick-wall limiter
#
# Loudness maximiser design (Waves L1/L2 style):
#
#   How it should feel to the user:
#     • Threshold — input sensitivity.  Lowering it increases the makeup gain
#       applied to the signal, making it louder.  Think of it as "how hard you
#       push the signal into the limiter".
#     • Ceiling — absolute brick-wall output cap.  The signal will never exceed
#       this level regardless of threshold or input level.
#
#   Implementation — two stages per block:
#     1. MAKEUP GAIN  (static, computed once from threshold + ceiling)
#        makeup = ceil_lin / thresh_lin
#        This boosts the signal so that a peak exactly at threshold_db ends up
#        exactly at ceiling_db.  Lowering the threshold raises this gain.
#
#     2. GAIN REDUCTION  (dynamic, envelope follower)
#        After the makeup boost, peaks that still exceed ceil_lin are brought
#        back down by a fast-attack envelope.  This is the true limiting action.
#        Attack: 0.1 ms (near-instant — catches transients before they clip)
#        Release: 50 ms (smooth recovery, no pumping on programme material)
#
#     3. HARD CLIP  (safety net)
#        np.clip to ±ceil_lin as an absolute final safety net.
#
# State is stored in _lim_state keyed by stem key.
# ----------------------------
_lim_state: dict = {}   # stem_key -> {"env": float}

def apply_limiter(chunk: np.ndarray,
                  key:   str,
                  enabled:      bool,
                  threshold_db: float,
                  ceiling_db:   float,
                  sr_now:       int) -> np.ndarray:
    """Loudness-maximiser limiter.

    threshold_db — lower = more makeup gain = louder output.
                   At -6 dB the signal is boosted by 6 dB (minus ceiling offset)
                   before the brick wall kicks in.
    ceiling_db   — absolute output ceiling; signal never exceeds this level.
    """
    if not enabled or len(chunk) == 0:
        return chunk

    # Clamp ceiling so it can never exceed 0 dBFS (true peak safety)
    ceiling_db   = min(ceiling_db, 0.0)
    # Threshold must be ≤ ceiling (can't push harder than the wall)
    threshold_db = min(threshold_db, ceiling_db)

    ceil_lin   = 10.0 ** (ceiling_db   / 20.0)
    thresh_lin = 10.0 ** (threshold_db / 20.0)

    # --- Stage 1: static makeup gain ---
    # Raises the whole signal so a peak at thresh_lin lands at ceil_lin.
    makeup = ceil_lin / max(thresh_lin, 1e-9)

    out  = chunk.copy().astype(np.float32)
    out *= makeup   # apply makeup before limiting

    # --- Stage 2: dynamic gain reduction (brick-wall envelope) ---
    sr_f  = float(sr_now)

    def _tc(ms):
        return float(np.exp(-1.0 / max(sr_f * ms * 0.001, 1.0)))

    atk_c = _tc(0.1)   # 0.1 ms — catches transients before they clip
    rel_c = _tc(50.0)  # 50 ms  — smooth, pump-free recovery

    st  = _lim_state.setdefault(key, {"env": 0.0})
    env = st["env"]

    BLOCK = 64
    peaks   = _block_peak(out, BLOCK)
    env_arr = _env_follow(peaks, env, atk_c, rel_c)
    env     = float(env_arr[-1]) if len(env_arr) else env
    # Blocks whose envelope stays under the ceiling are left alone.
    gains = np.where(env_arr > ceil_lin,
                     ceil_lin / np.maximum(env_arr, 1e-9), 1.0)
    _apply_block_gain(out, gains, BLOCK)

    st["env"] = env

    # --- Stage 3: hard clip (absolute safety net) ---
    np.clip(out, -ceil_lin, ceil_lin, out=out)
    return out


def load_hidden_layer():
    """Let the user pick an audio file to use as the Hidden Layer stem."""
    if _playable_length() == 0 or state.separating:
        return
    path = filedialog.askopenfilename(
        title="Import Hidden Layer",
        initialdir=state.last_hl_dir,
        filetypes=[("Audio files", "*.wav *.flac *.mp3 *.ogg *.aif *.aiff *.m4a *.aac *.mp4 *.m4v *.mov"), ("All files", "*.*")]
    )
    if not path:
        return
    state.last_hl_dir = os.path.dirname(path)
    try:
        audio, file_sr = _read_audio_file(path)
        state.hl_data = _resample_audio(audio, file_sr)
        state.hl_sr   = state.sr
        name_no_ext = os.path.splitext(os.path.basename(path))[0]
        hl_import_btn.configure(text=f"⬡ {name_no_ext[:18]}")
    except Exception as e:
        print("Hidden Layer load error:", e)




# ── ATMOS bed loaders ──────────────────────────────────────────────────────
# One per channel, same as the ANY slots: pick a file, resample it to the
# session rate and hand it to the mixer, which plays it from that moment on.
def _make_atmos_loader(key, label, btn_name):
    def _load():
        # Deliberately no "is a track loaded?" test: an ATMOS bed is imported
        # on its own, with nothing separated.
        if state.separating:
            return
        path = filedialog.askopenfilename(
            title=f"Import — {label}",
            initialdir=getattr(state, f"last_{key}_dir", None),
            filetypes=[("Audio files",
                        "*.wav *.flac *.mp3 *.ogg *.aif *.aiff *.m4a *.aac "
                        "*.mp4 *.m4v *.mov"),
                       ("All files", "*.*")])
        if not path:
            return
        setattr(state, f"last_{key}_dir", os.path.dirname(path))
        try:
            audio, file_sr = _read_audio_file(path)
            setattr(state, f"{key}_data", _resample_audio(audio, file_sr))
            setattr(state, f"{key}_sr", state.sr)
            name_no_ext = os.path.splitext(os.path.basename(path))[0]
            btn = globals().get(btn_name)
            if btn is not None:
                btn.configure(text=f"⬡ {name_no_ext[:18]}")
            # With nothing separated, imported audio is the track: give it a
            # waveform and a working transport.
            _refresh_import_waveform()
            _refresh_transport_state()
        except Exception as e:
            print(f"{label} load error:", e)
    return _load


load_atmos_fl = _make_atmos_loader("atmos_fl", "ATMOS FL", "atmos_fl_import_btn")
load_atmos_fr = _make_atmos_loader("atmos_fr", "ATMOS FR", "atmos_fr_import_btn")
load_atmos_c = _make_atmos_loader("atmos_c", "ATMOS C", "atmos_c_import_btn")
load_atmos_lfe = _make_atmos_loader("atmos_lfe", "ATMOS LFE", "atmos_lfe_import_btn")
load_atmos_bl = _make_atmos_loader("atmos_bl", "ATMOS BL", "atmos_bl_import_btn")
load_atmos_br = _make_atmos_loader("atmos_br", "ATMOS BR", "atmos_br_import_btn")


def load_any():
    """Let the user pick an audio file to use as the Synth stem."""
    if _playable_length() == 0 or state.separating:
        return
    path = filedialog.askopenfilename(
        title="Import — Any",
        initialdir=state.last_any_dir,
        filetypes=[("Audio files", "*.wav *.flac *.mp3 *.ogg *.aif *.aiff *.m4a *.aac *.mp4 *.m4v *.mov"), ("All files", "*.*")]
    )
    if not path:
        return
    state.last_any_dir = os.path.dirname(path)
    try:
        audio, file_sr = _read_audio_file(path)
        state.any_data = _resample_audio(audio, file_sr)
        state.any_sr   = state.sr
        name_no_ext = os.path.splitext(os.path.basename(path))[0]
        any_import_btn.configure(text=f"⬡ {name_no_ext[:18]}")
    except Exception as e:
        print("Synth load error:", e)



def load_any_plus():
    """Let the user pick an audio file to use as the Strings stem."""
    if _playable_length() == 0 or state.separating:
        return
    path = filedialog.askopenfilename(
        title="Import — Any+",
        initialdir=state.last_any_plus_dir,
        filetypes=[("Audio files", "*.wav *.flac *.mp3 *.ogg *.aif *.aiff *.m4a *.aac *.mp4 *.m4v *.mov"), ("All files", "*.*")]
    )
    if not path:
        return
    state.last_any_plus_dir = os.path.dirname(path)
    try:
        audio, file_sr = _read_audio_file(path)
        state.any_plus_data = _resample_audio(audio, file_sr)
        state.any_plus_sr   = state.sr
        name_no_ext = os.path.splitext(os.path.basename(path))[0]
        if any_plus_import_btn is not None:      # cell no longer built
            any_plus_import_btn.configure(text=f"⬡ {name_no_ext[:18]}")
    except Exception as e:
        print("Strings load error:", e)



def load_any_plusplus():
    """Let the user pick an audio file to use as the FX stem."""
    if _playable_length() == 0 or state.separating:
        return
    path = filedialog.askopenfilename(
        title="Import — Any++",
        initialdir=state.last_any_plusplus_dir,
        filetypes=[("Audio files", "*.wav *.flac *.mp3 *.ogg *.aif *.aiff *.m4a *.aac *.mp4 *.m4v *.mov"), ("All files", "*.*")]
    )
    if not path:
        return
    state.last_any_plusplus_dir = os.path.dirname(path)
    try:
        audio, file_sr = _read_audio_file(path)
        state.any_plusplus_data = _resample_audio(audio, file_sr)
        state.any_plusplus_sr   = state.sr
        name_no_ext = os.path.splitext(os.path.basename(path))[0]
        if any_plusplus_import_btn is not None:
            any_plusplus_import_btn.configure(text=f"⬡ {name_no_ext[:18]}")
    except Exception as e:
        print("FX load error:", e)


# ============================================================
# SESSION SAVE / LOAD
# Serialises every mixer parameter to a JSON file so sessions
# survive across restarts.
# ============================================================
_SESSION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ramma_session.json")

def _all_stem_keys():
    base = list(state.stems.keys()) if state.stems else []
    return base + ["front_vocals", "bg_vocals", "hidden_layer",
                   "any", "any+", "any++", "atmos_fl", "atmos_fr", "atmos_c", "atmos_lfe", "atmos_bl", "atmos_br"]

def save_session():
    """Write all current mixer settings to ramma_session.json."""
    keys = _all_stem_keys()
    s = state
    data = {
        "volume_master":  s.volume_master,
        "stereo_width":   s.stereo_width,
        "reverb_master":  s.reverb_master,
        "air_master":     s.air_master,
        "stem_volumes":   {k: s.stem_volumes.get(k, 1.0)  for k in keys},
        "stem_widths":    {k: s.stem_widths.get(k, 1.0)   for k in keys},
        "stem_reverbs":   {k: s.stem_reverbs.get(k, 0.0)  for k in keys},
        "stem_air":       {k: s.stem_air.get(k, 0.0)      for k in keys},
        "stem_pan":       {k: s.stem_pan.get(k, 0.0)      for k in keys},
        "stem_nudge":     {k: s.stem_nudge.get(k, 0)      for k in keys},
        "stem_lvl_enabled":   {k: s.stem_lvl_enabled.get(k, False) for k in keys},
        "stem_lvl_threshold": {k: s.stem_lvl_threshold.get(k, 0.3) for k in keys},
        "stem_lvl_amount":    {k: s.stem_lvl_amount.get(k, 0.5)    for k in keys},
        "stem_lim_enabled":   {k: s.stem_lim_enabled.get(k, False)                         for k in keys},
        "stem_lim_threshold": {k: s.stem_lim_threshold.get(k, _LIM_DEFAULTS["threshold"])  for k in keys},
        "stem_lim_ceiling":   {k: s.stem_lim_ceiling.get(k,   _LIM_DEFAULTS["ceiling"])    for k in keys},
        "stem_mute":      {k: s.stem_mute.get(k, False)   for k in keys},
        "stem_solo":      {k: s.stem_solo.get(k, False)   for k in keys},
        "stem_comp_enabled":  {k: s.stem_comp_enabled.get(k, False)                    for k in keys},
        "stem_comp_thresh":   {k: s.stem_comp_thresh.get(k, _COMP_DEFAULTS["thresh"])  for k in keys},
        "stem_comp_ratio":    {k: s.stem_comp_ratio.get(k, _COMP_DEFAULTS["ratio"])    for k in keys},
        "stem_comp_attack":   {k: s.stem_comp_attack.get(k, _COMP_DEFAULTS["attack"])  for k in keys},
        "stem_comp_release":  {k: s.stem_comp_release.get(k, _COMP_DEFAULTS["release"]) for k in keys},
        "stem_gate_enabled":  {k: s.stem_gate_enabled.get(k, False)                    for k in keys},
        "stem_gate_thresh":   {k: s.stem_gate_thresh.get(k, _GATE_DEFAULTS["thresh"])  for k in keys},
        "stem_gate_attack":   {k: s.stem_gate_attack.get(k, _GATE_DEFAULTS["attack"])  for k in keys},
        "stem_gate_release":  {k: s.stem_gate_release.get(k, _GATE_DEFAULTS["release"]) for k in keys},
        "eq_bands":       {k: s.eq_bands.get(k, [0]*5)    for k in s.eq_bands},
        "debleed":        {k: s.stem_debleed.get(k, {})   for k in keys},
    }
    try:
        with open(_SESSION_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        app.after(0, lambda: _session_status_lbl.configure(text="SESSION SAVED ✓"))
        app.after(2000, lambda: _session_status_lbl.configure(text=""))
    except Exception as e:
        print("Session save error:", e)

# The three import cells were once keyed "synth", "strings" and "fx". A
# session saved under those names is translated on load, so its settings land
# on the cells now called any / any+ / any++ instead of being dropped.
_LEGACY_STEM_KEYS = {"synth": "any", "strings": "any+", "fx": "any++"}


def _migrate_stem_keys(data):
    """Rename legacy stem keys throughout a loaded session/preset dict."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            nk = _LEGACY_STEM_KEYS.get(k, k) if isinstance(k, str) else k
            out[nk] = _migrate_stem_keys(v)
        return out
    if isinstance(data, list):
        return [_LEGACY_STEM_KEYS.get(x, x) if isinstance(x, str) else
                _migrate_stem_keys(x) for x in data]
    return data


def load_session():
    """Restore mixer settings from ramma_session.json."""
    try:
        with open(_SESSION_PATH, "r", encoding="utf-8") as f:
            data = _migrate_stem_keys(json.load(f))
    except Exception as e:
        print("Session load error:", e)
        app.after(0, lambda: _session_status_lbl.configure(text="NO SESSION FILE"))
        app.after(2000, lambda: _session_status_lbl.configure(text=""))
        return

    s = state
    s.volume_master = float(data.get("volume_master", 0.7))
    s.stereo_width  = float(data.get("stereo_width",  1.0))
    s.reverb_master = float(data.get("reverb_master", 0.0))
    s.air_master    = float(data.get("air_master",    0.0))

    for k, v in data.get("stem_volumes",  {}).items(): s.stem_volumes[k]  = float(v)
    for k, v in data.get("stem_widths",   {}).items(): s.stem_widths[k]   = float(v)
    for k, v in data.get("stem_reverbs",  {}).items(): s.stem_reverbs[k]  = float(v)
    for k, v in data.get("stem_air",      {}).items(): s.stem_air[k]      = float(v)
    for k, v in data.get("stem_pan",      {}).items(): s.stem_pan[k]      = float(v)
    for k, v in data.get("stem_nudge",    {}).items(): s.stem_nudge[k]    = int(v)
    for k, v in data.get("stem_lvl_enabled",   {}).items(): s.stem_lvl_enabled[k]   = bool(v)
    for k, v in data.get("stem_lvl_threshold",  {}).items(): s.stem_lvl_threshold[k]  = float(v)
    for k, v in data.get("stem_lvl_amount",     {}).items(): s.stem_lvl_amount[k]     = float(v)
    for k, v in data.get("stem_lim_enabled",   {}).items(): s.stem_lim_enabled[k]   = bool(v)
    for k, v in data.get("stem_lim_threshold", {}).items(): s.stem_lim_threshold[k] = float(v)
    for k, v in data.get("stem_lim_ceiling",   {}).items(): s.stem_lim_ceiling[k]   = float(v)
    for k, v in data.get("stem_mute",     {}).items(): s.stem_mute[k]     = bool(v)
    for k, v in data.get("stem_solo",     {}).items(): s.stem_solo[k]     = bool(v)
    for k, v in data.get("stem_comp_enabled",  {}).items(): s.stem_comp_enabled[k]  = bool(v)
    for k, v in data.get("stem_comp_thresh",   {}).items(): s.stem_comp_thresh[k]   = float(v)
    for k, v in data.get("stem_comp_ratio",    {}).items(): s.stem_comp_ratio[k]    = float(v)
    for k, v in data.get("stem_comp_attack",   {}).items(): s.stem_comp_attack[k]   = float(v)
    for k, v in data.get("stem_comp_release",  {}).items(): s.stem_comp_release[k]  = float(v)
    for k, v in data.get("stem_gate_enabled",  {}).items(): s.stem_gate_enabled[k]  = bool(v)
    for k, v in data.get("stem_gate_thresh",   {}).items(): s.stem_gate_thresh[k]   = float(v)
    for k, v in data.get("stem_gate_attack",   {}).items(): s.stem_gate_attack[k]   = float(v)
    for k, v in data.get("stem_gate_release",  {}).items(): s.stem_gate_release[k]  = float(v)
    for k, v in data.get("eq_bands",      {}).items(): s.eq_bands[k]      = list(v)
    for k, v in data.get("debleed",       {}).items(): s.stem_debleed[k]  = dict(v)

    app.after(0, lambda: _session_status_lbl.configure(text="SESSION LOADED ✓"))
    app.after(2000, lambda: _session_status_lbl.configure(text=""))


# ============================================================
# STEM NUDGE — per-stem time alignment offset (±200 ms)
# A positive nudge delays a stem; negative advances it.
# ============================================================

def _nudged_slice(data: np.ndarray, start: int, frames: int, nudge_samp: int) -> np.ndarray:
    """Read `frames` samples from `data` starting at `start + nudge_samp`,
    wrapping at buffer boundaries.  Returns a (frames, 2) float32 array."""
    buf_len = len(data)
    if buf_len == 0:
        return np.zeros((frames, 2), dtype=np.float32)
    s = (start + nudge_samp) % buf_len
    e = s + frames
    if e <= buf_len:
        chunk = data[s:e].copy()
    else:
        chunk = np.concatenate([data[s:], data[:e - buf_len]])
    if len(chunk) < frames:
        chunk = np.pad(chunk, ((0, frames - len(chunk)), (0, 0)))
    return chunk


# ============================================================
# DE-BLEED — spectral subtraction of one stem from another
# stem_debleed[target] = {source: amount}  where amount 0.0–1.0
# Applied after EQ in mix() via apply_debleed().
# ============================================================

def apply_debleed(chunk: np.ndarray, key: str,
                  source_chunks: dict, sr_now: int) -> np.ndarray:
    """Subtract a fraction of each registered source stem's spectrum
    from `chunk` to reduce separation bleed."""
    mapping = state.stem_debleed.get(key, {})
    if not mapping:
        return chunk
    n   = len(chunk)
    out = chunk.copy().astype(np.float32)
    fft_self = np.fft.rfft(out, axis=0)
    for src_key, amount in mapping.items():
        if abs(amount) < 0.001:
            continue
        src = source_chunks.get(src_key)
        if src is None or len(src) != n:
            continue
        fft_src = np.fft.rfft(src.astype(np.float32), axis=0)
        fft_self -= fft_src * float(amount)
    result = np.fft.irfft(fft_self, n=n, axis=0).astype(np.float32)
    np.clip(result, -1.0, 1.0, out=result)
    return result



# ============================================================
# BOUNCE TO STEM
# Renders FRT + BG + HID + Synth + Strings together (with all
# processing applied) into a single float32 buffer, then asks
# the user which import slot to load it into.
# ============================================================
def bounce_to_stem():
    """Render user-imported stems into a composite buffer and
    slot it back as a new importable stem."""
    if state.stems is None:
        return
    sr_i   = int(state.sr)
    length = next(iter(state.stems.values())).shape[0]

    import_slots = {
        "front_vocals":  ("FRT VOX",     lambda d: _bounce_load(d, "front_vocals")),
        "bg_vocals":     ("BG VOX",      lambda d: _bounce_load(d, "bg_vocals")),
        "hidden_layer":  ("HID LAYER",   lambda d: _bounce_load(d, "hidden_layer")),
        "any":         ("ANY",         lambda d: _bounce_load(d, "any")),
        "any+":       ("ANY+",        lambda d: _bounce_load(d, "any+")),
        "any++":            ("ANY++",       lambda d: _bounce_load(d, "any++")),
    }

    # Ask which slot to write to
    win = ctk.CTkToplevel(app)
    win.title("BOUNCE TO STEM")
    win.geometry("340x295")
    win.configure(fg_color=BG)
    win.attributes("-topmost", True)
    win.lift()

    ctk.CTkLabel(win, text="— BOUNCE TO STEM —",
                 font=FONT_TITLE, text_color=GLOW_RED).pack(pady=(14, 4))
    ctk.CTkLabel(win,
                 text="Renders FRT+BG+HID+SYNTH+STRINGS\ninto one stem. Choose target slot:",
                 font=FONT_SMALL, text_color=TEXT_DIM).pack(pady=(0, 10))

    for key, (label, fn) in import_slots.items():
        def _make_cb(f=fn, w=win):
            def _cb():
                w.destroy()
                threading.Thread(target=lambda: _do_bounce(f, sr_i, length),
                                 daemon=True).start()
            return _cb
        ctk.CTkButton(win, text=label, command=_make_cb(),
                      fg_color=RED, hover_color=BRIGHT_RED,
                      text_color=TEXT_MAIN, font=FONT_SMALL,
                      corner_radius=0, height=28).pack(fill="x", padx=30, pady=3)


def _do_bounce(load_fn, sr_i, length):
    """Background: render the mix of import stems only and call load_fn."""
    BLOCK = 4096
    rendered = np.zeros((length, 2), dtype=np.float32)
    for pos in range(0, length, BLOCK):
        frames = min(BLOCK, length - pos)
        blk = np.zeros((frames, 2), dtype=np.float32)
        # Collect import stems only (no Demucs stems, no master FX)
        for data, vol, key, eq_key in [
            (state.fv_data,      state.fv_volume,      "front_vocals", "front_vocals"),
            (state.bg_vocals_data, state.bg_vocals_volume, "bg_vocals", "bg_vocals"),
            (state.hl_data,      state.hl_volume,      "hidden_layer", "hidden_layer"),
            (state.any_data,   state.any_volume,   "any",        "any"),
            (state.any_plus_data, state.any_plus_volume, "any+",      "any+"),
            (state.any_plusplus_data,      state.any_plusplus_volume,      "any++",           "any++"),
        ]:
            if data is None or vol <= 0:
                continue
            buf_len = len(data)
            s = pos % buf_len
            e = s + frames
            ch = data[s:e].copy() if e <= buf_len else np.concatenate([data[s:], data[:e-buf_len]])
            if len(ch) < frames:
                ch = np.pad(ch, ((0, frames-len(ch)), (0, 0)))
            ch = apply_eq(ch, sr_i, state.eq_bands.get(eq_key, [0]*5))
            ch = apply_stem_width(ch, state.stem_widths.get(key, 1.0))
            ch = apply_pan(ch, state.stem_pan.get(key, 0.0))
            ch = apply_air(ch, state.stem_air.get(key, 0.0), sr_i)
            blk += ch * vol
        rendered[pos:pos+frames] = blk[:frames]
    np.clip(rendered, -1.0, 1.0, out=rendered)
    app.after(0, lambda: load_fn(rendered))


def _bounce_load(data: np.ndarray, key: str):
    """Load bounce result into the appropriate import slot."""
    sr_i = int(state.sr)
    if key == "front_vocals":
        state.fv_data, state.fv_sr = data, sr_i
        fv_import_btn.configure(text="⬡ BOUNCE", text_color=TEXT_MAIN)
    elif key == "bg_vocals":
        state.bg_vocals_data, state.bg_vocals_sr = data, sr_i
        bgv_import_btn.configure(text="⬡ BOUNCE", text_color=TEXT_MAIN)
    elif key == "hidden_layer":
        state.hl_data, state.hl_sr = data, sr_i
        hl_import_btn.configure(text="⬡ BOUNCE")
    elif key == "any":
        state.any_data, state.any_sr = data, sr_i
        any_import_btn.configure(text="⬡ BOUNCE")
    elif key == "any+":
        state.any_plus_data, state.any_plus_sr = data, sr_i
        if any_plus_import_btn is not None:
            any_plus_import_btn.configure(text="⬡ BOUNCE")
    elif key == "any++":
        state.any_plusplus_data, state.any_plusplus_sr = data, sr_i
        if any_plusplus_import_btn is not None:
            any_plusplus_import_btn.configure(text="⬡ BOUNCE")


_eq_window = None   # singleton reference

def _eq_raise(win):
    app.after(60, lambda: _fit_all(rescan=True))
    """Put the EQ window above RAMMA, and keep it there.

    CTkToplevel finishes its own set-up a couple of hundred milliseconds
    after creation (title-bar colour, icon), and that pass can drop the
    window behind its parent — which is why it sometimes opened underneath
    on the second and later opens. Re-assert afterwards rather than only at
    creation. transient() is what actually keeps it above RAMMA; topmost is
    used briefly to win the initial raise, then released so the window does
    not sit on top of every other application.
    """
    def _assert(final=False):
        try:
            if not win.winfo_exists():
                return
            win.deiconify()
            win.lift()
            win.attributes("-topmost", not final)
            if final:
                win.focus_force()
        except Exception:
            pass
    try:
        win.transient(app)
    except Exception:
        pass
    _assert()
    for _delay in (120, 300, 600):
        app.after(_delay, _assert)
    app.after(800, lambda: _assert(final=True))


def open_eq_window():
    global _eq_window
    if not running:
        return
    # #2 — Single instance: if already open, just raise it to front
    if _eq_window is not None and _eq_window.winfo_exists():
        _eq_raise(_eq_window)
        return
    try:
        win = ctk.CTkToplevel(app)
        win.title("EQ MATRIX")
        win.geometry("1140x900")
        win.resizable(True, True)
        win.configure(fg_color=BG)
        # #1 — Always draw on top of RAMMA (see _eq_raise)
        _eq_raise(win)
        # Unbind mousewheel when closed so it doesn't affect main window
        def _on_eq_close():
            # Deliberately does NOT touch the wheel bindings: they are global
            # and shared with the main window (see _wheel_target).
            _close_eq_window(win)
        win.protocol("WM_DELETE_WINDOW", _on_eq_close)
        _eq_window = win

        outer = ctk.CTkFrame(win, fg_color=PANEL,
                              corner_radius=0,
                              border_color=STEEL, border_width=1)
        outer.pack(fill="both", expand=True, padx=6, pady=6)

        # Scrollable canvas inside the outer frame
        _eq_canvas = tk.Canvas(outer, bg=PANEL, highlightthickness=0, bd=0)
        # The application-wide wheel handler routes by window and looks this
        # up by name; as a plain local it could never be found, so the wheel
        # always scrolled the main window instead.
        globals()["_eq_canvas"] = _eq_canvas
        _eq_vscroll = tk.Scrollbar(outer, orient="vertical",
                                    command=_eq_canvas.yview,
                                    bg="#111111", troughcolor="#0f0f0f",
                                    activebackground="#cc0000",
                                    relief="flat", bd=0, width=10)
        _eq_vscroll.pack(side="right", fill="y")
        _eq_canvas.pack(side="left", fill="both", expand=True)
        _eq_canvas.configure(yscrollcommand=_eq_vscroll.set)

        frame = tk.Frame(_eq_canvas, bg=PANEL)
        _eq_frame_id = _eq_canvas.create_window((0, 0), window=frame, anchor="nw")

        # Same approach as the main window: re-laying the contents out on
        # every <Configure> makes CustomTkinter redraw every widget in the
        # matrix for each pixel of a resize. Do it once the drag settles.
        _eq_job = [None]

        def _eq_apply_geometry():
            _eq_job[0] = None
            try:
                _eq_canvas.configure(scrollregion=_eq_canvas.bbox("all"))
                _eq_canvas.itemconfig(_eq_frame_id,
                                      width=_eq_canvas.winfo_width())
            except Exception:
                pass

        def _eq_schedule(delay=90):
            if _eq_job[0] is not None:
                try:
                    app.after_cancel(_eq_job[0])
                except Exception:
                    pass
            _eq_job[0] = app.after(delay, _eq_apply_geometry)

        def _eq_on_configure(event):
            _eq_schedule()

        def _eq_on_canvas_resize(event):
            _eq_schedule()
        def _eq_mousewheel(event):
            if event.num == 4:
                _eq_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                _eq_canvas.yview_scroll(1, "units")
            else:
                _eq_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        frame.bind("<Configure>", _eq_on_configure)
        _eq_canvas.bind("<Configure>", _eq_on_canvas_resize)
        # The wheel is handled by the application-wide binding, which routes
        # to this canvas while the pointer is over this window.

        bands = ["< 200 Hz", "200–500", "500–2k", "2k–6k", "> 6 kHz"]

        # HID LAYER is retired from the mixer, so it has no place here
        # either. state.eq_bands still carries its entry (presets and saved
        # sessions may hold one), hence the filter rather than a delete.
        _EQ_HIDDEN = ("hidden_layer",)
        _EQ_LABELS = {
            "front_vocals": "FRT VOX",
            "bg_vocals":    "BG VOX",
            "any":        "ANY",
            "atmos_fl":    "ATMOS FL",
            "atmos_fr":    "ATMOS FR",
            "atmos_c":    "ATMOS C",
            "atmos_lfe":  "ATMOS LFE",
            "atmos_bl":   "ATMOS BL",
            "atmos_br":   "ATMOS BR",
            "any+":      "ANY+",
            "any++":           "ANY++",
            "instrumental": "INST",
        }
        all_eq_stems = [k for k in state.eq_bands.keys() if k not in _EQ_HIDDEN]
        for stem in all_eq_stems:
            label = _EQ_LABELS.get(stem, stem.upper())
            if stem in ("front_vocals", "bg_vocals", "hidden_layer", "any", "any+", "any++"):
                tk.Frame(frame, bg=STEEL, height=1).pack(fill="x", padx=8, pady=(6, 0))
            ctk.CTkLabel(frame, text=f"── {label} ──",
                         font=FONT_TITLE,
                         text_color=GLOW_RED if stem not in ("front_vocals", "bg_vocals", "hidden_layer", "any", "any+", "any++") else BRIGHT_RED).pack(pady=(8, 0))
            row = ctk.CTkFrame(frame, fg_color=BG,
                               corner_radius=0,
                               border_color=STEEL, border_width=1)
            row.pack(fill="x", padx=8, pady=3)
            for i, band in enumerate(bands):
                col = ctk.CTkFrame(row, fg_color="transparent", corner_radius=0)
                col.pack(side="left", expand=True, fill="both", padx=3)
                ctk.CTkLabel(col, text=band,
                             font=FONT_SMALL,
                             text_color=TEXT_DIM).pack()
                sl = LockedSlider(col, from_=-1, to=1,
                                  button_color=RED,
                                  button_hover_color=GLOW_RED,
                                  progress_color=RED,
                                  fg_color=BORDER,
                                  command=lambda v, s=stem, idx=i:
                                      state.eq_bands[s].__setitem__(idx, float(v)))
                sl.set(state.eq_bands[stem][i])
                sl.pack(fill="x")
    except Exception as e:
        print("EQ window error:", e)

def _close_eq_window(win):
    global _eq_window
    _eq_window = None
    globals()["_eq_canvas"] = None
    try:
        win.destroy()
    except Exception:
        pass

# ============================================================
# HELPER: industrial-style button
# ============================================================
def _btn(parent, text, command, **kw):
    return ctk.CTkButton(
        parent, text=text, command=command,
        fg_color=RED,
        hover_color=BRIGHT_RED,
        text_color=TEXT_MAIN,
        font=FONT_LABEL,
        corner_radius=0,
        border_width=1,
        border_color=BRIGHT_RED,
        height=34,
        **kw
    )

# ============================================================
# TOP CHROME — header bar
# ============================================================
header = ctk.CTkFrame(app, fg_color=PANEL,
                      corner_radius=0,
                      border_color=RED, border_width=2,
                      height=42)
header.pack(fill="x", padx=0, pady=0)
header.pack_propagate(False)

ctk.CTkLabel(header,
             text="R · A · M · M · A  ──  STEM  ENGINE  v0.667.1",
             font=("Courier New", 14, "bold"),
             text_color=GLOW_RED).pack(side="left", padx=20)

ctk.CTkLabel(header,
             text="BS-RoFormer SW · 6-stem",
             font=FONT_SMALL,
             text_color=TEXT_DIM).pack(side="right", padx=20)


def _on_update_click():
    """Check GitHub now, off the UI thread, and say what was found."""
    btn = _update_btn
    try:
        btn.configure(text="CHECKING…", state="disabled")
    except Exception:
        pass

    def _work():
        try:
            check_for_update(quiet=False, manual=True)
        finally:
            if running:
                app.after(0, lambda: btn.configure(text="⟳ UPDATE",
                                                   state="normal"))

    threading.Thread(target=_work, daemon=True).start()


_update_btn = ctk.CTkButton(header, text="⟳ UPDATE", command=_on_update_click,
                            fg_color=STEEL, hover_color=STEEL_LIGHT,
                            text_color="#e6c000",
                            font=("Courier New", 12, "bold"),
                            corner_radius=0, border_width=1,
                            border_color=BORDER, height=26, width=110)
_update_btn.pack(side="right", padx=(0, 8))


def _ask_passphrase(parent, title, prompt):
    """A masked passphrase prompt; returns the text, or None if cancelled."""
    from tkinter import simpledialog
    return simpledialog.askstring(title, prompt, show="•", parent=parent)


_tips_win = [None]


def open_tips():
    """Show the tips, read-only; EDIT unlocks them for the author."""
    win = _tips_win[0]
    if win is not None and win.winfo_exists():
        win.deiconify()
        win.lift()
        return win

    win = ctk.CTkToplevel(app)
    win.title("R·A·M·M·A — TIPS")
    win.geometry("640x560")
    win.minsize(420, 320)
    win.configure(fg_color=BG)
    _tips_win[0] = win

    ctk.CTkLabel(win, text="— TIPS —", font=FONT_TITLE,
                 text_color="#e6c000").pack(pady=(10, 4))

    box = ctk.CTkTextbox(win, font=("Courier New", 13), wrap="word",
                         fg_color=PANEL, text_color=TEXT_MAIN,
                         border_color=STEEL, border_width=1, corner_radius=0)
    box.pack(fill="both", expand=True, padx=14, pady=(0, 8))

    def _show_text():
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", _read_tips())
        box.configure(state="disabled")     # read-only until unlocked
    _show_text()

    row = ctk.CTkFrame(win, fg_color="transparent")
    row.pack(fill="x", padx=14, pady=(0, 12))
    status = ctk.CTkLabel(row, text="", font=FONT_SMALL, text_color=TEXT_DIM)
    status.pack(side="left")

    editing = [False]

    def _edit_or_save():
        if not editing[0]:
            # Unlock: set the passphrase the first time, check it after.
            if not _TIPS_EDITOR_HASH:
                p1 = _ask_passphrase(win, "RAMMA — set the editor passphrase",
                                     "No editor passphrase is set yet.\n"
                                     "Choose one — you will need it to edit "
                                     "the tips from now on:")
                if not p1:
                    return
                p2 = _ask_passphrase(win, "RAMMA — confirm",
                                     "Type the passphrase again:")
                if p1 != p2:
                    messagebox.showerror("RAMMA", "The two passphrases did not "
                                         "match. Nothing was changed.",
                                         parent=win)
                    return
                try:
                    _store_editor_hash(_passphrase_hash(p1))
                except Exception as e:
                    messagebox.showerror("RAMMA", f"The passphrase could not "
                                         f"be saved: {e}", parent=win)
                    return
                print("[Tips] Editor passphrase set — its hash is now in "
                      "ramma.py")
            else:
                phrase = _ask_passphrase(win, "RAMMA — edit tips",
                                         "Editor passphrase:")
                if phrase is None:
                    return
                if _passphrase_hash(phrase) != _TIPS_EDITOR_HASH:
                    messagebox.showerror("RAMMA", "That passphrase is not "
                                         "right.", parent=win)
                    return
            editing[0] = True
            box.configure(state="normal")
            if box.get("1.0", "end").strip() == _TIPS_DEFAULT.strip():
                box.delete("1.0", "end")
            box.focus_set()
            edit_btn.configure(text="💾 SAVE", text_color=BRIGHT_GREEN)
            status.configure(text="Editing — press SAVE when done",
                             text_color="#e6c000")
        else:
            try:
                _write_tips(box.get("1.0", "end"))
            except Exception as e:
                messagebox.showerror("RAMMA", f"The tips could not be "
                                     f"saved: {e}", parent=win)
                return
            editing[0] = False
            _show_text()
            edit_btn.configure(text="✎ EDIT", text_color=TEXT_DIM)
            status.configure(text="Saved to tips.txt", text_color=BRIGHT_GREEN)
            print(f"[Tips] Saved to {_TIPS_PATH}")

    def _close():
        if editing[0] and not messagebox.askyesno(
                "RAMMA", "Close without saving your changes?", parent=win):
            return
        editing[0] = False
        edit_btn.configure(text="✎ EDIT", text_color=TEXT_DIM)
        status.configure(text="")
        _show_text()
        win.withdraw()

    ctk.CTkButton(row, text="CLOSE", command=_close, width=90, height=28,
                  fg_color=STEEL, hover_color=STEEL_LIGHT, text_color=TEXT_MAIN,
                  font=FONT_SMALL, corner_radius=0, border_width=1,
                  border_color=BORDER).pack(side="right")
    edit_btn = ctk.CTkButton(row, text="✎ EDIT", command=_edit_or_save,
                             width=90, height=28,
                             fg_color=STEEL, hover_color=STEEL_LIGHT,
                             text_color=TEXT_DIM, font=FONT_SMALL,
                             corner_radius=0, border_width=1,
                             border_color=BORDER)
    edit_btn.pack(side="right", padx=(0, 8))

    win.protocol("WM_DELETE_WINDOW", _close)
    win.transient(app)
    win.lift()
    return win


_tips_btn = ctk.CTkButton(header, text="💡 TIPS", command=open_tips,
                          fg_color=STEEL, hover_color=STEEL_LIGHT,
                          text_color="#e6c000",
                          font=("Courier New", 12, "bold"),
                          corner_radius=0, border_width=1,
                          border_color=BORDER, height=26, width=100)
_tips_btn.pack(side="right", padx=(0, 8))


def _on_locate_stems():
    """Choose the folder saved stems are kept in, and say what it holds."""
    folder = filedialog.askdirectory(
        title="Where are your saved stems?",
        initialdir=state.saved_stems_dir or state.export_folder or None)
    if not folder:
        return
    state.saved_stems_dir = folder
    _save_dirs()
    print(f"[Stems] Saved stems will be looked for in {folder}")

    path = _playlist_current[0]
    if not path:
        messagebox.showinfo(
            "RAMMA — saved stems",
            f"Saved stems will be looked for in:\n{folder}\n\n"
            f"Any song you load whose stems are there will use them instead "
            f"of separating again.")
        return
    found = find_saved_stems(path)
    song = os.path.splitext(os.path.basename(path))[0]
    if not found:
        messagebox.showinfo(
            "RAMMA — saved stems",
            f"No saved stems for \"{song}\" in that folder.\n\nFiles are "
            f"matched by name — {song}_vocals.wav, {song}_drums.wav and so "
            f"on, as EXPORT STEMS writes them.")
        return
    if messagebox.askyesno(
            "RAMMA — saved stems",
            f"Found saved stems for \"{song}\":\n  "
            + ", ".join(sorted(found)) +
            "\n\nLoad the song again now and use them?"):
        load_file_path(path)




# ============================================================
# SCROLLABLE CONTENT AREA — hosts everything below the header
# ============================================================
_scroll_outer = tk.Frame(app, bg=BG)
_scroll_outer.pack(fill="both", expand=True)

_scroll_canvas = tk.Canvas(_scroll_outer, bg=BG, highlightthickness=0, bd=0)

# Themed scrollbar — forge gold thumb on near-black track
_vscroll = tk.Scrollbar(
    _scroll_outer,
    orient="vertical",
    command=_scroll_canvas.yview,
    bg="#0a0a0a",           # track background — forge floor black
    troughcolor="#0f0f0f",  # trough
    activebackground="#e6c000",  # thumb when hovered
    relief="flat",
    bd=0,
    width=10,
)
_vscroll.pack(side="right", fill="y")

# Sideways scrolling, for when the mixer holds more cells than the window can
# show at a readable width. Packed only while it is needed.
_hbar = tk.Scrollbar(
    _scroll_outer,
    orient="horizontal",
    command=lambda *a: _scroll_canvas.xview(*a),
    bg="#0a0a0a",
    troughcolor="#0f0f0f",
    activebackground="#e6c000",
    relief="flat",
    bd=0,
    width=10,
)
_scroll_canvas.pack(side="left", fill="both", expand=True)
_scroll_canvas.configure(yscrollcommand=_vscroll.set,
                         xscrollcommand=_hbar.set)

# Inner frame — all content below the header is parented here
_sf = tk.Frame(_scroll_canvas, bg=BG)
_sf_id = _scroll_canvas.create_window((0, 0), window=_sf, anchor="nw")

_sf_resize_job = [None]


def _apply_sf_geometry():
    _sf_resize_job[0] = None
    try:
        # The content is normally exactly as wide as the window, with the
        # mixer's cells sharing whatever there is. They stop shrinking at
        # MIXER_MIN_CELL_W, and past that the canvas scrolls sideways.
        # Deliberately not _sf.winfo_reqwidth(): that is the width the cells
        # would *like* (their sliders and buttons at full size), which is far
        # more than they need to be usable.
        # Count the columns that actually hold a cell: grid_size() also
        # counts gaps left by retired cells, which would reserve width for
        # nothing.
        mixer_min = 0
        for _grid in (mixer_grid, globals().get("atmos_grid")):
            if _grid is None:
                continue
            try:
                cols = {int(w.grid_info()["column"]) for w in _grid.grid_slaves()}
            except Exception:
                cols = set()
            if cols:
                mixer_min = max(mixer_min,
                                len(cols) * (MIXER_MIN_CELL_W + 8) + 40)
        need = max(_scroll_canvas.winfo_width(), mixer_min)
        _scroll_canvas.itemconfig(_sf_id, width=need)
        _scroll_canvas.configure(scrollregion=_scroll_canvas.bbox("all"))
        _update_hscroll()
    except Exception:
        pass


def _update_hscroll():
    """Show the horizontal scrollbar only when there is something to reach."""
    try:
        # What the content was actually given, not what its widgets asked for.
        try:
            content = int(float(_scroll_canvas.itemcget(_sf_id, "width")))
        except Exception:
            content = _sf.winfo_reqwidth()
        need = content > _scroll_canvas.winfo_width() + 2
        if need and not _hbar.winfo_ismapped():
            _hbar.pack(side="bottom", fill="x")
        elif not need and _hbar.winfo_ismapped():
            _hbar.pack_forget()
            _scroll_canvas.xview_moveto(0)
    except Exception:
        pass


def _schedule_sf_geometry(delay=90):
    """Re-apply the content width and scroll region once things settle."""
    if _sf_resize_job[0] is not None:
        try:
            app.after_cancel(_sf_resize_job[0])
        except Exception:
            pass
    _sf_resize_job[0] = app.after(delay, _apply_sf_geometry)


def _on_sf_configure(event):
    # Collapsing a panel or resizing the window fires a burst of these; doing
    # the work once, after the burst, keeps scrolling smooth.
    _schedule_sf_geometry()


def _on_canvas_resize(event):
    # Deliberately does NOT set the content width here. Doing that on every
    # <Configure> reflowed the whole mixer for each pixel of a window drag,
    # and CustomTkinter redraws every rounded widget when it is re-laid out —
    # thousands of canvas operations per frame. The width is applied once the
    # drag settles instead; during the drag the view is simply clipped, which
    # costs nothing.
    _schedule_sf_geometry()

_sf.bind("<Configure>", _on_sf_configure)
_scroll_canvas.bind("<Configure>", _on_canvas_resize)

def _wheel_target(event):
    """Which canvas the wheel should scroll, judged by the window under it."""
    try:
        top = event.widget.winfo_toplevel()
    except Exception:
        return _scroll_canvas
    eqw = globals().get("_eq_window")
    if eqw is not None and top is eqw:
        eqc = globals().get("_eq_canvas")
        try:
            if eqc is not None and eqc.winfo_exists():
                return eqc
        except Exception:
            pass
    # The loadlist window scrolls its own list, and each dynamics window has
    # its own scrollable frame; everything else scrolls the main window.
    llw = globals().get("_loadlist_win")
    if llw is not None and top is llw:
        return None
    for _k, _w in (globals().get("_dyn_windows") or {}).items():
        try:
            if _w.winfo_exists() and top is _w:
                _c = (globals().get("_dyn_canvases") or {}).get(_k)
                return _c if (_c is not None and _c.winfo_exists()) else None
        except Exception:
            pass
    return _scroll_canvas


def _on_mousewheel(event):
    canvas = _wheel_target(event)
    if canvas is None:
        return            # a window that handles its own scrolling
    if event.num == 4:
        canvas.yview_scroll(-1, "units")
    elif event.num == 5:
        canvas.yview_scroll(1, "units")
    else:
        # Windows reports multiples of 120; three lines per notch feels right
        # without asking the canvas to redraw more than it must.
        canvas.yview_scroll(int(-1 * (event.delta / 120)) * 3, "units")


# One binding for the whole application, installed once. Nothing else may
# call bind_all/unbind_all for the wheel, or this stops working.
def _on_shift_mousewheel(event):
    """Shift + wheel scrolls the main view sideways."""
    canvas = _wheel_target(event)
    if canvas is None:
        return
    if event.num == 4:
        canvas.xview_scroll(-3, "units")
    elif event.num == 5:
        canvas.xview_scroll(3, "units")
    else:
        canvas.xview_scroll(int(-1 * (event.delta / 120)) * 3, "units")


_scroll_canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)
_scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
_scroll_canvas.bind_all("<Button-4>",   _on_mousewheel)
_scroll_canvas.bind_all("<Button-5>",   _on_mousewheel)

# ============================================================
# TRANSPORT CONTROLS
# ============================================================
transport_outer = ctk.CTkFrame(_sf, fg_color="transparent")
transport_outer.pack(pady=8, padx=20, fill="x")

# Inner frame is not packed to a side — it centres itself
transport = ctk.CTkFrame(transport_outer, fg_color="transparent")
transport.pack(anchor="center")

btn_load = ctk.CTkButton(
    transport, text="◼ LOAD", command=load_file,
    fg_color="#1a6a9a", hover_color="#2196c8",
    text_color=TEXT_MAIN,
    font=FONT_LABEL, corner_radius=0,
    border_width=1, border_color="#2196c8", height=34)
btn_load.pack(side="left", padx=4)

# Play and Stop start dimmed — brightened once audio is loaded
btn_play = ctk.CTkButton(
    transport, text="▶ PLAY", command=play,
    fg_color=STEEL, hover_color=STEEL_LIGHT,
    text_color=TEXT_DIM,
    font=FONT_LABEL, corner_radius=0,
    border_width=1, border_color=BORDER, height=34)
btn_play.pack(side="left", padx=4)

btn_stop = ctk.CTkButton(
    transport, text="■ STOP", command=stop,
    fg_color=STEEL, hover_color=STEEL_LIGHT,
    text_color=TEXT_DIM,
    font=FONT_LABEL, corner_radius=0,
    border_width=1, border_color=BORDER, height=34)
btn_stop.pack(side="left", padx=4)

_btn(transport, "≋ EQ",             open_eq_window).pack(side="left", padx=4)
_btn(transport, "⬡ EXPORT FOLDER",  choose_export_folder).pack(side="left", padx=4)
_btn(transport, "⌕ LOCATE STEMS",   _on_locate_stems).pack(side="left", padx=4)
_btn(transport, "💾 SAVE SESSION",  save_session).pack(side="left", padx=4)
_btn(transport, "📂 LOAD SESSION",  load_session).pack(side="left", padx=4)
_btn(transport, "⬡ BOUNCE",         bounce_to_stem).pack(side="left", padx=4)

# ── Loadlist toggle ───────────────────────────────────────────────────────────
_playlist_visible = [False]   # mutable box — toggled by button
_playlist_paths   = []        # up to 100 file paths
_playlist_current = [None]    # path of the song currently loaded, if any

def _show_loadlist():
    """Open the LOADLIST window, or bring it forward if it is already open."""
    w = _loadlist_win
    _playlist_visible[0] = True
    app.after(60, lambda: _fit_all(rescan=True))
    try:
        w.deiconify()
        w.transient(app)          # keeps it above RAMMA, not above everything
        w.lift()
        w.attributes("-topmost", True)
        # CTkToplevel finishes setting up a moment later and can drop the
        # window behind; release topmost only after that.
        app.after(400, lambda: w.attributes("-topmost", False))
        w.focus_force()
    except Exception:
        pass


def _hide_loadlist():
    _playlist_visible[0] = False
    try:
        _loadlist_win.withdraw()
    except Exception:
        pass


def _toggle_playlist():
    if _playlist_visible[0]:
        _hide_loadlist()
    else:
        _show_loadlist()

_playlist_toggle_btn = ctk.CTkButton(
    transport, text="📋 LOADLIST",
    command=_toggle_playlist,
    fg_color=STEEL, hover_color=STEEL_LIGHT,
    text_color=TEXT_MAIN, font=FONT_LABEL,
    corner_radius=0, border_width=1,
    border_color=BORDER, height=34)
_playlist_toggle_btn.pack(side="left", padx=4)

_fix_btn = ctk.CTkButton(
    transport, text="✂ STEM FIX",
    command=lambda: open_fix_window(),
    fg_color=STEEL, hover_color=STEEL_LIGHT,
    text_color=TEXT_MAIN, font=FONT_LABEL,
    corner_radius=0, border_width=1, border_color=BORDER,
    height=34, width=150)
_fix_btn.pack(side="left", padx=4)

# Tooltip for dimmed Play/Stop
_tooltip_win = None
def _show_tooltip(event):
    global _tooltip_win
    # Only when there is genuinely nothing to play. Imported cells count as
    # much as separated stems do, so an imported ATMOS bed (with nothing
    # separated) must not raise the warning.
    if _playable_length() > 0:
        return
    if _tooltip_win is not None:
        return
    x = event.widget.winfo_rootx() + 10
    y = event.widget.winfo_rooty() + event.widget.winfo_height() + 4
    _tooltip_win = tk.Toplevel(app)
    _tooltip_win.wm_overrideredirect(True)
    _tooltip_win.wm_geometry(f"+{x}+{y}")
    tk.Label(_tooltip_win,
             text="Audio must be loaded before attempting to play or stop it.",
             font=FONT_SMALL,
             bg="#1a1a1a", fg=TEXT_DIM,
             relief="flat", padx=6, pady=3).pack()

def _hide_tooltip(event):
    global _tooltip_win
    if _tooltip_win:
        _tooltip_win.destroy()
        _tooltip_win = None

for _tb in (btn_play, btn_stop):
    _tb.bind("<Enter>", _show_tooltip)
    _tb.bind("<Leave>", _hide_tooltip)

def _lock_transport():
    """Dim Play and Stop, as they are before anything is loaded."""
    btn_play.configure(fg_color=STEEL, hover_color=STEEL_LIGHT,
                       text_color=TEXT_MAIN, border_color=BORDER)
    btn_stop.configure(fg_color=STEEL, hover_color=STEEL_LIGHT,
                       text_color=TEXT_MAIN, border_color=BORDER)


def _unlock_transport():
    """Called after stems are loaded — brighten Play and Stop using current theme colours."""
    btn_play.configure(fg_color="#2e7d32", hover_color="#43a047",
                       text_color=TEXT_MAIN, border_color="#43a047")
    btn_stop.configure(fg_color="#5c0000", hover_color="#7a0000",
                       text_color=TEXT_MAIN, border_color="#7a0000")

# ============================================================
# STEM FIX WINDOW — move a mis-placed sound to the right stem
# ============================================================
_fix_win = None
_fix_src_menu = None
_fix_dst_menu = None
_fix_region_lbl = None
_fix_solo_lbl = None
_fix_list_frame = None


# Cells a fix can move audio between: the separated stems, plus the two
# karaoke halves, whose audio lives outside state.stems.
_FIX_EXTRA_BUFFERS = {
    "front_vocals": ("fv_data", "fv_sr"),
    "bg_vocals":    ("bg_vocals_data", "bg_vocals_sr"),
    "strings":      ("strings_data", "strings_sr"),
}
_FIX_DISPLAY = {"front_vocals": "FRT VOX", "bg_vocals": "BG VOX",
                "strings": "STRINGS"}


def _fix_label_for(key):
    return _FIX_DISPLAY.get(key, key.upper())


def _fix_key_for(label):
    label = (label or "").strip().upper()
    for key, name in _FIX_DISPLAY.items():
        if name == label:
            return key
    return label.lower()


def _fix_keys():
    """Every cell a fix can use, separated stems first."""
    keys = list((state.stems or {}).keys()) or list(_STEM_KEYS)
    return keys + list(_FIX_EXTRA_BUFFERS)


def _fix_buffer(key, create_len=0):
    """The audio array behind a cell, optionally creating an empty one.

    The karaoke halves may hold nothing yet — moving a backing vocal into an
    empty BG VOX should still work, so a silent buffer is made for it.
    """
    if state.stems and key in state.stems:
        return state.stems[key]
    attr = _FIX_EXTRA_BUFFERS.get(key, (None, None))[0]
    if attr is None:
        return None
    buf = getattr(state, attr, None)
    if buf is None and create_len > 0:
        buf = np.zeros((int(create_len), 2), dtype=np.float32)
        setattr(state, attr, buf)
        sr_attr = _FIX_EXTRA_BUFFERS[key][1]
        if getattr(state, sr_attr, None) is None:
            setattr(state, sr_attr, state.sr)
    return buf


def _soloed_stem():
    """The single soloed stem, when exactly one separated stem is soloed.

    Soloing is how you find the offending sound in the first place, so the
    stem you are listening to is the one the fix should take its audio from.
    """
    if not state.stems:
        return None
    soloed = [k for k in _fix_keys() if state.stem_solo.get(k, False)]
    return soloed[0] if len(soloed) == 1 else None


def _fix_stem_choices():
    return [_fix_label_for(k) for k in _fix_keys()]


def _fix_watch_loop():
    """Follow the loop markers while the fix window is open."""
    try:
        if _fix_win is not None and _fix_win.winfo_exists() and \
                _fix_win.state() == "normal":
            _refresh_fix_list()
    except Exception:
        pass
    if running:
        app.after(400, _fix_watch_loop)


def _refresh_fix_list():
    """Redraw the region line and the list of fixes for this track."""
    if _fix_win is None or not _fix_win.winfo_exists():
        return
    try:
        # FROM follows the soloed stem: what you are listening to is what
        # gets moved. With nothing (or several things) soloed, the menu is
        # yours to set.
        solo = _soloed_stem()
        if solo is not None:
            want = _fix_label_for(solo)
            if _fix_src_menu.get() != want:
                _fix_src_menu.set(want)
            _fix_solo_lbl.configure(
                text=f"FROM follows the soloed stem: {want}",
                text_color=SOLO_ON)
        else:
            _fix_solo_lbl.configure(
                text="Solo a stem to have FROM follow it",
                text_color=TEXT_DIM)

        sr = max(1, int(state.sr or 44100))
        a, b = state.loop_start, state.loop_end
        if a is not None and b is not None and b > a:
            _fix_region_lbl.configure(
                text=f"Marked: {a / sr:.2f}s to {b / sr:.2f}s  "
                     f"({(b - a) / sr:.2f}s)", text_color=BRIGHT_GREEN)
        else:
            _fix_region_lbl.configure(
                text="Nothing marked — drag a loop across the part to move",
                text_color=TEXT_DIM)

        choices = _fix_stem_choices()
        for menu in (_fix_src_menu, _fix_dst_menu):
            if list(menu.cget("values")) != choices:
                current = menu.get()
                menu.configure(values=choices)
                if current in choices:
                    menu.set(current)

        for w in _fix_list_frame.winfo_children():
            w.destroy()
        fixes = _fixes_for_current()
        if not fixes:
            ctk.CTkLabel(_fix_list_frame, text="No fixes saved for this track",
                         font=FONT_SMALL, text_color=TEXT_DIM).pack(pady=6)
            return
        for i, fix in enumerate(fixes):
            row = ctk.CTkFrame(_fix_list_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=_fix_describe(fix), font=FONT_SMALL,
                         text_color=TEXT_MAIN, anchor="w").pack(side="left",
                                                                fill="x", expand=True)
            ctk.CTkButton(row, text="UNDO", width=70, height=22,
                          command=lambda idx=i: remove_stem_fix(idx),
                          fg_color=STEEL, hover_color=MUTE_ON,
                          text_color=TEXT_MAIN, font=FONT_SMALL,
                          corner_radius=0, border_width=1,
                          border_color=BORDER).pack(side="left", padx=(6, 0))
    except Exception:
        pass


def open_fix_window():
    """Open (or re-show) the window for moving audio between stems."""
    global _fix_win, _fix_src_menu, _fix_dst_menu, _fix_region_lbl
    global _fix_solo_lbl, _fix_list_frame
    if _fix_win is not None and _fix_win.winfo_exists():
        _fix_win.deiconify()
        _fix_win.lift()
        _refresh_fix_list()
        return _fix_win

    win = ctk.CTkToplevel(app)
    win.title("R·A·M·M·A — STEM FIX")
    win.geometry("620x560")
    win.minsize(520, 420)
    win.configure(fg_color=BG)
    win.protocol("WM_DELETE_WINDOW", lambda: win.withdraw())
    _fix_win = win

    ctk.CTkLabel(win, text="— MOVE AUDIO TO ANOTHER STEM —",
                 font=FONT_TITLE, text_color=GLOW_RED).pack(pady=(10, 2))
    ctk.CTkLabel(win, wraplength=560, justify="left", font=FONT_SMALL,
                 text_color=TEXT_DIM,
                 text=("Solo the stem the wrong sound is in, mark the part on "
                       "the waveform with a loop, choose the stem it belongs "
                       "to, then press MOVE. Only the soloed stem's audio is "
                       "moved. The change is saved for this track and applied "
                       "again the next time you load it.")
                 ).pack(padx=16, pady=(0, 8), fill="x")

    _fix_region_lbl = ctk.CTkLabel(win, text="", font=FONT_LABEL,
                                   text_color=TEXT_DIM)
    _fix_region_lbl.pack(pady=(0, 2))

    _fix_solo_lbl = ctk.CTkLabel(win, text="", font=FONT_SMALL,
                                 text_color=TEXT_DIM)
    _fix_solo_lbl.pack(pady=(0, 8))

    row = ctk.CTkFrame(win, fg_color="transparent")
    row.pack(pady=(0, 10))
    choices = _fix_stem_choices()
    ctk.CTkLabel(row, text="FROM", font=FONT_SMALL,
                 text_color=TEXT_MAIN).pack(side="left", padx=(0, 4))
    _fix_src_menu = ctk.CTkOptionMenu(
        row, values=choices, width=150, font=FONT_SMALL, dropdown_font=FONT_SMALL,
        fg_color=STEEL, button_color=RED, button_hover_color=BRIGHT_RED,
        dropdown_fg_color=PANEL, dropdown_hover_color=STEEL,
        text_color=TEXT_MAIN, corner_radius=0)
    _fix_src_menu.pack(side="left", padx=(0, 12))
    ctk.CTkLabel(row, text="TO", font=FONT_SMALL,
                 text_color=TEXT_MAIN).pack(side="left", padx=(0, 4))
    _fix_dst_menu = ctk.CTkOptionMenu(
        row, values=choices, width=150, font=FONT_SMALL, dropdown_font=FONT_SMALL,
        fg_color=STEEL, button_color=RED, button_hover_color=BRIGHT_RED,
        dropdown_fg_color=PANEL, dropdown_hover_color=STEEL,
        text_color=TEXT_MAIN, corner_radius=0)
    if len(choices) > 1:
        _fix_dst_menu.set(choices[-1])
    _fix_dst_menu.pack(side="left")

    def _do_move():
        add_stem_fix(_fix_key_for(_fix_src_menu.get()),
                     _fix_key_for(_fix_dst_menu.get()))

    ctk.CTkButton(win, text="✂  MOVE THE MARKED PART", command=_do_move,
                  fg_color="#2e7d32", hover_color="#43a047",
                  text_color=TEXT_MAIN, font=FONT_LABEL,
                  corner_radius=0, border_width=1, border_color="#43a047",
                  height=34, width=300).pack(pady=(0, 10))

    ctk.CTkLabel(win, text="Saved fixes for this track",
                 font=FONT_LABEL, text_color=TEXT_MAIN).pack(pady=(4, 2))
    holder = ctk.CTkFrame(win, fg_color=PANEL, corner_radius=0,
                          border_color=STEEL, border_width=1)
    holder.pack(fill="both", expand=True, padx=14, pady=(0, 12))
    _fix_list_frame = ctk.CTkFrame(holder, fg_color="transparent")
    _fix_list_frame.pack(fill="both", expand=True, padx=8, pady=8)

    _refresh_fix_list()
    win.transient(app)
    win.lift()
    win.attributes("-topmost", True)
    app.after(400, lambda: win.attributes("-topmost", False))
    return win


# ============================================================
# LOADLIST WINDOW — up to 100 tracks, opened from the LOADLIST button
# ============================================================
# LOADLIST type: 3pt above the shared sizes it used to borrow, for easier
# reading in its own window. The emoji "icons" on the buttons scale with it.
LL_FONT_TITLE  = (FONT_TITLE[0],  FONT_TITLE[1]  + 3, "bold")
LL_FONT_SMALL  = (FONT_SMALL[0],  FONT_SMALL[1]  + 3)
LL_FONT_STATUS = ("Courier New", 16 + 3, "bold")

_loadlist_win = ctk.CTkToplevel(app)
_loadlist_win.withdraw()                     # hidden until asked for
_loadlist_win.title("R·A·M·M·A — LOADLIST")
_loadlist_win.geometry("840x560")
# Narrower than this and the button row and legend would be cut off.
_loadlist_win.minsize(640, 340)
_loadlist_win.configure(fg_color=BG)
_loadlist_win.protocol("WM_DELETE_WINDOW", _hide_loadlist)   # hide, never destroy

_playlist_frame = ctk.CTkFrame(_loadlist_win, fg_color=PANEL,
                                corner_radius=0,
                                border_color=STEEL, border_width=1)
_playlist_frame.pack(fill="both", expand=True, padx=8, pady=8)

tk.Frame(_playlist_frame, bg=RED, height=2).pack(fill="x")

_pl_hdr = ctk.CTkFrame(_playlist_frame, fg_color="transparent")
_pl_hdr.pack(fill="x", padx=10, pady=(6, 4))

ctk.CTkLabel(_pl_hdr, text="— 📋  LOADLIST  (MAX 100 TRACKS) —",
             font=LL_FONT_TITLE, text_color=GLOW_RED).pack(side="left", expand=True)

def _playlist_add_songs():
    """Open multi-file dialog and append tracks to the loadlist."""
    paths = filedialog.askopenfilenames(
        title="Add Tracks to Loadlist",
        initialdir=state.last_load_dir,
        filetypes=[("Audio files", "*.wav *.flac *.mp3 *.ogg *.aif *.aiff *.m4a *.aac *.mp4 *.m4v *.mov"),
                   ("All files", "*.*")]
    )
    added = []
    for p in paths:
        if p not in _playlist_paths and len(_playlist_paths) < 100:
            _playlist_paths.append(p)
            added.append(p)
    _playlist_refresh()

    # Nothing loaded yet? Open the first song straight away rather than
    # making the user hunt for it in the list.
    idle = (not state.separating and state.stems is None
            and state.instrumental is None and _playlist_current[0] is None)
    if idle and _playlist_paths:
        first = _playlist_paths[0]
        print(f"[Loadlist] Opening {os.path.basename(first)!r} automatically")
        load_file_path(first)
        return

    # Start pre-loading right away so the first song change is instant.

def _playlist_remove_selected():
    """Remove the currently highlighted item."""
    sel = _pl_listbox.curselection()
    if not sel:
        return
    idx = sel[0]
    if 0 <= idx < len(_playlist_paths):
        _playlist_paths.pop(idx)
    _playlist_refresh()

def _playlist_clear():
    """Wipe the entire loadlist."""
    _playlist_paths.clear()
    _playlist_refresh()

def _playlist_refresh():
    """Rebuild the listbox from _playlist_paths.

    Rows are marked ▶ for the loaded song and ◐ while one is separating.
    """
    try:
        sel = _pl_listbox.curselection()
    except Exception:
        sel = ()
    _pl_listbox.delete(0, "end")
    for i, p in enumerate(_playlist_paths):
        name = os.path.basename(p)
        if p == _playlist_current[0]:
            mark = "◐" if state.separating else "▶"
        else:
            mark = "·"
        _pl_listbox.insert("end", f" {mark} {i+1:>2}.  {name}")
    for i in sel:
        if i < len(_playlist_paths):
            _pl_listbox.selection_set(i)
    _playlist_sync_selector()


def _playlist_sync_selector():
    """Refresh the two status lines under the list."""
    try:
        _pl_now_lbl
    except NameError:
        return

    # "Now playing" line
    cur = _playlist_current[0]
    if cur and state.separating:
        _pl_now_lbl.configure(text="LOADING:  " + os.path.basename(cur),
                              text_color="#ffaa00")
    elif cur:
        _pl_now_lbl.configure(text="NOW PLAYING:  " + os.path.basename(cur),
                              text_color="#44cc44")
    else:
        _pl_now_lbl.configure(text="NOW PLAYING:  —", text_color=TEXT_DIM)

    # Second line: what the list holds
    _pl_pre_lbl.configure(
        text=f"{len(_playlist_paths)} track(s) — double-click one to load it",
        text_color=TEXT_DIM)


def _apply_pending_autosolo():
    """Apply a vocals->instrumental link that was asked for during loading."""
    if _inst_autosolo_pending[0]:
        _inst_autosolo_pending[0] = False
        _vocals_autosolo_check()


def _playlist_load_selected(event=None):
    """Load the selected (double-clicked or Enter) song."""
    sel = _pl_listbox.curselection()
    if not sel:
        return
    idx = sel[0]
    if 0 <= idx < len(_playlist_paths):
        load_file_path(_playlist_paths[idx])

_pl_btn_row = ctk.CTkFrame(_playlist_frame, fg_color="transparent")
_pl_btn_row.pack(fill="x", padx=10, pady=(0, 6))

ctk.CTkButton(_pl_btn_row, text="➕  ADD TRACKS",
              command=_playlist_add_songs,
              fg_color=RED, hover_color=BRIGHT_RED,
              text_color=TEXT_MAIN, font=FONT_LABEL,
              corner_radius=0, border_width=1,
              border_color=BRIGHT_RED, height=30,
              width=160).pack(side="left", padx=(0, 8))

ctk.CTkButton(_pl_btn_row, text="✕  REMOVE",
              command=_playlist_remove_selected,
              fg_color=STEEL, hover_color=STEEL_LIGHT,
              text_color=TEXT_MAIN, font=FONT_SMALL,
              corner_radius=0, border_width=1,
              border_color=BORDER, height=30,
              width=110).pack(side="left", padx=(0, 8))

ctk.CTkButton(_pl_btn_row, text="✕✕  CLEAR ALL",
              command=_playlist_clear,
              fg_color=STEEL, hover_color=STEEL_LIGHT,
              text_color=TEXT_DIM, font=FONT_SMALL,
              corner_radius=0, border_width=1,
              border_color=BORDER, height=30,
              width=130).pack(side="left", padx=(0, 12))

ctk.CTkLabel(_pl_btn_row,
             text="▶ loaded   ◐ separating",
             font=LL_FONT_SMALL, text_color=TEXT_DIM).pack(side="left")

# ── Status lines ──────────────────────────────────────────────────────────
_pl_status_row = ctk.CTkFrame(_playlist_frame, fg_color="transparent")
_pl_status_row.pack(fill="x", padx=10, pady=(0, 6))

_pl_now_lbl = ctk.CTkLabel(_pl_status_row, text="NOW PLAYING:  —",
                           font=LL_FONT_SMALL, text_color=TEXT_DIM)
_pl_now_lbl.pack(side="left", padx=(0, 16))

_pl_pre_lbl = ctk.CTkLabel(_pl_status_row, text="",
                           font=LL_FONT_STATUS,
                           text_color=TEXT_DIM)
_pl_pre_lbl.pack(side="left")

_pl_list_frame = tk.Frame(_playlist_frame, bg=PANEL)
_pl_list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

_pl_sb = tk.Scrollbar(_pl_list_frame, orient="vertical",
                       bg=STEEL, troughcolor="#0f0f0f",
                       activebackground=BRIGHT_RED,
                       relief="flat", bd=0, width=10)
_pl_sb.pack(side="right", fill="y")

_pl_listbox = tk.Listbox(
    _pl_list_frame,
    yscrollcommand=_pl_sb.set,
    bg="#0d0d0d", fg=TEXT_MAIN,
    selectbackground=RED, selectforeground="#ffffff",
    activestyle="none", relief="flat", bd=0,
    highlightthickness=1, highlightbackground=STEEL,
    font=LL_FONT_SMALL,
    height=8,
    exportselection=False,
)
_pl_listbox.pack(side="left", fill="both", expand=True)
_pl_sb.config(command=_pl_listbox.yview)

# The listbox scrolls itself with the wheel; the application-wide handler
# steps aside for this window (see _wheel_target).
def _pl_wheel(event):
    step = -1 if (event.num == 4 or getattr(event, "delta", 0) > 0) else 1
    _pl_listbox.yview_scroll(step * 3, "units")
    return "break"
for _seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
    _pl_listbox.bind(_seq, _pl_wheel)

_pl_listbox.bind("<Double-Button-1>", _playlist_load_selected)
_pl_listbox.bind("<Return>",          _playlist_load_selected)

def _pl_wheel(e):
    _pl_listbox.yview_scroll(
        int(-1 * (e.delta / 120)) if e.delta else (-1 if e.num == 4 else 1),
        "units")
    return "break"
_pl_listbox.bind("<MouseWheel>", _pl_wheel)
_pl_listbox.bind("<Button-4>",   _pl_wheel)
_pl_listbox.bind("<Button-5>",   _pl_wheel)

# ============================================================
# DIVIDER
# ============================================================
def _divider(parent):
    tk.Frame(parent, bg=STEEL, height=1).pack(fill="x", padx=20, pady=6)

_divider(_sf)

_build_wave_widgets(_sf)

# ============================================================
# STEM MIXER — grid layout
# ============================================================
mixer_outer = ctk.CTkFrame(_sf, fg_color=PANEL,
                            corner_radius=0,
                            border_color=STEEL, border_width=1)
mixer_outer.pack(fill="x", padx=20, pady=4)

# Clicking the heading replays the track-name flash over the waveform.
_mixer_title_lbl = ctk.CTkLabel(mixer_outer,
                                text="— STEM MIXER —",
                                font=FONT_TITLE,
                                text_color=GLOW_RED)
_mixer_title_lbl.pack(pady=(8, 4))


def _on_mixer_title_click(_event=None):
    """Show the loaded track's name again, on demand."""
    if state.current_audio_name:
        _flash_track_name(state.current_audio_name)


def _mixer_title_hover(entering):
    _mixer_title_lbl.configure(text_color="#ff7755" if entering else GLOW_RED)


# CTkLabel.bind() forwards to the canvas and the inner tk.Label it draws on,
# so one call here covers a click anywhere on the text.
_mixer_title_lbl.bind("<Button-1>", _on_mixer_title_click)
_mixer_title_lbl.bind("<Enter>",    lambda e: _mixer_title_hover(True))
_mixer_title_lbl.bind("<Leave>",    lambda e: _mixer_title_hover(False))

# Hand cursor over the heading, so it reads as clickable.
for _w in (getattr(_mixer_title_lbl, "_label", None),
           getattr(_mixer_title_lbl, "_canvas", None)):
    try:
        _w.configure(cursor="hand2")
    except Exception:
        pass

MIXER_MIN_CELL_W = 124   # px; below this a cell's title cannot be read


mixer_grid = ctk.CTkFrame(mixer_outer, fg_color="transparent")
mixer_grid.pack(fill="x", padx=10, pady=(0, 4))

# The ATMOS bed sits on its own row beneath the stem cells: eighteen cells on
# one row cannot be shown at a readable width without scrolling sideways.
# ── ATMOS row, folded away by default ─────────────────────────────────────
# Six cells plus their import bar is a lot of height for something most
# tracks never use, so the row starts collapsed behind this header.
_atmos_header_row = ctk.CTkFrame(mixer_outer, fg_color="transparent")
_atmos_header_row.pack(fill="x", padx=10, pady=(0, 2))

_atmos_open = [False]
_atmos_toggle_btn = ctk.CTkButton(
    _atmos_header_row, text="",
    fg_color=STEEL, hover_color=STEEL_LIGHT, text_color=TEXT_MAIN,
    font=FONT_LABEL, corner_radius=0, border_width=1,
    border_color=BORDER, height=26)
_atmos_toggle_btn.pack(fill="x")

atmos_grid = ctk.CTkFrame(mixer_outer, fg_color="transparent")

# One file, six cells: load an E-AC-3 (including the streams used for Dolby
# Atmos) or any multichannel file and its 5.1 bed is spread over the row.
_atmos_bar = ctk.CTkFrame(mixer_outer, fg_color="transparent")

ctk.CTkButton(_atmos_bar,
              text="⬡  IMPORT ATMOS BED  (E-AC-3 / 5.1 file)",
              command=lambda: import_atmos_bed(),
              fg_color=STEEL, hover_color=STEEL_LIGHT,
              text_color=TEXT_MAIN, font=("Courier New", 13, "bold"),
              corner_radius=0, border_width=1, border_color=BRIGHT_RED,
              height=26, width=340).pack(side="left", padx=(4, 8))

ctk.CTkButton(_atmos_bar, text="✕  CLEAR BED",
              command=lambda: [_clear_import(k, globals().get(f"{k}_import_btn"))
                               for k in _51_TO_CELL],
              fg_color=STEEL, hover_color=MUTE_ON,
              text_color=TEXT_DIM, font=("Courier New", 13, "bold"),
              corner_radius=0, border_width=1, border_color=BORDER,
              height=26, width=150).pack(side="left")

ctk.CTkLabel(_atmos_bar,
             text="FFmpeg decodes the 5.1 bed; Atmos height objects are not rendered",
             font=FONT_SMALL, text_color=TEXT_DIM).pack(side="left", padx=12)


def _atmos_set_open(open_it):
    """Show or hide the ATMOS row and its import bar together."""
    _atmos_open[0] = bool(open_it)
    if _atmos_open[0]:
        atmos_grid.pack(fill="x", padx=10, pady=(0, 2),
                        after=_atmos_header_row)
        _atmos_bar.pack(fill="x", padx=10, pady=(0, 8), after=atmos_grid)
        _atmos_toggle_btn.configure(text="▾  ATMOS BED  (5.1 — 6 cells)")
    else:
        atmos_grid.pack_forget()
        _atmos_bar.pack_forget()
        _atmos_toggle_btn.configure(text="▸  ATMOS BED  (5.1 — 6 cells)")
    # The row changes the content height, so let the scroll area catch up.
    try:
        _schedule_sf_geometry(30)
    except Exception:
        pass


_atmos_toggle_btn.configure(
    command=lambda: _atmos_set_open(not _atmos_open[0]))
_atmos_set_open(False)          # folded until asked for

STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]

stem_sliders = {}   # name -> CTkSlider, so reset buttons can reach them
_invert_btns = {}


def _toggle_invert(key):
    """Flip a cell's polarity."""
    state.stem_invert[key] = not state.stem_invert.get(key, False)
    _paint_invert(key)
    print(f"[Mixer] {key}: phase {'inverted' if state.stem_invert[key] else 'normal'}")


def _paint_invert(key):
    btn = _invert_btns.get(key)
    if btn is None:
        return
    on = state.stem_invert.get(key, False)
    try:
        btn.configure(fg_color="#7a3d00" if on else STEEL,
                      hover_color="#a35200" if on else STEEL_LIGHT,
                      text_color=BRIGHT_GREEN if on else TEXT_DIM,
                      text="Ø PHASE  ON" if on else "Ø PHASE")
    except Exception:
        pass


def _add_invert_button(cell, key, after=None):
    """A polarity switch for one cell."""
    btn = ctk.CTkButton(cell, text="Ø PHASE",
                        command=lambda k=key: _toggle_invert(k),
                        fg_color=STEEL, hover_color=STEEL_LIGHT,
                        text_color=TEXT_DIM,
                        font=("Courier New", 11, "bold"),
                        corner_radius=0, border_width=1,
                        border_color=BORDER, height=18)
    if after is not None:
        btn.pack(fill="x", padx=6, pady=(1, 0), after=after)
    else:
        btn.pack(fill="x", padx=6, pady=(1, 0))
    _invert_btns[key] = btn
    _paint_invert(key)
    return btn


def _paint_ms(key):
    """Colour a cell's M and S buttons from state.

    While a cell is soloed its mute is suspended, so the M button shows
    unlit even though the mute is still remembered and returns when solo is
    released. Hovering an engaged button shows a darker shade of its colour.
    """
    try:
        m = _mute_btns[key]
        s = _solo_btns[key]
    except (NameError, KeyError):
        return
    soloed     = bool(state.stem_solo.get(key, False))
    show_muted = bool(state.stem_mute.get(key, False)) and not soloed
    m.configure(fg_color=MUTE_ON if show_muted else STEEL,
                hover_color=MUTE_ON_HOVER if show_muted else MUTE_ON)
    s.configure(fg_color=SOLO_ON if soloed else STEEL,
                hover_color=SOLO_ON_HOVER if soloed else SOLO_ON)


def _toggle_mute(key):
    """What clicking a cell's M button does.

    On a soloed cell the mute is suspended and the M button shows unlit, so a
    plain toggle changed a mute you could not see or hear — the button
    looked dead. M now wins there: the cell leaves solo and is muted.
    """
    if state.stem_solo.get(key, False):
        state.stem_solo[key] = False
        state.stem_mute[key] = True
    else:
        state.stem_mute[key] = not state.stem_mute.get(key, False)
    _paint_ms(key)


def _paint_all_ms():
    for _k in list(_mute_btns.keys()):
        _paint_ms(_k)


_mute_btns   = {}   # name -> M button widget (for colour toggling)
_solo_btns   = {}   # name -> S button widget (for colour toggling)


def _make_leveller_panel(parent, key):
    """Build a collapsible LEVELLER panel for any stem cell.
    Same visual pattern as COMP/GATE — collapsed by default."""
    lvl_frame = ctk.CTkFrame(parent, fg_color=PANEL,
                              corner_radius=0,
                              border_color=STEEL, border_width=1)

    def _toggle_lvl(f=lvl_frame):
        if f.winfo_ismapped():
            f.pack_forget()
        else:
            f.pack(fill="x", padx=4, pady=(2, 0))

    toggle_btn = ctk.CTkButton(parent,
                                text="LEVELLER",
                                command=_toggle_lvl,
                                fg_color=STEEL,
                                hover_color=STEEL_LIGHT,
                                text_color=TEXT_DIM,
                                font=("Courier New", 14, "bold"),
                                corner_radius=0,
                                border_width=1,
                                border_color=BORDER,
                                height=24)
    toggle_btn.pack(fill="x", padx=4, pady=(3, 0))

    tk.Frame(lvl_frame, bg=STEEL, height=1).pack(fill="x")
    ctk.CTkLabel(lvl_frame, text="LEVELLER",
                 font=("Courier New", 12, "bold"),
                 text_color=TEXT_DIM).pack(pady=(2, 0))

    en_var = ctk.BooleanVar(value=False)
    def _on_en(k=key, v=en_var):
        state.stem_lvl_enabled[k] = v.get()
        # keep HL legacy alias in sync
        if k == "hidden_layer":
            setattr(state, "hl_lvl_enabled", v.get())
    ctk.CTkCheckBox(lvl_frame, text="ON",
                    variable=en_var, command=_on_en,
                    font=("Courier New", 13), text_color=TEXT_DIM,
                    fg_color=STEEL, hover_color=STEEL_LIGHT,
                    checkmark_color="#ffffff", corner_radius=0,
                    border_color=BORDER, border_width=1,
                    checkbox_width=16, checkbox_height=16).pack(pady=(2, 0))

    for lbl, attr, init, is_thr in [
        ("THR", "threshold", 0.3, True),
        ("AMT", "amount",    0.5, False),
    ]:
        r = ctk.CTkFrame(lvl_frame, fg_color="transparent")
        r.pack(fill="x", padx=3, pady=1)
        ctk.CTkLabel(r, text=lbl, font=("Courier New", 13),
                     text_color=TEXT_DIM, width=42).pack(side="left")

        sv = tk.StringVar(value=f"{init:.2f}")

        def _make_cmds(k=key, a=attr, s=sv):
            def _on_sl(v):
                if a == "threshold":
                    state.stem_lvl_threshold[k] = float(v)
                    if k == "hidden_layer":
                        setattr(state, "hl_lvl_threshold", float(v))
                else:
                    state.stem_lvl_amount[k] = float(v)
                    if k == "hidden_layer":
                        setattr(state, "hl_lvl_amount", float(v))
                s.set(f"{float(v):.2f}")
            def _on_entry(event, sl=None):
                try:
                    val = max(0.0, min(1.0, float(s.get())))
                    if a == "threshold":
                        state.stem_lvl_threshold[k] = val
                        if k == "hidden_layer":
                            setattr(state, "hl_lvl_threshold", val)
                    else:
                        state.stem_lvl_amount[k] = val
                        if k == "hidden_layer":
                            setattr(state, "hl_lvl_amount", val)
                    s.set(f"{val:.2f}")
                    if sl: sl.set(val)
                except ValueError:
                    cur = state.stem_lvl_threshold.get(k, 0.3) if a == "threshold" else state.stem_lvl_amount.get(k, 0.5)
                    s.set(f"{cur:.2f}")
            return _on_sl, _on_entry

        _sl_cb, _en_cb = _make_cmds()

        entry = ctk.CTkEntry(r, textvariable=sv,
                              width=56, height=24,
                              font=("Courier New", 13),
                              fg_color=BG, border_color=STEEL,
                              text_color=TEXT_MAIN, corner_radius=0)
        entry.pack(side="left", padx=(2, 2))

        sl = LockedSlider(r, from_=0.0, to=1.0, height=11,
                          button_color=STEEL, button_hover_color=STEEL_LIGHT,
                          progress_color=STEEL, command=_sl_cb)
        sl.set(init)
        sl.pack(side="left", fill="x", expand=True)

        entry.bind("<Return>",   lambda e, s=sl, f=_en_cb: f(e, s))
        entry.bind("<FocusOut>", lambda e, s=sl, f=_en_cb: f(e, s))

    tk.Frame(lvl_frame, bg=BG, height=3).pack()


def _make_limiter_panel(parent, key):
    """Build a collapsible brick-wall limiter panel for one stem cell.

    Controls:
      ON  — enable/disable checkbox
      THR — threshold in dBFS (-40…0); editable entry + slider
      CEL — ceiling  in dBFS (-40…0); editable entry + slider

    Threshold is where gain reduction begins (fast attack envelope).
    Ceiling is the absolute hard-clip level after gain reduction.
    """
    lim_frame = ctk.CTkFrame(parent, fg_color=PANEL,
                              corner_radius=0,
                              border_color=STEEL, border_width=1)

    def _toggle_lim(f=lim_frame):
        if f.winfo_ismapped():
            f.pack_forget()
        else:
            f.pack(fill="x", padx=4, pady=(2, 0))

    toggle_btn = ctk.CTkButton(parent,
                                text="LIMITER",
                                command=_toggle_lim,
                                fg_color=STEEL,
                                hover_color=STEEL_LIGHT,
                                text_color=TEXT_DIM,
                                font=("Courier New", 14, "bold"),
                                corner_radius=0,
                                border_width=1,
                                border_color=BORDER,
                                height=24)
    toggle_btn.pack(fill="x", padx=4, pady=(3, 0))

    tk.Frame(lim_frame, bg=STEEL, height=1).pack(fill="x")
    ctk.CTkLabel(lim_frame, text="LIMITER",
                 font=("Courier New", 12, "bold"),
                 text_color=TEXT_DIM).pack(pady=(2, 0))

    en_var = ctk.BooleanVar(value=False)
    def _on_lim_en(k=key, v=en_var):
        state.stem_lim_enabled[k] = v.get()
    ctk.CTkCheckBox(lim_frame, text="ON",
                    variable=en_var, command=_on_lim_en,
                    font=("Courier New", 13), text_color=TEXT_DIM,
                    fg_color=STEEL, hover_color=STEEL_LIGHT,
                    checkmark_color="#ffffff", corner_radius=0,
                    border_color=BORDER, border_width=1,
                    checkbox_width=16, checkbox_height=16).pack(pady=(2, 0))

    # THR and CEL rows — each has a numeric entry + slider (dBFS, -40…0)
    for lbl, attr_key, init_db in [
        ("THR", "threshold", _LIM_DEFAULTS["threshold"]),
        ("CEL", "ceiling",   _LIM_DEFAULTS["ceiling"]),
    ]:
        r = ctk.CTkFrame(lim_frame, fg_color="transparent")
        r.pack(fill="x", padx=3, pady=1)
        ctk.CTkLabel(r, text=lbl, font=("Courier New", 13),
                     text_color=TEXT_DIM, width=42).pack(side="left")

        sv = tk.StringVar(value=f"{init_db:.1f}")

        def _make_lim_cmds(k=key, a=attr_key, s=sv):
            def _on_sl(v):
                fv = float(v)
                if a == "threshold":
                    state.stem_lim_threshold[k] = fv
                else:
                    state.stem_lim_ceiling[k] = fv
                s.set(f"{fv:.1f}")

            def _on_entry(event, sl=None):
                try:
                    val = max(-40.0, min(0.0, float(s.get())))
                    if a == "threshold":
                        state.stem_lim_threshold[k] = val
                    else:
                        state.stem_lim_ceiling[k] = val
                    s.set(f"{val:.1f}")
                    if sl:
                        sl.set(val)
                except ValueError:
                    cur = (state.stem_lim_threshold.get(k, _LIM_DEFAULTS["threshold"])
                           if a == "threshold"
                           else state.stem_lim_ceiling.get(k, _LIM_DEFAULTS["ceiling"]))
                    s.set(f"{cur:.1f}")

            return _on_sl, _on_entry

        _sl_cb, _en_cb = _make_lim_cmds()

        entry = ctk.CTkEntry(r, textvariable=sv,
                              width=56, height=24,
                              font=("Courier New", 13),
                              fg_color=BG, border_color=STEEL,
                              text_color=TEXT_MAIN, corner_radius=0)
        entry.pack(side="left", padx=(2, 2))

        sl = LockedSlider(r, from_=-40.0, to=0.0, height=11,
                          button_color=STEEL, button_hover_color=STEEL_LIGHT,
                          progress_color=STEEL, command=_sl_cb)
        sl.set(init_db)
        sl.pack(side="left", fill="x", expand=True)

        entry.bind("<Return>",   lambda e, s=sl, f=_en_cb: f(e, s))
        entry.bind("<FocusOut>", lambda e, s=sl, f=_en_cb: f(e, s))

    tk.Frame(lim_frame, bg=BG, height=3).pack()


# Dynamics windows, one per cell, built the first time it is asked for and
# then hidden rather than destroyed: the panels inside register widgets and
# per-stem state, so rebuilding them on every open would pile up duplicates.
_dyn_windows: dict = {}
_dyn_canvases: dict = {}   # key -> the scrolling canvas inside that window

_DYN_TITLES = {
    "vocals": "VOCALS", "drums": "DRUMS", "bass": "BASS", "guitar": "GUITAR",
    "piano": "PIANO", "other": "OTHER", "instrumental": "INST",
    "front_vocals": "FRT VOX", "bg_vocals": "BG VOX", "hidden_layer": "HID LAYER",
    "any": "ANY", "any+": "ANY+", "any++": "ANY++",
}


def _dyn_window_title(key):
    return _DYN_TITLES.get(key, str(key).upper())


def _open_dynamics_window(key):
    """Show this cell's COMP / GATE, LEVELLER and LIMITER in their own window."""
    win = _dyn_windows.get(key)
    if win is None or not win.winfo_exists():
        win = ctk.CTkToplevel(app)
        win.title(f"R·A·M·M·A — {_dyn_window_title(key)}  ·  CMP / LVL / LIM")
        win.geometry("460x860")
        win.minsize(380, 420)
        win.configure(fg_color=BG)
        win.protocol("WM_DELETE_WINDOW", lambda k=key: _hide_dynamics_window(k))

        head = ctk.CTkLabel(win, text=f"— {_dyn_window_title(key)} —",
                            font=FONT_TITLE, text_color=GLOW_RED)
        head.pack(pady=(10, 2))

        # Scrollable, so the three panels always reach their controls even on
        # a short screen. A plain canvas rather than CTkScrollableFrame: this
        # one re-lays its contents out only once a resize settles, where the
        # CTk version does it on every <Configure> and makes dragging the
        # window's edge stutter.
        holder = ctk.CTkFrame(win, fg_color=PANEL, corner_radius=0,
                              border_color=STEEL, border_width=1)
        holder.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        canvas = tk.Canvas(holder, bg=PANEL, highlightthickness=0, bd=0)
        canvas.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(holder, orient="vertical", command=canvas.yview,
                          bg=STEEL, troughcolor="#0f0f0f",
                          activebackground=BRIGHT_RED, relief="flat",
                          bd=0, width=10)
        sb.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=sb.set)

        body = ctk.CTkFrame(canvas, fg_color=PANEL, corner_radius=0)
        body_id = canvas.create_window((0, 0), window=body, anchor="nw")

        _dyn_canvases[key] = canvas

        _job = [None]

        def _apply_geometry(c=canvas, b=body, bid=body_id, j=_job):
            j[0] = None
            try:
                c.configure(scrollregion=c.bbox("all"))
                c.itemconfig(bid, width=c.winfo_width())
            except Exception:
                pass

        def _schedule(_event=None, j=_job, fn=_apply_geometry):
            if j[0] is not None:
                try:
                    app.after_cancel(j[0])
                except Exception:
                    pass
            j[0] = app.after(90, fn)

        body.bind("<Configure>", _schedule)
        canvas.bind("<Configure>", _schedule)
        _schedule()

        _make_dynamics_panel(body, key)
        _make_leveller_panel(body, key)
        _make_limiter_panel(body, key)

        # The three sections fold themselves away by default, which makes
        # sense inside a narrow cell but not in a window opened to see them.
        def _expand(w):
            for c in w.winfo_children():
                if isinstance(c, ctk.CTkButton) and c.cget("text") in (
                        "COMP/GATE", "LEVELLER", "LIMITER"):
                    try:
                        c.invoke()
                    except Exception:
                        pass
                _expand(c)
        _expand(body)

        _dyn_windows[key] = win

    app.after(60, lambda: _fit_all(rescan=True))
    try:
        win.deiconify()
        win.transient(app)      # stays above RAMMA, not above everything
        win.lift()
        win.attributes("-topmost", True)
        # CTkToplevel finishes its own set-up a moment later and can drop
        # behind; release topmost only after that has happened.
        app.after(400, lambda: win.attributes("-topmost", False))
        win.focus_force()
    except Exception:
        pass
    return win


def _hide_dynamics_window(key):
    win = _dyn_windows.get(key)
    try:
        if win is not None and win.winfo_exists():
            win.withdraw()
    except Exception:
        pass


def _make_dynamics_group(cell, key):
    """The button that opens this cell's dynamics window.

    The three panels used to unfold inside the cell itself; they now live in
    a window of their own, so a cell stays the same height whatever is open.
    """
    wrap = ctk.CTkFrame(cell, fg_color="transparent")
    wrap.pack(fill="x")

    header = ctk.CTkButton(wrap, text="",
                           command=lambda k=key: _open_dynamics_window(k),
                           fg_color=STEEL, hover_color=STEEL_LIGHT,
                           text_color=TEXT_MAIN,
                           font=("Courier New", 12, "bold"),
                           corner_radius=0, border_width=1,
                           border_color=BORDER, height=20)
    header.pack(fill="x", padx=4, pady=(4, 0))
    header._own_fit = True      # the general text fitter leaves this one alone

    import tkinter.font as _tkf
    _LABEL_FULL    = "CMP / LVL / LIM"
    _LABEL_COMPACT = "CMP/LVL/LIM"      # same words, no spaces, for narrow cells
    _fit = {"w": None}

    def _fit_text(_event=None, force=False):
        """Largest header text that fits the cell's current width."""
        avail = wrap.winfo_width() - 16          # padding + border
        if avail <= 1 or (avail == _fit["w"] and not force):
            return
        _fit["w"] = avail
        for text, floor in ((f"⚙  {_LABEL_FULL}", 8),
                            (f"⚙ {_LABEL_COMPACT}", 7)):
            size = 12
            while size > floor and _tkf.Font(family="Courier New", size=size,
                                             weight="bold").measure(text) > avail:
                size -= 1
            if _tkf.Font(family="Courier New", size=size,
                         weight="bold").measure(text) <= avail:
                break
        header.configure(text=text, font=("Courier New", size, "bold"))

    header.configure(text=f"⚙  {_LABEL_FULL}")
    _fit_text(force=True)
    wrap.bind("<Configure>", _fit_text, add="+")
    return header


def _make_dynamics_panel(parent, key):
    """Build a collapsible compressor + gate panel for one stem cell.
    Returns the toggle button so callers can register it with the theme."""
    # --- toggle button ---
    comp_frame = ctk.CTkFrame(parent, fg_color=PANEL,
                               corner_radius=0,
                               border_color=STEEL, border_width=1)

    def _toggle_comp(f=comp_frame):
        if f.winfo_ismapped():
            f.pack_forget()
        else:
            f.pack(fill="x", padx=4, pady=(2, 0))

    toggle_btn = ctk.CTkButton(parent,
                                text="COMP/GATE",
                                command=_toggle_comp,
                                fg_color=STEEL,
                                hover_color=STEEL_LIGHT,
                                text_color=TEXT_DIM,
                                font=("Courier New", 14, "bold"),
                                corner_radius=0,
                                border_width=1,
                                border_color=BORDER,
                                height=24)
    toggle_btn.pack(fill="x", padx=4, pady=(3, 0))

    # --- compressor rows ---
    tk.Frame(comp_frame, bg=STEEL, height=1).pack(fill="x")
    ctk.CTkLabel(comp_frame, text="COMP",
                 font=("Courier New", 12, "bold"),
                 text_color=TEXT_DIM).pack(pady=(2, 0))

    comp_en_var = ctk.BooleanVar(value=False)
    def _on_comp_en(k=key, v=comp_en_var):
        state.stem_comp_enabled[k] = v.get()
    ctk.CTkCheckBox(comp_frame, text="ON",
                    variable=comp_en_var, command=_on_comp_en,
                    font=("Courier New", 13), text_color=TEXT_DIM,
                    fg_color=STEEL, hover_color=STEEL_LIGHT,
                    checkmark_color="#ffffff", corner_radius=0,
                    border_color=BORDER, border_width=1,
                    checkbox_width=16, checkbox_height=16).pack(pady=(1, 0))

    def _comp_row(label, lo, hi, init, attr):
        r = ctk.CTkFrame(comp_frame, fg_color="transparent")
        r.pack(fill="x", padx=3, pady=0)
        ctk.CTkLabel(r, text=label, font=("Courier New", 13),
                     text_color=TEXT_DIM, width=42).pack(side="left")
        def _cmd(v, k=key, a=attr):
            if a == "thresh":   state.stem_comp_thresh[k]   = float(v)
            elif a == "ratio":  state.stem_comp_ratio[k]    = float(v)
            elif a == "atk":    state.stem_comp_attack[k]   = float(v)
            elif a == "rel":    state.stem_comp_release[k]  = float(v)
        sl = LockedSlider(r, from_=lo, to=hi, height=11,
                          button_color=STEEL, button_hover_color=STEEL_LIGHT,
                          progress_color=STEEL,
                          command=_cmd)
        sl.set(init)
        sl.pack(side="left", fill="x", expand=True)

    _comp_row("THR", -60, 0,    _COMP_DEFAULTS["thresh"],  "thresh")
    _comp_row("RAT",   1, 20,   _COMP_DEFAULTS["ratio"],   "ratio")
    _comp_row("ATK",   1, 200,  _COMP_DEFAULTS["attack"],  "atk")
    _comp_row("REL",  10, 1000, _COMP_DEFAULTS["release"], "rel")

    # --- gate rows ---
    tk.Frame(comp_frame, bg=STEEL, height=1).pack(fill="x", pady=(3, 0))
    ctk.CTkLabel(comp_frame, text="GATE",
                 font=("Courier New", 12, "bold"),
                 text_color=TEXT_DIM).pack(pady=(2, 0))

    gate_en_var = ctk.BooleanVar(value=False)
    def _on_gate_en(k=key, v=gate_en_var):
        state.stem_gate_enabled[k] = v.get()
    ctk.CTkCheckBox(comp_frame, text="ON",
                    variable=gate_en_var, command=_on_gate_en,
                    font=("Courier New", 13), text_color=TEXT_DIM,
                    fg_color=STEEL, hover_color=STEEL_LIGHT,
                    checkmark_color="#ffffff", corner_radius=0,
                    border_color=BORDER, border_width=1,
                    checkbox_width=16, checkbox_height=16).pack(pady=(1, 2))

    def _gate_row(label, lo, hi, init, attr):
        r = ctk.CTkFrame(comp_frame, fg_color="transparent")
        r.pack(fill="x", padx=3, pady=0)
        ctk.CTkLabel(r, text=label, font=("Courier New", 13),
                     text_color=TEXT_DIM, width=42).pack(side="left")
        def _cmd(v, k=key, a=attr):
            if a == "thresh":  state.stem_gate_thresh[k]   = float(v)
            elif a == "atk":   state.stem_gate_attack[k]   = float(v)
            elif a == "rel":   state.stem_gate_release[k]  = float(v)
        sl = LockedSlider(r, from_=lo, to=hi, height=11,
                          button_color=STEEL, button_hover_color=STEEL_LIGHT,
                          progress_color=STEEL, command=_cmd)
        sl.set(init)
        sl.pack(side="left", fill="x", expand=True)

    _gate_row("THR", -80, 0,   _GATE_DEFAULTS["thresh"],  "thresh")
    _gate_row("ATK",   1, 200, _GATE_DEFAULTS["attack"],  "atk")
    _gate_row("REL",  10, 500, _GATE_DEFAULTS["release"], "rel")
    tk.Frame(comp_frame, bg=BG, height=3).pack()

    return toggle_btn


for col_idx, name in enumerate(STEMS):
    cell = ctk.CTkFrame(mixer_grid,
                        fg_color=BG,
                        corner_radius=0,
                        border_color=STEEL, border_width=1)
    cell.grid(row=0, column=col_idx, padx=4, pady=4, sticky="nsew")
    mixer_grid.columnconfigure(col_idx, weight=1, minsize=MIXER_MIN_CELL_W)

    # ── Top bar + label row with M / S buttons ──────────────────────────
    tk.Frame(cell, bg=RED, height=3).pack(fill="x")

    _hdr = ctk.CTkFrame(cell, fg_color="transparent")
    _hdr.pack(fill="x", padx=2, pady=(2, 0))

    # Packed AFTER the M/S buttons below: pack() serves earlier widgets first,
    # so the buttons keep their space however narrow the cell gets.
    _name_lbl = ctk.CTkLabel(_hdr,
                             text=name.upper(),
                             font=FONT_LABEL, anchor="center", width=1,
                             text_color=BRIGHT_RED)

    # MUTE button
    def _make_mute(n):
        def _cmd():
            _toggle_mute(n)
            _inst_muted_by_us.discard(n)
            if n == "vocals":
                if _split_halves_exist():
                    # The split halves are the alternative to this stem, so
                    # the button swaps between them rather than soloing
                    # INSTRUM (which would silence the halves too).
                    _vocals_toggled_with_split()
                else:
                    _vocals_autosolo_check()
        return _cmd
    _mb = ctk.CTkButton(_hdr, text="M", command=_make_mute(name),
                         fg_color=STEEL, hover_color=MUTE_ON,
                         text_color=TEXT_MAIN,
                         font=FONT_MS_BTN,
                         corner_radius=0, border_width=1,
                         border_color=BORDER, height=MS_BTN_H, width=MS_BTN_W)
    _mb.pack(side="right", padx=(1, 2), pady=(6, 0))

    # SOLO button
    def _make_solo(n):
        def _cmd():
            state.stem_solo[n] = not state.stem_solo.get(n, False)
            _paint_ms(n)
        return _cmd
    _sb = ctk.CTkButton(_hdr, text="S", command=_make_solo(name),
                         fg_color=STEEL, hover_color=SOLO_ON,
                         text_color=TEXT_MAIN,
                         font=FONT_MS_BTN,
                         corner_radius=0, border_width=1,
                         border_color=BORDER, height=MS_BTN_H, width=MS_BTN_W)
    _sb.pack(side="right", padx=(0, 1), pady=(6, 0))
    _name_lbl.pack(side="left", fill="x", expand=True)

    _mute_btns[name] = _mb
    _solo_btns[name] = _sb
    if name == "vocals":
        _vocals_cell = cell          # needed for the refinement indicator
    if name == "other":
        _other_cell = cell           # needed for its phase button
        _vocals_hdr_ref = [_hdr]     # the indicator sits under the header

    # Sub-label, in the same place and style as the model credits on the
    # INST, FRT VOX and BG VOX cells.
    _sub_lbl = ctk.CTkLabel(cell, text="To be improved",
                            font=("Courier New", 13, "bold"),
                            text_color=BRIGHT_GREEN,
                            wraplength=140, justify="center")
    _sub_lbl.pack(pady=(3, 0), fill="x")
    if name == "vocals":
        _vocals_sub_ref = [_sub_lbl]   # the model row is packed under this

    # ── Volume slider ────────────────────────────────────────────────────
    def _make_vol_cmd(n):
        def _cmd(v):
            state.stem_volumes[n] = float(v)
        return _cmd

    sl = LockedSlider(cell, from_=0, to=2,
                       button_color=RED,
                       button_hover_color=GLOW_RED,
                       progress_color=RED,
                       command=_make_vol_cmd(name))
    sl.set(1.0)
    sl.pack(fill="x", padx=6, pady=(2, 1))
    stem_sliders[name] = sl

    # ── Pan slider ───────────────────────────────────────────────────────
    _pan_row = ctk.CTkFrame(cell, fg_color="transparent")
    _pan_row.pack(fill="x", padx=4, pady=0)
    ctk.CTkLabel(_pan_row, text="PAN",
                 font=("Courier New", 13, "bold"),
                 text_color=TEXT_DIM, width=22).pack(side="left")
    def _make_pan_cmd(n):
        def _cmd(v): state.stem_pan[n] = float(v)
        return _cmd
    _pan_sl = LockedSlider(_pan_row, from_=-1.0, to=1.0, height=12,
                            button_color=STEEL, button_hover_color=STEEL_LIGHT,
                            progress_color=STEEL,
                            command=_make_pan_cmd(name))
    _pan_sl.set(0.0)
    _pan_sl.pack(side="left", fill="x", expand=True)

    # ── SW / REV / AIR ───────────────────────────────────────────────────
    _sw_row = ctk.CTkFrame(cell, fg_color="transparent")
    _sw_row.pack(fill="x", padx=4, pady=0)
    ctk.CTkLabel(_sw_row, text="SW",
                 font=("Courier New", 13, "bold"),
                 text_color=TEXT_DIM, width=22).pack(side="left")
    def _make_sw_cmd(n):
        def _cmd(v): state.stem_widths[n] = float(v)
        return _cmd
    _sw_sl = LockedSlider(_sw_row, from_=0.0, to=2.0, height=12,
                           button_color=STEEL, button_hover_color=STEEL_LIGHT,
                           progress_color=STEEL, command=_make_sw_cmd(name))
    _sw_sl.set(1.0)
    _sw_sl.pack(side="left", fill="x", expand=True)

    _rev_row = ctk.CTkFrame(cell, fg_color="transparent")
    _rev_row.pack(fill="x", padx=4, pady=0)
    ctk.CTkLabel(_rev_row, text="REV",
                 font=("Courier New", 13, "bold"),
                 text_color=TEXT_DIM, width=22).pack(side="left")
    def _make_rev_cmd(n):
        def _cmd(v): state.stem_reverbs[n] = float(v)
        return _cmd
    LockedSlider(_rev_row, from_=0.0, to=1.0, height=12,
                 button_color=STEEL, button_hover_color=STEEL_LIGHT,
                 progress_color=STEEL,
                 command=_make_rev_cmd(name)).pack(side="left", fill="x", expand=True)

    _air_row = ctk.CTkFrame(cell, fg_color="transparent")
    _air_row.pack(fill="x", padx=4, pady=0)
    ctk.CTkLabel(_air_row, text="AIR",
                 font=("Courier New", 13, "bold"),
                 text_color=TEXT_DIM, width=22).pack(side="left")
    def _make_air_cmd(n):
        def _cmd(v): state.stem_air[n] = float(v)
        return _cmd
    _air_s = LockedSlider(_air_row, from_=-1.0, to=1.0, height=12,
                          button_color=STEEL, button_hover_color=STEEL_LIGHT,
                          progress_color=STEEL, command=_make_air_cmd(name))
    _air_s.set(0.0)
    _air_s.pack(side="left", fill="x", expand=True)

    # NDG (nudge) sliders removed. state.stem_nudge stays empty, so the
    # mixer treats every stem as un-nudged.

    # ── RST + meter ──────────────────────────────────────────────────────
    def _make_reset(n, s):
        def _reset():
            s.set(1.0)
            state.stem_volumes[n] = 1.0
        return _reset

    rst_meter_row = ctk.CTkFrame(cell, fg_color="transparent")
    rst_meter_row.pack(pady=(0, 2))
    ctk.CTkButton(rst_meter_row, text="RST",
                  command=_make_reset(name, sl),
                  fg_color=STEEL, hover_color=STEEL_LIGHT,
                  text_color=TEXT_DIM,
                  font=("Courier New", 13, "bold"),
                  corner_radius=0, border_width=1,
                  border_color=BORDER, height=18, width=36
                  ).pack(side="left", padx=(0, 4))
    make_meter(rst_meter_row, name).pack(side="left")

    # ── COMP / LEVELLER / LIMITER, folded behind one header ─────────────
    # Each of the three already folds its own controls away, but their
    # three buttons were always on show in every cell. They now sit inside
    # one group that folds away entirely, starting folded.
    _make_dynamics_group(cell, name)

    # VOC FOCUS used to sit here in the VOCALS cell; it has been removed.
    tk.Frame(cell, bg=BG, height=6).pack()

# FRT VOX — user-imported audio stem cell (column 6)
fv_col = len(STEMS) + 1      # STRINGS sits at len(STEMS), just left of here
mixer_grid.columnconfigure(fv_col, weight=1, minsize=MIXER_MIN_CELL_W)

fv_cell = ctk.CTkFrame(mixer_grid,
                        fg_color=BG,
                        corner_radius=0,
                        border_color=STEEL, border_width=1)
fv_cell.grid(row=0, column=fv_col, padx=4, pady=4, sticky="nsew")

tk.Frame(fv_cell, bg=RED, height=3).pack(fill="x")

_fv_hdr = ctk.CTkFrame(fv_cell, fg_color="transparent")
_fv_hdr.pack(fill="x", padx=2, pady=(2, 0))
_fv_name_lbl = ctk.CTkLabel(_fv_hdr, text="FRT VOX", anchor="center", width=1,
                            font=FONT_LABEL, text_color=BRIGHT_RED)
def _fv_mute():
    _toggle_mute("front_vocals")
    _halves_muted_by_vocals.discard("front_vocals")
    # Bringing a vocal half in takes over from the full VOCALS stem, so the
    # two never play together. (Muting a half leaves VOCALS as it is — you
    # may well want all the vocals silent.)
    if _split_vocals_active() and not getattr(state, "vocals_is_vca", False):
        state.stem_mute["vocals"] = True
        _paint_ms("vocals")
def _fv_solo():
    state.stem_solo["front_vocals"] = not state.stem_solo.get("front_vocals", False)
    _paint_ms("front_vocals")
_fv_mb = ctk.CTkButton(_fv_hdr, text="M", command=_fv_mute,
                        fg_color=STEEL, hover_color=MUTE_ON, text_color=TEXT_MAIN,
                        font=FONT_MS_BTN, corner_radius=0,
                        border_width=1, border_color=BORDER, height=MS_BTN_H, width=MS_BTN_W)
_fv_mb.pack(side="right", padx=(1, 2), pady=(6, 0))
_fv_sb = ctk.CTkButton(_fv_hdr, text="S", command=_fv_solo,
                        fg_color=STEEL, hover_color=SOLO_ON, text_color=TEXT_MAIN,
                        font=FONT_MS_BTN, corner_radius=0,
                        border_width=1, border_color=BORDER, height=MS_BTN_H, width=MS_BTN_W)
_fv_sb.pack(side="right", padx=(0, 1), pady=(6, 0))
_fv_name_lbl.pack(side="left", fill="x", expand=True)
_mute_btns["front_vocals"] = _fv_mb
_solo_btns["front_vocals"] = _fv_sb

# Import button — disabled until main stems are loaded
def _update_fv_button():
    """Label the FRT VOX button with what the karaoke model is doing."""
    try:
        btn = fv_import_btn
    except NameError:
        return
    if _kara_separating:
        btn.configure(text="⟳ SPLITTING…", text_color="#ffaa00")
    elif state.fv_data is not None:
        btn.configure(text="⟳ RE-SPLIT", text_color=BRIGHT_GREEN)
    elif kara_model is None and kara_model_ready:
        btn.configure(text="⬡ NO MODEL", text_color=TEXT_DIM)
    else:
        btn.configure(text="⟳ SPLIT VOX", text_color=TEXT_MAIN)


fv_import_btn = ctk.CTkButton(
    fv_cell,
    text="⟳ SPLIT VOX",
    command=lambda: _bgv_separate_now(),
    fg_color=STEEL,
    hover_color=STEEL_LIGHT,
    text_color=TEXT_MAIN,
    font=("Courier New", 13, "bold"),
    corner_radius=0,
    border_width=1,
    border_color=BORDER,
    height=20,
    state="disabled",
)
fv_import_btn.pack(fill="x", padx=6, pady=(0, 4))
# Sub-label: karaoke model credit, in the same form as the INST cell's.
_kara_title_fv_cell = ctk.CTkLabel(fv_cell, text=_KARA_TITLE,
             font=("Courier New", 11), text_color=STEEL_LIGHT,
             wraplength=140, justify="center")
_kara_title_fv_cell.pack(pady=(2, 0), fill="x")
# "by" stays in the dim credit colour; the name itself is picked out.
_kara_credit_row_fv_cell = ctk.CTkFrame(fv_cell, fg_color="transparent")
_kara_credit_row_fv_cell.pack()
_kby_fv_cell, _, _kwho_fv_cell = _KARA_CREDIT.partition(" ")
ctk.CTkLabel(_kara_credit_row_fv_cell, text=_kby_fv_cell + " ",
             font=("Courier New", 11), text_color=TEXT_DIM
             ).pack(side="left")
_kara_author_fv_cell = ctk.CTkLabel(
    _kara_credit_row_fv_cell, text=_kwho_fv_cell or _KARA_CREDIT,
    font=("Courier New", 13, "bold"), text_color=BRIGHT_GREEN)
_kara_author_fv_cell.pack(side="left")
_register_model_credit("karaoke", _kara_title_fv_cell, _kara_author_fv_cell)
add_model_picker(fv_cell, "karaoke", after=_kara_credit_row_fv_cell)
ctk.CTkButton(fv_cell, text="⇄ SWAP FRT/BG", command=swap_vocal_halves,
              fg_color=STEEL, hover_color=STEEL_LIGHT, text_color=TEXT_DIM,
              font=("Courier New", 11, "bold"), corner_radius=0,
              border_width=1, border_color=BORDER, height=18
              ).pack(fill="x", padx=6, pady=(1, 0))


# Karaoke-split progress. One pass fills both cells, so both show the same
# figure; each is packed only while the split runs.
_fv_prog_lbl = ctk.CTkLabel(fv_cell, text="0%",
                            font=("Courier New", 16, "bold"),
                            text_color=GLOW_RED)

# Volume slider
fv_vol_sl = LockedSlider(fv_cell, from_=0, to=2,
                          button_color=RED,
                          button_hover_color=GLOW_RED,
                          progress_color=RED,
                          command=lambda v: setattr(state, "fv_volume", float(v)))
fv_vol_sl.set(1.0)
fv_vol_sl.pack(fill="x", padx=6, pady=(0, 1))

# Pan
_fv_pan_row = ctk.CTkFrame(fv_cell, fg_color="transparent")
_fv_pan_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_fv_pan_row, text="PAN", font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
_fv_pan_sl = LockedSlider(_fv_pan_row, from_=-1.0, to=1.0, height=12,
                           button_color=STEEL, button_hover_color=STEEL_LIGHT,
                           progress_color=STEEL,
                           command=lambda v: state.stem_pan.__setitem__("front_vocals", float(v)))
_fv_pan_sl.set(0.0)
_fv_pan_sl.pack(side="left", fill="x", expand=True)

# Stereo Width slider
_fv_sw_row = ctk.CTkFrame(fv_cell, fg_color="transparent")
_fv_sw_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_fv_sw_row, text="SW",
             font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
_fv_sw_sl = LockedSlider(_fv_sw_row, from_=0.0, to=2.0, height=12,
                          button_color=STEEL, button_hover_color=STEEL_LIGHT,
                          progress_color=STEEL,
                          command=lambda v: state.stem_widths.__setitem__("front_vocals", float(v)))
_fv_sw_sl.set(1.0)
_fv_sw_sl.pack(side="left", fill="x", expand=True)

# Reverb
_fv_rev_row = ctk.CTkFrame(fv_cell, fg_color="transparent")
_fv_rev_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_fv_rev_row, text="REV", font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
LockedSlider(_fv_rev_row, from_=0.0, to=1.0, height=12,
             button_color=STEEL, button_hover_color=STEEL_LIGHT, progress_color=STEEL,
             command=lambda v: state.stem_reverbs.__setitem__("front_vocals", float(v))
             ).pack(side="left", fill="x", expand=True)

# Air
_fv_air_row = ctk.CTkFrame(fv_cell, fg_color="transparent")
_fv_air_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_fv_air_row, text="AIR", font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
_fv_air_sl = LockedSlider(_fv_air_row, from_=-1.0, to=1.0, height=12,
                           button_color=STEEL, button_hover_color=STEEL_LIGHT, progress_color=STEEL,
                           command=lambda v: state.stem_air.__setitem__("front_vocals", float(v)))
_fv_air_sl.set(0.0)
_fv_air_sl.pack(side="left", fill="x", expand=True)

# NDG (nudge) sliders removed. state.stem_nudge stays empty, so the
# mixer treats every stem as un-nudged.
# RST button
def _reset_fv():
    fv_vol_sl.set(1.0)
    setattr(state, "fv_volume", 1.0)

_fv_rst_row = ctk.CTkFrame(fv_cell, fg_color="transparent")
_fv_rst_row.pack(pady=(0, 2))
ctk.CTkButton(_fv_rst_row,
              text="RST",
              command=_reset_fv,
              fg_color=STEEL,
              hover_color=STEEL_LIGHT,
              text_color=TEXT_DIM,
              font=("Courier New", 13, "bold"),
              corner_radius=0,
              border_width=1,
              border_color=BORDER,
              height=18,
              width=36
              ).pack(side="left", padx=(0, 4))
make_meter(_fv_rst_row, "front_vocals").pack(side="left")
# COMP / LEVELLER / LIMITER, folded behind one header like the main cells
_make_dynamics_group(fv_cell, "front_vocals")

# (VOC FOCUS and its BKG slider were removed from this cell.)
tk.Frame(fv_cell, bg=BG, height=6).pack()

# BG VOX — user-imported audio stem cell (column 7)
bgv_col = len(STEMS) + 2
mixer_grid.columnconfigure(bgv_col, weight=1, minsize=MIXER_MIN_CELL_W)

bgv_cell = ctk.CTkFrame(mixer_grid,
                         fg_color=BG,
                         corner_radius=0,
                         border_color=STEEL, border_width=1)
bgv_cell.grid(row=0, column=bgv_col, padx=4, pady=4, sticky="nsew")

tk.Frame(bgv_cell, bg=RED, height=3).pack(fill="x")

_bgv_hdr = ctk.CTkFrame(bgv_cell, fg_color="transparent")
_bgv_hdr.pack(fill="x", padx=2, pady=(2, 0))
_bgv_name_lbl = ctk.CTkLabel(_bgv_hdr, text="BG VOX", anchor="center", width=1,
                             font=FONT_LABEL, text_color=BRIGHT_RED)
def _bgv_mute():
    _toggle_mute("bg_vocals")
    _halves_muted_by_vocals.discard("bg_vocals")
    # Bringing a vocal half in takes over from the full VOCALS stem, so the
    # two never play together. (Muting a half leaves VOCALS as it is — you
    # may well want all the vocals silent.)
    if _split_vocals_active() and not getattr(state, "vocals_is_vca", False):
        state.stem_mute["vocals"] = True
        _paint_ms("vocals")
def _bgv_solo():
    state.stem_solo["bg_vocals"] = not state.stem_solo.get("bg_vocals", False)
    _paint_ms("bg_vocals")
_bgv_mb = ctk.CTkButton(_bgv_hdr, text="M", command=_bgv_mute,
                         fg_color=STEEL, hover_color=MUTE_ON, text_color=TEXT_MAIN,
                         font=FONT_MS_BTN, corner_radius=0,
                         border_width=1, border_color=BORDER, height=MS_BTN_H, width=MS_BTN_W)
_bgv_mb.pack(side="right", padx=(1, 2), pady=(6, 0))
_bgv_sb = ctk.CTkButton(_bgv_hdr, text="S", command=_bgv_solo,
                         fg_color=STEEL, hover_color=SOLO_ON, text_color=TEXT_MAIN,
                         font=FONT_MS_BTN, corner_radius=0,
                         border_width=1, border_color=BORDER, height=MS_BTN_H, width=MS_BTN_W)
_bgv_sb.pack(side="right", padx=(0, 1), pady=(6, 0))
_bgv_name_lbl.pack(side="left", fill="x", expand=True)
_mute_btns["bg_vocals"] = _bgv_mb
_solo_btns["bg_vocals"] = _bgv_sb

# Import button — disabled until main stems are loaded
def _bgv_separate_now():
    """Re-run the karaoke split for the loaded song."""
    if state.separating or _kara_separating:
        return
    if not state.stems or state.stems.get("vocals") is None:
        print("[Karaoke] No vocals stem yet")
        return
    def _resplit():
        _lower_thread_priority()
        separate_bg_vocals()
    threading.Thread(target=_resplit, daemon=True).start()
    if running:
        app.after(0, _update_bgv_button)


def _update_bgv_button():
    """Label the button with what the karaoke model is doing."""
    try:
        btn = bgv_import_btn
    except NameError:
        return
    if _kara_separating:
        btn.configure(text="⟳ SPLITTING…", text_color="#ffaa00")
    elif state.bg_vocals_data is not None:
        btn.configure(text="⟳ RE-SPLIT", text_color=BRIGHT_GREEN)
    elif kara_model is None and kara_model_ready:
        btn.configure(text="⬡ NO MODEL", text_color=TEXT_DIM)
    else:
        btn.configure(text="⟳ SPLIT VOX", text_color=TEXT_MAIN)


bgv_import_btn = ctk.CTkButton(
    bgv_cell,
    text="⟳ SPLIT VOX",
    command=_bgv_separate_now,
    fg_color=STEEL,
    hover_color=STEEL_LIGHT,
    text_color=TEXT_MAIN,
    font=("Courier New", 13, "bold"),
    corner_radius=0,
    border_width=1,
    border_color=BORDER,
    height=20,
    state="disabled",
)
bgv_import_btn.pack(fill="x", padx=6, pady=(0, 4))
# Sub-label: karaoke model credit, in the same form as the INST cell's.
_kara_title_bgv_cell = ctk.CTkLabel(bgv_cell, text=_KARA_TITLE,
             font=("Courier New", 11), text_color=STEEL_LIGHT,
             wraplength=140, justify="center")
_kara_title_bgv_cell.pack(pady=(2, 0), fill="x")
# "by" stays in the dim credit colour; the name itself is picked out.
_kara_credit_row_bgv_cell = ctk.CTkFrame(bgv_cell, fg_color="transparent")
_kara_credit_row_bgv_cell.pack()
_kby_bgv_cell, _, _kwho_bgv_cell = _KARA_CREDIT.partition(" ")
ctk.CTkLabel(_kara_credit_row_bgv_cell, text=_kby_bgv_cell + " ",
             font=("Courier New", 11), text_color=TEXT_DIM
             ).pack(side="left")
_kara_author_bgv_cell = ctk.CTkLabel(
    _kara_credit_row_bgv_cell, text=_kwho_bgv_cell or _KARA_CREDIT,
    font=("Courier New", 13, "bold"), text_color=BRIGHT_GREEN)
_kara_author_bgv_cell.pack(side="left")
_register_model_credit("karaoke", _kara_title_bgv_cell, _kara_author_bgv_cell)
add_model_picker(bgv_cell, "karaoke", after=_kara_credit_row_bgv_cell)
ctk.CTkButton(bgv_cell, text="⇄ SWAP FRT/BG", command=swap_vocal_halves,
              fg_color=STEEL, hover_color=STEEL_LIGHT, text_color=TEXT_DIM,
              font=("Courier New", 11, "bold"), corner_radius=0,
              border_width=1, border_color=BORDER, height=18
              ).pack(fill="x", padx=6, pady=(1, 0))

_backing_src_btn = ctk.CTkButton(
    bgv_cell,
    text="BG: RESIDUE" if _KARA_BACKING_FROM_RESIDUE else "BG: MODEL",
    command=toggle_backing_source,
    fg_color=STEEL, hover_color=STEEL_LIGHT, text_color=TEXT_DIM,
    font=("Courier New", 11, "bold"), corner_radius=0,
    border_width=1, border_color=BORDER, height=18)
_backing_src_btn.pack(fill="x", padx=6, pady=(1, 0))


_bgv_prog_lbl = ctk.CTkLabel(bgv_cell, text="0%",
                             font=("Courier New", 16, "bold"),
                             text_color=GLOW_RED)


def _set_split_progress(frac):
    """Show how far the lead/backing split has got, on both cells."""
    pct = f"{int(round(max(0.0, min(1.0, frac)) * 100))}%"
    for _lbl, _cell, _after in ((_fv_prog_lbl,  fv_cell,  fv_import_btn),
                                (_bgv_prog_lbl, bgv_cell, bgv_import_btn)):
        try:
            if not _lbl.winfo_ismapped():
                _lbl.pack(pady=(0, 2), after=_after)
            _lbl.configure(text=pct)
        except Exception:
            pass


def _clear_split_progress():
    for _lbl in (_fv_prog_lbl, _bgv_prog_lbl):
        try:
            _lbl.configure(text="0%")
            _lbl.pack_forget()
        except Exception:
            pass

# Volume slider
bgv_vol_sl = LockedSlider(bgv_cell, from_=0, to=2,
                            button_color=RED,
                            button_hover_color=GLOW_RED,
                            progress_color=RED,
                            command=lambda v: setattr(state, "bg_vocals_volume", float(v)))
bgv_vol_sl.set(1.0)
bgv_vol_sl.pack(fill="x", padx=6, pady=(0, 1))

# Pan
_bgv_pan_row = ctk.CTkFrame(bgv_cell, fg_color="transparent")
_bgv_pan_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_bgv_pan_row, text="PAN", font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
_bgv_pan_sl = LockedSlider(_bgv_pan_row, from_=-1.0, to=1.0, height=12,
                            button_color=STEEL, button_hover_color=STEEL_LIGHT,
                            progress_color=STEEL,
                            command=lambda v: state.stem_pan.__setitem__("bg_vocals", float(v)))
_bgv_pan_sl.set(0.0)
_bgv_pan_sl.pack(side="left", fill="x", expand=True)

# Stereo Width slider
_bgv_sw_row = ctk.CTkFrame(bgv_cell, fg_color="transparent")
_bgv_sw_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_bgv_sw_row, text="SW",
             font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
_bgv_sw_sl = LockedSlider(_bgv_sw_row, from_=0.0, to=2.0, height=12,
                           button_color=STEEL, button_hover_color=STEEL_LIGHT,
                           progress_color=STEEL,
                           command=lambda v: state.stem_widths.__setitem__("bg_vocals", float(v)))
_bgv_sw_sl.set(1.0)
_bgv_sw_sl.pack(side="left", fill="x", expand=True)

# Reverb
_bgv_rev_row = ctk.CTkFrame(bgv_cell, fg_color="transparent")
_bgv_rev_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_bgv_rev_row, text="REV", font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
LockedSlider(_bgv_rev_row, from_=0.0, to=1.0, height=12,
             button_color=STEEL, button_hover_color=STEEL_LIGHT, progress_color=STEEL,
             command=lambda v: state.stem_reverbs.__setitem__("bg_vocals", float(v))
             ).pack(side="left", fill="x", expand=True)

# Air
_bgv_air_row = ctk.CTkFrame(bgv_cell, fg_color="transparent")
_bgv_air_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_bgv_air_row, text="AIR", font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
_bgv_air_sl = LockedSlider(_bgv_air_row, from_=-1.0, to=1.0, height=12,
                            button_color=STEEL, button_hover_color=STEEL_LIGHT, progress_color=STEEL,
                            command=lambda v: state.stem_air.__setitem__("bg_vocals", float(v)))
_bgv_air_sl.set(0.0)
_bgv_air_sl.pack(side="left", fill="x", expand=True)

# NDG (nudge) sliders removed. state.stem_nudge stays empty, so the
# mixer treats every stem as un-nudged.
# RST button
def _reset_bgv():
    bgv_vol_sl.set(1.0)
    setattr(state, "bg_vocals_volume", 1.0)

_bgv_rst_row = ctk.CTkFrame(bgv_cell, fg_color="transparent")
_bgv_rst_row.pack(pady=(0, 2))
ctk.CTkButton(_bgv_rst_row,
              text="RST",
              command=_reset_bgv,
              fg_color=STEEL,
              hover_color=STEEL_LIGHT,
              text_color=TEXT_DIM,
              font=("Courier New", 13, "bold"),
              corner_radius=0,
              border_width=1,
              border_color=BORDER,
              height=18,
              width=36
              ).pack(side="left", padx=(0, 4))
make_meter(_bgv_rst_row, "bg_vocals").pack(side="left")
# COMP / LEVELLER / LIMITER, folded behind one header like the main cells
_make_dynamics_group(bgv_cell, "bg_vocals")

# (VOC FOCUS and its BKG slider were removed from this cell.)
tk.Frame(bgv_cell, bg=BG, height=6).pack()

# HIDDEN LAYER — user-imported audio stem cell (column 8)
hl_col = len(STEMS) + 3
# HID LAYER retired: constructed into a frame that is never shown, so the
# widgets the rest of the file refers to still exist while nothing appears in
# the mixer. Its grid column is deliberately left unconfigured so it takes no
# space either.
_retired_cells = ctk.CTkFrame(app, fg_color=BG)

hl_cell = ctk.CTkFrame(_retired_cells,
                        fg_color=BG,
                        corner_radius=0,
                        border_color=STEEL, border_width=1)
hl_cell.grid(row=0, column=hl_col, padx=4, pady=4, sticky="nsew")

tk.Frame(hl_cell, bg=RED, height=3).pack(fill="x")

_hl_hdr = ctk.CTkFrame(hl_cell, fg_color="transparent")
_hl_hdr.pack(fill="x", padx=2, pady=(2, 0))
_hl_name_lbl = ctk.CTkLabel(_hl_hdr, text="HID LAYER", anchor="center", width=1,
                            font=FONT_LABEL, text_color=BRIGHT_RED)
def _hl_mute():
    _toggle_mute("hidden_layer")
def _hl_solo():
    state.stem_solo["hidden_layer"] = not state.stem_solo.get("hidden_layer", False)
    _paint_ms("hidden_layer")
_hl_mb = ctk.CTkButton(_hl_hdr, text="M", command=_hl_mute,
                        fg_color=STEEL, hover_color=MUTE_ON, text_color=TEXT_MAIN,
                        font=FONT_MS_BTN, corner_radius=0,
                        border_width=1, border_color=BORDER, height=MS_BTN_H, width=MS_BTN_W)
_hl_mb.pack(side="right", padx=(1, 2), pady=(6, 0))
_hl_sb = ctk.CTkButton(_hl_hdr, text="S", command=_hl_solo,
                        fg_color=STEEL, hover_color=SOLO_ON, text_color=TEXT_MAIN,
                        font=FONT_MS_BTN, corner_radius=0,
                        border_width=1, border_color=BORDER, height=MS_BTN_H, width=MS_BTN_W)
_hl_sb.pack(side="right", padx=(0, 1), pady=(6, 0))
_hl_name_lbl.pack(side="left", fill="x", expand=True)
_mute_btns["hidden_layer"] = _hl_mb
_solo_btns["hidden_layer"] = _hl_sb

# Import button — disabled until main stems are loaded
hl_import_btn = ctk.CTkButton(
    hl_cell,
    text="⬡ IMPORT",
    command=load_hidden_layer,
    fg_color=STEEL,
    hover_color=STEEL_LIGHT,
    text_color=TEXT_MAIN,
    font=("Courier New", 13, "bold"),
    corner_radius=0,
    border_width=1,
    border_color=BORDER,
    height=20,
    state="disabled",
)
hl_import_btn.pack(fill="x", padx=6, pady=(0, 4))

# Volume slider
hl_vol_sl = LockedSlider(hl_cell, from_=0, to=2,
                          button_color=RED,
                          button_hover_color=GLOW_RED,
                          progress_color=RED,
                          command=lambda v: setattr(state, "hl_volume", float(v)))
hl_vol_sl.set(1.0)
hl_vol_sl.pack(fill="x", padx=6, pady=(0, 1))

# Pan
_hl_pan_row = ctk.CTkFrame(hl_cell, fg_color="transparent")
_hl_pan_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_hl_pan_row, text="PAN", font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
_hl_pan_sl = LockedSlider(_hl_pan_row, from_=-1.0, to=1.0, height=12,
                           button_color=STEEL, button_hover_color=STEEL_LIGHT,
                           progress_color=STEEL,
                           command=lambda v: state.stem_pan.__setitem__("hidden_layer", float(v)))
_hl_pan_sl.set(0.0)
_hl_pan_sl.pack(side="left", fill="x", expand=True)

# Stereo Width slider
_hl_sw_row = ctk.CTkFrame(hl_cell, fg_color="transparent")
_hl_sw_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_hl_sw_row, text="SW",
             font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
_hl_sw_sl = LockedSlider(_hl_sw_row, from_=0.0, to=2.0, height=12,
                          button_color=STEEL, button_hover_color=STEEL_LIGHT,
                          progress_color=STEEL,
                          command=lambda v: state.stem_widths.__setitem__("hidden_layer", float(v)))
_hl_sw_sl.set(1.0)
_hl_sw_sl.pack(side="left", fill="x", expand=True)

# Reverb
_hl_rev_row = ctk.CTkFrame(hl_cell, fg_color="transparent")
_hl_rev_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_hl_rev_row, text="REV", font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
LockedSlider(_hl_rev_row, from_=0.0, to=1.0, height=12,
             button_color=STEEL, button_hover_color=STEEL_LIGHT, progress_color=STEEL,
             command=lambda v: state.stem_reverbs.__setitem__("hidden_layer", float(v))
             ).pack(side="left", fill="x", expand=True)

# Air
_hl_air_row = ctk.CTkFrame(hl_cell, fg_color="transparent")
_hl_air_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_hl_air_row, text="AIR", font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
_hl_air_sl = LockedSlider(_hl_air_row, from_=-1.0, to=1.0, height=12,
                           button_color=STEEL, button_hover_color=STEEL_LIGHT, progress_color=STEEL,
                           command=lambda v: state.stem_air.__setitem__("hidden_layer", float(v)))
_hl_air_sl.set(0.0)
_hl_air_sl.pack(side="left", fill="x", expand=True)

# NDG (nudge) sliders removed. state.stem_nudge stays empty, so the
# mixer treats every stem as un-nudged.
# RST button
def _reset_hl():
    hl_vol_sl.set(1.0)
    setattr(state, "hl_volume", 1.0)

_hl_rst_row = ctk.CTkFrame(hl_cell, fg_color="transparent")
_hl_rst_row.pack(pady=(0, 2))
ctk.CTkButton(_hl_rst_row,
              text="RST",
              command=_reset_hl,
              fg_color=STEEL,
              hover_color=STEEL_LIGHT,
              text_color=TEXT_DIM,
              font=("Courier New", 13, "bold"),
              corner_radius=0,
              border_width=1,
              border_color=BORDER,
              height=18,
              width=36
              ).pack(side="left", padx=(0, 4))
make_meter(_hl_rst_row, "hidden_layer").pack(side="left")
# COMP / LEVELLER / LIMITER, folded behind one header like the main cells
_make_dynamics_group(hl_cell, "hidden_layer")

# ── VOC FOCUS — collapsible ──────────────────────────────────────────────
_hl_extra_frame = ctk.CTkFrame(hl_cell, fg_color=PANEL,
                                corner_radius=0,
                                border_color=STEEL, border_width=1)

def _toggle_hl_extra(f=_hl_extra_frame):
    if f.winfo_ismapped():
        f.pack_forget()
    else:
        f.pack(fill="x", padx=4, pady=(2, 0))

ctk.CTkButton(hl_cell,
              text="VOC FOCUS",
              command=_toggle_hl_extra,
              fg_color=STEEL,
              hover_color=STEEL_LIGHT,
              text_color=TEXT_DIM,
              font=("Courier New", 12, "bold"),
              corner_radius=0,
              border_width=1,
              border_color=BORDER,
              height=16).pack(fill="x", padx=4, pady=(3, 0))

# --- VOC FOCUS inside the panel ---
tk.Frame(_hl_extra_frame, bg=STEEL, height=1).pack(fill="x")
ctk.CTkLabel(_hl_extra_frame, text="VOC FOCUS",
             font=("Courier New", 12, "bold"),
             text_color=TEXT_DIM).pack(pady=(2, 0))

hl_vff_var = ctk.BooleanVar(value=False)
def _on_hl_vff(v=hl_vff_var):
    state.hl_vff_enabled = v.get()
ctk.CTkCheckBox(_hl_extra_frame,
                text="ON",
                variable=hl_vff_var,
                command=_on_hl_vff,
                font=("Courier New", 10),
                text_color=TEXT_DIM,
                fg_color=STEEL,
                hover_color=STEEL_LIGHT,
                checkmark_color="#ffffff",
                corner_radius=0,
                border_color=BORDER,
                border_width=1,
                checkbox_width=12,
                checkbox_height=12).pack(pady=(1, 0))

for _hl_lbl, _hl_attr, _hl_init in [
    ("LCD", "hl_vff_lead_cut", state.hl_vff_lead_cut),
    ("BDY", "hl_vff_body_cut", state.hl_vff_body_cut),
    ("PRS", "hl_vff_presence",  state.hl_vff_presence),
]:
    _hl_row = ctk.CTkFrame(_hl_extra_frame, fg_color="transparent")
    _hl_row.pack(fill="x", padx=3, pady=0)
    ctk.CTkLabel(_hl_row, text=_hl_lbl,
                 font=("Courier New", 10),
                 text_color=TEXT_DIM,
                 width=30).pack(side="left")
    def _make_hl_vff_cmd(attr):
        def _cmd(v):
            setattr(state, attr, float(v))
        return _cmd
    LockedSlider(_hl_row, from_=0.0, to=1.0, height=11,
                 button_color=STEEL, button_hover_color=STEEL_LIGHT,
                 progress_color=STEEL,
                 command=_make_hl_vff_cmd(_hl_attr)).pack(side="left", fill="x", expand=True)

tk.Frame(_hl_extra_frame, bg=STEEL, height=1).pack(fill="x", padx=3, pady=(3, 0))
_bkg_hl_r = ctk.CTkFrame(_hl_extra_frame, fg_color="transparent")
_bkg_hl_r.pack(fill="x", padx=3, pady=0)
ctk.CTkLabel(_bkg_hl_r, text="BKG",
             font=("Courier New", 10),
             text_color=TEXT_DIM, width=30).pack(side="left")
bkg_hl_sl = LockedSlider(_bkg_hl_r, from_=0.0, to=3.0, height=11,
                          button_color=RED, button_hover_color=GLOW_RED,
                          progress_color=RED,
                          command=lambda v: setattr(state, "hl_vff_bkg_vol", float(v)))
bkg_hl_sl.set(state.hl_vff_bkg_vol)
bkg_hl_sl.pack(side="left", fill="x", expand=True)
tk.Frame(_hl_extra_frame, bg=BG, height=3).pack()
tk.Frame(hl_cell, bg=BG, height=4).pack()


def _make_import_cell(col_idx, label, key, vol_global, load_fn,
                      import_btn_var_name, grid=None):
    """Build a standard import-stem mixer cell (the ANY and ATMOS slots)."""
    grid = grid if grid is not None else mixer_grid
    grid.columnconfigure(col_idx, weight=1, minsize=MIXER_MIN_CELL_W)
    cell = ctk.CTkFrame(grid, fg_color=BG, corner_radius=0,
                         border_color=STEEL, border_width=1)
    cell.grid(row=0, column=col_idx, padx=4, pady=4, sticky="nsew")

    tk.Frame(cell, bg=RED, height=3).pack(fill="x")
    _hdr = ctk.CTkFrame(cell, fg_color="transparent")
    _hdr.pack(fill="x", padx=2, pady=(2, 0))
    _imp_lbl = ctk.CTkLabel(_hdr, text=label, font=FONT_LABEL, anchor="center",
                            width=1, text_color=BRIGHT_RED)

    def _mute_fn():
        _toggle_mute(key)
    def _solo_fn():
        state.stem_solo[key] = not state.stem_solo.get(key, False)
        _paint_ms(key)

    _mb = ctk.CTkButton(_hdr, text="M", command=_mute_fn,
                         fg_color=STEEL, hover_color=MUTE_ON, text_color=TEXT_MAIN,
                         font=FONT_MS_BTN, corner_radius=0,
                         border_width=1, border_color=BORDER, height=MS_BTN_H, width=MS_BTN_W)
    _mb.pack(side="right", padx=(1, 2), pady=(6, 0))
    _sb = ctk.CTkButton(_hdr, text="S", command=_solo_fn,
                         fg_color=STEEL, hover_color=SOLO_ON, text_color=TEXT_MAIN,
                         font=FONT_MS_BTN, corner_radius=0,
                         border_width=1, border_color=BORDER, height=MS_BTN_H, width=MS_BTN_W)
    _sb.pack(side="right", padx=(0, 1), pady=(6, 0))
    _imp_lbl.pack(side="left", fill="x", expand=True)
    _mute_btns[key] = _mb
    _solo_btns[key] = _sb

    # Enabled from the start: an imported file does not need a separated
    # track to exist, and plays on its own.
    _imp_row = ctk.CTkFrame(cell, fg_color="transparent")
    _imp_row.pack(fill="x", padx=6, pady=(2, 4))

    import_btn = ctk.CTkButton(_imp_row, text="⬡ IMPORT", command=load_fn,
                                fg_color=STEEL, hover_color=STEEL_LIGHT,
                                text_color=TEXT_MAIN,
                                font=("Courier New", 13, "bold"),
                                corner_radius=0, border_width=1,
                                border_color=BORDER, height=20)
    globals()[import_btn_var_name] = import_btn

    # The clear button is packed first, against the right edge, so the
    # import button expands into what is left. Packed the other way round
    # the expanding button takes everything and squeezes this to a sliver.
    ctk.CTkButton(_imp_row, text="✕",
                  command=lambda k=key, b=import_btn: _clear_import(k, b),
                  fg_color=STEEL, hover_color=MUTE_ON, text_color=TEXT_DIM,
                  font=("Courier New", 12, "bold"),
                  corner_radius=0, border_width=1, border_color=BORDER,
                  height=20, width=26).pack(side="right", padx=(2, 0))
    import_btn.pack(side="left", fill="x", expand=True)

    vol_sl = LockedSlider(cell, from_=0, to=2,
                           button_color=RED, button_hover_color=GLOW_RED,
                           progress_color=RED,
                           command=lambda v, k=key: setattr(state, vol_global, float(v)))
    vol_sl.set(1.0)
    vol_sl.pack(fill="x", padx=6, pady=(0, 1))

    for lbl2, from2, to2, init2, dict_ref, def_val in [
        ("PAN", -1.0, 1.0, 0.0,  state.stem_pan,    0.0),
        ("SW",   0.0, 2.0, 1.0,  state.stem_widths, 1.0),
        ("REV",  0.0, 1.0, 0.0,  state.stem_reverbs,0.0),
        ("AIR", -1.0, 1.0, 0.0,  state.stem_air,    0.0),
    ]:
        row = ctk.CTkFrame(cell, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=0)
        ctk.CTkLabel(row, text=lbl2, font=("Courier New", 13, "bold"),
                     text_color=TEXT_DIM, width=22).pack(side="left")
        def _cmd_factory(d=dict_ref, k=key):
            def _cmd(v):
                if d is not None:
                    d[k] = float(v)
            return _cmd
        sl2 = LockedSlider(row, from_=from2, to=to2, height=12,
                            button_color=STEEL, button_hover_color=STEEL_LIGHT,
                            progress_color=STEEL, command=_cmd_factory())
        sl2.set(init2)
        sl2.pack(side="left", fill="x", expand=True)

    rst_row = ctk.CTkFrame(cell, fg_color="transparent")
    rst_row.pack(pady=(0, 2))

    def _rst(vg=vol_global, vs=vol_sl):
        setattr(state, vg, 1.0)
        vs.set(1.0)
    ctk.CTkButton(rst_row, text="RST", command=_rst,
                  fg_color=STEEL, hover_color=STEEL_LIGHT, text_color=TEXT_DIM,
                  font=("Courier New", 13, "bold"), corner_radius=0,
                  border_width=1, border_color=BORDER, height=18, width=36
                  ).pack(side="left", padx=(0, 4))
    make_meter(rst_row, key).pack(side="left")

    # COMP / LEVELLER / LIMITER, folded behind one header like the main cells
    _make_dynamics_group(cell, key)
    tk.Frame(cell, bg=BG, height=4).pack()
    return cell


any_col   = len(STEMS) + 4
any_plus_col = len(STEMS) + 5          # kept: the removed ANY+ cell's column
any_plusplus_col      = len(STEMS) + 6 # kept: the removed ANY++ cell's column

any_import_btn   = None   # filled by _make_import_cell
any_plus_import_btn = None
any_plusplus_import_btn      = None

_make_import_cell(any_col,   "ANY",     "any",   "any_volume",   load_any,   "any_import_btn")
# ANY+ and ANY++ have been replaced by the STRINGS cell below. Their loaders
# and state stay, so an old session mentioning them still loads.

# ── STRINGS — gilliaan's bowed strings model ───────────────────────────────
# Laid out like the FRT VOX and BG VOX cells, with the INST cell's AUTO
# SEPARATE switch and SEPARATE NOW button. Nothing is imported here: the
# cell is filled by the model, so it has no IMPORT or clear button.
# Immediately right of OTHER and left of FRT VOX.
strings_col = len(STEMS)
mixer_grid.columnconfigure(strings_col, weight=1, minsize=MIXER_MIN_CELL_W)

_strings_cell = ctk.CTkFrame(mixer_grid, fg_color=BG, corner_radius=0,
                             border_color=STEEL, border_width=1)
_strings_cell.grid(row=0, column=strings_col, padx=4, pady=4, sticky="nsew")

tk.Frame(_strings_cell, bg=RED, height=3).pack(fill="x")

_strings_hdr = ctk.CTkFrame(_strings_cell, fg_color="transparent")
_strings_hdr.pack(fill="x", padx=2, pady=(2, 0))
_strings_name_lbl = ctk.CTkLabel(_strings_hdr, text="STRINGS", anchor="center",
                                 width=1, font=FONT_LABEL,
                                 text_color=BRIGHT_RED)


def _strings_mute():
    _toggle_mute("strings")


def _strings_solo():
    state.stem_solo["strings"] = not state.stem_solo.get("strings", False)
    _paint_ms("strings")


_strings_mb = ctk.CTkButton(_strings_hdr, text="M", command=_strings_mute,
                            fg_color=STEEL, hover_color=MUTE_ON,
                            text_color=TEXT_MAIN, font=FONT_MS_BTN,
                            corner_radius=0, border_width=1,
                            border_color=BORDER, height=MS_BTN_H, width=MS_BTN_W)
_strings_mb.pack(side="right", padx=(1, 2), pady=(6, 0))
_strings_sb = ctk.CTkButton(_strings_hdr, text="S", command=_strings_solo,
                            fg_color=STEEL, hover_color=SOLO_ON,
                            text_color=TEXT_MAIN, font=FONT_MS_BTN,
                            corner_radius=0, border_width=1,
                            border_color=BORDER, height=MS_BTN_H, width=MS_BTN_W)
_strings_sb.pack(side="right", padx=(0, 1), pady=(6, 0))
_strings_name_lbl.pack(side="left", fill="x", expand=True)
_mute_btns["strings"] = _strings_mb
_solo_btns["strings"] = _strings_sb

# The model's name and author, directly under the cell's own name.
_str_title_lbl = ctk.CTkLabel(_strings_cell, text=_STR_TITLE,
             font=("Courier New", 11), text_color=STEEL_LIGHT,
             wraplength=140, justify="center")
_str_title_lbl.pack(pady=(2, 0), fill="x")
_str_credit_row = ctk.CTkFrame(_strings_cell, fg_color="transparent")
_str_credit_row.pack()
_sby, _, _swho = _STR_CREDIT.partition(" ")
ctk.CTkLabel(_str_credit_row, text=_sby + " ",
             font=("Courier New", 11), text_color=TEXT_DIM).pack(side="left")
_str_author_lbl = ctk.CTkLabel(_str_credit_row, text=_swho or _STR_CREDIT,
             font=("Courier New", 13, "bold"), text_color=BRIGHT_GREEN)
_str_author_lbl.pack(side="left")
_register_model_credit("strings", _str_title_lbl, _str_author_lbl)

# What the model is doing, worded as on the INST cell.
_strings_status_lbl = ctk.CTkLabel(_strings_cell, text="WAITING",
                                   font=("Courier New", 10, "bold"),
                                   text_color=TEXT_DIM)
_strings_status_lbl.pack(pady=(2, 0))


def _update_strings_status_label():
    lbl = globals().get("_strings_status_lbl")
    if lbl is None:
        return
    try:
        if _strings_separating:
            lbl.configure(text="SEPARATING…", text_color="#ffaa00")
        elif state.strings_data is not None:
            if getattr(state, "strings_is_quick", False):
                lbl.configure(text="QUICK MIX", text_color="#ffaa00")
            else:
                lbl.configure(text="READY", text_color=BRIGHT_GREEN)
        elif not _str_find_files()[0]:
            lbl.configure(text="NO MODEL", text_color=TEXT_DIM)
        elif not _STR_AUTO:
            lbl.configure(text="AUTO OFF", text_color=TEXT_DIM)
        else:
            lbl.configure(text="WAITING", text_color=TEXT_DIM)
    except Exception:
        pass


def _update_strings_button():
    """Kept under its old name: the separation code calls this when it ends."""
    _update_strings_status_label()


# Progress for this cell's pass, shown only while it runs.
_strings_prog_row = ctk.CTkFrame(_strings_cell, fg_color="transparent")
_strings_prog_bar = ctk.CTkProgressBar(_strings_prog_row, progress_color=GLOW_RED,
                                       fg_color=PANEL, corner_radius=0, height=6)
_strings_prog_bar.set(0)
_strings_prog_bar.pack(fill="x", padx=2)
_strings_prog_lbl = ctk.CTkLabel(_strings_prog_row, text="0%",
                                 font=("Courier New", 16, "bold"),
                                 text_color=GLOW_RED)
_strings_prog_lbl.pack()


def _set_strings_progress(frac):
    try:
        if not _strings_prog_row.winfo_ismapped():
            _strings_prog_row.pack(fill="x", padx=6, pady=(2, 0),
                                   after=_strings_status_lbl)
        _strings_prog_bar.set(max(0.0, min(1.0, frac)))
        _strings_prog_lbl.configure(text=f"{int(round(frac * 100))}%")
    except Exception:
        pass


def _clear_strings_progress():
    try:
        _strings_prog_bar.set(0)
        _strings_prog_lbl.configure(text="0%")
        _strings_prog_row.pack_forget()
    except Exception:
        pass


_strings_vol_sl = LockedSlider(_strings_cell, from_=0, to=2,
                          button_color=RED,
                          button_hover_color=GLOW_RED,
                          progress_color=RED,
                          command=lambda v: setattr(state, "strings_volume", float(v)))
_strings_vol_sl.set(1.0)
_strings_vol_sl.pack(fill="x", padx=6, pady=(0, 1))

# Pan
_strings_pan_row = ctk.CTkFrame(_strings_cell, fg_color="transparent")
_strings_pan_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_strings_pan_row, text="PAN", font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
_strings_pan_sl = LockedSlider(_strings_pan_row, from_=-1.0, to=1.0, height=12,
                           button_color=STEEL, button_hover_color=STEEL_LIGHT,
                           progress_color=STEEL,
                           command=lambda v: state.stem_pan.__setitem__("strings", float(v)))
_strings_pan_sl.set(0.0)
_strings_pan_sl.pack(side="left", fill="x", expand=True)

# Stereo Width slider
_strings_sw_row = ctk.CTkFrame(_strings_cell, fg_color="transparent")
_strings_sw_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_strings_sw_row, text="SW",
             font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
_strings_sw_sl = LockedSlider(_strings_sw_row, from_=0.0, to=2.0, height=12,
                          button_color=STEEL, button_hover_color=STEEL_LIGHT,
                          progress_color=STEEL,
                          command=lambda v: state.stem_widths.__setitem__("strings", float(v)))
_strings_sw_sl.set(1.0)
_strings_sw_sl.pack(side="left", fill="x", expand=True)

# Reverb
_strings_rev_row = ctk.CTkFrame(_strings_cell, fg_color="transparent")
_strings_rev_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_strings_rev_row, text="REV", font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
LockedSlider(_strings_rev_row, from_=0.0, to=1.0, height=12,
             button_color=STEEL, button_hover_color=STEEL_LIGHT, progress_color=STEEL,
             command=lambda v: state.stem_reverbs.__setitem__("strings", float(v))
             ).pack(side="left", fill="x", expand=True)

# Air
_strings_air_row = ctk.CTkFrame(_strings_cell, fg_color="transparent")
_strings_air_row.pack(fill="x", padx=4, pady=0)
ctk.CTkLabel(_strings_air_row, text="AIR", font=("Courier New", 13, "bold"),
             text_color=TEXT_DIM, width=22).pack(side="left")
_strings_air_sl = LockedSlider(_strings_air_row, from_=-1.0, to=1.0, height=12,
                           button_color=STEEL, button_hover_color=STEEL_LIGHT, progress_color=STEEL,
                           command=lambda v: state.stem_air.__setitem__("strings", float(v)))
_strings_air_sl.set(0.0)
_strings_air_sl.pack(side="left", fill="x", expand=True)

# NDG (nudge) sliders removed. state.stem_nudge stays empty, so the
# mixer treats every stem as un-nudged.
# RST button
def _reset_strings():
    _strings_vol_sl.set(1.0)
    setattr(state, "strings_volume", 1.0)

_strings_rst_row = ctk.CTkFrame(_strings_cell, fg_color="transparent")
_strings_rst_row.pack(pady=(0, 2))
ctk.CTkButton(_strings_rst_row,
              text="RST",
              command=_reset_strings,
              fg_color=STEEL,
              hover_color=STEEL_LIGHT,
              text_color=TEXT_DIM,
              font=("Courier New", 13, "bold"),
              corner_radius=0,
              border_width=1,
              border_color=BORDER,
              height=18,
              width=36
              ).pack(side="left", padx=(0, 4))
make_meter(_strings_rst_row, "strings").pack(side="left")
# COMP / LEVELLER / LIMITER, folded behind one header like the main cells
_make_dynamics_group(_strings_cell, "strings")

# The model's own controls sit under the dynamics button, at the foot of the
# cell.
# AUTO SEPARATE — run after every track, as the INST cell does.
_strings_auto_var = ctk.BooleanVar(value=_STR_AUTO)


def _on_strings_auto():
    global _STR_AUTO
    _STR_AUTO = _strings_auto_var.get()
    _update_strings_status_label()


ctk.CTkCheckBox(_strings_cell, text="AUTO SEPARATE",
                variable=_strings_auto_var, command=_on_strings_auto,
                font=("Courier New", 11), text_color=TEXT_MAIN,
                fg_color=RED, hover_color=BRIGHT_RED,
                checkmark_color="#ffffff", corner_radius=0,
                border_color=STEEL, checkbox_width=14, checkbox_height=14
                ).pack(padx=6, pady=(2, 0))


_strings_first_var = ctk.BooleanVar(value=_STR_FIRST)


def _on_strings_first():
    global _STR_FIRST
    _STR_FIRST = _strings_first_var.get()
    _update_strings_status_label()


ctk.CTkCheckBox(_strings_cell, text="LOAD FIRST",
                variable=_strings_first_var, command=_on_strings_first,
                font=("Courier New", 11), text_color=TEXT_MAIN,
                fg_color=RED, hover_color=BRIGHT_RED,
                checkmark_color="#ffffff", corner_radius=0,
                border_color=STEEL, checkbox_width=14, checkbox_height=14
                ).pack(padx=6, pady=(0, 0))


def _strings_separate_now():
    path = _playlist_current[0]
    if not path or state.separating or _strings_separating:
        return
    threading.Thread(target=_run_low_priority,
                     args=(separate_strings, path), daemon=True).start()
    _update_strings_status_label()


ctk.CTkButton(_strings_cell, text="⟳ SEPARATE NOW", command=_strings_separate_now,
              fg_color=STEEL, hover_color=STEEL_LIGHT, text_color=BRIGHT_GREEN,
              font=("Courier New", 12, "bold"), corner_radius=0,
              border_width=1, border_color=BORDER, height=20
              ).pack(fill="x", padx=6, pady=(3, 2))

tk.Frame(_strings_cell, bg=BG, height=4).pack()

_update_strings_status_label()

# ── ATMOS bed cells ────────────────────────────────────────────────────────
# Six import slots for the channels of an Atmos 5.1 bed, on their own row
# beneath the stem cells. Same controls as the ANY cells.
_atmos_col = 0
atmos_fl_import_btn   = None   # filled by _make_import_cell
atmos_fr_import_btn   = None
atmos_c_import_btn   = None
atmos_lfe_import_btn = None
atmos_bl_import_btn  = None
atmos_br_import_btn  = None
for _akey, _alabel, _aload in (
        ("atmos_fl",  "ATMOS FL",  load_atmos_fl),
        ("atmos_bl",  "ATMOS BL",  load_atmos_bl),
        ("atmos_c",   "ATMOS C",   load_atmos_c),
        ("atmos_fr",  "ATMOS FR",  load_atmos_fr),
        ("atmos_br",  "ATMOS BR",  load_atmos_br),
        ("atmos_lfe", "ATMOS LFE", load_atmos_lfe)):
    _make_import_cell(_atmos_col, _alabel, _akey, f"{_akey}_volume", _aload,
                      f"{_akey}_import_btn", grid=atmos_grid)
    _atmos_col += 1

# ── VOCALS ↔ INSTRUM link ──────────────────────────────────────────────────
# Muting the vocals, or pulling their fader to the minimum, solos (and
# un-mutes) the instrumental cell, so you hear the backing track on its own.
# Un-muting the vocals or raising the fader puts both back to normal.
_inst_autosolo_active  = False
_inst_autosolo_pending = [False]   # link asked for while a song was loading


def _vocals_autosolo_check():
    """Solo the instrumental when the VOCALS cell is muted.

    Driven by the M button only. Pulling the volume slider to zero used to
    trigger it as well, which made the fader jump the whole mix into
    instrumental-only as it passed the bottom; the slider now just sets the
    volume.
    """
    global _inst_autosolo_active
    if "instrumental" not in _mute_btns or "instrumental" not in _solo_btns:
        return   # instrumental cell not built yet

    vocals_down = bool(state.stem_mute.get("vocals", False))

    # While a song is separating there is no instrumental to solo yet: the
    # cell is empty, so switching solo on would mute every stem and leave the
    # player silent, with the mixer churning through a stem that holds
    # nothing. Remember the intent and apply it when the song is ready.
    if vocals_down and (state.separating or state.instrumental is None):
        _inst_autosolo_pending[0] = True
        return
    if not vocals_down:
        _inst_autosolo_pending[0] = False

    if vocals_down and not _inst_autosolo_active:
        _inst_autosolo_active = True
        state.stem_solo["instrumental"] = True
        state.stem_mute["instrumental"] = False
        _paint_ms("instrumental")
        _sync_instrumental_overlap()
    elif not vocals_down and _inst_autosolo_active:
        # Only undo what we switched on ourselves.
        _inst_autosolo_active = False
        state.stem_solo["instrumental"] = False
        state.stem_mute["instrumental"] = True
        _paint_ms("instrumental")
        _sync_instrumental_overlap()


# ── INSTRUMENTAL CELL ───────────────────────────────────────────────────
# Auto-populated by separate_inst() — no manual import button needed.
# ---------------------------------------------------------------------------
# The instrumental duplicates the other stems, so it starts muted and is
# un-muted automatically when the vocals fader is pulled all the way down
# (see _vocals_autosolo_check).
state.stem_mute["instrumental"] = True

inst_col = len(STEMS) + 7
mixer_grid.columnconfigure(inst_col, weight=1, minsize=MIXER_MIN_CELL_W)

_inst_cell = ctk.CTkFrame(mixer_grid, fg_color=BG, corner_radius=0,
                           border_color=STEEL, border_width=1)
_inst_cell.grid(row=0, column=inst_col, padx=4, pady=4, sticky="nsew")

# Red accent bar at top
tk.Frame(_inst_cell, bg=RED, height=3).pack(fill="x")

# Header row: label + M/S buttons
_inst_hdr = ctk.CTkFrame(_inst_cell, fg_color="transparent")
_inst_hdr.pack(fill="x", padx=2, pady=(2, 0))
_inst_name_lbl = ctk.CTkLabel(_inst_hdr, text="INST", font=FONT_LABEL,
                              text_color=BRIGHT_RED, anchor="center", width=1)

_inst_key = "instrumental"

def _inst_mute():
    global _inst_autosolo_active
    _inst_autosolo_active = False   # manual click wins over the vocals link
    _toggle_mute(_inst_key)
    _sync_instrumental_overlap()

def _inst_solo():
    """Solo works both ways: it un-mutes the instrumental and mutes the
    vocals, so one click swaps between the full mix and the backing track.
    Un-soloing restores both."""
    global _inst_autosolo_active
    # Solo already makes INSTRUM the only thing heard — muted or not, since
    # solo suspends a cell's mute — and silences the vocals with everything
    # else. So nothing needs muting or un-muting here. Doing so is what made
    # un-soloing throw away an un-mute you had set yourself, flipping the
    # backing stems back on; releasing solo now simply returns every cell to
    # the mute it had before.
    soloed = not state.stem_solo.get(_inst_key, False)
    state.stem_solo[_inst_key] = soloed
    _paint_ms(_inst_key)

    # A click on S is your decision, not the vocals link's: stop the link
    # from later undoing it when the vocals are touched.
    _inst_autosolo_active = False

_d_mb = ctk.CTkButton(_inst_hdr, text="M", command=_inst_mute,
                       fg_color=MUTE_ON, hover_color=MUTE_ON,
                       text_color=TEXT_MAIN,
                       font=FONT_MS_BTN, corner_radius=0,
                       border_width=1, border_color=BORDER,
                       height=MS_BTN_H, width=MS_BTN_W)
_d_mb.pack(side="right", padx=(1, 2), pady=(6, 0))
_d_sb = ctk.CTkButton(_inst_hdr, text="S", command=_inst_solo,
                       fg_color=STEEL, hover_color=SOLO_ON,
                       text_color=TEXT_MAIN,
                       font=FONT_MS_BTN, corner_radius=0,
                       border_width=1, border_color=BORDER,
                       height=MS_BTN_H, width=MS_BTN_W)
_d_sb.pack(side="right", padx=(0, 1), pady=(6, 0))
# Label last: pack gives space to earlier widgets first, so M/S always fit.
_inst_name_lbl.pack(side="left", fill="x", expand=True)
_mute_btns[_inst_key] = _d_mb
_solo_btns[_inst_key] = _d_sb

# Sub-label: instrumental model credit
_inst_title_lbl = ctk.CTkLabel(_inst_cell, text=_INST_TITLE,
             font=("Courier New", 11), text_color=STEEL_LIGHT,
             wraplength=140, justify="center")
_inst_title_lbl.pack(pady=(2, 0), fill="x")
# "by" stays in the dim credit colour; the name itself is picked out.
_inst_credit_row = ctk.CTkFrame(_inst_cell, fg_color="transparent")
_inst_credit_row.pack()
_by, _, _who = _INST_CREDIT.partition(" ")
ctk.CTkLabel(_inst_credit_row, text=_by + " ",
             font=("Courier New", 11), text_color=TEXT_DIM
             ).pack(side="left")
_inst_author_lbl = ctk.CTkLabel(_inst_credit_row, text=_who or _INST_CREDIT,
             font=("Courier New", 13, "bold"), text_color=BRIGHT_GREEN)
_inst_author_lbl.pack(side="left")
_register_model_credit("instrumental", _inst_title_lbl, _inst_author_lbl)

# Status pill — updated by _update_inst_status_label()
_inst_status_lbl = ctk.CTkLabel(_inst_cell, text="WAITING",
                                 font=("Courier New", 12, "bold"),
                                 text_color=TEXT_DIM)
_inst_status_lbl.pack(pady=(2, 0))

# How far this cell's own separation has got. Packed only while it runs, so
# the cell keeps its usual height the rest of the time.
_inst_prog_row = ctk.CTkFrame(_inst_cell, fg_color="transparent")
_inst_prog_bar = ctk.CTkProgressBar(_inst_prog_row, progress_color=GLOW_RED,
                                    fg_color=PANEL, corner_radius=0, height=6)
_inst_prog_bar.set(0)
_inst_prog_bar.pack(fill="x", padx=2)
_inst_prog_lbl = ctk.CTkLabel(_inst_prog_row, text="0%",
                              font=("Courier New", 16, "bold"),
                              text_color=GLOW_RED)
_inst_prog_lbl.pack()


def _set_inst_progress(frac):
    """Show the instrumental pass's progress on its own cell."""
    try:
        if not _inst_prog_row.winfo_ismapped():
            _inst_prog_row.pack(fill="x", padx=6, pady=(2, 0),
                                after=_inst_status_lbl)
        _inst_prog_bar.set(max(0.0, min(1.0, frac)))
        _inst_prog_lbl.configure(text=f"{int(round(frac * 100))}%")
    except Exception:
        pass


def _clear_inst_progress():
    try:
        _inst_prog_bar.set(0)
        _inst_prog_lbl.configure(text="0%")
        _inst_prog_row.pack_forget()
    except Exception:
        pass

# Volume slider
_inst_vol_sl = LockedSlider(_inst_cell, from_=0, to=2,
                             button_color=RED, button_hover_color=GLOW_RED,
                             progress_color=RED,
                             command=lambda v: setattr(state, "instrumental_vol", float(v)))
_inst_vol_sl.set(1.0)
_inst_vol_sl.pack(fill="x", padx=6, pady=(4, 1))

# PAN / SW / REV / AIR mini-sliders — same as all other cells
for _lbl2, _from2, _to2, _init2, _dict2, _def2 in [
    ("PAN", -1.0, 1.0, 0.0, state.stem_pan,    0.0),
    ("SW",   0.0, 2.0, 1.0, state.stem_widths, 1.0),
    ("REV",  0.0, 1.0, 0.0, state.stem_reverbs,0.0),
    ("AIR", -1.0, 1.0, 0.0, state.stem_air,    0.0),
]:
    _row2 = ctk.CTkFrame(_inst_cell, fg_color="transparent")
    _row2.pack(fill="x", padx=4, pady=0)
    ctk.CTkLabel(_row2, text=_lbl2, font=("Courier New", 13, "bold"),
                 text_color=TEXT_DIM, width=22).pack(side="left")
    def _d_cmd_factory(d=_dict2, k=_inst_key):
        def _cmd(v):
            if d is not None:
                d[k] = float(v)
        return _cmd
    _sl2 = LockedSlider(_row2, from_=_from2, to=_to2, height=12,
                         button_color=STEEL, button_hover_color=STEEL_LIGHT,
                         progress_color=STEEL, command=_d_cmd_factory())
    _sl2.set(_init2)
    _sl2.pack(side="left", fill="x", expand=True)

# RST + meter row
_inst_rst_row = ctk.CTkFrame(_inst_cell, fg_color="transparent")
_inst_rst_row.pack(pady=(0, 2))

def _inst_rst(vs=_inst_vol_sl):
    state.instrumental_vol = 1.0
    vs.set(1.0)

ctk.CTkButton(_inst_rst_row, text="RST", command=_inst_rst,
              fg_color=STEEL, hover_color=STEEL_LIGHT, text_color=TEXT_DIM,
              font=("Courier New", 13, "bold"), corner_radius=0,
              border_width=1, border_color=BORDER, height=18, width=36
              ).pack(side="left", padx=(0, 4))
make_meter(_inst_rst_row, _inst_key).pack(side="left")



def _inst_separate_now():
    """Run the instrumental pass for the song that's loaded."""
    if state.separating:
        _inst_export_flash("BUSY", BRIGHT_RED)
        return
    path = _playlist_current[0]
    if not path or not os.path.isfile(path):
        _inst_export_flash("NO SONG", BRIGHT_RED)
        return
    threading.Thread(target=_run_low_priority, args=(separate_inst, path),
                     daemon=True).start()
    if running:
        app.after(0, _update_inst_status_label)


def _inst_export_flash(msg, colour):
    """Show a short message in the status pill, then restore it."""
    _inst_status_lbl.configure(text=msg, text_color=colour)
    if running:
        app.after(2000, _update_inst_status_label)


# AUTO switch — skip the instrumental pass entirely to halve GPU work per song.
_inst_auto_var = ctk.BooleanVar(value=_INST_AUTO)

def _on_inst_auto():
    global _INST_AUTO, _INST_ONLY
    _INST_AUTO = _inst_auto_var.get()
    if not _INST_AUTO and _INST_ONLY:
        _INST_ONLY = False
        _inst_only_var.set(False)
    _update_inst_status_label()

ctk.CTkCheckBox(_inst_cell, text="AUTO SEPARATE",
                variable=_inst_auto_var, command=_on_inst_auto,
                font=("Courier New", 11), text_color=TEXT_MAIN,
                fg_color=RED, hover_color=BRIGHT_RED,
                checkmark_color="#ffffff", corner_radius=0,
                border_color=STEEL, checkbox_width=14, checkbox_height=14
                ).pack(padx=6, pady=(2, 0))

# Run the instrumental before the six stems, so the backing track is ready
# (and playable on its own) first.
_inst_first_var = ctk.BooleanVar(value=_INST_FIRST)

def _on_inst_first():
    global _INST_FIRST
    _INST_FIRST = _inst_first_var.get()

# Instrumental only — skip the six-stem model completely.
_inst_only_var = ctk.BooleanVar(value=_INST_ONLY)

def _on_inst_only():
    global _INST_ONLY, _INST_AUTO
    _INST_ONLY = _inst_only_var.get()
    if _INST_ONLY and not _INST_AUTO:
        # "Only" is meaningless without the pass itself.
        _INST_AUTO = True
        _inst_auto_var.set(True)
    _update_inst_status_label()

ctk.CTkCheckBox(_inst_cell, text="INST ONLY",
                variable=_inst_only_var, command=_on_inst_only,
                font=("Courier New", 11), text_color=TEXT_MAIN,
                fg_color=RED, hover_color=BRIGHT_RED,
                checkmark_color="#ffffff", corner_radius=0,
                border_color=STEEL, checkbox_width=14, checkbox_height=14
                ).pack(padx=6, pady=(0, 0))

ctk.CTkCheckBox(_inst_cell, text="LOAD FIRST",
                variable=_inst_first_var, command=_on_inst_first,
                font=("Courier New", 11), text_color=TEXT_MAIN,
                fg_color=RED, hover_color=BRIGHT_RED,
                checkmark_color="#ffffff", corner_radius=0,
                border_color=STEEL, checkbox_width=14, checkbox_height=14
                ).pack(padx=6, pady=(0, 0))

# EXPORT STEM removed — the instrumental is exported from the EXPORT STEMS
# panel like any other stem (tick INSTRUM, then WRITE STEMS TO DISK).
ctk.CTkButton(_inst_cell, text="⟳ SEPARATE NOW", command=_inst_separate_now,
              fg_color=STEEL, hover_color=STEEL_LIGHT,
              text_color=BRIGHT_GREEN,
              font=("Courier New", 12, "bold"), corner_radius=0,
              border_width=1, border_color=BORDER, height=20
              ).pack(fill="x", padx=6, pady=(3, 2))

# COMP / LEVELLER / LIMITER, folded behind one header like the main cells
_make_dynamics_group(_inst_cell, _inst_key)
tk.Frame(_inst_cell, bg=BG, height=4).pack()


master_frame = ctk.CTkFrame(_sf, fg_color="transparent")
master_frame.pack(fill="x", padx=20, pady=2)

master_frame.columnconfigure(0, weight=1)
master_frame.columnconfigure(1, weight=1)
master_frame.columnconfigure(2, weight=1)
master_frame.columnconfigure(3, weight=1)

# ── Master control titles ──────────────────────────────────────────────────
# Size of "MASTER VOLUME", "STEREO WIDTH", "MASTER REVERB" and "MASTER AIR".
# Change the number after "Courier New" to make them bigger or smaller.
# (If the window is too narrow for the size you pick, the text shrinks to fit
# automatically and grows back when there is room.)
FONT_MASTER_LABEL = ("Courier New", 17, "bold")


def _labeled_slider(parent, label, from_, to, init_val, command, col):
    f = ctk.CTkFrame(parent, fg_color=PANEL,
                     corner_radius=0,
                     border_color=STEEL, border_width=1)
    f.grid(row=0, column=col, padx=6, sticky="ew")
    tk.Frame(f, bg=STEEL, height=2).pack(fill="x")
    ctk.CTkLabel(f, text=label, font=FONT_MASTER_LABEL,
                 text_color=TEXT_MAIN).pack(pady=(6, 2))
    sl = LockedSlider(f, from_=from_, to=to,
                       button_color=RED,
                       button_hover_color=GLOW_RED,
                       progress_color=RED,
                       command=command)
    sl.set(init_val)
    sl.pack(fill="x", padx=10, pady=(0, 8))

_labeled_slider(master_frame, "MASTER VOLUME", 0, 1, state.volume_master,
                lambda v: setattr(state, "volume_master", float(v)), 0)
_labeled_slider(master_frame, "STEREO WIDTH",  0, 2, state.stereo_width,
                lambda v: setattr(state, "stereo_width", float(v)), 1)
_labeled_slider(master_frame, "MASTER REVERB", 0, 1, state.reverb_master,
                lambda v: setattr(state, "reverb_master", float(v)), 2)
_labeled_slider(master_frame, "MASTER AIR",   -1, 1, state.air_master,
                lambda v: setattr(state, "air_master", float(v)), 3)


# ============================================================
# EXPORT SECTION
# ============================================================
export_outer = ctk.CTkFrame(_sf, fg_color=PANEL,
                             corner_radius=0,
                             border_color=STEEL, border_width=1)
export_outer.pack(fill="x", padx=20, pady=4)

_export_title_lbl = ctk.CTkLabel(export_outer,
                                 text="— EXPORT STEMS —  ▲",
                                 font=FONT_TITLE,
                                 text_color=GLOW_RED)
_export_title_lbl.pack(pady=(8, 4))

check_row = ctk.CTkFrame(export_outer, fg_color="transparent")
check_row.pack(pady=(0, 6))

stem_check_vars = {}
for name in STEMS:
    var = ctk.BooleanVar(value=False)
    stem_check_vars[name] = var
    ctk.CTkCheckBox(check_row,
                    text=name.upper(),
                    variable=var,
                    font=FONT_SMALL,
                    text_color=TEXT_MAIN,
                    fg_color=RED,
                    hover_color=BRIGHT_RED,
                    checkmark_color="#ffffff",
                    corner_radius=0,
                    border_color=STEEL
                    ).pack(side="left", padx=8)

# The instrumental isn't one of state.stems, but export_selected_stems()
# understands the key, so it gets a checkbox like everything else.
_inst_check_var = ctk.BooleanVar(value=False)
stem_check_vars["instrumental"] = _inst_check_var
ctk.CTkCheckBox(check_row,
                text="INST",
                variable=_inst_check_var,
                font=FONT_SMALL,
                text_color=TEXT_MAIN,
                fg_color=RED,
                hover_color=BRIGHT_RED,
                checkmark_color="#ffffff",
                corner_radius=0,
                border_color=STEEL
                ).pack(side="left", padx=8)

# The karaoke halves and the strings cell, same as the instrumental: their
# audio lives outside state.stems, but export_selected_stems() knows where to
# find it (see _EXPORT_EXTRAS).
for _xkey, _xlabel in (("front_vocals", "FRT VOX"), ("bg_vocals", "BG VOX"),
                       ("strings", "STRINGS")):
    _xvar = ctk.BooleanVar(value=False)
    stem_check_vars[_xkey] = _xvar
    ctk.CTkCheckBox(check_row,
                    text=_xlabel,
                    variable=_xvar,
                    font=FONT_SMALL,
                    text_color=TEXT_MAIN,
                    fg_color=RED,
                    hover_color=BRIGHT_RED,
                    checkmark_color="#ffffff",
                    corner_radius=0,
                    border_color=STEEL
                    ).pack(side="left", padx=8)

fmt_row = ctk.CTkFrame(export_outer, fg_color="transparent")
fmt_row.pack(pady=(4, 6))

# WAV 24-bit checkbox (left of button)
fmt_wav24_var = ctk.BooleanVar(value=state.export_fmt_wav24)
def _on_fmt_wav24():
    state.export_fmt_wav24 = fmt_wav24_var.get()
    _save_dirs()
_fmt_wav24_cb = ctk.CTkCheckBox(fmt_row,
                text="WAV 24-bit",
                variable=fmt_wav24_var,
                command=_on_fmt_wav24,
                font=FONT_SMALL,
                text_color=TEXT_MAIN,
                fg_color=RED,
                hover_color=BRIGHT_RED,
                checkmark_color="#ffffff",
                corner_radius=0,
                border_color=STEEL)
_fmt_wav24_cb.pack(side="left", padx=(0, 12))

# Write button (centre)
_write_btn = _btn(fmt_row, "⬇  WRITE STEMS TO DISK", export_selected_stems, width=260)
_write_btn.pack(side="left", padx=0)

# MP3 320k checkbox (right of button)
fmt_mp3_var = ctk.BooleanVar(value=state.export_fmt_mp3)
def _on_fmt_mp3():
    state.export_fmt_mp3 = fmt_mp3_var.get()
    _save_dirs()
_fmt_mp3_cb = ctk.CTkCheckBox(fmt_row,
               text="MP3 320k",
               variable=fmt_mp3_var,
               command=_on_fmt_mp3,
               font=FONT_SMALL,
               text_color=TEXT_MAIN,
               fg_color=RED,
               hover_color=BRIGHT_RED,
               checkmark_color="#ffffff",
               corner_radius=0,
               border_color=STEEL)
_fmt_mp3_cb.pack(side="left", padx=(12, 0))

# MP3 256k checkbox (far right of stems row)
fmt_mp3_256_var = ctk.BooleanVar(value=state.export_fmt_mp3_256)
def _on_fmt_mp3_256():
    state.export_fmt_mp3_256 = fmt_mp3_256_var.get()
    _save_dirs()
_fmt_mp3_256_cb = ctk.CTkCheckBox(fmt_row,
                text="MP3 256k",
                variable=fmt_mp3_256_var,
                command=_on_fmt_mp3_256,
                font=FONT_SMALL,
                text_color=TEXT_MAIN,
                fg_color=RED,
                hover_color=BRIGHT_RED,
                checkmark_color="#ffffff",
                corner_radius=0,
                border_color=STEEL)
_fmt_mp3_256_cb.pack(side="left", padx=(12, 0))

# Export progress — hidden until an export starts
_export_progress_frame = ctk.CTkFrame(export_outer, fg_color="transparent")
# Not packed yet — shown dynamically during export

_export_status_lbl = ctk.CTkLabel(
    _export_progress_frame,
    text="",
    font=FONT_SMALL,
    text_color=GLOW_RED)
_export_status_lbl.pack(pady=(2, 1))

_export_progress_bar = ctk.CTkProgressBar(
    _export_progress_frame,
    progress_color=GLOW_RED,
    fg_color=PANEL,
    corner_radius=0,
    height=10)
_export_progress_bar.set(0)
_export_progress_bar.pack(fill="x", padx=10, pady=(0, 6))


# ============================================================
# BOTTOM STATUS ROW
# ============================================================
status_row = ctk.CTkFrame(_sf, fg_color=PANEL,
                           corner_radius=0,
                           border_color=STEEL, border_width=1,
                           height=36)
status_row.pack(fill="x", padx=20, pady=6)
status_row.pack_propagate(False)

gpu_var = ctk.BooleanVar(value=True)
ctk.CTkCheckBox(status_row,
                text="USE GPU",
                variable=gpu_var,
                command=set_gpu_mode,
                font=FONT_SMALL,
                text_color=TEXT_DIM,
                fg_color=RED,
                hover_color=BRIGHT_RED,
                checkmark_color="#ffffff",
                corner_radius=0,
                border_color=STEEL
                ).pack(side="left", padx=16, pady=4)

ctk.CTkLabel(status_row,
             text="BY: Sai & Eidii & many more!",
             font=FONT_SMALL,
             text_color=TEXT_DIM).pack(side="right", padx=16)

# The TAP TEMPO row has been removed. Its right-hand end also showed the
# "SESSION SAVED ✓" / "SESSION LOADED ✓" confirmations, so that label now
# lives on the status row above instead.
_session_status_lbl = ctk.CTkLabel(status_row, text="",
                                    font=FONT_SMALL, text_color=GLOW_RED)
_session_status_lbl.pack(side="right", padx=16)

# ============================================================
# RUN
# ============================================================

# ============================================================
# THEME ENGINE — Easter egg colour schemes
# "eidii"     → ULTRA-EVIL PURPLE
# "RAMMSTEIN" → GOTHIC
# "sai"       → revert everything to default
#
# Every widget that needs recolouring registers itself via
# _reg(widget, role) immediately after creation.  _apply_theme()
# walks the registry and applies the correct colour for each role,
# so themes are always applied perfectly regardless of current state.
# ============================================================

# --- Colour palettes ---
_THEME_DEFAULT = {
    "bg":                    "#0a0a0a",
    "panel":                 "#111111",
    "border":                "#2a2a2a",
    "accent":                "#8B0000",
    "accent_hover":          "#cc0000",
    "accent_glow":           "#ff2200",
    "util":                  "#3a3a3a",
    "util_hover":            "#555555",
    "text_main":             "#c8c8c8",
    "text_dim":              "#666666",
    "label_accent":          "#ff2200",
    "label_bright":          "#cc0000",
    "canvas_bg":             "#060606",
    "wave_bg":               "#0a0a0a",
    "slider_accent":         "#8B0000",
    "slider_accent_h":       "#cc0000",
    "util_slider_accent":    "#3a3a3a",   # sub-slider thumb — same as util (steel)
    "util_slider_accent_h":  "#555555",
    "util_slider_track":     "#2a2a2a",   # sub-slider track (unfilled bar)
}

_THEME_EIDII = {
    "bg":                    "#0d0020",
    "panel":                 "#1a0035",
    "border":                "#7700ee",
    "accent":                "#aa00ff",
    "accent_hover":          "#cc44ff",
    "accent_glow":           "#ff00ff",
    "util":                  "#3d1066",
    "util_hover":            "#5a1a99",
    "text_main":             "#f0d0ff",
    "text_dim":              "#9955cc",
    "label_accent":          "#ff00ff",
    "label_bright":          "#dd44ff",
    "canvas_bg":             "#060010",
    "wave_bg":               "#0d0020",
    "slider_accent":         "#aa00ff",
    "slider_accent_h":       "#cc44ff",
    "util_slider_accent":    "#5a1a99",   # mid-purple sub-sliders
    "util_slider_accent_h":  "#7722bb",
    "util_slider_track":     "#2a0055",
}

_THEME_RAMMSTEIN = {
    "bg":                    "#080808",
    "panel":                 "#100c0c",
    "border":                "#4a3000",
    "accent":                "#1a1a1a",
    "accent_hover":          "#2d2d2d",
    "accent_glow":           "#c8a000",
    "util":                  "#0f0f0f",
    "util_hover":            "#222222",
    "text_main":             "#e8e0d0",
    "text_dim":              "#666050",
    "label_accent":          "#c8a000",
    "label_bright":          "#a07800",
    "canvas_bg":             "#040404",
    "wave_bg":               "#080808",
    "slider_accent":         "#c8a000",   # accent sliders: gold
    "slider_accent_h":       "#e6c000",
    "util_slider_accent":    "#c8a000",   # BDY/LCD/PRS/SW/REV/AIR: also gold — stand-out
    "util_slider_accent_h":  "#e6c000",
    "util_slider_track":     "#4a3000",   # track bar: dark gold — visible on near-black
}

# --- Widget registry ---
# Each entry: (widget, role)
# Roles: "accent_btn", "util_btn", "panel_frame", "bg_frame",
#        "border_panel", "border_bg", "label_main", "label_dim",
#        "label_accent", "label_bright", "accent_slider", "util_slider",
#        "checkbox", "entry", "progressbar", "accent_bar", "util_bar",
#        "bg_bar", "canvas", "toplevel_panel"
_widget_registry: list = []

def _reg(widget, role: str):
    """Register a widget with its semantic role for theme recolouring."""
    _widget_registry.append((widget, role))
    return widget   # passthrough so it can wrap existing expressions inline


def _apply_theme(t: dict):
    """Apply a palette dict to every registered widget."""
    global BG, PANEL, BORDER, RED, BRIGHT_RED, GLOW_RED
    global STEEL, STEEL_LIGHT, TEXT_DIM, TEXT_MAIN

    BG          = t["bg"]
    PANEL       = t["panel"]
    BORDER      = t["border"]
    RED         = t["accent"]
    BRIGHT_RED  = t["accent_hover"]
    GLOW_RED    = t["accent_glow"]
    STEEL       = t["util"]
    STEEL_LIGHT = t["util_hover"]
    TEXT_DIM    = t["text_dim"]
    TEXT_MAIN   = t["text_main"]

    # slider_accent may differ from button accent (e.g. RAMMSTEIN = gold)
    sl_acc        = t.get("slider_accent",         RED)
    sl_acc_h      = t.get("slider_accent_h",       BRIGHT_RED)
    sl_util       = t.get("util_slider_accent",    STEEL)
    sl_util_h     = t.get("util_slider_accent_h",  STEEL_LIGHT)
    sl_util_track = t.get("util_slider_track",     STEEL)

    app.configure(fg_color=BG)

    for widget, role in _widget_registry:
        try:
            if role == "accent_btn":
                widget.configure(
                    fg_color=RED, hover_color=BRIGHT_RED,
                    text_color=TEXT_MAIN, border_color=BRIGHT_RED)
            elif role == "util_btn":
                widget.configure(
                    fg_color=STEEL, hover_color=STEEL_LIGHT,
                    text_color=TEXT_DIM, border_color=BORDER)
            elif role == "panel_frame":
                widget.configure(fg_color=PANEL)
            elif role == "bg_frame":
                widget.configure(fg_color=BG)
            elif role == "border_panel":
                widget.configure(fg_color=PANEL, border_color=BORDER)
            elif role == "border_accent":
                widget.configure(fg_color=PANEL, border_color=RED)
            elif role == "cell_frame":
                widget.configure(fg_color=BG, border_color=BORDER)
            elif role == "label_main":
                widget.configure(text_color=TEXT_MAIN)
            elif role == "label_dim":
                widget.configure(text_color=TEXT_DIM)
            elif role == "label_accent":
                widget.configure(text_color=t["label_accent"])
            elif role == "label_bright":
                widget.configure(text_color=t["label_bright"])
            elif role == "accent_slider":
                widget.configure(
                    button_color=sl_acc, button_hover_color=sl_acc_h,
                    progress_color=sl_acc, fg_color=sl_acc)
            elif role == "util_slider":
                widget.configure(
                    button_color=sl_util, button_hover_color=sl_util_h,
                    progress_color=sl_util, fg_color=sl_util_track)
            elif role == "checkbox":
                widget.configure(
                    fg_color=RED, hover_color=BRIGHT_RED,
                    checkmark_color="#ffffff",
                    text_color=TEXT_DIM, border_color=BORDER)
            elif role == "entry":
                widget.configure(
                    fg_color=BG, border_color=STEEL, text_color=TEXT_MAIN)
            elif role == "progressbar":
                widget.configure(progress_color=GLOW_RED, fg_color=PANEL)
            elif role == "accent_bar":
                widget.configure(bg=RED)
            elif role == "util_bar":
                widget.configure(bg=STEEL)
            elif role == "bg_bar":
                widget.configure(bg=BG)
            elif role == "canvas":
                widget.configure(
                    bg=t["canvas_bg"], highlightbackground=STEEL)
            elif role == "wave_slot":
                widget.configure(bg=t["wave_bg"])
            elif role == "scrollbar":
                widget.configure(
                    button_color=RED, button_hover_color=BRIGHT_RED)
        except Exception:
            pass

    # #9 — Play/Stop buttons: only recolour to match transport state.
    # If no audio loaded → keep them in their dimmed appearance (STEEL-coloured).
    # If audio is loaded → use active accent colours so they stay recognisable.
    try:
        if state.stems is not None:
            # Audio loaded — buttons should be active/bright in any theme
            btn_play.configure(fg_color="#2e7d32", hover_color="#43a047",
                               text_color=TEXT_MAIN, border_color="#43a047")
            btn_stop.configure(fg_color="#5c0000", hover_color="#7a0000",
                               text_color=TEXT_MAIN, border_color="#7a0000")
        else:
            # No audio — keep dimmed but still themed correctly
            btn_play.configure(fg_color=STEEL, hover_color=STEEL_LIGHT,
                               text_color=TEXT_DIM, border_color=BORDER)
            btn_stop.configure(fg_color=STEEL, hover_color=STEEL_LIGHT,
                               text_color=TEXT_DIM, border_color=BORDER)
    except Exception:
        pass

    # Force wave canvas to show updated scanlines on next draw cycle
    global _scanlines_drawn
    _scanlines_drawn = False

    # If the EQ window is open, rebuild it so its sliders adopt the new theme
    if _eq_window is not None:
        try:
            if _eq_window.winfo_exists():
                _close_eq_window(_eq_window)
                app.after(50, open_eq_window)
        except Exception:
            pass


# Now register every existing widget that was already built.
# We walk the tree once and tag each widget by inspecting its current colours
# against the default palette so we know its role.
def _auto_register_all():
    """One-time scan: register all existing widgets by matching default colours."""
    D = _THEME_DEFAULT

    def _walk(widget):
        cls = widget.__class__.__name__
        try:
            if cls == "CTkButton":
                fg = widget.cget("fg_color")
                is_util = str(fg) in (D["util"], "#3a3a3a", D["util_hover"], "#555555")
                _reg(widget, "util_btn" if is_util else "accent_btn")

            elif cls == "CTkFrame":
                fg     = str(widget.cget("fg_color"))
                border = ""
                try: border = str(widget.cget("border_color"))
                except Exception: pass
                if fg == "transparent":
                    pass   # layout helper frames — leave alone
                elif border in (D["accent"], D["accent_glow"], "#8B0000", "#ff2200", "#cc0000"):
                    _reg(widget, "border_accent")
                elif border in (D["border"], D["util"], "#2a2a2a", "#3a3a3a"):
                    role = "cell_frame" if fg in (D["bg"], "#0a0a0a") else "border_panel"
                    _reg(widget, role)
                elif fg in (D["panel"], "#111111"):
                    _reg(widget, "panel_frame")
                elif fg in (D["bg"], "#0a0a0a"):
                    _reg(widget, "bg_frame")

            elif cls == "CTkLabel":
                tc = str(widget.cget("text_color"))
                if tc in (D["label_accent"], "#ff2200"):
                    _reg(widget, "label_accent")
                elif tc in (D["label_bright"], "#cc0000"):
                    _reg(widget, "label_bright")
                elif tc in (D["text_dim"], "#666666"):
                    _reg(widget, "label_dim")
                elif tc in (D["text_main"], "#c8c8c8"):
                    _reg(widget, "label_main")

            elif cls == "CTkCheckBox":
                _reg(widget, "checkbox")

            elif cls in ("CTkSlider", "LockedSlider"):
                bc = str(widget.cget("button_color"))
                is_util = bc in (D["util"], "#3a3a3a")
                _reg(widget, "util_slider" if is_util else "accent_slider")

            elif cls == "CTkEntry":
                _reg(widget, "entry")

            elif cls == "CTkProgressBar":
                _reg(widget, "progressbar")

            elif cls == "CTkScrollbar":
                _reg(widget, "scrollbar")

            elif cls == "Frame":
                bg = str(widget.cget("bg"))
                if bg in (D["accent"], "#8B0000"):
                    _reg(widget, "accent_bar")
                elif bg in (D["util"], "#3a3a3a"):
                    _reg(widget, "util_bar")
                elif bg in (D["bg"], "#0a0a0a"):
                    _reg(widget, "bg_bar")

            elif cls == "Canvas":
                if widget is wave_canvas:
                    _reg(widget, "canvas")
                # wave_slot is a tk.Frame
        except Exception:
            pass

        for child in widget.winfo_children():
            _walk(child)

    _walk(app)
    # Register the special containers directly
    _reg(wave_slot,  "wave_slot")
    _reg(header,     "border_accent")
    _reg(status_row, "border_panel")

_auto_register_all()


# --- Keystroke listener ---
_key_buf      = ""
_active_theme = "default"   # "default" | "eidii" | "rammstein"

def _on_key(event):
    global _key_buf, _active_theme
    ch = event.char
    if not ch:
        return
    # Buffer long enough for "RAMMSTEIN" (9)
    _key_buf = (_key_buf + ch)[-9:]
    lower9   = _key_buf.lower()

    if lower9.endswith("sai"):
        if _active_theme != "default":
            _active_theme = "default"
            _apply_theme(_THEME_DEFAULT)
            _update_rammstein_ui()

    elif lower9.endswith("eidii") and _active_theme != "eidii":
        _active_theme = "eidii"
        _apply_theme(_THEME_EIDII)
        _update_rammstein_ui()

    elif _key_buf.endswith("RAMMSTEIN") and _active_theme != "rammstein":
        _active_theme = "rammstein"
        _apply_theme(_THEME_RAMMSTEIN)
        _update_rammstein_ui()

app.bind("<Key>", _on_key)

# ============================================================
# THEME PANELS — Export Mix button, shown below status_row
# One panel per theme; _update_theme_panel() shows the right one.
# ============================================================

# ── SAI (default) panel ──────────────────────────────────────────────────
_sai_panel = ctk.CTkFrame(_sf, fg_color=PANEL,
                           corner_radius=0,
                           border_color=STEEL, border_width=1)
# Not packed yet

tk.Frame(_sai_panel, bg=RED, height=2).pack(fill="x")
_sai_title_lbl = ctk.CTkLabel(_sai_panel,
                              text="— SAI ENGINE —  ▲",
                              font=FONT_TITLE,
                              text_color=GLOW_RED)
_sai_title_lbl.pack(pady=(6, 2))

_sai_btn_row = ctk.CTkFrame(_sai_panel, fg_color="transparent")
_sai_btn_row.pack(pady=(4, 10))

_sai_wav24_var = ctk.BooleanVar(value=state.export_fmt_wav24)
def _on_sai_wav24():
    state.export_fmt_wav24 = _sai_wav24_var.get()
    fmt_wav24_var.set(state.export_fmt_wav24)
    _save_dirs()
ctk.CTkCheckBox(_sai_btn_row, text="WAV 24-bit",
                variable=_sai_wav24_var, command=_on_sai_wav24,
                font=FONT_SMALL, text_color=TEXT_MAIN,
                fg_color=RED, hover_color=BRIGHT_RED,
                checkmark_color="#ffffff", corner_radius=0,
                border_color=STEEL).pack(side="left", padx=(0, 10))

_sai_export_mix_btn = ctk.CTkButton(
    _sai_btn_row,
    text="⬇  EXPORT MIX",
    command=export_mix,
    fg_color=RED,
    hover_color=BRIGHT_RED,
    text_color=TEXT_MAIN,
    font=FONT_LABEL,
    corner_radius=0,
    border_width=2,
    border_color=BRIGHT_RED,
    height=34,
    width=220)
_sai_export_mix_btn.pack(side="left")

_sai_mp3_320_var = ctk.BooleanVar(value=state.export_fmt_mp3)
def _on_sai_mp3_320():
    state.export_fmt_mp3 = _sai_mp3_320_var.get()
    fmt_mp3_var.set(state.export_fmt_mp3)
    _save_dirs()
ctk.CTkCheckBox(_sai_btn_row, text="MP3 320k",
                variable=_sai_mp3_320_var, command=_on_sai_mp3_320,
                font=FONT_SMALL, text_color=TEXT_MAIN,
                fg_color=RED, hover_color=BRIGHT_RED,
                checkmark_color="#ffffff", corner_radius=0,
                border_color=STEEL).pack(side="left", padx=(10, 0))

_sai_mp3_256_var = ctk.BooleanVar(value=state.export_fmt_mp3_256)
def _on_sai_mp3_256():
    state.export_fmt_mp3_256 = _sai_mp3_256_var.get()
    fmt_mp3_256_var.set(state.export_fmt_mp3_256)
    _save_dirs()
ctk.CTkCheckBox(_sai_btn_row, text="MP3 256k",
                variable=_sai_mp3_256_var, command=_on_sai_mp3_256,
                font=FONT_SMALL, text_color=TEXT_MAIN,
                fg_color=RED, hover_color=BRIGHT_RED,
                checkmark_color="#ffffff", corner_radius=0,
                border_color=STEEL).pack(side="left", padx=(10, 0))

# ── EIDII panel ───────────────────────────────────────────────────────────
_eidii_panel = ctk.CTkFrame(_sf, fg_color="#1a0035",
                              corner_radius=0,
                              border_color="#7700ee", border_width=1)
# Not packed yet

tk.Frame(_eidii_panel, bg="#aa00ff", height=2).pack(fill="x")
ctk.CTkLabel(_eidii_panel,
             text="— ✦  EIDII NEXUS  ✦ —",
             font=FONT_TITLE,
             text_color="#ff00ff").pack(pady=(6, 2))

_eidii_btn_row = ctk.CTkFrame(_eidii_panel, fg_color="transparent")
_eidii_btn_row.pack(pady=(4, 10))

_eidii_wav24_var = ctk.BooleanVar(value=state.export_fmt_wav24)
def _on_eidii_wav24():
    state.export_fmt_wav24 = _eidii_wav24_var.get()
    fmt_wav24_var.set(state.export_fmt_wav24)
    _save_dirs()
ctk.CTkCheckBox(_eidii_btn_row, text="WAV 24-bit",
                variable=_eidii_wav24_var, command=_on_eidii_wav24,
                font=FONT_SMALL, text_color="#ff00ff",
                fg_color="#7700ee", hover_color="#aa00ff",
                checkmark_color="#ffffff", corner_radius=0,
                border_color="#7700ee").pack(side="left", padx=(0, 10))

_eidii_export_mix_btn = ctk.CTkButton(
    _eidii_btn_row,
    text="✦  EXPORT MIX",
    command=export_mix,
    fg_color="#1a0035",
    hover_color="#2d0055",
    text_color="#ff00ff",
    font=FONT_LABEL,
    corner_radius=0,
    border_width=2,
    border_color="#aa00ff",
    height=34,
    width=220)
_eidii_export_mix_btn.pack(side="left")

_eidii_mp3_320_var = ctk.BooleanVar(value=state.export_fmt_mp3)
def _on_eidii_mp3_320():
    state.export_fmt_mp3 = _eidii_mp3_320_var.get()
    fmt_mp3_var.set(state.export_fmt_mp3)
    _save_dirs()
ctk.CTkCheckBox(_eidii_btn_row, text="MP3 320k",
                variable=_eidii_mp3_320_var, command=_on_eidii_mp3_320,
                font=FONT_SMALL, text_color="#ff00ff",
                fg_color="#7700ee", hover_color="#aa00ff",
                checkmark_color="#ffffff", corner_radius=0,
                border_color="#7700ee").pack(side="left", padx=(10, 0))

_eidii_mp3_256_var = ctk.BooleanVar(value=state.export_fmt_mp3_256)
def _on_eidii_mp3_256():
    state.export_fmt_mp3_256 = _eidii_mp3_256_var.get()
    fmt_mp3_256_var.set(state.export_fmt_mp3_256)
    _save_dirs()
ctk.CTkCheckBox(_eidii_btn_row, text="MP3 256k",
                variable=_eidii_mp3_256_var, command=_on_eidii_mp3_256,
                font=FONT_SMALL, text_color="#ff00ff",
                fg_color="#7700ee", hover_color="#aa00ff",
                checkmark_color="#ffffff", corner_radius=0,
                border_color="#7700ee").pack(side="left", padx=(10, 0))

# ── RAMMSTEIN panel ───────────────────────────────────────────────────────
_rammstein_panel = ctk.CTkFrame(_sf, fg_color=PANEL,
                                 corner_radius=0,
                                 border_color=STEEL, border_width=1)
# Not packed yet

tk.Frame(_rammstein_panel, bg="#c8a000", height=2).pack(fill="x")
_rammstein_title_lbl = ctk.CTkLabel(_rammstein_panel,
                                    text="— ⛧  RAMMSTEIN FORGE  ⛧ —  ▲",
                                    font=FONT_TITLE,
                                    text_color="#c8a000")
_rammstein_title_lbl.pack(pady=(6, 2))

_forge_btn_row = ctk.CTkFrame(_rammstein_panel, fg_color="transparent")
_forge_btn_row.pack(pady=(4, 10))

# ── EXPORT MIX — the same function the SAI panel offers, in forge colours ──
# The format switches are shared with every other panel, so ticking one here
# ticks it in the EXPORT STEMS panel too.
_forge_wav24_var = ctk.BooleanVar(value=state.export_fmt_wav24)


def _on_forge_wav24():
    state.export_fmt_wav24 = _forge_wav24_var.get()
    fmt_wav24_var.set(state.export_fmt_wav24)
    _sai_wav24_var.set(state.export_fmt_wav24)
    _save_dirs()


ctk.CTkCheckBox(_forge_btn_row, text="WAV 24-bit",
                variable=_forge_wav24_var, command=_on_forge_wav24,
                font=FONT_SMALL, text_color="#c8a000",
                fg_color="#7a6200", hover_color="#c8a000",
                checkmark_color="#1a1a1a", corner_radius=0,
                border_color="#7a6200").pack(side="left", padx=(0, 10))

_export_mix_btn = ctk.CTkButton(
    _forge_btn_row,
    text="⬇  EXPORT MIX",
    command=export_mix,
    fg_color="#c8a000",
    hover_color="#e6c000",
    text_color="#1a1a1a",
    font=FONT_LABEL,
    corner_radius=0,
    border_width=2,
    border_color="#e6c000",
    height=34,
    width=260)
_export_mix_btn.pack(side="left")

_forge_mp3_var = ctk.BooleanVar(value=state.export_fmt_mp3)


def _on_forge_mp3():
    state.export_fmt_mp3 = _forge_mp3_var.get()
    fmt_mp3_var.set(state.export_fmt_mp3)
    _sai_mp3_320_var.set(state.export_fmt_mp3)
    _save_dirs()


ctk.CTkCheckBox(_forge_btn_row, text="MP3 320k",
                variable=_forge_mp3_var, command=_on_forge_mp3,
                font=FONT_SMALL, text_color="#c8a000",
                fg_color="#7a6200", hover_color="#c8a000",
                checkmark_color="#1a1a1a", corner_radius=0,
                border_color="#7a6200").pack(side="left", padx=(10, 0))


def _update_rammstein_ui():
    """Show the correct theme panel below the status row, hide the others."""
    # Hide all panels first
    for _p in (_sai_panel, _eidii_panel, _rammstein_panel):
        try:
            _p.pack_forget()
        except Exception:
            pass

    # Show the panel that matches the active theme.
    # Only scroll to the bottom when the user switches themes mid-session —
    # not on the initial startup call (when _startup is True).
    if _active_theme == "default":
        _sai_panel.pack(fill="x", padx=20, pady=4, before=status_row)
        if not _startup:
            app.after(50, lambda: _scroll_canvas.yview_moveto(1.0))
    elif _active_theme == "eidii":
        _eidii_panel.pack(fill="x", padx=20, pady=4, before=status_row)
        if not _startup:
            app.after(50, lambda: _scroll_canvas.yview_moveto(1.0))
    elif _active_theme == "rammstein":
        _rammstein_panel.pack(fill="x", padx=20, pady=4, before=status_row)
        if not _startup:
            app.after(50, lambda: _scroll_canvas.yview_moveto(1.0))

_startup = True   # cleared after the splash, so theme changes work normally

# Show the default panel on startup
_update_rammstein_ui()

# ── SPACEBAR — play / stop toggle ────────────────────────────────────────────
def _on_spacebar(event):
    # Ignore spacebar when focus is on a text-entry widget
    if isinstance(app.focus_get(), (tk.Entry, ctk.CTkEntry)):
        return
    if state.stream is not None:
        stop()
    else:
        play()

app.bind("<space>", _on_spacebar)
# ─────────────────────────────────────────────────────────────────────────────

# ============================================================
# TEXT FITTING
# At small window sizes the mixer's thirteen columns share very little width
# (about 93px each at 1280x800), and fixed font sizes clip. Rather than tune
# every size by hand, every text widget is fitted to the width it has
# actually been given: the type shrinks only as far as needed, and grows back
# to its designed size as soon as there is room again.
# ============================================================
import tkinter.font as _tkf_fit

_FIT_MIN_PT  = 7          # never smaller than this
_fit_base    = {}         # widget -> (family, designed size, weight)
_fit_seen    = {}         # widget -> (text, width) last fitted for
_fit_quick   = {}         # widget -> cheap change-detection key
_fit_widgets = []         # cached list of text widgets, refreshed rarely
_fit_walked  = [0.0]      # when the list was last rebuilt
_FIT_PAD     = {"CTkButton": 8, "CTkCheckBox": 30, "CTkLabel": 0, "Label": 0}


def _fit_inner(w):
    """The Tk label that actually draws a CTk widget's text."""
    for attr in ("_text_label", "_label"):
        inner = getattr(w, attr, None)
        if inner is not None:
            return inner
    return w


def _fit_designed(w):
    """Designed font of *w*, recorded the first time it is seen."""
    base = _fit_base.get(w)
    if base is None:
        f = w.cget("font")
        if isinstance(f, (tuple, list)) and len(f) >= 2:
            base = (f[0], int(f[1]), "bold" if "bold" in f[2:] else "normal")
        else:
            try:
                fo = _tkf_fit.Font(font=f)
                base = (fo.actual("family"), abs(int(fo.actual("size"))),
                        fo.actual("weight"))
            except Exception:
                return None
        _fit_base[w] = base
    return base


def _fit_padx(info):
    """Total horizontal padding from a pack/grid info dict."""
    p = info.get("padx", 0)
    try:
        if isinstance(p, (tuple, list)):
            return sum(int(float(x)) for x in p)
        parts = str(p).split()
        if len(parts) == 2:
            return int(float(parts[0])) + int(float(parts[1]))
        return 2 * int(float(parts[0])) if parts else 0
    except Exception:
        return 0


def _fit_border(c):
    try:
        return 2 * int(float(c.cget("border_width")))
    except Exception:
        return 0


def _fit_container_width(c, depth=0):
    """Width a container can actually offer its contents.

    A container that merely wraps its contents (packed without fill) is as
    wide as whatever is inside it, so its own width says nothing about the
    room available — look past it to the container that decides the width.
    """
    try:
        if depth > 8 or isinstance(c, (tk.Tk, tk.Toplevel)):
            return c.winfo_width()
        mgr = c.winfo_manager()
        if mgr == "pack":
            info = c.pack_info()
            fills = str(info.get("fill", "none")) in ("x", "both")
            if fills or str(info.get("side", "top")) in ("left", "right"):
                return c.winfo_width()
            return _fit_container_width(c.master, depth + 1) - _fit_padx(info)
        if mgr == "grid":
            info = c.grid_info()
            sticky = str(info.get("sticky", ""))
            if "e" in sticky and "w" in sticky:
                return c.winfo_width()
            bb = c.master.grid_bbox(int(info["column"]), int(info["row"]))
            return (bb[2] if bb and bb[2] > 1 else c.winfo_width()) - _fit_padx(info)
        return c.winfo_width()
    except Exception:
        try:
            return c.winfo_width()
        except Exception:
            return 0


def _fit_space(w):
    """Width *w* could be given if its text needed it.

    Deliberately not w's own width: a widget sized to its text has exactly
    the width of its current — possibly already shrunk — text, and using
    that would let it shrink but never grow back.
    """
    try:
        parent = w.master
        pw = _fit_container_width(parent) - _fit_border(parent)
        mgr = w.winfo_manager()
        if mgr == "pack":
            info = w.pack_info()
            side = str(info.get("side", "top"))
            space = pw - _fit_padx(info)
            if side in ("left", "right"):
                # Share the row with the other widgets packed beside it.
                for sib in parent.pack_slaves():
                    if sib is w:
                        continue
                    try:
                        si = sib.pack_info()
                    except Exception:
                        continue
                    if str(si.get("side", "top")) in ("left", "right"):
                        space -= sib.winfo_width() + _fit_padx(si)
            return space
        if mgr == "grid":
            info = w.grid_info()
            bb = parent.grid_bbox(int(info["column"]), int(info["row"]))
            if bb and bb[2] > 1:
                return bb[2] - _fit_padx(info)
        return w.winfo_width()
    except Exception:
        try:
            return w.winfo_width()
        except Exception:
            return 0


def _fit_one(w):
    if getattr(w, "_own_fit", False):
        return                          # fits its own text (e.g. CMP / LVL / LIM)
    try:
        if not w.winfo_ismapped():
            return
        text = w.cget("text")
        # Cheap guard first: if the text, the widget's own width and the
        # width of what holds it are all unchanged, nothing can have moved,
        # so skip the costly part. This is what makes the periodic pass
        # affordable — working out the available space walks the widget's
        # ancestry, and doing that for every label every time cost more than
        # the fitting itself.
        quick = (text, w.winfo_width(), w.master.winfo_width())
        if _fit_quick.get(w) == quick:
            return
        _fit_quick[w] = quick
    except Exception:
        return
    if not isinstance(text, str) or not text.strip() or "\n" in text:
        return
    try:
        if int(float(w.cget("wraplength") or 0)) > 0:
            return                      # wraps on purpose
    except Exception:
        pass
    kind = type(w).__name__
    avail = _fit_space(w) - _FIT_PAD.get(kind, 0)
    if avail < 8:
        return
    key = (text, avail)
    if _fit_seen.get(w) == key:
        return
    _fit_seen[w] = key

    base = _fit_designed(w)
    if base is None:
        return
    fam, size0, weight = base

    # CTk scales fonts and draws them at a pixel size (a negative Tk size),
    # while the sizes we set are points. Measure candidates in exactly the
    # unit the inner label draws in, or the estimate is off wherever pixels
    # and points differ.
    inner = _fit_inner(w)
    pixel_based, scale = False, 1.0
    try:
        cur = w.cget("font")
        cur_pt = int(cur[1]) if isinstance(cur, (tuple, list)) else size0
        cfg = int(_tkf_fit.Font(font=inner.cget("font")).cget("size"))
        pixel_based = cfg < 0
        if cfg:
            scale = abs(cfg) / max(1, cur_pt)
    except Exception:
        pass

    def needs(pt):
        n = max(1, int(round(pt * scale)))
        return _tkf_fit.Font(family=fam, size=-n if pixel_based else n,
                             weight=weight).measure(text)

    size = size0
    while size > _FIT_MIN_PT and needs(size) > avail:
        size -= 1
    try:
        cur = w.cget("font")
        if not (isinstance(cur, (tuple, list)) and int(cur[1]) == size):
            w.configure(font=(fam, size, weight) if weight == "bold" else (fam, size))
    except Exception:
        pass


def _fit_walk(root):
    try:
        import customtkinter as _ctkm
        kinds = (_ctkm.CTkLabel, _ctkm.CTkButton, _ctkm.CTkCheckBox)
    except Exception:
        kinds = ()
    for c in root.winfo_children():
        if isinstance(c, tk.Toplevel):
            continue                # other windows are walked separately
        if kinds and isinstance(c, kinds):
            _fit_one(c)
        elif isinstance(c, tk.Label) and not isinstance(c.master, kinds):
            _fit_one(c)             # a plain label, not a CTk widget's insides
        _fit_walk(c)


def _fit_collect():
    """Rebuild the list of text widgets (rare: they are created once)."""
    try:
        import customtkinter as _ctkm
        kinds = (_ctkm.CTkLabel, _ctkm.CTkButton, _ctkm.CTkCheckBox)
    except Exception:
        kinds = ()
    found = []

    def walk(root):
        for c in root.winfo_children():
            if isinstance(c, tk.Toplevel):
                continue
            if (kinds and isinstance(c, kinds)) or (
                    isinstance(c, tk.Label) and not isinstance(c.master, kinds)):
                found.append(c)
            walk(c)

    try:
        walk(app)
        for name in ("_loadlist_win", "_eq_window"):
            win = globals().get(name)
            if win is not None and win.winfo_exists():
                walk(win)
        for win in (globals().get("_dyn_windows") or {}).values():
            if win.winfo_exists() and win.state() == "normal":
                walk(win)
    except Exception:
        pass
    _fit_widgets[:] = found
    _fit_walked[0] = time.monotonic()


def _fit_all(rescan=False):
    """Refit everything visible.

    Walking the widget tree is the expensive part, so it happens only when
    asked for or every few seconds; in between, the cached list is used and
    each widget is dismissed in a couple of cheap calls unless it changed.
    """
    try:
        if rescan or not _fit_widgets or time.monotonic() - _fit_walked[0] > 5.0:
            _fit_collect()
        dead = False
        for w in _fit_widgets:
            try:
                _fit_one(w)
            except Exception:
                dead = True
        if dead:
            _fit_widgets[:] = [w for w in _fit_widgets if w.winfo_exists()]
    except Exception:
        pass


_fit_resize_job = [None]


def _fit_after_resize(_event=None):
    """Refit shortly after the window stops changing size."""
    if _fit_resize_job[0] is not None:
        try:
            app.after_cancel(_fit_resize_job[0])
        except Exception:
            pass
    _fit_resize_job[0] = app.after(150, _fit_run_now)


def _fit_run_now():
    # No rescan here: resizing moves widgets, it does not create them, and
    # re-walking the tree is the expensive half of a fitting pass. New
    # windows ask for a rescan themselves, and the periodic pass picks up
    # anything else within a few seconds.
    _fit_resize_job[0] = None
    _fit_all()


def _fit_loop():
    # A slow background pass catches labels whose text changes on its own
    # ("SPLIT VOCALS" -> "SPLITTING…", status pills and so on); resizing and
    # folding trigger a prompt pass of their own.
    _fit_all()
    if running:
        app.after(1000, _fit_loop)


# ============================================================
# COLLAPSIBLE PANELS
# Clicking a panel's title folds its contents away and back. Set up here,
# after everything has been built, so the body is known.
# ============================================================
def _make_collapsible(panel, title_lbl, title_text, start_open=True):
    """Fold *panel*'s contents away when its title is clicked.

    The title itself, and the thin red rule some panels start with, stay
    visible; everything else is un-packed and re-packed with the geometry
    options it was created with, so expanding restores the original layout.
    """
    keep = {title_lbl}
    for _c in panel.winfo_children():
        # The 2px rule at the top of a panel is part of its frame, not body.
        if isinstance(_c, tk.Frame) and _c.winfo_reqheight() <= 3:
            keep.add(_c)

    body = []
    for _c in panel.winfo_children():
        if _c in keep:
            continue
        try:
            info = _c.pack_info()
        except Exception:
            continue          # placed or gridded, not packed: leave it alone
        body.append((_c, dict(info)))

    open_state = [bool(start_open)]

    def _apply():
        if open_state[0]:
            for _w, _info in body:
                _info.pop("in", None)
                try:
                    _w.pack(**_info)
                except Exception:
                    pass
            title_lbl.configure(text=f"{title_text}  ▲")
        else:
            for _w, _ in body:
                try:
                    _w.pack_forget()
                except Exception:
                    pass
            title_lbl.configure(text=f"{title_text}  ▼")

    def _toggle(_event=None):
        open_state[0] = not open_state[0]
        _apply()

    title_lbl.bind("<Button-1>", _toggle)
    title_lbl.bind("<Enter>", lambda e: title_lbl.configure(text_color="#ff7755"))
    title_lbl.bind("<Leave>", lambda e: title_lbl.configure(text_color=GLOW_RED))
    for _w in (getattr(title_lbl, "_label", None), getattr(title_lbl, "_canvas", None)):
        try:
            _w.configure(cursor="hand2")
        except Exception:
            pass
    _apply()
    return _toggle


# ── VOCALS cell: progress for the dedicated vocals model ──────────────────
# The six-stem vocals are usable straight away; this says that a better
# version is on its way, and shows how far along it is.
_vocals_status_lbl = ctk.CTkLabel(_vocals_cell, text="",
                                  font=("Courier New", 10, "bold"),
                                  text_color=TEXT_DIM)
_vocals_prog_row = ctk.CTkFrame(_vocals_cell, fg_color="transparent")
_vocals_prog_bar = ctk.CTkProgressBar(_vocals_prog_row, progress_color=GLOW_RED,
                                      fg_color=PANEL, corner_radius=0, height=5)
_vocals_prog_bar.set(0)
_vocals_prog_bar.pack(fill="x", padx=2)
_vocals_prog_lbl = ctk.CTkLabel(_vocals_prog_row, text="0%",
                                font=("Courier New", 13, "bold"),
                                text_color=GLOW_RED)
_vocals_prog_lbl.pack()


def _vocals_status_tick():
    """Keep the VOCALS cell's line current.

    It was only refreshed from inside the refinement pass, so with no vocals
    model installed — or before one had run — the line never appeared.
    """
    try:
        _update_vocals_status()
    except Exception:
        pass
    if running:
        app.after(500, _vocals_status_tick)


# Pick the vocals model by hand when the name-based search picks wrongly.
# The VOCALS cell carries no model name or credit: the stem it shows comes
# from the six-stem model first and the vocals model later, so naming one of
# them would be wrong half the time.
_voc_model_wrap = _vocals_sub_ref[0] if _vocals_sub_ref[0] is not None \
    else _vocals_hdr_ref[0]


def _update_vocals_status():
    """Say whether these are the rough vocals or the refined ones."""
    try:
        if getattr(state, "vocals_is_vca", False):
            _vocals_status_lbl.configure(text="VCA → FRT + BG",
                                         text_color=BRIGHT_GREEN)
        elif _vocals_refining:
            _vocals_status_lbl.configure(text="REFINING VOCALS…",
                                         text_color="#ffaa00")
        elif getattr(state, "vocals_refined", False):
            _vocals_status_lbl.configure(text="REFINED", text_color=BRIGHT_GREEN)
        elif state.stems:
            _vocals_status_lbl.configure(text="QUICK (6-STEM)",
                                         text_color=TEXT_DIM)
        else:
            _vocals_status_lbl.configure(text="")
        if _vocals_status_lbl.cget("text") and not _vocals_status_lbl.winfo_ismapped():
            _vocals_status_lbl.pack(after=_voc_model_wrap, pady=(1, 0))
        elif not _vocals_status_lbl.cget("text"):
            _vocals_status_lbl.pack_forget()
    except Exception:
        pass


def _set_vocals_progress(frac):
    try:
        if not _vocals_prog_row.winfo_ismapped():
            _vocals_prog_row.pack(fill="x", padx=6, pady=(1, 0),
                                  after=_vocals_status_lbl)
        _vocals_prog_bar.set(max(0.0, min(1.0, frac)))
        _vocals_prog_lbl.configure(text=f"{int(round(frac * 100))}%")
    except Exception:
        pass


def _clear_vocals_progress():
    try:
        _vocals_prog_bar.set(0)
        _vocals_prog_lbl.configure(text="0%")
        _vocals_prog_row.pack_forget()
    except Exception:
        pass


# ── The VOCALS cell steps aside once the split has replaced it ────────────
def _hide_vocals_cell():
    """Take VOCALS out of the mixer: FRT VOX and BG VOX hold it now.

    Leaving it in place means the same singing plays from three cells at
    once, which is both louder than it should be and confusing.
    """
    try:
        if _vocals_cell.winfo_ismapped():
            _vocals_cell.grid_remove()
            print("[Mixer] VOCALS replaced by FRT VOX + BG VOX")
    except Exception:
        pass


def _show_vocals_cell():
    try:
        if not _vocals_cell.winfo_ismapped():
            _vocals_cell.grid()
    except Exception:
        pass


def report_model_files():
    """Print the file each model role resolved to, so a miss is visible."""
    print("[Models] Files found in the models folder:")
    for role, finder in (("vocals",       _voc_find_files),
                         ("karaoke",      _kara_find_files),
                         ("instrumental", _inst_find_files),
                         ("strings",      _str_find_files)):
        try:
            ckpt = finder()[0]
        except Exception:
            ckpt = ""
        name = os.path.basename(ckpt) if ckpt else "— none found —"
        print(f"[Models]   {role:13} {name}")


report_model_files()

# No polarity switches: OTHER has the strings taken out of it automatically
# once gilliaan's model has run, so the two cells no longer share material
# and there is nothing to cancel.

# Credit whichever model each cell is actually going to load.
sync_model_credits()

# Paint every M/S button from state once, so the first frame shows exactly
# what the mixer will do (some cells start muted).
_paint_all_ms()

_make_collapsible(export_outer, _export_title_lbl, "— EXPORT STEMS —")
_make_collapsible(_sai_panel,   _sai_title_lbl,    "— SAI ENGINE —")
_make_collapsible(_rammstein_panel, _rammstein_title_lbl,
                  "— ⛧  RAMMSTEIN FORGE  ⛧ —")

# Signal that the UI is fully built, then start polling for model ready
_splash_set(0.35, "UI BUILT — LOADING MODEL…")
app.after(200, _poll_model_ready)
_load_stem_fixes()          # fixes saved from previous sessions

# Look for a newer ramma.py on GitHub, in the background so a slow or
# unreachable network never delays start-up.
# Always say what the updater is doing: a check that is disabled, finds
# nothing new, or fails used to look exactly the same — silence.
if not _UPDATE_CHECK:
    print("[Update] Checking at start-up is off (_UPDATE_CHECK = False)")
elif not _UPDATE_REPO:
    print("[Update] Disabled — set _UPDATE_REPO near the top of ramma.py "
          "to your GitHub repository, e.g. \"yourname/ramma\"")
else:
    _n = len(_update_targets())
    print(f"[Update] Checking {_UPDATE_REPO} ({_UPDATE_BRANCH}) — "
          f"{_n} file{'s' if _n != 1 else ''} — for newer versions…")
    # quiet=False only means the outcome is printed to the console; the
    # dialog still appears only when there is actually something to install.
    threading.Thread(target=lambda: check_for_update(quiet=False),
                     daemon=True).start()
app.after(1200, _fix_watch_loop)
app.after(1000, _vocals_status_tick)   # keeps the VOCALS line up to date

app.after(800, _fit_loop)   # keep every label and button inside its space
app.bind("<Configure>", _fit_after_resize, add="+")

app.mainloop()

