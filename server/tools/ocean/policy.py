"""PufferNet policies extracted from PufferLib ocean web demos.

PufferLib 5.0 ships no checkpoints, but every demo on puffer.ai/ocean.html
embeds its trained policy as flat fp32 weights inside the Emscripten
preload bundle (assets/<env>/game.data), with the file manifest in
game.js. This module downloads a bundle, slices out <env>_weights.bin (and
the packed config ini when present), and runs the policy in numpy.

The forward pass mirrors src/puffercpu.c exactly: bias-free Linear encoder
-> MinGRU stack (state carried across steps, zeroed on terminal) -> Linear
decoder whose last output is a fused value head. Weight file order:
encoder, decoder, [logstd if continuous], mingru projections, each carved
with an 8-float alignment step.
"""

import re
import urllib.request
from pathlib import Path

import numpy as np

SITE = "https://puffer.ai"
DEFAULT_CACHE = Path("/tmp/puffer_ocean_cache")


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as r:
        dest.write_bytes(r.read())


def bundle_files(js_path):
    """Manifest from a demo game.js: filename -> (start, end)."""
    entries = re.findall(r'\{filename:"([^"]+)",start:(\d+),end:(\d+)\}', js_path.read_text())
    if not entries:
        raise ValueError(f"no preload manifest in {js_path}")
    return {name: (int(s), int(e)) for name, s, e in entries}


def extract_weights(env, cache_dir=DEFAULT_CACHE):
    """(weights float32 ndarray, ini text or None) for an ocean env."""
    cache = Path(cache_dir)
    js, data = cache / f"{env}_game.js", cache / f"{env}_game.data"
    if not js.exists():
        _download(f"{SITE}/assets/{env}/game.js", js)
    if not data.exists():
        _download(f"{SITE}/assets/{env}/game.data", data)
    files = bundle_files(js)
    blob = data.read_bytes()

    def slice_bytes(name):
        if name not in files:
            return None
        start, end = files[name]
        return blob[start:end]

    weights = slice_bytes(f"/resources/{env}/{env}_weights.bin")
    if weights is None:
        raise KeyError(f"{env}_weights.bin not in demo bundle")
    ini = slice_bytes(f"/config/{env}_web.ini") or slice_bytes(f"/config/{env}.ini")
    return (np.frombuffer(weights, dtype="<f4").astype(np.float32), ini.decode() if ini else None)


def align8(n):
    return (n + 7) & ~7


def weight_count(obs_size, act_sizes, hidden, layers):
    """Total floats in a PufferNet weight file (puffercpu.c parity)."""
    atn_sum = sum(act_sizes)
    n = align8(hidden * obs_size)
    n += align8((atn_sum + 1) * hidden)
    for _ in range(layers):
        n += align8(3 * hidden * hidden)
    return n


def infer_shapes(n_floats, obs_size, act_sizes, max_hidden=2048, max_layers=8):
    """Unique (hidden, layers) whose weight_count matches the file.

    The demo bundles omit the policy ini more often than not, but the
    count equation has one solution in the sane range.
    """
    candidates = [
        (hidden, layers)
        for hidden in range(8, max_hidden + 1, 8)
        for layers in range(1, max_layers + 1)
        if weight_count(obs_size, act_sizes, hidden, layers) == n_floats
    ]
    if not candidates:
        raise ValueError(
            f"no PufferNet shape matches a {n_floats}-float file for obs={obs_size} act={act_sizes}"
        )
    if len(candidates) > 1:
        raise ValueError(f"ambiguous weight file ({n_floats} floats): {candidates}")
    return candidates[0]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


class PufferNet:
    """Single-agent PufferNet. step() returns (logits, value).

    Keep one instance per playout: the MinGRU state persists across steps,
    matching the C eval loop, and resets via the terminal flag.
    """

    def __init__(self, weights, obs_size, act_sizes, hidden, layers):
        self.obs_size = obs_size
        self.act_sizes = list(act_sizes)
        self.atn_sum = sum(act_sizes)
        self.hidden = hidden
        self.layers = layers
        if len(weights) != weight_count(obs_size, act_sizes, hidden, layers):
            raise ValueError("weight count does not match architecture")

        idx = 0

        def take(rows, cols):
            nonlocal idx
            w = weights[idx : idx + rows * cols].reshape(rows, cols)
            idx = align8(idx + rows * cols)
            return w

        self.encoder = take(hidden, obs_size)
        self.decoder = take(self.atn_sum + 1, hidden)
        self.proj = [take(3 * hidden, hidden) for _ in range(layers)]
        if idx != len(weights):
            raise ValueError("trailing weights after carving")
        self.state = np.zeros((layers, hidden), dtype=np.float64)

    def reset(self):
        self.state[:] = 0.0

    def step(self, obs, terminal=False):
        """One decision. obs: float array of obs_size; terminal zeroes the
        carried state first, exactly like mingru_zero_term before the first
        decision of a new episode."""
        if terminal:
            self.state[:] = 0.0
        x = self.encoder @ np.asarray(obs, dtype=np.float64)
        for layer in range(self.layers):
            combined = self.proj[layer] @ x
            hidden, gate, hw = (
                combined[: self.hidden],
                combined[self.hidden : 2 * self.hidden],
                combined[2 * self.hidden :],
            )
            gate_s = _sigmoid(gate)
            # puffercpu.c quirk: positive side stays linear-ish, negative
            # side squashes through sigmoid
            h_tilde = np.where(hidden >= 0.0, hidden + 0.5, _sigmoid(hidden))
            mingru_out = self.state[layer] + gate_s * (h_tilde - self.state[layer])
            hw_s = _sigmoid(hw)
            x = hw_s * mingru_out + (1.0 - hw_s) * x
            self.state[layer] = mingru_out
        out = self.decoder @ x
        return out[: self.atn_sum], float(out[self.atn_sum])


def load_policy(env, obs_size, act_sizes, cache_dir=DEFAULT_CACHE, ini=None):
    """Extract, shape-solve, and build the net for an ocean env."""
    weights, bundle_ini = extract_weights(env, cache_dir)
    if ini is None:
        ini = bundle_ini or ""
    m = re.search(r"hidden_size\s*=\s*(\d+)", ini)
    m2 = re.search(r"num_layers\s*=\s*(\d+)", ini)
    if m and m2:
        hidden, layers = int(m.group(1)), int(m2.group(1))
        if weight_count(obs_size, act_sizes, hidden, layers) != len(weights):
            raise ValueError(
                f"{env}: ini says hidden={hidden} layers={layers} but the "
                f"weight file has {len(weights)} floats"
            )
    else:
        hidden, layers = infer_shapes(len(weights), obs_size, act_sizes)
    net = PufferNet(weights, obs_size, act_sizes, hidden, layers)
    net.env, net.arch = env, f"PufferNet h={hidden} l={layers}"
    net.provenance = {
        "weights": f"{SITE}/assets/{env}/game.data",
        "arch": net.arch,
        "floats": int(len(weights)),
    }
    return net


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / temperature
    e = np.exp(z - z.max())
    return e / e.sum()
