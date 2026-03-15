import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


L = 1024  # target length


def buffers_str_to_bytes(s: str) -> bytes:
    """
    Your dataset stores `buffers` as a JSON string that already contains control chars (e.g. \u0018).
    Best effort: interpret it as latin-1 byte string (0..255 codepoints -> 1 byte).
    Fallback to utf-8 if needed.
    """
    try:
        return s.encode("latin-1")
    except UnicodeEncodeError:
        return s.encode("utf-8", errors="replace")


def pad_or_crop(b: bytes, pad_byte: int, crop_mode: str = "none") -> bytes:
    """
    Ensure length exactly L.
    If longer -> crop depending on crop_mode.
    If shorter -> pad with pad_byte.
    """
    n = len(b)
    if n > L:
        if crop_mode == "tail":
            b = b[-L:]
        elif crop_mode == "head":
            b = b[:L]
        else:
            # "none" means: still crop (otherwise can't make positional stats),
            # but do it consistently (head crop by default)
            b = b[:L]
    elif n < L:
        b = b + bytes([pad_byte]) * (L - n)
    return b


def load_records(json_path: Path, dataset_key: str = "dataset"):
    """
    Load records from a JSON file that contains a top-level object with a list
    under `dataset_key`.
    """
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get(dataset_key, [])
    if not isinstance(records, list):
        raise ValueError(f"Expected '{dataset_key}' to be a list in {json_path}")
    return records


def infer_pad_byte(json_path: Path, buffers_field="buffers", sample=2000, dataset_key="dataset") -> int:
    """
    Auto-detect padding byte by looking at trailing runs in raw (variable-length) buffers.
    Heuristic: take last byte of each record if there's a long run of the same byte at the end.
    Choose the most common candidate.
    """
    candidates = []
    records = load_records(json_path, dataset_key=dataset_key)
    for i, rec in enumerate(records):
        if i >= sample:
            break
        raw = buffers_str_to_bytes(rec[buffers_field])
        if not raw:
            continue

        # measure trailing run length of the last byte
        last = raw[-1]
        run = 1
        j = len(raw) - 2
        while j >= 0 and raw[j] == last and run < 512:
            run += 1
            j -= 1

        # If we see a "meaningful" trailing run, treat it as padding candidate
        if run >= 16:
            candidates.append(last)

    if not candidates:
        # Fallback: most common last byte overall
        # (works OK for text-like payloads where padding exists)
        for i, rec in enumerate(records):
            if i >= sample:
                break
            raw = buffers_str_to_bytes(rec[buffers_field])
            if raw:
                candidates.append(raw[-1])

    cnt = Counter(candidates)
    pad_byte, _ = cnt.most_common(1)[0]
    return int(pad_byte)


def entropy_from_counts(counts: np.ndarray) -> np.ndarray:
    """
    counts: (L,256) int
    return: (L,) float Shannon entropy per position
    """
    N = counts.sum(axis=1, keepdims=True).astype(np.float64)  # (L,1)
    p = counts / np.maximum(N, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log2(p, where=(p > 0))
    H = -(p * logp).sum(axis=1)
    return H


def js_divergence(p: np.ndarray, q: np.ndarray, eps=1e-12) -> np.ndarray:
    """
    Jensen–Shannon divergence per position.
    p, q: (L,256) probabilities
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)

    def kl(a, b):
        return (a * (np.log2(a) - np.log2(b))).sum(axis=1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def plot_heatmap(vec: np.ndarray, title: str, out_path: Path, mode="32x32"):
    """
    vec shape: (L,)
    mode:
      - "32x32" => reshape to (32,32)
      - "1x1024" => reshape to (1,L)
    """
    if mode == "32x32":
        img = vec.reshape(32, 32)
        aspect = "equal"
    else:
        img = vec.reshape(1, L)
        aspect = "auto"

    plt.figure()
    plt.imshow(img, aspect=aspect)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(
    attack_json_path: str,
    benign_json_path: str,
    out_dir: str = "pos_stats_out",
    buffers_field: str = "buffers",
    pad_byte="auto",          # "auto" or int 0..255
    crop_mode: str = "none",  # "none"|"head"|"tail"
    heatmap_mode: str = "32x32",
    dataset_key: str = "dataset",
):
    attack_json_path = Path(attack_json_path)
    benign_json_path = Path(benign_json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if pad_byte == "auto":
        pad_a = infer_pad_byte(attack_json_path, buffers_field=buffers_field, dataset_key=dataset_key)
        pad_b = infer_pad_byte(benign_json_path, buffers_field=buffers_field, dataset_key=dataset_key)
        pad = int(Counter([pad_a, pad_b]).most_common(1)[0][0])
    else:
        pad = int(pad_byte)

    counts_b = np.zeros((L, 256), dtype=np.int64)
    counts_a = np.zeros((L, 256), dtype=np.int64)
    n_b = 0
    n_a = 0
    idx = np.arange(L)

    shorter_a = 0
    longer_a = 0
    shorter_b = 0
    longer_b = 0

    attack_records = load_records(attack_json_path, dataset_key=dataset_key)
    for rec in attack_records:
        raw = buffers_str_to_bytes(rec[buffers_field])

        if len(raw) < L:
            shorter_a += 1
        elif len(raw) > L:
            longer_a += 1

        b = pad_or_crop(raw, pad, crop_mode=crop_mode)
        x = np.frombuffer(b, dtype=np.uint8)  # (L,)

        counts_a[idx, x] += 1
        n_a += 1

    benign_records = load_records(benign_json_path, dataset_key=dataset_key)
    for rec in benign_records:
        raw = buffers_str_to_bytes(rec[buffers_field])

        if len(raw) < L:
            shorter_b += 1
        elif len(raw) > L:
            longer_b += 1

        b = pad_or_crop(raw, pad, crop_mode=crop_mode)
        x = np.frombuffer(b, dtype=np.uint8)  # (L,)

        counts_b[idx, x] += 1
        n_b += 1

    H_b = entropy_from_counts(counts_b)
    H_a = entropy_from_counts(counts_a)
    H_diff = H_a - H_b

    pad_rate_b = counts_b[:, pad] / max(n_b, 1)
    pad_rate_a = counts_a[:, pad] / max(n_a, 1)
    pad_rate_diff = pad_rate_a - pad_rate_b

    p_b = counts_b / np.maximum(counts_b.sum(axis=1, keepdims=True), 1)
    p_a = counts_a / np.maximum(counts_a.sum(axis=1, keepdims=True), 1)
    JS = js_divergence(p_a, p_b)

    plot_heatmap(H_b, f"Entropy per position (benign) N={n_b}", out_dir / "entropy_benign.png", mode=heatmap_mode)
    plot_heatmap(H_a, f"Entropy per position (attack) N={n_a}", out_dir / "entropy_attack.png", mode=heatmap_mode)
    plot_heatmap(H_diff, "Entropy diff (attack - benign)", out_dir / "entropy_diff.png", mode=heatmap_mode)

    plot_heatmap(pad_rate_b, f"Pad rate per position (benign), pad=0x{pad:02x}", out_dir / "pad_rate_benign.png", mode=heatmap_mode)
    plot_heatmap(pad_rate_a, f"Pad rate per position (attack), pad=0x{pad:02x}", out_dir / "pad_rate_attack.png", mode=heatmap_mode)
    plot_heatmap(pad_rate_diff, "Pad rate diff (attack - benign)", out_dir / "pad_rate_diff.png", mode=heatmap_mode)

    plot_heatmap(JS, "JS divergence per position (attack vs benign)", out_dir / "js_divergence.png", mode=heatmap_mode)

    # Small report
    report = (
        f"pad_byte = {pad} (0x{pad:02x})\n"
        f"benign N = {n_b}\n"
        f"attack N = {n_a}\n"
        f"benign records shorter than {L}: {shorter_b}\n"
        f"benign records longer than {L}: {longer_b}\n"
        f"attack records shorter than {L}: {shorter_a}\n"
        f"attack records longer than {L}: {longer_a}\n"
    )
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    # Example:
    main(
        "./dataset/attack/attack_train.json",
        "./dataset/benign/benign_train.json",
        pad_byte="auto",
        crop_mode="none",
        heatmap_mode="32x32",
    )
    pass
