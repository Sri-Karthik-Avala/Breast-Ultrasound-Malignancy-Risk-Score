import os, sys, csv, math, time, random, socket

__author__ = "Karthik"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
try:
    socket.setdefaulttimeout(6)
except Exception:
    pass

import numpy as np

T0 = time.time()
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
CWD = os.getcwd()

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_root():
    cands = []
    for b in [CWD, HERE, os.path.join(CWD, ".."), os.path.join(HERE, "..")]:
        for sub in ["", "public", os.path.join("dataset", "public"), "input", os.path.join("dataset")]:
            cands.append(os.path.normpath(os.path.join(b, sub)))
    seen = set()
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(os.path.join(c, "train.csv")) and os.path.isfile(os.path.join(c, "test.csv")):
            return c
    return CWD


ROOT = find_root()


def build_image_index(root, needed=None):
    idx = {}
    search_dirs = []
    for b in [root, CWD, HERE, os.path.join(root, ".."), os.path.join(CWD, "..")]:
        for sub in ["images", os.path.join("public", "images"), os.path.join("dataset", "public", "images")]:
            search_dirs.append(os.path.normpath(os.path.join(b, sub)))
    for b in [root, CWD, HERE]:
        search_dirs.append(os.path.normpath(b))
    need = set(needed) if needed is not None else None
    seen = set()
    for d in search_dirs:
        if d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        try:
            for fn in os.listdir(d):
                low = fn.lower()
                if low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png"):
                    key = os.path.splitext(fn)[0]
                    if (need is None or key in need) and key not in idx:
                        idx[key] = os.path.join(d, fn)
        except Exception:
            pass
        if need is not None and len(idx) >= len(need):
            break
    return idx


def read_csv_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for line in r:
            if line:
                rows.append(line)
    return header, rows


def load_data():
    htr, rtr = read_csv_rows(os.path.join(ROOT, "train.csv"))
    hte, rte = read_csv_rows(os.path.join(ROOT, "test.csv"))
    tr_ids = [x[0] for x in rtr]
    tr_y = np.array([int(float(x[1])) for x in rtr], dtype=np.float32)
    te_ids = [x[0] for x in rte]
    ssub = os.path.join(ROOT, "sample_submission.csv")
    if os.path.isfile(ssub):
        _, rss = read_csv_rows(ssub)
        order = [x[0] for x in rss if x and x[0] in set(te_ids)]
        if len(order) == len(te_ids):
            te_ids = order
    return tr_ids, tr_y, te_ids


def write_submission(te_ids, risks):
    risks = np.asarray(risks, dtype=np.float64)
    risks = np.where(np.isfinite(risks), risks, 0.37)
    risks = np.clip(risks, 0.0, 1.0)
    out_dirs = [os.path.join(CWD, "working"), os.path.join(HERE, "working")]
    targets = []
    for od in out_dirs:
        try:
            os.makedirs(od, exist_ok=True)
            targets.append(os.path.join(od, "submission.csv"))
        except Exception:
            pass
    targets.append(os.path.join(CWD, "submission.csv"))
    done = set()
    for tpath in targets:
        ap = os.path.abspath(tpath)
        if ap in done:
            continue
        done.add(ap)
        try:
            with open(tpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "risk"])
                for i, idv in enumerate(te_ids):
                    w.writerow([idv, "%.6f" % float(risks[i])])
        except Exception:
            pass


TR_IDS, TR_Y, TE_IDS = load_data()
PREV = float(TR_Y.mean())


def metric_base(risk, y, w=None, lo=0.20, hi=0.60, step=0.01, tsoft=2.0):
    risk = np.clip(np.asarray(risk, dtype=np.float64), 0, 1)
    y = np.asarray(y, dtype=np.float64)
    if w is None:
        w = np.ones_like(y)
    Wsum = w.sum()
    prev = (w * y).sum() / Wsum
    pts = np.arange(lo, hi + 1e-9, step)
    s = np.empty(len(pts))
    for i, pt in enumerate(pts):
        odds = pt / (1.0 - pt)
        dec = (risk >= pt).astype(np.float64)
        tp = (w * dec * y).sum() / Wsum
        fp = (w * dec * (1 - y)).sum() / Wsum
        yld = tp - fp * odds
        triv = max(prev - (1 - prev) * odds, 0.0)
        den = prev - triv
        s[i] = 0.0 if den <= 1e-12 else min(max((yld - triv) / den, 0.0), 1.0)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * (pts - lo) / (hi - lo))
    A = (hann * s).sum() / hann.sum() if hann.sum() > 1e-12 else s.mean()
    m = -(1.0 / tsoft) * np.log(np.mean(np.exp(-tsoft * s)) + 1e-12)
    return 0.6 * A + 0.4 * m


def fast_auc(score, y):
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y)
    order = np.argsort(score, kind="mergesort")
    s_sorted = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    n = len(score)
    rk = np.arange(1, n + 1, dtype=np.float64)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = rk[i:j + 1].mean()
        i = j + 1
    npos = (y == 1).sum()
    nneg = (y == 0).sum()
    if npos == 0 or nneg == 0:
        return 0.5
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_platt(z, y, lam=1e-2, iters=80):
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Np = y.sum()
    Nn = len(y) - Np
    tpos = (Np + 1.0) / (Np + 2.0)
    tneg = 1.0 / (Nn + 2.0)
    t = np.where(y == 1, tpos, tneg)
    a = 1.0
    b = math.log((Np + 1.0) / (Nn + 1.0))
    for _ in range(iters):
        eta = a * z + b
        p = sigmoid(eta)
        wv = np.clip(p * (1 - p), 1e-9, None)
        ga = (z * (p - t)).sum() + lam * (a - 1.0)
        gb = (p - t).sum()
        Haa = (wv * z * z).sum() + lam
        Hab = (wv * z).sum()
        Hbb = wv.sum()
        det = Haa * Hbb - Hab * Hab
        if abs(det) < 1e-12:
            break
        da = (Hbb * ga - Hab * gb) / det
        db = (-Hab * ga + Haa * gb) / det
        a -= da
        b -= db
        if abs(da) + abs(db) < 1e-9:
            break
    if not (np.isfinite(a) and np.isfinite(b)):
        return 1.0, 0.0
    return float(a), float(b)


def prevalence_weights(y, target):
    y = np.asarray(y, dtype=np.float64)
    base = y.mean()
    wp = target / max(base, 1e-6)
    wn = (1 - target) / max(1 - base, 1e-6)
    w = np.where(y == 1, wp, wn)
    return w * (len(y) / w.sum())


def stratified_folds(y, n_splits, seed):
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_splits)]
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for k, ix in enumerate(idx):
            folds[k % n_splits].append(ix)
    return [np.array(sorted(f)) for f in folds]


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from PIL import Image
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

if HAS_TORCH:
    torch.manual_seed(SEED)
    try:
        torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cpu"


def load_cache(all_ids, index):
    arr = np.zeros((len(all_ids), 112, 112, 3), dtype=np.uint8)
    for i, idv in enumerate(all_ids):
        p = index.get(idv)
        if p is None:
            continue
        try:
            im = Image.open(p).convert("RGB").resize((112, 112), Image.BILINEAR)
            arr[i] = np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    return t


def _construct(name, weights):
    fn = getattr(torchvision.models, name, None)
    if fn is None:
        return None
    return fn(weights=weights)


def build_model(name, drop=0.4):
    m = None
    pretrained = False
    for wo in ["DEFAULT", "IMAGENET1K_V1"]:
        try:
            m = _construct(name, wo)
            if m is not None:
                pretrained = True
                break
        except Exception:
            m = None
    if m is None:
        try:
            m = _construct(name, None)
        except Exception:
            m = None
    if m is None:
        return None, False
    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        nf = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(nf, 1))
    elif hasattr(m, "classifier"):
        cl = m.classifier
        if isinstance(cl, nn.Linear):
            nf = cl.in_features
            m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(nf, 1))
        else:
            last = None
            for i in range(len(cl) - 1, -1, -1):
                if isinstance(cl[i], nn.Linear):
                    last = i
                    break
            nf = cl[last].in_features
            keep = list(cl[:last])
            m.classifier = nn.Sequential(*keep, nn.Dropout(drop), nn.Linear(nf, 1))
    else:
        return None, False
    return m, pretrained


def head_params(model):
    hp = []
    bp = []
    for n, p in model.named_parameters():
        if n.startswith("fc.") or n.startswith("classifier"):
            hp.append(p)
        else:
            bp.append(p)
    return bp, hp


if HAS_TORCH:
    GEN = torch.Generator(device=DEVICE)
    GEN.manual_seed(SEED)
    MEAN_T = torch.tensor(MEAN, device=DEVICE).view(1, 3, 1, 1)
    STD_T = torch.tensor(STD, device=DEVICE).view(1, 3, 1, 1)


def rnd(shape):
    return torch.rand(shape, generator=GEN, device=DEVICE)


def augment(x, size):
    B = x.shape[0]
    area = 0.45 + 0.55 * rnd(B)
    cf = torch.sqrt(area)
    flip = torch.where(rnd(B) < 0.5, torch.ones(B, device=DEVICE), -torch.ones(B, device=DEVICE))
    rot = (rnd(B) - 0.5) * 0.30
    tx = (1 - cf) * (2 * rnd(B) - 1)
    ty = (1 - cf) * (2 * rnd(B) - 1)
    cr = torch.cos(rot)
    sr = torch.sin(rot)
    theta = torch.zeros(B, 2, 3, device=DEVICE)
    theta[:, 0, 0] = cf * flip * cr
    theta[:, 0, 1] = -cf * sr
    theta[:, 1, 0] = cf * flip * sr
    theta[:, 1, 1] = cf * cr
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (B, 3, size, size), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")
    if float(rnd(1)) < 0.6:
        ds = int(size * (0.5 + 0.5 * float(rnd(1))))
        ds = max(48, min(size, ds))
        x = F.interpolate(x, size=ds, mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    br = (0.7 + 0.6 * rnd(B)).view(B, 1, 1, 1)
    x = x * br
    mn = x.mean(dim=(1, 2, 3), keepdim=True)
    ct = (0.7 + 0.6 * rnd(B)).view(B, 1, 1, 1)
    x = (x - mn) * ct + mn
    sg = (0.01 + 0.04 * rnd(B)).view(B, 1, 1, 1)
    x = x + torch.randn(x.shape, generator=GEN, device=DEVICE) * sg
    x = x.clamp(0, 1)
    er = rnd(B) < 0.30
    for i in range(B):
        if bool(er[i]):
            eh = int((0.1 + 0.25 * float(rnd(1))) * size)
            ew = int((0.1 + 0.25 * float(rnd(1))) * size)
            top = int(float(rnd(1)) * (size - eh))
            left = int(float(rnd(1)) * (size - ew))
            x[i, :, top:top + eh, left:left + ew] = float(rnd(1))
    return x


def normalize(x):
    return (x - MEAN_T) / STD_T


def prep_eval(x, size):
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return normalize(x)


def predict_logits(model, cache, idx, size, bs=256):
    model.eval()
    out = np.zeros(len(idx), dtype=np.float64)
    use_amp = (DEVICE == "cuda")
    scales = [size, int(round(size * 1.15))]
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            sel = idx[s:s + bs]
            base = cache[sel].to(DEVICE).float() / 255.0
            accum = None
            nv = 0
            for sc in scales:
                xb = prep_eval(base, sc)
                if use_amp:
                    with torch.autocast("cuda"):
                        l1 = model(xb)
                        l2 = model(torch.flip(xb, dims=[3]))
                else:
                    l1 = model(xb)
                    l2 = model(torch.flip(xb, dims=[3]))
                v = ((l1 + l2) / 2.0).float().view(-1)
                accum = v if accum is None else accum + v
                nv += 1
            lg = (accum / nv).cpu().numpy()
            out[s:s + len(sel)] = lg
    return out


def train_fold(cache, tr_idx, y_tr, val_idx, te_idx, name, size, epochs, bs, deadline, mixup=0.0):
    model, pretrained = build_model(name)
    if model is None:
        return None, None
    model = model.to(DEVICE)
    bp, hp = head_params(model)
    opt = torch.optim.AdamW(
        [{"params": bp, "lr": 3e-4}, {"params": hp, "lr": 1e-3}],
        weight_decay=0.04,
    )
    steps = max(1, len(tr_idx) // bs)
    total = steps * epochs
    warm = steps * 2
    use_amp = (DEVICE == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ls = 0.05
    mrng = np.random.RandomState(SEED + 4242)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    gstep = 0
    y_tr_t = torch.tensor(y_tr, device=DEVICE, dtype=torch.float32)
    order = np.arange(len(tr_idx))
    for ep in range(epochs):
        if time.time() > deadline:
            break
        np.random.RandomState(SEED + ep).shuffle(order)
        model.train()
        for s in range(steps):
            bidx = order[s * bs:(s + 1) * bs]
            if len(bidx) == 0:
                continue
            sel = tr_idx[bidx]
            xb = cache[sel].to(DEVICE).float() / 255.0
            xb = augment(xb, size)
            xb = normalize(xb)
            yb = y_tr_t[bidx].view(-1, 1)
            if mixup > 0 and xb.shape[0] > 1 and mrng.rand() < 0.6:
                lam = float(mrng.beta(mixup, mixup))
                perm = torch.randperm(xb.shape[0], device=DEVICE, generator=GEN)
                xb = lam * xb + (1 - lam) * xb[perm]
                yb = lam * yb + (1 - lam) * yb[perm]
            yb = yb * (1 - ls) + 0.5 * ls
            lr_scale = (gstep + 1) / warm if gstep < warm else 0.5 * (1 + math.cos(math.pi * (gstep - warm) / max(1, total - warm)))
            lr_scale = max(lr_scale, 1e-3)
            opt.param_groups[0]["lr"] = 3e-4 * lr_scale
            opt.param_groups[1]["lr"] = 1e-3 * lr_scale
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast("cuda"):
                    out = model(xb)
                    loss = F.binary_cross_entropy_with_logits(out, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                out = model(xb)
                loss = F.binary_cross_entropy_with_logits(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            d = min(0.999, (1 + gstep) / (10 + gstep))
            with torch.no_grad():
                sd = model.state_dict()
                for k in ema:
                    v = sd[k]
                    if v.dtype.is_floating_point:
                        ema[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
                    else:
                        ema[k] = v.detach().clone()
            gstep += 1
    bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        model.load_state_dict({k: ema[k].to(bak[k].dtype) for k in bak})
    except Exception:
        model.load_state_dict(bak)
    val_lg = predict_logits(model, cache, val_idx, size)
    te_lg = predict_logits(model, cache, te_idx, size)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return val_lg, te_lg


def calibrate(oof_logits, y, test_logits, names):
    a_b = {}
    for nm in names:
        z = oof_logits[nm]
        mask = np.isfinite(z)
        if mask.sum() < 10:
            a_b[nm] = (1.0, 0.0)
        else:
            a_b[nm] = fit_platt(z[mask], y[mask])

    def oof_prob(tscale, c):
        acc = np.zeros(len(y))
        cnt = np.zeros(len(y))
        for nm in names:
            a, b = a_b[nm]
            z = oof_logits[nm]
            mask = np.isfinite(z)
            p = sigmoid((a * z[mask] + b) / tscale + c)
            acc[mask] += p
            cnt[mask] += 1
        cnt = np.clip(cnt, 1, None)
        return acc / cnt

    base_t = PREV
    prev_grid = [max(0.2, base_t - 0.07), base_t - 0.045, base_t - 0.02, base_t, base_t + 0.015]
    best = (-1e9, 1.0, 0.0)
    for tscale in np.linspace(0.8, 1.4, 25):
        for c in np.linspace(-0.4, 0.2, 31):
            p = oof_prob(tscale, c)
            sc = min(metric_base(p, y, prevalence_weights(y, pv)) for pv in prev_grid)
            if sc > best[0]:
                best = (sc, tscale, c)
    _, tscale, c = best

    n_test = len(test_logits[names[0]][0])
    acc = np.zeros(n_test)
    cnt = 0
    for nm in names:
        a, b = a_b[nm]
        for tl in test_logits[nm]:
            acc += sigmoid((a * tl + b) / tscale + c)
            cnt += 1
    final = acc / max(cnt, 1)
    oof_final = oof_prob(tscale, c)
    return final, oof_final, (tscale, c)


def run():
    if not HAS_TORCH:
        return None
    all_ids = TR_IDS + TE_IDS
    index = build_image_index(ROOT, needed=set(all_ids))
    cache = load_cache(all_ids, index)
    n_tr = len(TR_IDS)
    te_idx = np.arange(n_tr, n_tr + len(TE_IDS))

    if DEVICE == "cuda":
        configs = [("densenet121", 192), ("resnet18", 192), ("resnet18", 224)]
        epochs = 26
        bs = 32
        n_splits = 5
        cap = 3400.0
        mix = 0.2
    else:
        configs = [("resnet18", 160)]
        epochs = 7
        bs = 32
        n_splits = 5
        cap = 1500.0
        mix = 0.0

    deadline = T0 + cap
    folds = stratified_folds(TR_Y, n_splits, SEED)

    oof_logits = {}
    test_logits = {}
    used = []
    for (nm, size) in configs:
        chk, _ = build_model(nm)
        if chk is None:
            continue
        del chk
        key = "%s_%d" % (nm, size)
        oof = np.full(n_tr, np.nan)
        te_list = []
        ok = False
        for f in range(n_splits):
            if time.time() > deadline:
                break
            val_idx = folds[f]
            tr_idx = np.array([i for i in range(n_tr) if i not in set(val_idx.tolist())])
            vl, tl = None, None
            for cur_bs in [bs, bs // 2, max(8, bs // 4)]:
                try:
                    vl, tl = train_fold(cache, tr_idx, TR_Y[tr_idx], val_idx, te_idx, nm, size, epochs, cur_bs, deadline, mixup=mix)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if DEVICE == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    raise
            if vl is None:
                continue
            oof[val_idx] = vl
            te_list.append(tl)
            ok = True
        if ok and len(te_list) > 0:
            oof_logits[key] = oof
            test_logits[key] = te_list
            used.append(key)
            try:
                m = np.isfinite(oof)
                print("config", key, "oof_auc", round(fast_auc(oof[m], TR_Y[m]), 4), "t", round(time.time() - T0, 1))
            except Exception:
                pass

    if not used:
        return None

    final, oof_final, params = calibrate(oof_logits, TR_Y, test_logits, used)
    try:
        print("calib", params, "oof_base", round(metric_base(oof_final, TR_Y), 4),
              "oof_auc", round(fast_auc(oof_final, TR_Y), 4))
    except Exception:
        pass

    final = np.where(np.isfinite(final), final, PREV)
    if not (np.std(final) > 1e-4):
        r = np.argsort(np.argsort(final))
        final = 0.05 + 0.9 * r / max(1, len(final) - 1)
    final = np.clip(final, 0.02, 0.98)
    return final


def main():
    try:
        risks = run()
    except Exception as e:
        try:
            print("run failed:", repr(e))
        except Exception:
            pass
        risks = None
    if risks is None:
        risks = np.full(len(TE_IDS), PREV, dtype=np.float64)
        risks = np.clip(risks, 0.02, 0.98)
    write_submission(TE_IDS, risks)
    try:
        print("wrote submission", len(TE_IDS), "rows; risk mean", round(float(np.mean(risks)), 4))
    except Exception:
        pass


if __name__ == "__main__":
    main()
