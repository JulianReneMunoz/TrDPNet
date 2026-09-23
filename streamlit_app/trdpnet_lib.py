# Librería TrDPNet generada desde el notebook del proyecto.
import math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DIST_CAM, FOCAL = 2.0, 1.6
CFG = {'modo': 'rapido', 'seed': 42, 'categorias': ['avion', 'banca', 'armario', 'auto', 'silla', 'pantalla', 'lampara', 'parlante', 'rifle', 'sofa', 'mesa', 'telefono', 'embarcacion'], 'n_puntos': 1024, 'img_size': 48, 'patch': 8, 'n_train_por_cat': 80, 'n_test_por_cat': 12, 'dim': 128, 'depth': 4, 'heads': 4, 'img_depth': 2, 'M_centros': 64, 'K_vecinos': 24, 'radio': 0.22, 'T': 200, 'beta_0': 0.00085, 'beta_T': 0.012, 'warmup': 0.1, 'p_uncond': 0.1, 'omega': 3.0, 'pasos': 1500, 'batch': 16, 'lr': 0.0003, 'weight_decay': 0.01, 'tau': 0.05, 'n_eval_por_cat': 2, 'omegas_ablacion': [1.0, 2.0, 3.0, 5.0], 'ablacion_self_attention': False, 'datos': 'ShapeNet-R2N2 real'}

def farthest_point_sample(xyz, M):
    '''xyz (B,N,3) -> índices (B,M) de los puntos más separados entre sí.'''
    B, N, _ = xyz.shape
    idx = torch.zeros(B, M, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    lejano = torch.zeros(B, dtype=torch.long, device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(M):
        idx[:, i] = lejano
        centro = xyz[ar, lejano].unsqueeze(1)
        d = ((xyz - centro) ** 2).sum(-1)
        dist = torch.minimum(dist, d)          # distancia al conjunto ya elegido
        lejano = dist.argmax(-1)               # el punto peor cubierto
    return idx


def index_points(x, idx):
    '''Indexado por lotes: x (B,N,C), idx (B,...) -> (B,...,C).'''
    B = x.shape[0]
    ar = torch.arange(B, device=x.device).view(B, *([1] * (idx.dim() - 1)))
    return x[ar, idx]


def ball_query(xyz, centros, radio, K):
    '''Para cada centro, K vecinos dentro de 'radio' (relleno con el más cercano).'''
    d = torch.cdist(centros, xyz)                       # (B,M,N)
    idx = d.argsort(-1)[:, :, :K]                       # K vecinos más próximos
    dk = torch.gather(d, 2, idx)
    primero = idx[:, :, :1].expand(-1, -1, K)
    return torch.where(dk <= radio, idx, primero), dk


def timestep_embedding(t, dim):
    '''t (B,) -> (B,dim) con pares coseno/seno de frecuencias geométricas.'''
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class CapaSA(nn.Module):
    '''Nube ruidosa (B,N,3) -> tokens (B,M,D) + centros (B,M,3).'''
    def __init__(self, M, K, radio, dim):
        super().__init__()
        self.M, self.K, self.radio = M, K, radio
        self.mlp = nn.Sequential(
            nn.Conv2d(3, 64, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, dim, 1))
        self.pos = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, xyz):
        fps = farthest_point_sample(xyz, self.M)
        kp = index_points(xyz, fps)                                   # (B,M,3) puntos clave
        idx, _ = ball_query(xyz, kp, self.radio, self.K)              # (B,M,K)
        rel = index_points(xyz, idx) - kp.unsqueeze(2)                # coords. relativas
        f = self.mlp(rel.permute(0, 3, 1, 2))                         # (B,D,M,K)
        z = f.max(dim=-1).values.transpose(1, 2)                      # max-pool -> (B,M,D)
        return z + self.pos(kp), kp, idx

class CodificadorImagen(nn.Module):
    '''Imagen (B,1,H,W) -> tokens visuales (B,I,D).'''
    def __init__(self, dim, img, patch, depth, heads, in_ch=1):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.n_tokens = (img // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        capa = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                          norm_first=True, activation="gelu", dropout=0.0)
        self.enc = nn.TransformerEncoder(capa, depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, img):
        x = self.proj(img).flatten(2).transpose(1, 2) + self.pos   # parches + posición
        return self.norm(self.enc(x))

class BloqueTransformer(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, usar_self=True, modo_fusion="cross"):
        super().__init__()
        self.usar_self, self.modo_fusion = usar_self, modo_fusion
        self.n1 = nn.LayerNorm(dim)
        self.auto = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim); self.nc = nn.LayerNorm(dim)
        self.cruz = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, z, c, devolver_attn=False):
        M = z.shape[1]
        if self.modo_fusion == "concat":
            # Variante literal de la Ecuación (6): Splice(Attn(Cat(z,c)), M)
            h = self.n1(torch.cat([z, self.nc(c)], dim=1))
            o, w = self.auto(h, h, h, need_weights=devolver_attn, average_attn_weights=True)
            z = z + o[:, :M]
            z = z + self.ffn(self.n3(z))
            return z, (w[:, :M, M:] if w is not None else None)
        if self.usar_self:
            h = self.n1(z)
            z = z + self.auto(h, h, h, need_weights=False)[0]
        h, ck = self.n2(z), self.nc(c)
        o, w = self.cruz(h, ck, ck, need_weights=devolver_attn, average_attn_weights=True)
        z = z + o
        z = z + self.ffn(self.n3(z))
        return z, w

class CapaFP(nn.Module):
    '''(X_t, centros, z, ẑ) -> ruido predicho (B,N,3).'''
    def __init__(self, dim, k=3):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(nn.Linear(2 * dim + 3, dim), nn.GELU(),
                                 nn.Linear(dim, dim // 2), nn.GELU(),
                                 nn.Linear(dim // 2, 3))

    def forward(self, xyz, centros, z, z_hat):
        d = torch.cdist(xyz, centros)                       # (B,N,M)
        dk, idx = d.topk(self.k, dim=-1, largest=False)     # 3 centros más cercanos
        w = 1.0 / (dk + 1e-8); w = w / w.sum(-1, keepdim=True)
        f  = (index_points(z,     idx) * w.unsqueeze(-1)).sum(2)
        fh = (index_points(z_hat, idx) * w.unsqueeze(-1)).sum(2)
        return self.mlp(torch.cat([xyz, f, fh], dim=-1)), w, idx

class TrDPNet(nn.Module):
    def __init__(self, cfg, usar_self=True, modo_fusion="cross"):
        super().__init__()
        d = cfg["dim"]
        self.sa  = CapaSA(cfg["M_centros"], cfg["K_vecinos"], cfg["radio"], d)
        self.img = CodificadorImagen(d, cfg["img_size"], cfg["patch"], cfg["img_depth"], cfg["heads"])
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.nulo = nn.Parameter(torch.zeros(1, self.img.n_tokens, d))   # token nulo (CFG)
        nn.init.trunc_normal_(self.nulo, std=0.02)
        self.bloques = nn.ModuleList([BloqueTransformer(d, cfg["heads"], 4.0, usar_self, modo_fusion)
                                      for _ in range(cfg["depth"])])
        self.fp = CapaFP(d)
        self.dim = d

    def forward(self, x_t, imagen, t, descartar=None, devolver_attn=False):
        z, centros, _ = self.sa(x_t)                       # encoder geométrico
        tokens_img = self.img(imagen)                      # encoder visual
        if descartar is not None:                          # guía sin clasificador
            m = descartar.view(-1, 1, 1).float()
            tokens_img = m * self.nulo.expand(tokens_img.shape[0], -1, -1) + (1 - m) * tokens_img
        t_tok = self.t_mlp(timestep_embedding(t, self.dim)).unsqueeze(1)
        c = torch.cat([tokens_img, t_tok], dim=1)          # condición (I+1) tokens
        z_hat, attn = z, None
        for blk in self.bloques:                           # cascada de L bloques
            z_hat, w = blk(z_hat, c, devolver_attn)
            if w is not None: attn = w
        eps, _, _ = self.fp(x_t, centros, z, z_hat)        # decoder geométrico
        return (eps, attn) if devolver_attn else (eps, None)

class Difusion:
    '''Calendario DDPM lineal con warmup, según la Sección 4.2 del artículo.'''
    def __init__(self, T, b0, b1, warmup, device):
        betas = torch.linspace(b0, b1, T)
        nw = max(1, int(T * warmup))
        betas[:nw] = torch.linspace(b0 * 0.1, b0, nw)     # arranque suave
        self.T = T
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.ab = torch.cumprod(self.alphas, dim=0)       # alpha barra
        self.sqrt_ab = self.ab.sqrt()
        self.sqrt_1mab = (1 - self.ab).sqrt()

    def q_sample(self, x0, t, ruido):
        '''Ecuación (2): difusión directa en forma cerrada.
        Los coeficientes se mueven al dispositivo de x0, de modo que la misma
        instancia sirve para tensores en CPU (celdas de demostración) y en GPU
        (entrenamiento y muestreo) sin mezclar dispositivos.'''
        dev = x0.device
        a = self.ab.to(dev)[t.to(dev)].view(-1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * ruido

@torch.no_grad()
def muestrea(modelo, imagen, n_puntos, dif, omega=3.0, guardar=(), semilla=None, verbose=False):
    '''Algoritmo 2 del artículo. Devuelve (X_0, instantáneas).'''
    modelo.eval()
    if semilla is not None: torch.manual_seed(semilla)
    B = imagen.shape[0]
    x = torch.randn(B, n_puntos, 3, device=DEVICE)
    cond   = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    uncond = torch.ones(B,  dtype=torch.bool, device=DEVICE)
    snaps = {}
    for i in reversed(range(dif.T)):
        t = torch.full((B,), i, device=DEVICE, dtype=torch.long)
        e_c, _ = modelo(x, imagen, t, descartar=cond)
        if omega != 1.0:
            e_u, _ = modelo(x, imagen, t, descartar=uncond)
            eps = e_u + omega * (e_c - e_u)          # Ecuación (5)
        else:
            eps = e_c
        a, ab = dif.alphas[i], dif.ab[i]
        x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()     # Ecuación (4)
        if i > 0:
            x = x + dif.betas[i].sqrt() * torch.randn_like(x)
        if i in guardar: snaps[i] = x.clone().cpu().numpy()
        if verbose and i % max(1, dif.T // 5) == 0: print(f"    t={i}", flush=True)
    return x, snaps


def chamfer(pred, gt):
    '''CD simétrica por muestra. pred (B,N,3), gt (B,M,3) -> (B,)'''
    d = torch.cdist(pred, gt)
    return (d.min(2).values ** 2).mean(1) + (d.min(1).values ** 2).mean(1)


def f_score(pred, gt, tau):
    '''F-score@tau con precisión y recall. Devuelve (F, P, R), cada uno (B,).'''
    d = torch.cdist(pred, gt)
    p = (d.min(2).values < tau).float().mean(1)      # precisión
    r = (d.min(1).values < tau).float().mean(1)      # recall
    return 2 * p * r / (p + r + 1e-9), p, r


def _caja(n, cx, cy, cz, sx, sy, sz, rng):
    '''Puntos sobre la superficie de una caja (no en su interior).'''
    p = rng.uniform(-0.5, 0.5, size=(n, 3)) * np.array([sx, sy, sz])
    eje = rng.integers(0, 3, size=n); signo = rng.choice([-0.5, 0.5], size=n)
    for a in range(3):
        m = eje == a
        p[m, a] = signo[m] * [sx, sy, sz][a]
    return p + np.array([cx, cy, cz])


def _cilindro(n, cx, cy, cz, r, h, rng, eje=1):
    '''Puntos sobre el manto de un cilindro alineado a un eje.'''
    th = rng.uniform(0, 2 * np.pi, n); hh = rng.uniform(-h / 2, h / 2, n)
    a, b = r * np.cos(th), r * np.sin(th)
    p = {0: np.stack([hh, a, b], 1), 1: np.stack([a, hh, b], 1), 2: np.stack([a, b, hh], 1)}[eje]
    return p + np.array([cx, cy, cz])


def genera_forma(cat, n, rng):
    '''Devuelve una nube (n,3) normalizada de la categoría pedida.'''
    if cat == "avion":
        L, W = rng.uniform(0.9, 1.15), rng.uniform(0.8, 1.25)
        partes = [_cilindro(int(n * .35), 0, 0, 0, 0.06, L, rng, eje=2),          # fuselaje
                  _caja(int(n * .35), 0, 0, 0.0, W, 0.02, 0.25, rng),             # alas
                  _caja(int(n * .15), 0, 0, -L / 2 + .1, 0.42, 0.02, 0.15, rng),  # estabilizador
                  _caja(int(n * .15), 0, .12, -L / 2 + .1, 0.02, 0.26, 0.15, rng)]# deriva
    elif cat == "silla":
        h, w = rng.uniform(0.5, 0.8), rng.uniform(0.45, 0.62)
        partes = [_caja(int(n * .30), 0, 0, 0, w, 0.05, w, rng),                  # asiento
                  _caja(int(n * .30), 0, h / 2, -w / 2, w, h, 0.05, rng)]         # respaldo
        for dx in (-1, 1):
            for dz in (-1, 1):
                partes.append(_cilindro(int(n * .10), dx * w / 2 * .85, -.25, dz * w / 2 * .85, .03, .5, rng))
    elif cat == "mesa":
        w, d, h = rng.uniform(0.8, 1.15), rng.uniform(0.5, 0.85), rng.uniform(0.4, 0.62)
        partes = [_caja(int(n * .40), 0, 0, 0, w, 0.06, d, rng)]                  # tablero
        for dx in (-1, 1):
            for dz in (-1, 1):
                partes.append(_cilindro(int(n * .15), dx * w / 2 * .9, -h / 2, dz * d / 2 * .85, .035, h, rng))
    elif cat == "lampara":
        h, r = rng.uniform(0.6, 0.95), rng.uniform(0.25, 0.42)
        th = rng.uniform(0, 2 * np.pi, int(n * .45)); u = rng.uniform(0, 1, int(n * .45))
        pantalla = np.stack([r * u * np.cos(th), h / 2 + .25 * (1 - u), r * u * np.sin(th)], 1)
        partes = [pantalla,
                  _cilindro(int(n * .35), 0, 0, 0, .03, h, rng),                  # vástago
                  _cilindro(int(n * .20), 0, -h / 2, 0, .22, .04, rng)]           # base
    else:  # sofa
        w, d = rng.uniform(0.9, 1.25), rng.uniform(0.5, 0.68)
        partes = [_caja(int(n * .35), 0, 0, 0, w, .25, d, rng),
                  _caja(int(n * .25), 0, .25, -d / 2, w, .5, .12, rng),
                  _caja(int(n * .20), -w / 2, .15, 0, .12, .4, d, rng),
                  _caja(int(n * .20),  w / 2, .15, 0, .12, .4, d, rng)]
    pts = np.concatenate(partes, 0)
    if len(pts) < n:                                   # completar por remuestreo
        pts = np.concatenate([pts, pts[rng.integers(0, len(pts), n - len(pts))]], 0)
    pts = pts[rng.permutation(len(pts))[:n]]
    pts -= pts.mean(0, keepdims=True)                  # centrar
    pts /= (np.abs(pts).max() + 1e-8) / 0.5            # escalar al cubo [-0.5, 0.5]
    return pts.astype(np.float32)


def base_camara(elev_deg, azim_deg):
    '''Devuelve la matriz de rotación mundo->cámara (3x3).'''
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    frente = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    arriba = np.array([0.0, 1.0, 0.0])
    derecha = np.cross(frente, arriba); derecha /= np.linalg.norm(derecha)
    arriba2 = np.cross(derecha, frente)
    return np.stack([derecha, arriba2, frente], 0)


def renderiza(points, img=None, elev=20.0, azim=35.0):
    '''Nube (N,3) -> imagen de profundidad (1,H,W) en [0,1]. 0 = fondo.'''
    img = img or CFG["img_size"]
    cam = points @ base_camara(elev, azim).T
    z = cam[:, 2] + DIST_CAM
    u = (cam[:, 0] / z) * FOCAL
    v = (cam[:, 1] / z) * FOCAL
    ix = np.clip(((u + 0.5) * img).astype(int), 0, img - 1)
    iy = np.clip(((0.5 - v) * img).astype(int), 0, img - 1)
    prof = np.full((img, img), np.inf)
    np.minimum.at(prof, (iy, ix), z)          # z-buffer: el punto más cercano gana
    m = np.isfinite(prof)
    out = np.zeros((img, img), np.float32)
    if m.any():
        d = prof[m]
        out[m] = 1.0 - (d - d.min()) / (d.max() - d.min() + 1e-8) * 0.7
    return out[None].astype(np.float32)


def _redimensiona(a, lado):
    '''Redimensionado por vecino más cercano, sin dependencias externas.'''
    h, w = a.shape[:2]
    fi = (np.arange(lado) * h / lado).astype(int).clip(0, h - 1)
    fj = (np.arange(lado) * w / lado).astype(int).clip(0, w - 1)
    return a[fi][:, fj]


def carga_imagen(ruta, lado=None):
    '''Lee .png/.jpg/.bmp/.tif/.npy y devuelve (1,lado,lado) float32 en [0,1].'''
    lado = lado or CFG["img_size"]
    ruta = Path(ruta)
    if ruta.suffix.lower() == ".npy":
        a = np.load(ruta).astype(np.float32)
        if a.ndim == 3 and a.shape[0] in (1, 3): a = a.transpose(1, 2, 0)
    else:
        try:
            from PIL import Image
            a = np.asarray(Image.open(ruta).convert("L"), dtype=np.float32)
        except ImportError:
            if ruta.suffix.lower() != ".png":
                raise RuntimeError(f"{ruta.suffix} necesita pillow: pip install pillow")
            a = plt.imread(ruta).astype(np.float32)
    if a.ndim == 3: a = a[..., :3].mean(-1)          # a escala de grises
    if a.max() > 1.5: a = a / 255.0                  # normalizar si venía en 0-255
    a = _redimensiona(a, lado)
    return np.clip(a, 0, 1)[None].astype(np.float32)


def carga_nube(ruta):
    '''Lee .ply (ASCII), .npy, .csv o .txt y devuelve un array (N,3) float32.'''
    ruta = Path(ruta)
    s = ruta.suffix.lower()
    if s == ".npy":
        p = np.load(ruta)
    elif s == ".ply":
        lineas = ruta.read_text(encoding="utf-8", errors="ignore").splitlines()
        i = next(k for k, l in enumerate(lineas) if l.strip().lower() == "end_header")
        p = np.array([[float(v) for v in l.split()[:3]] for l in lineas[i + 1:] if l.strip()])
    else:
        txt = [l for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
        if any(c.isalpha() for c in txt[0]): txt = txt[1:]        # descartar cabecera
        sep = "," if "," in txt[0] else None
        p = np.array([[float(v) for v in (l.split(sep) if sep else l.split())[:3]] for l in txt])
    return np.asarray(p, dtype=np.float32).reshape(-1, 3)
