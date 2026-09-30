# Biblioteca de la aplicación: la MISMA arquitectura y el MISMO muestreo del notebook.
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as Fnn
from pathlib import Path

# =============================================================================
# 3 · ARQUITECTURA OFICIAL EN PYTORCH PURO
# =============================================================================
# ---- operaciones geométricas (equivalentes a las extensiones CUDA) ------------
def muestreo_mas_lejano(coords, m):
    '''Muestreo por punto más lejano. coords (B,3,N) -> centros (B,3,M).'''
    B, _, N = coords.shape
    p = coords.transpose(1, 2)                                  # (B,N,3)
    idx = torch.zeros(B, m, dtype=torch.long, device=coords.device)
    dist = torch.full((B, N), 1e10, device=coords.device)
    lejano = torch.zeros(B, dtype=torch.long, device=coords.device)
    lote = torch.arange(B, device=coords.device)
    for i in range(m):
        idx[:, i] = lejano
        centro = p[lote, lejano].unsqueeze(1)                    # (B,1,3)
        d = ((p - centro) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        lejano = dist.argmax(-1)
    return torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3)).transpose(1, 2).contiguous()

def consulta_esfera(centros, puntos, radio, k):
    '''Consulta por esfera. Devuelve (B,M,k) índices: los k primeros puntos dentro del radio.

    Reproduce la semántica de la extensión CUDA: se recorren los puntos en orden y se toman
    los primeros k que caen dentro del radio; si hay menos, se repite el primero encontrado.'''
    B, _, M = centros.shape
    N = puntos.shape[2]
    d2 = torch.cdist(centros.transpose(1, 2), puntos.transpose(1, 2)) ** 2      # (B,M,N)
    dentro = d2 <= radio ** 2
    orden = torch.arange(N, device=puntos.device).view(1, 1, N).expand(B, M, N)
    clave = torch.where(dentro, orden, orden + N)               # los de fuera van al final
    idx = clave.argsort(dim=-1)[:, :, :k]
    hay = dentro.any(-1, keepdim=True)
    primero = idx[:, :, :1]
    valido = torch.gather(dentro, 2, idx)
    idx = torch.where(valido, idx, primero.expand_as(idx))      # relleno con el primero
    return torch.where(hay, idx, torch.zeros_like(idx))

def agrupa(features, idx):
    '''Agrupa rasgos por índices. features (B,C,N), idx (B,M,k) -> (B,C,M,k).'''
    B, C, N = features.shape
    M, k = idx.shape[1], idx.shape[2]
    plano = idx.reshape(B, 1, M * k).expand(B, C, M * k)
    return torch.gather(features, 2, plano).reshape(B, C, M, k)

def voxeliza(features, coords, r, normalizar=True, eps=0.0):
    '''Voxelización por promedio. Devuelve (rejilla (B,C,r,r,r), coordenadas normalizadas).'''
    coords = coords.detach()
    nc = coords - coords.mean(2, keepdim=True)
    if normalizar:
        nc = nc / (nc.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
    else:
        nc = (nc + 1) / 2.0
    nc = torch.clamp(nc * r, 0, r - 1)
    vox = torch.round(nc).long()
    B, C, N = features.shape
    ind = vox[:, 0] * r * r + vox[:, 1] * r + vox[:, 2]          # misma fórmula que el CUDA
    salida = torch.zeros(B, C, r ** 3, device=features.device, dtype=features.dtype)
    cuenta = torch.zeros(B, 1, r ** 3, device=features.device, dtype=features.dtype)
    salida.scatter_add_(2, ind.unsqueeze(1).expand(B, C, N), features)
    cuenta.scatter_add_(2, ind.unsqueeze(1), torch.ones(B, 1, N, device=features.device, dtype=features.dtype))
    salida = salida / cuenta.clamp(min=1.0)
    return salida.view(B, C, r, r, r), nc

def desvoxeliza(rejilla, coords, r):
    '''Devoxelización trilineal: muestrea la rejilla en las coordenadas continuas (B,3,N).'''
    B, C = rejilla.shape[:2]
    c = coords.clamp(0, r - 1)
    g = 2.0 * c / max(r - 1, 1) - 1.0                            # a [-1,1]
    # grid_sample espera el triple (x_w, y_h, z_d) = (z, y, x) para una rejilla (B,C,X,Y,Z)
    malla = torch.stack([g[:, 2], g[:, 1], g[:, 0]], dim=-1).view(B, 1, 1, -1, 3)
    out = Fnn.grid_sample(rejilla, malla, mode="bilinear", align_corners=True, padding_mode="border")
    return out.view(B, C, -1)

def interpola_vecinos(coords_puntos, coords_centros, rasgos_centros):
    '''Interpolación por los tres vecinos más cercanos, ponderada por el inverso de la distancia.'''
    d = torch.cdist(coords_puntos.transpose(1, 2), coords_centros.transpose(1, 2))   # (B,N,M)
    dist, idx = torch.topk(d, k=min(3, d.shape[-1]), dim=-1, largest=False)
    w = 1.0 / dist.clamp(min=1e-10)
    w = w / w.sum(-1, keepdim=True)                                                  # (B,N,3)
    B, C, M = rasgos_centros.shape
    N = idx.shape[1]
    vec = torch.gather(rasgos_centros.unsqueeze(2).expand(B, C, N, M), 3,
                       idx.unsqueeze(1).expand(B, C, N, idx.shape[-1]))
    return (vec * w.unsqueeze(1)).sum(-1)

# ---- módulos con los nombres de parámetros del repositorio oficial -----------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SharedMLP(nn.Module):
    '''MLP compartido: Conv1d/Conv2d + GroupNorm + Swish por cada salida.'''
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        capas = []
        for oc in out_channels:
            capas += [conv(in_channels, oc, 1), nn.GroupNorm(8, oc), Swish()]
            in_channels = oc
        self.layers = nn.Sequential(*capas)

    def forward(self, x):
        return self.layers(x)

class SE3d(nn.Module):
    '''Recalibración de canales (squeeze and excitation) en 3D.'''
    def __init__(self, channel, reduction=8, use_relu=True):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.ReLU(True) if use_relu else Swish(),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(-1).mean(-1).mean(-1)).view(x.shape[0], x.shape[1], 1, 1, 1)

class PVConv(nn.Module):
    '''Convolución punto-vóxel: rama volumétrica (3D) más rama por punto, sumadas.'''
    def __init__(self, in_channels, out_channels, kernel_size=3, resolution=32,
                 dropout=0.1, with_se=True, with_se_relu=True, normalize=True, eps=0):
        super().__init__()
        self.resolution, self.normalize, self.eps = int(resolution), normalize, eps
        capas = [nn.Conv3d(in_channels, out_channels, kernel_size, 1, kernel_size // 2),
                 nn.GroupNorm(8, out_channels), Swish()]
        if dropout is not None:
            capas += [nn.Dropout(dropout)]
        capas += [nn.Conv3d(out_channels, out_channels, kernel_size, 1, kernel_size // 2),
                  nn.GroupNorm(8, out_channels), Swish()]
        if with_se:
            capas.append(SE3d(out_channels, use_relu=with_se_relu))
        self.voxel_layers = nn.Sequential(*capas)
        self.point_features = SharedMLP(in_channels, out_channels, dim=1)

    def forward(self, entrada):
        features, coords = entrada
        rejilla, nc = voxeliza(features, coords, self.resolution, self.normalize, self.eps)
        rejilla = self.voxel_layers(rejilla)
        vox = desvoxeliza(rejilla, nc, self.resolution)
        return vox + self.point_features(features), coords

class PointNetSAModule(nn.Module):
    '''Capa de abstracción: centros por punto más lejano, vecinos por esfera y max-pool.'''
    def __init__(self, num_centers, radius, num_neighbors, in_channels, out_channels,
                 include_coordinates=True):
        super().__init__()
        self.num_centers, self.radius, self.num_neighbors = num_centers, radius, num_neighbors
        self.include_coordinates = include_coordinates
        self.groupers = nn.ModuleList([nn.Identity()])          # marcador de posición sin parámetros
        self.mlps = nn.ModuleList([SharedMLP(in_channels + (3 if include_coordinates else 0),
                                             out_channels, dim=2)])

    def forward(self, entrada):
        features, coords = entrada
        centros = muestreo_mas_lejano(coords, self.num_centers)
        idx = consulta_esfera(centros, coords, self.radius, self.num_neighbors)
        vecinos = agrupa(coords, idx) - centros.unsqueeze(-1)
        rasgos = torch.cat([vecinos, agrupa(features, idx)], 1) if self.include_coordinates \
            else agrupa(features, idx)
        return self.mlps[0](rasgos).max(dim=-1).values, centros

class PointNetFPModule(nn.Module):
    '''Capa de propagación: devuelve los rasgos de los centros a todos los puntos.'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP(in_channels, out_channels, dim=1)

    def forward(self, entrada):
        coords_puntos, coords_centros, rasgos_centros, rasgos_puntos = entrada
        x = interpola_vecinos(coords_puntos, coords_centros, rasgos_centros.float())
        if rasgos_puntos is not None and rasgos_puntos.shape[1] > 0:
            x = torch.cat([x, rasgos_puntos], 1)
        return self.mlp(x), coords_puntos

class PVCNN2SA(nn.Module):
    '''Codificador geométrico: dos PVConv y una capa de abstracción con 512 centros.'''
    def __init__(self, dim=768, dropout=0.1, resolucion=32, centros=512, radio=0.1, vecinos=32):
        super().__init__()
        self.sa_layers = nn.ModuleList([nn.Sequential(
            PVConv(3, 32, 3, resolucion, dropout=dropout),
            PVConv(32, 32, 3, resolucion, dropout=dropout),
            PointNetSAModule(centros, radio, vecinos, 32, [128, dim]))])
        self.sa_in_channels = [3]
        self.channels_sa_features = dim

    def forward(self, pts):
        entradas = pts.transpose(-1, -2)                        # (B,3,N)
        coords = entradas[:, :3, :].contiguous()
        features = entradas
        lista_rasgos, lista_coords = [], []
        for capa in self.sa_layers:
            lista_rasgos.append(features)
            lista_coords.append(coords)
            features, coords = capa((features, coords))
        lista_rasgos[0] = entradas[:, 3:, :].contiguous()       # vacío: no hay rasgos extra
        return lista_rasgos, lista_coords, features, coords

class PVCNN2FP(nn.Module):
    '''Decodificador geométrico: interpolación, dos PVConv y MLP de salida a 3 canales.'''
    def __init__(self, num_classes=3, channels_sa_features=768, dropout=0.1, resolucion=32):
        super().__init__()
        self.fp_layers = nn.ModuleList([nn.Sequential(
            PointNetFPModule(channels_sa_features, [128]),
            PVConv(128, 32, 3, resolucion, dropout=dropout),
            PVConv(32, 32, 3, resolucion, dropout=dropout))])
        self.classifier = nn.Sequential(SharedMLP(32, 128, dim=1), nn.Dropout(dropout),
                                        nn.Conv1d(128, num_classes, 1))

    def forward(self, lista_rasgos, lista_coords, features, coords):
        for i, bloques in enumerate(self.fp_layers):
            features, coords_pts = bloques[0]((lista_coords[-1 - i], coords, features,
                                               lista_rasgos[-1 - i]))
            for pv in bloques[1:]:
                features, coords_pts = pv((features, coords_pts))
            coords = coords_pts
        return self.classifier(features).transpose(1, 2)

# ---- Transformer del artículo -------------------------------------------------
def codifica_paso(embed_dim, timesteps, device):
    '''Codificación sinusoidal del paso de difusión, idéntica a la del repositorio original.'''
    half = embed_dim // 2
    emb = np.log(10000) / (half - 1)
    emb = torch.from_numpy(np.exp(np.arange(0, half) * -emb)).float().to(device)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:
        emb = Fnn.pad(emb, (0, 1), "constant", 0)
    return emb

class Attention(nn.Module):
    '''Atención multi-cabeza con una sola matriz qkv, como en el original.'''
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.ultimo_mapa = None                                  # para el análisis de atención

    def forward(self, x, guardar=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if guardar:
            a = (q @ k.transpose(-2, -1)) * self.scale
            a = a.softmax(-1)
            self.ultimo_mapa = a.detach()
            x = (a @ v).transpose(1, 2).reshape(B, N, C)
        else:
            x = Fnn.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class TransformerBlock(nn.Module):
    '''Bloque del artículo: auto-atención sobre [c, x], atención cruzada sobre [x, c] y MLP.'''
    def __init__(self, dim=768, mlp_dim=3072, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attention = Attention(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop_rate),
                                 nn.Linear(mlp_dim, dim), nn.Dropout(drop_rate))

    def forward(self, x, c, guardar=False):
        x = x + self.attn(self.norm1(torch.cat((c, x), dim=1)))[:, c.shape[1]:, :]
        x = x + self.cross_attention(self.norm2(torch.cat((x, c), dim=1)), guardar)[:, :x.shape[1], :]
        return x + self.mlp(self.norm3(x))

class TransformerEncoder(nn.Module):
    def __init__(self, dim=768, mlp_dim=3072, drop_rate=0.0, depth=4):
        super().__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(dim, mlp_dim, drop_rate) for _ in range(depth)])

    def forward(self, x, c, guardar=False):
        todos = []
        for b in self.blocks:
            x = b(x, c, guardar)
            todos.append(x)
        return x, todos

class TrDPNetOficial(nn.Module):
    '''Red de predicción de ruido del artículo, con los nombres de parámetros originales.'''
    def __init__(self, dim=768, depth=4, mlp_dim=3072, drop_rate=0.1, n_centros=512,
                 radio=0.1, vecinos=32, resolucion=32):
        super().__init__()
        self.dim, self.depth = dim, depth
        self.pointnet_sa = PVCNN2SA(dim=dim, dropout=drop_rate, resolucion=resolucion,
                                    centros=n_centros, radio=radio, vecinos=vecinos)
        self.pointnet_fp = PVCNN2FP(3, dim, dropout=drop_rate, resolucion=resolucion)
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, dim))
        self.encoder = TransformerEncoder(dim=dim, depth=depth, mlp_dim=mlp_dim, drop_rate=drop_rate)

    def forward(self, pts, image_feature, time_step, guardar_atencion=False):
        lista_rasgos, lista_coords, tokens, centros = self.pointnet_sa(pts)
        tokens = tokens.transpose(-1, -2)                        # (B,M,D)
        centros = centros.transpose(-1, -2)                      # (B,M,3)
        t_emb = codifica_paso(self.dim, time_step, pts.device).unsqueeze(1)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(tokens.size(0), -1, -1)
        x = torch.cat((cls, tokens), dim=1)
        pos = torch.cat((cls_pos, self.pos_embed(centros)), dim=1)
        condicion = torch.cat((t_emb, image_feature), dim=1)
        x, _ = self.encoder(x + pos, condicion, guardar_atencion)
        return self.pointnet_fp(lista_rasgos, lista_coords,
                                x[:, x.shape[1] - tokens.size(1):, :].transpose(-1, -2),
                                centros.transpose(-1, -2))

# ---- codificador de imagen (ViT-B/16 congelado) -------------------------------
class BloqueViT(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)      # timm usa eps=1e-6, no el 1e-5 por defecto
        self.attn = Attention(dim, num_heads=heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        oculto = int(dim * mlp_ratio)
        self.mlp = nn.Sequential()
        self.mlp.fc1 = nn.Linear(dim, oculto)
        self.mlp.act = nn.GELU()
        self.mlp.fc2 = nn.Linear(oculto, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp.fc2(self.mlp.act(self.mlp.fc1(self.norm2(x))))

class ViT(nn.Module):
    '''Vision Transformer con los nombres de parámetros de timm, para cargar los pesos MAE.'''
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        n = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.blocks = nn.ModuleList([BloqueViT(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + self.pos_embed
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

class ModeloRasgos(nn.Module):
    '''Envoltorio del ViT congelado, con la normalización de ImageNet del original.'''
    MEDIA = (0.485, 0.456, 0.406)
    DESV = (0.229, 0.224, 0.225)

    def __init__(self, **kw):
        super().__init__()
        self.model = ViT(**kw)

    def forward(self, img):
        media = torch.tensor(self.MEDIA, device=img.device).view(1, 3, 1, 1)
        desv = torch.tensor(self.DESV, device=img.device).view(1, 3, 1, 1)
        return self.model((img - media) / desv)

# ---- difusión y muestreo ------------------------------------------------------
class DifusionOficial:
    '''Programa DDPM del artículo (1000 pasos, betas lineales de 8.5e-4 a 1.2e-2).'''
    def __init__(self, T=1000, beta_start=0.00085, beta_end=0.012, device="cpu"):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        self.betas = betas.float().to(device)
        self.alphas = (1.0 - self.betas)
        self.ab = torch.cumprod(self.alphas, 0)

    def q_sample(self, x0, t, ruido):
        ab = self.ab[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * ruido

@torch.no_grad()
def muestrea_oficial(modelo, rasgos_img, n_puntos, dif, w=3.0, pasos=None, escala=7.0,
                     semilla=None, callback=None, device="cpu", traza=None, cada=None):
    '''Algoritmo 2 del artículo: muestreo inverso con guía sin clasificador.

    rasgos_img: tokens de la imagen (B, I, D). Si pasos < T se recorre el programa a saltos
    con la actualización determinista de DDIM, que es la forma estándar de acelerar.'''
    if semilla is not None:
        torch.manual_seed(semilla)
    B = rasgos_img.shape[0]
    x = torch.randn(B, n_puntos, 3, device=device)
    tiempos = list(range(dif.T))
    if pasos is not None and pasos < dif.T:
        tiempos = sorted(set(np.linspace(0, dif.T - 1, pasos).round().astype(int).tolist()))
    c = torch.cat([rasgos_img, torch.zeros_like(rasgos_img)], 0)
    for j, i in enumerate(reversed(tiempos)):
        t = torch.full((B * 2,), i, device=device, dtype=torch.long)
        e = modelo(x.repeat(2, 1, 1), c, t)
        e_c, e_u = e.chunk(2)
        eps = e_u + w * (e_c - e_u)
        ab_t = dif.ab[i]
        prev = tiempos[len(tiempos) - 2 - j] if j + 2 <= len(tiempos) else None
        ab_prev = dif.ab[prev] if prev is not None else torch.tensor(1.0, device=device)
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps      # actualización determinista
        if traza is not None and (j % (cada or max(1, len(tiempos) // 12)) == 0 or j == len(tiempos) - 1):
            traza.append((i, (x / escala).detach().cpu().clone()))
        if callback is not None and j % 10 == 0:
            callback(j / len(tiempos))
    return x / escala

# El propio código de esta celda se guarda para reutilizarlo en la aplicación web y
# para poder mostrarlo más adelante sin depender de un archivo en disco.

def fuente_de(nombre):
    '''Extrae el código de una clase o función definida en la celda anterior.'''
    import re
    m = re.search(rf"^(class|def) {nombre}\b.*?(?=\n(?:class |def |# =)|\Z)",
                  CODIGO_ARQUITECTURA, re.S | re.M)
    return m.group(0).rstrip() if m else f"[código de {nombre} no disponible]"