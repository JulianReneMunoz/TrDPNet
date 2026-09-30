
# Interfaz web de inferencia de TrDPNet con los pesos oficiales.
import sys, pickle, time, numpy as np, torch, streamlit as st
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trdpnet_lib import TrDPNetOficial, ModeloRasgos, DifusionOficial, muestrea_oficial

RAIZ = Path(__file__).resolve().parent.parent
CANDIDATOS = [RAIZ / "weights" / "TrDPNet_shapenet.pth", RAIZ / "TrDPNet_shapenet.pth"]
CFG = {'dim': 768, 'depth': 4, 'mlp_dim': 3072, 'n_centros': 512, 'radio': 0.1, 'vecinos': 32, 'resolucion': 32, 'img_size': 224, 'vit_dim': 768, 'vit_depth': 12, 'vit_heads': 12, 'T': 1000, 'beta_0': 0.00085, 'beta_T': 0.012, 'escala': 7.0, 'omega': 3.0, 'pasos_muestreo': 250}

st.set_page_config(page_title="TrDPNet · pesos oficiales", layout="wide")
st.title("TrDPNet · de una imagen a una nube de puntos 3D")
st.caption("Inferencia con los pesos publicados por los autores del artículo.")

class _Fantasma:
    # sustituye a las clases del repositorio original que aquí no existen
    def __init__(self, *a, **k): pass
    def __setstate__(self, estado): self.__dict__.update(estado if isinstance(estado, dict) else {})

class _Desempaquetador(pickle.Unpickler):
    def find_class(self, modulo, nombre):
        try:
            return super().find_class(modulo, nombre)
        except Exception:
            return _Fantasma

class _pickle_tolerante:
    Unpickler = _Desempaquetador
    @staticmethod
    def load(f, **kw): return _Desempaquetador(f, **kw).load()

def _es_tensor(v): return hasattr(v, "shape") and hasattr(v, "numel")

def _busca_sd(obj, prof=0):
    # busca el diccionario con más tensores dentro del objeto cargado
    if prof > 4 or _es_tensor(obj): return None
    mejor, cands = None, ([obj] if isinstance(obj, dict) else [])
    if hasattr(obj, "__dict__") and not isinstance(obj, dict): cands.append(vars(obj))
    for d in cands:
        propios = {k: v for k, v in d.items() if isinstance(k, str) and _es_tensor(v)}
        if len(propios) >= 8 and (mejor is None or len(propios) > len(mejor)): mejor = propios
        for v in d.values():
            if _es_tensor(v): continue
            sub = _busca_sd(v, prof + 1)
            if sub and (mejor is None or len(sub) > len(mejor)): mejor = sub
    return mejor

@st.cache_resource
def carga():
    ruta = next((p for p in CANDIDATOS if p.exists()), None)
    if ruta is None:
        return None, None, None
    ck = torch.load(ruta, map_location="cpu", weights_only=False, pickle_module=_pickle_tolerante)
    sd = _busca_sd(ck)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = TrDPNetOficial(dim=CFG["dim"], depth=CFG["depth"], mlp_dim=CFG["mlp_dim"],
                       n_centros=CFG["n_centros"], radio=CFG["radio"], vecinos=CFG["vecinos"],
                       resolucion=CFG["resolucion"])
    pref = "point_cloud_model."
    m.load_state_dict({k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)} or
                      {k: v for k, v in sd.items()
                       if k.startswith(("pointnet_", "encoder.", "pos_embed", "cls_"))}, strict=True)
    v = ModeloRasgos(img_size=CFG["img_size"], embed_dim=CFG["vit_dim"], depth=CFG["vit_depth"],
                     num_heads=CFG["vit_heads"])
    pv = "feature_model.model."
    v.model.load_state_dict({k[len(pv):]: val for k, val in sd.items() if k.startswith(pv)}, strict=False)
    return m.to(dev).eval(), v.to(dev).eval(), DifusionOficial(CFG["T"], CFG["beta_0"], CFG["beta_T"], dev)

modelo, vit, dif = carga()
if modelo is None:
    st.error("No se encontró TrDPNet_shapenet.pth en la carpeta weights/.")
    st.stop()
dev = next(modelo.parameters()).device

col1, col2 = st.columns([1, 1.4])
with col1:
    archivo = st.file_uploader("Imagen del objeto", type=["png", "jpg", "jpeg"])
    ejemplos = sorted((RAIZ / "ejemplos").glob("*.png"))
    elegido = st.selectbox("…o un ejemplo", ["(ninguno)"] + [e.name for e in ejemplos])
    omega = st.slider("Escala de guía ω", 1.0, 5.0, float(CFG["omega"]), .5)
    pasos = st.select_slider("Pasos de muestreo", [50, 100, 250, 500, 1000], value=CFG["pasos_muestreo"])
    puntos = st.select_slider("Puntos", [1024, 2048, 4096, 8192], value=2048)
    lanzar = st.button("Reconstruir", type="primary")

ruta_img = None
if archivo is not None:
    ruta_img = archivo
elif elegido != "(ninguno)":
    ruta_img = RAIZ / "ejemplos" / elegido
if ruta_img is not None:
    im = Image.open(ruta_img).convert("RGB")
    col1.image(im, caption="entrada", width=220)

if lanzar and ruta_img is not None:
    x = np.asarray(im.resize((CFG["img_size"], CFG["img_size"])), np.float32) / 255.0
    t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(dev)
    barra = col2.progress(0.0, text="muestreando…")
    with torch.no_grad():
        pred = muestrea_oficial(modelo, vit(t), puntos, dif, w=omega, pasos=pasos,
                                escala=CFG["escala"], device=dev,
                                callback=lambda p: barra.progress(min(p, 1.0), text=f"muestreando… {100*p:.0f} %"))
    barra.empty()
    p = pred[0].cpu().numpy()
    p = (p - p.mean(0)) / (np.abs(p).max() + 1e-8) * 0.5
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter3d(x=p[:, 0], y=p[:, 2], z=p[:, 1], mode="markers",
                                     marker=dict(size=1.6, color=p[:, 1], colorscale="Viridis")))
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                          scene=dict(aspectmode="data"))
        col2.plotly_chart(fig, use_container_width=True)
    except Exception:
        col2.write(p[:10])
    # --- se guardan las dos versiones en disco y se ofrecen para descarga -------
    obj = ("# nube de puntos generada por TrDPNet\n"
           + "\n".join(f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in p) + "\n")
    ply = ("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
           "property float z\nend_header\n" % len(p)
           + "\n".join(f"{a:.6f} {b:.6f} {c:.6f}" for a, b, c in p) + "\n")
    carpeta = RAIZ / "outputs" / "nubes"
    carpeta.mkdir(parents=True, exist_ok=True)
    marca = time.strftime("%Y%m%d_%H%M%S")
    base_nombre = Path(getattr(ruta_img, "name", str(ruta_img))).stem
    ruta_obj = carpeta / f"{base_nombre}_{marca}.obj"
    ruta_ply = carpeta / f"{base_nombre}_{marca}.ply"
    ruta_obj.write_text(obj, encoding="utf-8")
    ruta_ply.write_text(ply, encoding="utf-8")
    col2.success(f"Guardado en disco: `{ruta_obj.relative_to(RAIZ)}` y `{ruta_ply.relative_to(RAIZ)}`")
    d1, d2 = col2.columns(2)
    d1.download_button("Descargar .obj", data=obj, file_name=ruta_obj.name, mime="text/plain")
    d2.download_button("Descargar .ply", data=ply, file_name=ruta_ply.name, mime="text/plain")
