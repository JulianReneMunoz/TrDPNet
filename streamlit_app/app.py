# Interfaz web de TrDPNet · ejecutar:  streamlit run streamlit_app/app.py
from pathlib import Path
import numpy as np, torch, streamlit as st
import trdpnet_lib as T
st.set_page_config(page_title="TrDPNet · imagen a 3D", page_icon="🧊", layout="wide")
RAIZ = Path(__file__).resolve().parent.parent
PESOS = next((p for p in [RAIZ / "weights" / "trdpnet_shapenet.pt",
                          RAIZ / "weights" / "trdpnet_replica.pt"] if p.exists()),
             RAIZ / "weights" / "trdpnet_shapenet.pt")

@st.cache_resource
def cargar():
    ck = torch.load(PESOS, map_location=T.DEVICE, weights_only=False)
    T.CFG = ck["cfg"]; m = T.TrDPNet(ck["cfg"]).to(T.DEVICE); m.load_state_dict(ck["state_dict"]); m.eval()
    c = ck["cfg"]; return m, T.Difusion(c["T"], c["beta_0"], c["beta_T"], c["warmup"], T.DEVICE), c

st.title("🧊 TrDPNet · de una imagen a una nube de puntos 3D")
st.caption("Transformer encoder–decoder + difusión · Li et al., J. Vis. Commun. Image R. 111 (2025)")
if not PESOS.exists():
    st.error(f"No se encontró {PESOS}. Ejecutar antes el notebook."); st.stop()
modelo, dif, cfg = cargar()
with st.sidebar:
    st.header("Parámetros")
    omega = st.slider("Escala de guía ω", 1.0, 5.0, float(cfg["omega"]), .5)
    semilla = st.number_input("Semilla", 0, 9999, 7)
    st.info("Imagen: .png .jpg .jpeg .bmp .tif .npy\nReferencia (opcional): .ply .npy .csv .txt")
c1, c2 = st.columns([1, 1.6])
with c1:
    st.subheader("1 · Imagen de entrada")
    ejemplos = sorted((RAIZ / "ejemplos").glob("*.png"))
    fuente = st.radio("Origen", ["Ejemplo del proyecto", "Subir archivo"], horizontal=True)
    ref = None
    if fuente == "Ejemplo del proyecto" and ejemplos:
        e = st.selectbox("Ejemplo", ejemplos, format_func=lambda p: p.stem)
        img = T.carga_imagen(e, cfg["img_size"])
        if e.with_suffix(".ply").exists(): ref = T.carga_nube(e.with_suffix(".ply"))
    else:
        up = st.file_uploader("Imagen", type=["png", "jpg", "jpeg", "bmp", "tif", "npy"])
        upr = st.file_uploader("Nube de referencia (opcional)", type=["ply", "npy", "csv", "txt"])
        if up is None: st.stop()
        tmp = RAIZ / "outputs" / ("_subida" + Path(up.name).suffix); tmp.write_bytes(up.getvalue())
        img = T.carga_imagen(tmp, cfg["img_size"])
        if upr is not None:
            tr = RAIZ / "outputs" / ("_ref" + Path(upr.name).suffix); tr.write_bytes(upr.getvalue()); ref = T.carga_nube(tr)
    st.image((img[0] * 255).astype(np.uint8), caption="lo que ve el modelo", width=260)
with c2:
    st.subheader("2 · Reconstrucción 3D")
    if st.button("Reconstruir", type="primary"):
        with st.spinner(f"Muestreando {dif.T} pasos…"):
            p, _ = T.muestrea(modelo, torch.from_numpy(img[None]).to(T.DEVICE), cfg["n_puntos"], dif, omega=omega, semilla=int(semilla))
        p = p.cpu()[0].numpy()
        import plotly.graph_objects as go
        fg = go.Figure(go.Scatter3d(x=p[:, 0], y=p[:, 2], z=p[:, 1], mode="markers", marker=dict(size=2, color=p[:, 1], colorscale="Viridis")))
        fg.update_layout(height=520, scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fg, use_container_width=True)
        if ref is not None:
            f, pr, rc = T.f_score(torch.from_numpy(p)[None], torch.from_numpy(ref)[None], cfg["tau"])
            a, b, c = st.columns(3); a.metric("F-score", f"{f.item():.3f}"); b.metric("Precisión", f"{pr.item():.3f}"); c.metric("Recall", f"{rc.item():.3f}")
        st.download_button("Descargar nube (.csv)", "x,y,z\n" + "\n".join(f"{x:.5f},{y:.5f},{z:.5f}" for x, y, z in p), "nube.csv")
