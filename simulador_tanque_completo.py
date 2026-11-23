"""SIMULADOR UNIVERSITARIO DE EDO: VACIADO GRAVITACIONAL DE UN TANQUE.

Universidad Técnica de Ambato | Ingeniería Mecánica | Análisis Matemático II.

Este programa es INDEPENDIENTE: no necesita importar otros .py del proyecto.

Instalación (una sola vez):
    python -m pip install streamlit numpy pandas scipy plotly

Ejecución desde la carpeta donde guardaste este archivo:
    python -m streamlit run simulador_tanque_completo.py

Modelo: A dh/dt = - C_d a sqrt(2 g h) ; h(0) = h_0.
La descarga se interrumpe al llegar a h=0. Salida libre a la atmósfera,
fluido incompresible, tanque de sección constante y Cd constante.

Las cuatro situaciones son ejercicios DIDÁCTICOS PROPUESTOS, no están
especificadas individualmente por la hoja original del docente.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
import streamlit as st

# ---------------------------------------------------------------------------
# CONSTANTES Y ENUNCIADOS PREDEFINIDOS
# ---------------------------------------------------------------------------
GRAVEDAD = 9.81  # m/s^2
AZUL = "#60C6FA"
VERDE = "#35D6A2"
AMARILLO = "#F4BB63"
ROJO = "#FF7B85"
VIOLETA = "#C19BFF"
FONDO = "#0E1522"
PANEL = "#172234"
BLANCO = "#EFF5FF"

EJERCICIOS = {
    "1 · Tanque de referencia": {
        "numero": 1, "R": 0.40, "d_mm": 25.0, "h0": 1.20, "Cd": 0.62,
        "titulo": "Vaciado de un tanque cilíndrico de referencia",
        "enunciado": (
            "Un tanque cilíndrico industrial tiene radio R = 0,40 m, diámetro de "
            "salida d = 25 mm y altura inicial h₀ = 1,20 m. Si Cᵈ = 0,62, "
            "halle su ecuación diferencial, la función h(t) y el tiempo total de vaciado."
        ),
        "interpretacion": (
            "Este caso es la referencia del proyecto. La altura no decrece a velocidad "
            "constante: la salida pierde presión hidrostática a medida que baja el nivel."
        ),
    },
    "2 · Orificio más grande": {
        "numero": 2, "R": 0.40, "d_mm": 35.0, "h0": 1.20, "Cd": 0.62,
        "titulo": "Influencia del diámetro del orificio de descarga",
        "enunciado": (
            "Se utiliza el mismo tanque del ejercicio 1, pero se cambia el diámetro "
            "de salida de 25 mm a 35 mm. Encuentre h(t), el tiempo de vaciado y "
            "el porcentaje de reducción respecto al tanque inicial."
        ),
        "interpretacion": (
            "El área del orificio aumenta con el cuadrado de su diámetro. Con el "
            "mismo Cᵈ, un orificio más grande permite un mayor caudal y reduce el tiempo."
        ),
    },
    "3 · Tanque más ancho": {
        "numero": 3, "R": 0.55, "d_mm": 25.0, "h0": 1.20, "Cd": 0.62,
        "titulo": "Influencia del radio del depósito",
        "enunciado": (
            "Se reemplaza el tanque original por otro de radio R = 0,55 m; "
            "el diámetro del orificio sigue siendo d = 25 mm y la altura h₀ = 1,20 m. "
            "Calcule h(t), el tiempo de vaciado y el cambio porcentual."
        ),
        "interpretacion": (
            "Al aumentar el área transversal, existe más volumen de agua por cada metro "
            "de altura. Si el orificio es igual, la altura disminuye más despacio."
        ),
    },
    "4 · Mayor altura inicial": {
        "numero": 4, "R": 0.40, "d_mm": 25.0, "h0": 2.00, "Cd": 0.62,
        "titulo": "Influencia de la altura inicial de llenado",
        "enunciado": (
            "Se mantiene el tanque inicial (R = 0,40 m y d = 25 mm), pero se llena "
            "hasta h₀ = 2,00 m. Determine h(t), el tiempo de vaciado y la "
            "diferencia porcentual con el ejercicio 1."
        ),
        "interpretacion": (
            "Una columna inicial más alta aporta más volumen y más presión al comienzo. "
            "El tiempo total crece en proporción a la raíz cuadrada de h₀."
        ),
    },
}

# ---------------------------------------------------------------------------
# MODELO MATEMÁTICO. TODAS LAS FUNCIONES NUMÉRICAS SON INDEPENDIENTES DE LA UI.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tanque:
    radio: float              # m
    diametro: float           # m
    h0: float                 # m
    cd: float = 0.62          # sin unidades
    g: float = GRAVEDAD       # m/s^2

    def __post_init__(self):
        if self.radio <= 0 or self.diametro <= 0 or self.h0 <= 0 or self.g <= 0:
            raise ValueError("Radio, diámetro de salida, altura inicial y gravedad deben ser positivos.")
        if self.diametro >= 2 * self.radio:
            raise ValueError("El diámetro de salida debe ser menor que el diámetro del tanque.")
        if not 0 < self.cd <= 1:
            raise ValueError("El coeficiente de descarga debe cumplir 0 < Cᵈ ≤ 1.")

    @property
    def A(self):
        """Área transversal del depósito, en m²."""
        return math.pi * self.radio**2

    @property
    def a(self):
        """Área geométrica del orificio, en m²."""
        return math.pi * (self.diametro / 2)**2

    @property
    def K(self):
        """Constante de la EDO, unidades raíz(m)/s."""
        return (self.cd * self.a / self.A) * math.sqrt(2 * self.g)

    @property
    def T(self):
        """Tiempo total de vaciado predicho por el modelo (s)."""
        return 2 * math.sqrt(self.h0) / self.K

    @property
    def V0(self):
        return self.A * self.h0

    @property
    def Q0(self):
        return self.cd * self.a * math.sqrt(2 * self.g * self.h0)

    def h(self, tiempos):
        """Solución analítica, solo t >= 0. Tras T, el nivel queda en cero."""
        t = np.asarray(tiempos, dtype=float)
        if np.any(t < 0):
            raise ValueError("El tiempo no puede ser negativo.")
        salida = np.maximum(math.sqrt(self.h0) - self.K * t / 2, 0.0)**2
        return float(salida) if salida.ndim == 0 else salida

    def Q(self, tiempos):
        """Caudal volumétrico efectivo, m³/s."""
        return self.cd * self.a * np.sqrt(2 * self.g * self.h(tiempos))

    def v_ideal(self, tiempos):
        """Velocidad IDEAL de Torricelli, m/s (no Cd por la velocidad)."""
        return np.sqrt(2 * self.g * self.h(tiempos))

    def V(self, tiempos):
        return self.A * self.h(tiempos)

    def dhdt(self, tiempos):
        """Tasa de variación del nivel; se fija en 0 después del vaciado."""
        return -self.K * np.sqrt(self.h(tiempos))

    def fraccion_vaciada(self, tiempos):
        return 1.0 - self.h(tiempos) / self.h0

    def tiempo_para_fraccion_altura(self, fraccion):
        """Instante de alcanzar un porcentaje de la ALTURA inicial."""
        if not 0 <= fraccion <= 1:
            raise ValueError("Fracción fuera de [0, 1].")
        return self.T * (1.0 - math.sqrt(fraccion))


def serie_datos(p: Tanque, n=750):
    t = np.linspace(0.0, p.T, int(n))
    h = p.h(t)
    return pd.DataFrame({
        "Tiempo (s)": t,
        "Tiempo (min)": t / 60,
        "Altura analítica h (m)": h,
        "Volumen restante (L)": 1000 * p.A * h,
        "Caudal Q (L/s)": 1000 * p.Q(t),
        "Velocidad ideal (m/s)": p.v_ideal(t),
        "Tasa dh/dt (m/s)": p.dhdt(t),
        "Porcentaje vaciado (%)": 100 * p.fraccion_vaciada(t),
    })


def euler_explicito(p: Tanque, pasos=4000):
    """Euler sobre la EDO original, con h>=0 como restricción física."""
    n = int(pasos)
    if n < 100:
        raise ValueError("Se requieren por lo menos 100 pasos para Euler.")
    ts = np.linspace(0.0, p.T, n + 1)
    hs = np.zeros(n + 1)
    hs[0] = p.h0
    dt = p.T / n
    for i in range(n):
        hs[i + 1] = max(0.0, hs[i] - dt * p.K * math.sqrt(hs[i]))
    return ts, hs


def rk45_sobre_edo(p: Tanque, n=650):
    """Integra dh/dt=-K sqrt(h) por RK45 hasta el 98% de T.

    Se evita exactamente h=0 porque la raíz cuadrada deja de ser Lipschitz
    allí y la dinámica física impone el estado seco h=0. Este recorte está
    documentado para que los errores que se muestren no sean engañosos.
    """
    t_fin = 0.98 * p.T
    ts = np.linspace(0.0, t_fin, n)

    def derivada(_t, y):
        return [-p.K * math.sqrt(max(0.0, y[0]))]

    sol = solve_ivp(derivada, (0.0, t_fin), (p.h0,), method="RK45",
                    t_eval=ts, rtol=1e-10, atol=1e-12,
                    max_step=t_fin / 350)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, np.maximum(sol.y[0], 0.0)


def parametros_caso(datos):
    return Tanque(datos["R"], datos["d_mm"] / 1000,
                  datos["h0"], datos["Cd"])


def tabla_comparacion():
    b = parametros_caso(next(iter(EJERCICIOS.values())))
    rows = []
    for e in EJERCICIOS.values():
        p = parametros_caso(e)
        rows.append({
            "Ejercicio": f"{e['numero']}. {e['titulo']}",
            "Radio R (m)": p.radio,
            "Diámetro d (mm)": 1000 * p.diametro,
            "Altura inicial (m)": p.h0,
            "Volumen inicial (L)": 1000 * p.V0,
            "Caudal inicial (L/s)": 1000 * p.Q0,
            "Tiempo (s)": p.T,
            "Tiempo (min)": p.T / 60,
            "Variación vs. ejercicio 1 (%)": (p.T / b.T - 1) * 100,
        })
    return pd.DataFrame(rows)


def interpretar(numero, modificado=False):
    frases = {
        1: "Los datos iniciales determinan el nivel, el caudal de salida y el tiempo de vaciado; el caudal disminuye durante el proceso.",
        2: "El diámetro pasa de 25 mm a 35 mm; el área del orificio aumenta casi al doble. Por ello el tiempo de vaciado se reduce.",
        3: "El radio del tanque aumenta de 0,40 m a 0,55 m. La sección transversal crece y el tanque contiene más volumen a la misma altura.",
        4: "La altura aumenta de 1,20 m a 2,00 m, y el tiempo aumenta según la raíz cuadrada de la altura inicial.",
    }
    if modificado:
        return ("Has editado los parámetros. La conclusión correspondiente al enunciado "
                "original ya no es necesariamente válida; interpreta los resultados "
                "mostrados con tus nuevos valores.")
    return frases[numero]


def texto_desarrollo(p: Tanque, e: dict, alterado: bool):
    raiz = math.sqrt(p.h0)
    mitad = p.K / 2
    base = parametros_caso(EJERCICIOS["1 · Tanque de referencia"])
    dif = 100 * (p.T / base.T - 1)
    lines = [
        "UNIVERSIDAD TÉCNICA DE AMBATO – ANÁLISIS MATEMÁTICO II",
        "Proyecto: modelado y simulación del vaciado gravitacional de un tanque",
        f"EJERCICIO {e['numero']} – {e['titulo']}",
        "",
        "ENUNCIADO PROPUESTO: " + e["enunciado"],
        "OBSERVACIÓN: los valores han sido modificados." if alterado else "DATOS COINCIDENTES CON EL EJERCICIO PROPUESTO.",
        "",
        "DATOS",
        f"Radio R = {p.radio:.6f} m; orificio d = {1000*p.diametro:.3f} mm",
        f"Altura inicial h0 = {p.h0:.6f} m; Cd = {p.cd:.4f}; g = {p.g:.2f} m/s²",
        "",
        "1. HIPÓTESIS Y ECUACIÓN DE CONTINUIDAD",
        "Tanque cilíndrico de sección constante, agua incompresible, orificio pequeño,",
        "descarga libre a presión atmosférica y coeficiente de descarga constante.",
        "V = A*h; dV/dt = A*dh/dt = -Q",
        "Ley de Torricelli con coeficiente efectivo: Q = Cd*a*sqrt(2*g*h)",
        "EDO: dh/dt = -(Cd*a/A)*sqrt(2*g*h) = -K*sqrt(h)",
        "",
        "2. CÁLCULO DE ÁREAS Y CONSTANTE",
        f"A = pi*R² = pi*({p.radio:.6f})² = {p.A:.10f} m²",
        f"a = pi*(d/2)² = pi*({p.diametro:.6f}/2)² = {p.a:.12f} m²",
        f"K = (Cd*a/A)*sqrt(2*g) = {p.K:.12f} sqrt(m)/s",
        f"EDO particular: dh/dt = -{p.K:.12f}*sqrt(h)",
        "",
        "3. SEPARACIÓN DE VARIABLES",
        "dh/sqrt(h) = -K*dt",
        "integral(h^(-1/2) dh) = integral(-K dt)",
        "2*sqrt(h) = -K*t + C",
        f"Condición inicial: h(0)={p.h0:.6f} --> C=2*sqrt(h0)={2*raiz:.10f}",
        "",
        "4. SOLUCIÓN PARTICULAR",
        f"h(t) = ({raiz:.10f} - {mitad:.12f}*t)²   para 0 <= t <= T",
        "h(t) = 0 para t >= T.",
        "AVISO: no prolongar el cuadrado más allá del instante de vaciado.",
        "",
        "5. TIEMPO DE VACIADO Y RESULTADOS",
        f"T = 2*sqrt(h0)/K = {p.T:.6f} s = {p.T/60:.6f} min",
        f"V0 = A*h0 = {p.V0:.9f} m³ = {1000*p.V0:.4f} L",
        f"Q0 = Cd*a*sqrt(2*g*h0) = {p.Q0:.9f} m³/s = {1000*p.Q0:.6f} L/s",
        f"Tiempo para llegar a h0/2 = {p.tiempo_para_fraccion_altura(.5):.6f} s",
        f"Cambio del tiempo frente a referencia = {dif:+.3f} %",
        "",
        "6. COMPROBACIÓN DE LA EDO",
        "h'(t) = -K*sqrt(h(t)) mientras h>0.",
        "Balance de volumen: A*h'(t) + Q(t) = 0.",
        "",
        "7. INTERPRETACIÓN PARA DEFENSA",
        interpretar(e["numero"], alterado),
        "",
        "LÍMITES DEL MODELO",
        "No considera variación de Cd, pérdidas adicionales en tuberías, bombas,",
        "presión adicional, geometría variable ni efectos de superficie cerca de la salida.",
        "La velocidad graficada corresponde a la ideal v=sqrt(2*g*h),",
        "mientras el caudal efectivo usa Cd: Q=Cd*a*v_ideal.",
    ]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# VISUALIZACIÓN EN 2D Y 3D (PLOTLY)
# ---------------------------------------------------------------------------
def configurar_fig(fig, alto=440, titulo=None):
    fig.update_layout(
        template="plotly_dark", height=alto, title=titulo,
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=BLANCO),
        margin=dict(l=30, r=25, t=55 if titulo else 25, b=28),
    )
    return fig


def dibujo_fisico_2d(p: Tanque, t: float):
    """Corte didáctico a escala proporcional del nivel; paredes esquemáticas."""
    h = float(p.h(t))
    porcentaje = h / p.h0
    H = 3.0
    x0, x1 = 0.65, 2.85
    y0 = 0.33
    nivel = y0 + H * porcentaje
    fig = go.Figure()
    # Agua transparente, nivel variable
    if porcentaje > 0:
        fig.add_shape(type="rect", x0=x0+.05, x1=x1-.05,
                      y0=y0+.04, y1=max(y0+.045, nivel),
                      fillcolor="rgba(42,167,222,0.60)", line=dict(width=0))
        fig.add_trace(go.Scatter(x=[x0+.05, x1-.05], y=[nivel, nivel],
                                 mode="lines", line=dict(color=AZUL, width=4),
                                 name="Superficie del agua"))
    # Paredes y fondo; se deja abierto en la parte de arriba
    fig.add_trace(go.Scatter(x=[x0, x0, x1, x1],
                             y=[y0+H, y0, y0, y0+H], mode="lines",
                             line=dict(color="#EAF1FA", width=6),
                             name="Paredes del depósito"))
    # Tubería corta de salida dibujada en el fondo lateral
    fig.add_trace(go.Scatter(x=[x1, x1+.55], y=[y0+.12, y0+.12],
                             mode="lines", line=dict(color="#C2CFDD", width=13),
                             name="Salida"))
    # Chorro: representación cualitativa para Q>0, sin predecir trayectoria real
    if h > 1e-10:
        rr = np.linspace(0, 0.72, 40)
        fig.add_trace(go.Scatter(x=x1+.55+rr, y=y0+.12-.2*rr**2,
                                 mode="lines", line=dict(color=AZUL, width=5),
                                 name="Chorro (ilustrativo)"))
    fig.add_annotation(x=(x0+x1)/2, y=y0+H+.28,
                       text="TANQUE CILÍNDRICO", showarrow=False,
                       font=dict(size=16, color=BLANCO))
    fig.add_annotation(x=x0-.22, y=max(y0+.14, nivel),
                       text=f"h = {h:.4f} m", showarrow=False,
                       textangle=-90, font=dict(color=AZUL, size=13))
    fig.add_annotation(x=x1+.72, y=y0+.5,
                       text=f"Q = {1000*float(p.Q(t)):.4f} L/s", showarrow=False,
                       font=dict(color=VERDE, size=13))
    fig.add_annotation(x=1.75, y=y0-0.2,
                       text=f"R={p.radio:g} m  |  d={p.diametro*1000:g} mm  |  {100*porcentaje:.1f}% de altura",
                       showarrow=False, font=dict(color=BLANCO, size=12))
    fig.update_xaxes(range=[-.1, 4.9], visible=False)
    fig.update_yaxes(range=[-.2, 4.05], visible=False, scaleanchor="x", scaleratio=1)
    configurar_fig(fig, alto=440, titulo="Corte lateral (esquema didáctico)")
    fig.update_layout(showlegend=False)
    return fig


def dibujo_fisico_3d(p: Tanque, t: float):
    """Tanque cilíndrico representado en 3D; agua a la altura h(t)."""
    h = float(p.h(t))
    theta = np.linspace(0, 2*math.pi, 90)
    H = p.h0 * 1.11
    R = p.radio
    fig = go.Figure()

    # Estructura transparente, con circunferencias de base y borde superior
    for z, nombre in [(0., "Base del tanque"), (H, "Borde superior")]:
        fig.add_trace(go.Scatter3d(x=R*np.cos(theta), y=R*np.sin(theta),
                                   z=np.full_like(theta, z), mode="lines",
                                   line=dict(color="#D8E8FA", width=5),
                                   name=nombre, showlegend=False))
    for ang in np.linspace(0, 2*math.pi, 13)[:-1]:
        fig.add_trace(go.Scatter3d(x=[R*math.cos(ang)]*2,
                                   y=[R*math.sin(ang)]*2, z=[0,H],
                                   mode="lines", line=dict(color="#C4DAEA", width=3),
                                   showlegend=False))
    if h > 1e-10:
        # Superficie lateral azul con semitransparencia
        zz = np.array([0.0, h])
        X = np.tile((R*.985*np.cos(theta)), (2,1))
        Y = np.tile((R*.985*np.sin(theta)), (2,1))
        Z = np.tile(zz.reshape(-1,1), (1, len(theta)))
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=.55,
                                 colorscale=[[0,"#1693CF"],[1,"#3DB8F2"]],
                                 showscale=False, name="Agua", hoverinfo="skip"))
        # Superficie superior del agua, disco
        rr = np.linspace(0, R*.985, 18)
        XX = rr[:,None] * np.cos(theta)[None,:]
        YY = rr[:,None] * np.sin(theta)[None,:]
        ZZ = np.full_like(XX, h)
        fig.add_trace(go.Surface(x=XX, y=YY, z=ZZ, opacity=.85,
                                 colorscale=[[0,"#63CDFA"],[1,"#63CDFA"]],
                                 showscale=False, name="Superficie de agua", hoverinfo="skip"))

    # Salida lateral a la altura del fondo; esquema, no un CAD dimensionado
    fig.add_trace(go.Scatter3d(x=[R, R + 0.6*R], y=[0,0], z=[.055*H]*2,
                               mode="lines", line=dict(color=AMARILLO, width=12),
                               name="Tubo de salida"))
    if h > 1e-10:
        dx = np.linspace(0, .65*R, 35)
        fig.add_trace(go.Scatter3d(x=R+.6*R+dx, y=np.zeros_like(dx),
                                   z=np.maximum(.055*H-.15*H*(dx/(.65*R))**2, -.1*H),
                                   mode="lines", line=dict(color=AZUL, width=7),
                                   name="Chorro ilustrativo"))
    fig.update_layout(
        template="plotly_dark", height=510, paper_bgcolor=PANEL,
        margin=dict(l=0,r=0,t=50,b=0),
        title=f"Vista 3D · h(t) = {h:.4f} m · t = {t:.2f} s",
        scene=dict(
            xaxis=dict(title="X (m)", showbackground=False, gridcolor="#2C3F52"),
            yaxis=dict(title="Y (m)", showbackground=False, gridcolor="#2C3F52"),
            zaxis=dict(title="Altura (m)", range=[0, max(H*1.25, .1)],
                       showbackground=False, gridcolor="#2C3F52"),
            bgcolor=PANEL,
            camera=dict(eye=dict(x=1.55,y=1.65,z=1.15)),
            aspectmode="manual", aspectratio=dict(x=1.4, y=1.4, z=1.55),
        ),
        showlegend=False,
    )
    return fig


def grafico_serie(p: Tanque, t_sel: float):
    df = serie_datos(p)
    tx = df["Tiempo (min)"].to_numpy()
    marcas = [
        ("Altura analítica h (m)", "Altura del agua (m)", AZUL),
        ("Caudal Q (L/s)", "Caudal efectivo (L/s)", VERDE),
        ("Volumen restante (L)", "Volumen restante (L)", AMARILLO),
        ("Velocidad ideal (m/s)", "Velocidad ideal de Torricelli (m/s)", VIOLETA),
    ]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[d[1] for d in marcas],
                        vertical_spacing=.17, horizontal_spacing=.13)
    for n, (col, _, color) in enumerate(marcas):
        r, c = n//2+1, n%2+1
        fig.add_trace(go.Scatter(x=tx, y=df[col], name=col, mode="lines",
                                 line=dict(color=color,width=3), showlegend=False), r, c)
        fig.add_vline(x=t_sel/60, line_dash="dash", line_color=ROJO, row=r, col=c)
        fig.update_xaxes(title_text="Tiempo (min)", row=r,col=c)
        fig.update_yaxes(title_text=col, rangemode="tozero", row=r,col=c)
    configurar_fig(fig, alto=765, titulo="Evolución del vaciado")
    return fig


def grafico_comparativo():
    fig = go.Figure()
    colores = [AZUL, VERDE, AMARILLO, VIOLETA]
    for e, color in zip(EJERCICIOS.values(), colores):
        p = parametros_caso(e)
        tt = np.linspace(0, p.T, 500)
        fig.add_trace(go.Scatter(x=tt/60, y=p.h(tt), mode="lines",
                                 line=dict(color=color, width=3),
                                 name=f"Ej. {e['numero']} · T={p.T/60:.2f} min"))
    configurar_fig(fig, 460, "Comparación de la altura h(t) en los cuatro ejercicios originales")
    fig.update_xaxes(title_text="Tiempo (min)")
    fig.update_yaxes(title_text="Altura h (m)", rangemode="tozero")
    fig.update_layout(legend=dict(orientation="h",y=-.24))
    return fig


def grafico_validacion(p: Tanque, numero_pasos):
    te, he = euler_explicito(p, numero_pasos)
    tr, hr = rk45_sobre_edo(p)
    t0 = np.linspace(0, p.T, 750)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t0/60, y=p.h(t0), name="Solución analítica",
                             line=dict(color=AZUL,width=3)))
    fig.add_trace(go.Scatter(x=te[::max(1,len(te)//500)]/60,
                             y=he[::max(1,len(he)//500)], name="Euler explícito",
                             line=dict(color=AMARILLO, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=tr/60, y=hr, name="RK45 hasta el 98% de T",
                             line=dict(color=VERDE,width=2,dash="dot")))
    configurar_fig(fig, 480, "Verificación: solución exacta frente a Euler y RK45")
    fig.update_xaxes(title_text="Tiempo (min)")
    fig.update_yaxes(title_text="Altura h (m)", rangemode="tozero")
    fig.update_layout(legend=dict(orientation="h",y=-.24))
    return fig, (te, he, tr, hr)

# ---------------------------------------------------------------------------
# INTERFAZ STREAMLIT
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Simulador EDO | Vaciado de tanque",
                   page_icon="💧", layout="wide")
st.markdown("""
<style>
  .stApp {background:#0E1522;color:#EFF5FF}
  section[data-testid="stSidebar"]{background:#172234}
  .block-container {max-width:1430px;padding-top:1.35rem}
  div[data-testid="stMetric"] {background:#172234;padding:.7rem 1rem;
        border:1px solid #304157;border-radius:12px}
  .hero-panel {padding:1.5rem 1.7rem;background:linear-gradient(105deg,#16304B,#165776);
        border-radius:16px;border:1px solid #295C7B;margin-bottom:1.2rem}
  .hero-panel h1{color:white;font-size:2.18rem;margin:0 0 .35rem}
  .hero-panel p{color:#D8F0FF;margin:0}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-panel">
  <h1>💧 Laboratorio virtual · Vaciado gravitacional de un tanque</h1>
  <p>Ingeniería Mecánica · Análisis Matemático II · EDO no lineal de primer orden · Python</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuración del sistema")
seleccion = st.sidebar.selectbox("Ejercicio asignado", list(EJERCICIOS.keys()))
preset = EJERCICIOS[seleccion]
num = preset["numero"]
st.sidebar.caption(preset["titulo"])
st.sidebar.markdown("### Propiedades geométricas")


def entrada(clave, titulo, minimo, maximo, paso, formato):
    return st.sidebar.number_input(titulo, min_value=minimo, max_value=maximo,
                                   value=float(preset[clave]), step=paso,
                                   format=formato, key=f"ej{num}_{clave}")

R = entrada("R", "Radio del tanque R (m)", .05, 5.0, .05, "%.3f")
d_mm = entrada("d_mm", "Diámetro del orificio d (mm)", 1., 200., 1., "%.1f")
h0 = entrada("h0", "Altura inicial h₀ (m)", .05, 10., .05, "%.3f")
st.sidebar.markdown("### Propiedades de descarga")
Cd = entrada("Cd", "Coeficiente de descarga Cᵈ", .10, 1.0, .01, "%.3f")
st.sidebar.caption("g = 9,81 m/s², agua incompresible y descarga a la atmósfera.")

try:
    p = Tanque(R, d_mm/1000, h0, Cd)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

alterado = any(not math.isclose(float(valor), float(preset[clave]),
                                abs_tol=1e-10,rel_tol=0)
               for clave,valor in [("R",R),("d_mm",d_mm),("h0",h0),("Cd",Cd)])

st.subheader(f"Ejercicio {num}. {preset['titulo']}")
st.info(preset["enunciado"])
if alterado:
    st.warning("Estás explorando un escenario modificado. Los cálculos se actualizan "
               "correctamente, pero los valores ya no corresponden al enunciado original.")

v1,v2,v3,v4,v5 = st.columns(5)
v1.metric("Tiempo de vaciado", f"{p.T/60:.2f} min")
v2.metric("Volumen inicial", f"{1000*p.V0:.2f} L")
v3.metric("Caudal inicial", f"{1000*p.Q0:.3f} L/s")
v4.metric("Área del tanque", f"{p.A:.4f} m²")
v5.metric("Constante K", f"{p.K:.7f} √m/s")

st.divider()
modelo_tab, desarrollo_tab, visor_tab, graficas_tab, comparacion_tab, numerico_tab, exposicion_tab = st.tabs([
    "📐 Modelo físico", "🧮 Resolución paso a paso", "🧊 Vista 2D y 3D",
    "📈 Gráficas", "📊 Comparación", "✅ Validación y datos", "🎓 Guía de exposición",
])

with modelo_tab:
    st.subheader("Descripción física y origen de la ecuación diferencial")
    c1, c2 = st.columns([1,1])
    with c1:
        st.plotly_chart(dibujo_fisico_2d(p, 0.20*p.T), use_container_width=True)
    with c2:
        st.markdown("""
**Sistema:** tanque cilíndrico vertical, con agua descargando por un orificio pequeño
cercano al fondo y expuesto a la atmósfera.

**Hipótesis adoptadas:** sección transversal uniforme, fluido incompresible,
coeficiente de descarga constante, área de salida pequeña frente al área del
depósito, sin bomba ni alimentación, y superficie libre a presión atmosférica.

**Variable dependiente:** altura de agua **h(t)** [m].

**Variable independiente:** tiempo **t** [s].

El dibujo representa un **volumen de control**, no un diagrama de cuerpo libre
como el del amortiguador: aquí empleamos continuidad y la ley de Torricelli.
        """)
    st.markdown("#### 1. Geometría del sistema")
    st.latex(r"A=\pi R^2,\qquad a=\pi\left(\frac{d}{2}\right)^2,\qquad V(t)=Ah(t)")
    st.markdown("#### 2. Caudal de salida")
    st.latex(r"v_{\mathrm{ideal}}=\sqrt{2gh},\qquad Q(t)=C_d\,a\sqrt{2gh(t)}")
    st.caption("v_ideal es la velocidad ideal de Torricelli; el coeficiente de descarga "
               "corrige el caudal efectivo. No debe interpretarse Cd·v_ideal como una "
               "medición directa de la velocidad local del chorro.")
    st.markdown("#### 3. Conservación del volumen")
    st.latex(r"\frac{dV}{dt}=-Q,\qquad A\frac{dh}{dt}=-C_d a\sqrt{2gh}")
    st.latex(r"\boxed{\frac{dh}{dt}=-\frac{C_da}{A}\sqrt{2gh}=-K\sqrt h}")
    st.latex(rf"K=\frac{{C_da}}{{A}}\sqrt{{2g}}={p.K:.11f}\ \mathrm{{\sqrt{{m}}/s}}")
    st.warning("El signo menos indica descenso del nivel. Este modelo predice "
               "h=0 cuando el tanque se vacía; después el agua permanece en cero.")

with desarrollo_tab:
    st.subheader("Resolución de la ecuación diferencial por separación de variables")
    st.markdown("**Paso 1. Identificar los datos del ejercicio.**")
    datos = pd.DataFrame({
        "Parámetro": ["Radio del tanque (m)", "Diámetro de salida (mm)",
                      "Altura inicial (m)", "Coeficiente de descarga", "Gravedad (m/s²)"],
        "Valor": [p.radio,1000*p.diametro,p.h0,p.cd,p.g]
    })
    st.dataframe(datos, hide_index=True, use_container_width=True)
    st.markdown("**Paso 2. Calcular el área del tanque y del orificio.**")
    st.latex(rf"A=\pi({p.radio:g})^2={p.A:.9f}\ \mathrm{{m^2}}")
    st.latex(rf"a=\pi\left(\frac{{{p.diametro:g}}}{{2}}\right)^2={p.a:.12f}\ \mathrm{{m^2}}")
    st.markdown("**Paso 3. Sustituir en la EDO original.**")
    st.latex(rf"\frac{{dh}}{{dt}}=-\frac{{({p.cd:g})({p.a:.10f})}}{{{p.A:.9f}}}\sqrt{{2({p.g:g})h}}")
    st.latex(rf"\boxed{{\frac{{dh}}{{dt}}=-{p.K:.10f}\sqrt{{h}}}}")
    st.markdown("**Paso 4. Separar las variables.**")
    st.latex(r"\frac{dh}{\sqrt h}=-K\,dt \quad (h>0)")
    st.markdown("**Paso 5. Integrar ambos miembros.**")
    st.latex(r"\int h^{-1/2}\,dh=-K\int dt")
    st.latex(r"2\sqrt h=-Kt+C")
    st.markdown("**Paso 6. Aplicar la condición inicial h(0)=h₀.**")
    st.latex(rf"C=2\sqrt{{h_0}}=2\sqrt{{{p.h0:g}}}={2*math.sqrt(p.h0):.9f}")
    st.markdown("**Paso 7. Despejar h(t).**")
    st.latex(r"\sqrt h=\sqrt{h_0}-\frac K2t")
    st.latex(r"\boxed{h(t)=\left(\sqrt{h_0}-\frac K2t\right)^2\quad\text{para }0\le t\le T}")
    st.latex(rf"\boxed{{h(t)=\left({math.sqrt(p.h0):.9f}-{p.K/2:.11f}t\right)^2}}")
    st.markdown("**Paso 8. Determinar el tiempo total de vaciado.**")
    st.latex(r"0=\sqrt{h_0}-\frac K2T\ \Longrightarrow\ \boxed{T=\frac{2\sqrt{h_0}}K}")
    st.latex(rf"T=\frac{{2\sqrt{{{p.h0:g}}}}}{{{p.K:.10f}}}={p.T:.6f}\ \mathrm{{s}}={p.T/60:.6f}\ \mathrm{{min}}")
    st.success(f"Resultado: T = {p.T:.2f} s = {p.T/60:.2f} minutos.")
    st.caption("Dominio físico: después de t=T se utiliza h(t)=0. Si se "
               "prolongara el cuadrado más allá de T, aparecería un aumento "
               "matemático de altura sin justificación física.")
    st.markdown("**Paso 9. Verificación analítica y dimensional.**")
    st.latex(r"h'(t)=-K\left(\sqrt{h_0}-\frac{Kt}{2}\right)=-K\sqrt{h(t)}\quad(0\le t\le T)")
    st.latex(r"A\,h'(t)+Q(t)=0")
    st.markdown(f"La constante K tiene unidades de **√m/s**, de modo que "
                f"K√h tiene unidades de **m/s**, iguales a dh/dt.")

with visor_tab:
    st.subheader("Simulación visual: avance del tiempo")
    st.markdown("Mueve el control para observar el agua en **dos vistas sincronizadas**. "
                "Puedes rotar el modelo 3D con el mouse.")
    avance = st.slider("Porcentaje del proceso transcurrido", min_value=0, max_value=100,
                       value=25, step=1, key=f"tiempo_e{num}")
    momento = p.T*avance/100
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tiempo actual",f"{momento:.2f} s")
    c2.metric("Altura actual",f"{float(p.h(momento)):.5f} m")
    c3.metric("Caudal efectivo",f"{1000*float(p.Q(momento)):.5f} L/s")
    c4.metric("Volumen restante",f"{1000*float(p.V(momento)):.3f} L")
    v2,v3 = st.columns(2)
    with v2:
        st.plotly_chart(dibujo_fisico_2d(p, momento), use_container_width=True)
    with v3:
        st.plotly_chart(dibujo_fisico_3d(p, momento), use_container_width=True)
    st.caption("Las formas son ilustrativas: la geometría no es un plano CAD ni "
               "una predicción detallada de la trayectoria del chorro. El nivel sí "
               "se calcula con h(t) para los parámetros escogidos.")

with graficas_tab:
    st.subheader("Gráficas del comportamiento físico")
    avance_g = st.slider("Línea de tiempo para las gráficas (%)", 0, 100, 25,
                         key=f"graf_t_e{num}")
    st.plotly_chart(grafico_serie(p,p.T*avance_g/100), use_container_width=True)
    st.markdown("#### Interpretación física")
    st.markdown(
        "La **altura** y el **volumen restante** disminuyen durante el vaciado. "
        "La velocidad ideal y el caudal dependen de √h y también disminuyen, "
        "por lo que no se debe asumir un caudal constante. El descenso se "
        "vuelve más lento cerca del final dentro de las hipótesis del modelo."
    )
    st.markdown("#### Datos adicionales")
    df = serie_datos(p)
    tab_resumen = pd.DataFrame({
        "Instante": ["Inicio", "25% del tiempo", "50% del tiempo", "75% del tiempo", "Fin"],
        "Tiempo (s)": [p.T*f for f in [0,.25,.5,.75,1.]],
        "Altura (m)": [float(p.h(p.T*f)) for f in [0,.25,.5,.75,1.]],
        "Caudal (L/s)": [1000*float(p.Q(p.T*f)) for f in [0,.25,.5,.75,1.]],
        "Volumen (L)": [1000*float(p.V(p.T*f)) for f in [0,.25,.5,.75,1.]],
    })
    st.dataframe(tab_resumen.style.format(precision=5),use_container_width=True,hide_index=True)
    st.download_button("⬇️ Descargar serie completa en CSV",
                       df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"tanque_ejercicio_{num}_datos.csv",mime="text/csv")

with comparacion_tab:
    st.subheader("Comparación de los cuatro ejercicios originales")
    st.plotly_chart(grafico_comparativo(), use_container_width=True)
    tbl = tabla_comparacion()
    st.dataframe(tbl.style.format(precision=3),use_container_width=True,hide_index=True)
    st.download_button("⬇️ Descargar comparación en CSV",
                       tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="comparacion_4_ejercicios.csv",mime="text/csv")
    st.markdown("### ¿Qué demuestra cada cambio?")
    st.latex(r"T=\frac{2A\sqrt{h_0}}{C_d\,a\sqrt{2g}}\ \propto\ \frac{R^2\sqrt{h_0}}{d^2}")
    st.markdown("**Caso 2:** un diámetro mayor reduce el tiempo de vaciado (relación inversa con d²).")
    st.markdown("**Caso 3:** un radio mayor eleva el tiempo de vaciado (relación directa con R²).")
    st.markdown("**Caso 4:** una altura mayor aumenta el tiempo de vaciado (relación con √h₀).")
    st.info("La tabla compara los cuatro ejercicios **originales**. Si cambiaste los "
            "valores en el panel lateral, el escenario personalizado se visualiza en "
            "las demás pestañas, sin alterar la comparación de referencia.")

with numerico_tab:
    st.subheader("Validación matemática de la solución analítica")
    st.markdown("El método de **Euler explícito** aproxima la EDO mediante pasos de tiempo. "
                "**RK45** es un integrador adaptativo de SciPy utilizado como segunda comprobación.")
    st.latex(r"h_{n+1}=\max\left(0,\ h_n-\Delta t\,K\sqrt{h_n}\right)")
    pasos = st.slider("Número de pasos de Euler", 500, 10000, 4000,
                      step=500, key=f"pasos_e{num}")
    figura, (t_e,h_e,t_r,h_r) = grafico_validacion(p,pasos)
    exact_e = p.h(t_e)
    exact_r = p.h(t_r)
    error_e = np.max(np.abs(h_e-exact_e))
    error_r = np.max(np.abs(h_r-exact_r))
    dc1,dc2,dc3 = st.columns(3)
    dc1.metric("Error máximo Euler", f"{error_e:.3e} m")
    dc2.metric("Error máximo RK45", f"{error_r:.3e} m")
    dc3.metric("Paso de Euler", f"{p.T/pasos:.4f} s")
    st.plotly_chart(figura, use_container_width=True)
    st.caption("RK45 integra la ecuación original únicamente hasta el 98% del "
               "tiempo de vaciado: cerca de h=0 la raíz cuadrada requiere un "
               "tratamiento numérico cuidadoso. El tramo final se obtiene "
               "de la solución analítica; Euler sí se extiende hasta T.")
    st.markdown("#### Verificación del balance de caudal")
    muestra = p.T * .35
    residuo = p.A * float(p.dhdt(muestra)) + float(p.Q(muestra))
    st.latex(r"\underbrace{A\frac{dh}{dt}}_{\text{variación del volumen}}+\underbrace{Q}_{\text{caudal de salida}}=0")
    st.success(f"En t = {muestra:.2f} s, el residuo del balance es {residuo:+.3e} m³/s (≈ 0).")
    t_data = np.linspace(0,p.T,700)
    tabla = pd.DataFrame({
        "t (s)":t_data,
        "h analítica (m)":p.h(t_data),
        "h Euler (m)":np.interp(t_data,t_e,h_e),
        "error abs Euler (m)":np.abs(p.h(t_data)-np.interp(t_data,t_e,h_e)),
        "Caudal Q (L/s)":1000*p.Q(t_data),
        "Volumen (L)":1000*p.V(t_data),
    })
    st.dataframe(tabla.style.format(precision=8),use_container_width=True,height=340)
    st.download_button("⬇️ Descargar validación numérica CSV",
                       tabla.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"tanque_ejercicio_{num}_validacion.csv",mime="text/csv")

with exposicion_tab:
    st.subheader(f"Guía para defender el ejercicio {num} ante el ingeniero")
    st.markdown("**Guion sugerido para explicar el modelo:**")
    st.markdown("""
1. **Presentación:** «Modelamos cómo se vacía un tanque cilíndrico por gravedad».
2. **Identificación:** explica R, d, h₀, Cᵈ, gravedad y qué representa h(t).
3. **Modelo físico:** muestra que V=Ah, por tanto A·h'=−Q.
4. **Ley física:** explica Q=Cᵈ·a·√(2gh) y el signo negativo de la EDO.
5. **Método de solución:** separa variables, integra y aplica h(0)=h₀.
6. **Resultado:** calcula la función h(t) y el tiempo T con h(T)=0.
7. **Simulación:** mueve el control del tiempo, enseña la vista 2D y 3D.
8. **Gráficas:** interpreta por qué la altura, el caudal y el volumen decrecen.
9. **Validación:** compara solución analítica con Euler y RK45; explica el error.
10. **Conclusión:** compara los cuatro casos y menciona las limitaciones físicas.
    """)
    st.markdown("**Explicación específica de este ejercicio:**")
    st.info(interpretar(num, alterado))
    st.markdown("**Preguntas que podría hacerte el ingeniero:**")
    with st.expander("¿Por qué es una ecuación diferencial no lineal?"):
        st.write("Porque aparece √h y la derivada depende de una función no lineal de h.")
    with st.expander("¿Qué significa el signo negativo?"):
        st.write("Indica que el nivel disminuye a medida que transcurre el tiempo.")
    with st.expander("¿Por qué se aplica separación de variables?"):
        st.write("Porque podemos escribir h^(-1/2) dh = −K dt e integrar ambos lados.")
    with st.expander("¿Por qué el tanque no se vacía a caudal constante?"):
        st.write("Porque Q = Cᵈ a √(2gh): al bajar h, disminuye el caudal.")
    with st.expander("¿Por qué no continúan la fórmula cuadrática después del vaciado?"):
        st.write("La expresión se deduce mientras hay agua. Para t≥T imponemos h=0; "
                 "extender el cuadrado daría un aumento de altura ficticio.")
    with st.expander("¿Qué representa Cᵈ = 0,62?"):
        st.write("Un coeficiente de descarga adoptado como supuesto ilustrativo, "
                 "que aproxima el caudal efectivo frente al ideal. No se midió en un tanque real.")
    with st.expander("¿Cómo se comprueba que el resultado está correcto?"):
        st.write("Sustituyendo h(t) y su derivada en la EDO, revisando h(0)=h₀ "
                 "y h(T)=0, y comparando con los métodos Euler y RK45.")
    with st.expander("¿En qué condiciones podría fallar este modelo?"):
        st.write("Si el área cambia con la altura, hay una bomba, el tanque está "
                 "presurizado, el orificio es grande o Cᵈ no es aproximadamente constante.")
    st.markdown("#### Desarrollo listo para guardar o estudiar")
    guia_txt = texto_desarrollo(p,preset,alterado)
    st.code(guia_txt,language=None)
    st.download_button("⬇️ Descargar desarrollo del ejercicio en TXT",
                       guia_txt.encode("utf-8"),
                       file_name=f"desarrollo_ejercicio_{num}_tanque.txt",
                       mime="text/plain")
    st.caption("Alcance: modelo idealizado con geometría cilíndrica y coeficiente "
               "de descarga supuesto constante. Las vistas 2D/3D son didácticas.")

st.divider()
st.caption("UTA · Proyecto académico de Análisis Matemático II · "
           "Ecuación diferencial de primer orden · "
           "Cuatro ejercicios de diseño propuestos para la demostración.")
