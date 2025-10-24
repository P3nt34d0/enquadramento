# app.py
import io
import zipfile
import xml.etree.ElementTree as ET
import datetime as dt
from datetime import timedelta
from io import BytesIO

import polars as pl
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import hashlib
import tempfile
from pathlib import Path

@st.cache_data(show_spinner=False)
def _file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

st.set_page_config(page_title="Projeção - FIDC", layout="wide")
st.title("Simulação de Projeção de Caixa/DC - FIDC's")

# =========================
# Sidebar — parâmetros
# =========================
with st.sidebar:
    st.header("Parâmetros")
    limite = st.number_input(
        "Limite mínimo de DC/PL (ex.: 0.67 = 67%)",
        min_value=0.0, max_value=1.0, value=0.67, step=0.01, format="%.2f"
    )
    horizonte_meses = st.slider("Horizonte da simulação (meses à frente)", 1, 18, 12)

    st.subheader("Fundo")
    fundo = st.selectbox(
        "Selecione o Fundo",
        ["FIDC Franquias", "FIDC Clubcard"],
    )

    st.divider()
    st.subheader("Uploads")
    estoque_zip = st.file_uploader("Estoque (ZIP com 1 CSV)", type=["zip"])
    carteira_xml = st.file_uploader("Carteira (ANBIMA XML v4.*)", type=["xml"])

# =========================
# Buckets por FAIXA de dias (percentuais distribuídos igualmente por dia)
# Última faixa (≥181): tudo no D+181 (um único dia), conforme solicitado
# =========================

RANGED_BUCKETS_CLUBCARD = [
    (  4,  15, 0.0430),
    ( 16,  30, 0.2573),
    ( 31,  45, 0.0636),
    ( 46,  60, 0.1956),
    ( 61,  75, 0.0383),
    ( 76,  90, 0.1142),
    ( 91, 105, 0.0199),
    (106, 120, 0.0724),
    (121, 135, 0.0126),
    (136, 150, 0.0491),
    (151, 165, 0.0097),
    (166, 180, 0.0310),
    (181, None, 0.0933),  # ≥181: tudo no D+181
]

RANGED_BUCKETS_FRANQUIAS = [
    (  4,  15, 0.0136),
    ( 16,  30, 0.1141),
    ( 31,  45, 0.1224),
    ( 46,  60, 0.1389),
    ( 61,  75, 0.0954),
    ( 76,  90, 0.1142),
    ( 91, 105, 0.0745),
    (106, 120, 0.0884),
    (121, 135, 0.0530),
    (136, 150, 0.0602),
    (151, 165, 0.0391),
    (166, 180, 0.0399),
    (181, None, 0.0463),  # ≥181: tudo no D+181
]

def pick_range_buckets(nome_fundo: str):
    return RANGED_BUCKETS_CLUBCARD if nome_fundo == "FIDC Clubcard" else RANGED_BUCKETS_FRANQUIAS

# =========================
# Pesos de aquisição por dia da semana (somam ~1.0 na semana útil)
# 0=Seg, 1=Ter, 2=Qua, 3=Qui, 4=Sex
# =========================

WEEK_WEIGHTS_CLUBCARD = {
    0: 0.4311,  # Seg
    1: 0.1394,  # Ter
    2: 0.1467,  # Qua
    3: 0.1338,  # Qui
    4: 0.1490,  # Sex
}

WEEK_WEIGHTS_FRANQUIAS = {
    0: 0.4190,  # Seg
    1: 0.1506,  # Ter
    2: 0.1534,  # Qua
    3: 0.1435,  # Qui
    4: 0.1335,  # Sex
}

def pick_week_weights(nome_fundo: str) -> dict[int, float]:
    return WEEK_WEIGHTS_CLUBCARD if nome_fundo == "FIDC Clubcard" else WEEK_WEIGHTS_FRANQUIAS

# =========================
# Utilitários
# =========================
SOBERANO_ISINS = {"BRUTOPCTF005", "BRITSBCTF001"}
PL_DAILY_GROWTH = 0.000547507101747335  # 0,0547507101747335% ao dia

def distribute_monthly_acq_by_weekday(mes_primeiro_dia: dt.date,
                                      valor_mensal: float,
                                      week_weights: dict[int, float]) -> dict[dt.date, float]:
    """
    Divide o valor mensal de aquisições entre os dias úteis do mês,
    ponderando pelos pesos do dia da semana:
      - Soma os pesos apenas nos dias úteis que de fato existem no mês.
      - Calcula valor_dia = valor_mensal * peso_do_dia / soma_pesos_no_mes.
      - Garante soma exata (ajusta o residual no último dia).
      - Se por acaso cair num dia não útil, joga para o próximo dia útil.
    Retorna: {data: valor_nesse_dia}
    """
    if valor_mensal <= 0:
        return {}

    bdays = business_days_in_month(mes_primeiro_dia.replace(day=1))
    if not bdays:
        return {}

    # pesos apenas para os dias úteis do mês
    day_weights = [week_weights.get(d.weekday(), 0.0) for d in bdays]
    total_w = sum(day_weights)
    if total_w <= 0:
        # fallback: divide igual
        per_day = valor_mensal / len(bdays)
        return {d: per_day for d in bdays}

    # valores proporcionais (mantém soma com ajuste no último dia)
    valores = [valor_mensal * w / total_w for w in day_weights]
    # Ajuste de arredondamento (centavos, se quiser)
    # Aqui deixo sem arredondar para preservar a precisão.

    # Monta o mapa, empurrando para próximo dia útil se necessário
    out: dict[dt.date, float] = {}
    for d, v in zip(bdays, valores):
        dd = d if is_business_day_calendar(d) else next_business_day_calendar(d)
        out[dd] = out.get(dd, 0.0) + float(v)

    # Garante soma exata (ajuste no último dia útil do mês)
    soma = sum(out.values())
    resid = valor_mensal - soma
    if abs(resid) > 1e-6:
        last_day = bdays[-1]
        out[last_day] = out.get(last_day, 0.0) + resid

    return out

def add_to_map(m: dict, key: dt.date, value: float):
    m[key] = m.get(key, 0.0) + float(value or 0.0)

def distribute_acq_by_ranges(d_acq: dt.date, aq_diaria: float, faixas: list[tuple[int, int | None, float]], add_fn):
    """
    Para cada faixa:
      - se max_d is None (última faixa ≥181): lança TODO o % em D+181 (ajustado para dia útil)
      - caso contrário: distribui igualmente por dia em [min_d, max_d], cada data ajustada para próximo dia útil
    add_fn(date, valor) deve acumular no mapa de liquidações
    """
    if aq_diaria <= 0:
        return
    for min_d, max_d, pct in faixas:
        if pct <= 0:
            continue
        if max_d is None or (min_d >= 181):
            # bucket final: tudo em D+181 (um dia)
            target = next_business_day_calendar(d_acq + dt.timedelta(days=181))
            add_fn(target, aq_diaria * pct)
        else:
            ndays = int(max_d - min_d + 1)
            if ndays <= 0:
                continue
            per_day = (aq_diaria * pct) / ndays
            # distribui 1..N dentro da faixa
            for off in range(min_d, max_d + 1):
                target = next_business_day_calendar(d_acq + dt.timedelta(days=off))
                add_fn(target, per_day)

# ======= FERIADOS (carrega uma vez no início do app) =======
def load_holidays_from_xlsx(path: str = "./feriados_nacionais.xlsx") -> set[dt.date]:
    """Carrega o calendário nacional de feriados de um arquivo Excel"""
    try:
        dfh = pd.read_excel(path)
        col = dfh.columns[0]
        dates = (
            pd.to_datetime(dfh[col], errors="coerce")
            .dt.date.dropna()
            .tolist()
        )
        return set(dates)
    except Exception:
        return set()

HOLIDAYS = load_holidays_from_xlsx()

# ======= Funções baseadas no calendário =======
def is_business_day_calendar(d: dt.date) -> bool:
    """Retorna True se for dia útil (Seg–Sex e não feriado)"""
    return (d.weekday() < 5) and (d not in HOLIDAYS)

def next_business_day_calendar(d: dt.date) -> dt.date:
    """Avança até encontrar o próximo dia útil (considerando feriados)"""
    while not is_business_day_calendar(d):
        d += timedelta(days=1)
    return d

def business_days_in_month(mes_inicio: dt.date) -> list[dt.date]:
    """Retorna todos os dias úteis do mês (considerando feriados)"""
    primeiro_dia = mes_inicio.replace(day=1)
    if mes_inicio.month == 12:
        proximo_mes = dt.date(mes_inicio.year + 1, 1, 1)
    else:
        proximo_mes = dt.date(mes_inicio.year, mes_inicio.month + 1, 1)
    dias = pd.date_range(primeiro_dia, proximo_mes - timedelta(days=1), freq="D")
    dias = [d.date() for d in dias if is_business_day_calendar(d.date())]
    return dias

def build_liq_bucket_df(agenda_df: pd.DataFrame, dia_zero: dt.date | None) -> pd.DataFrame:
    """
    Monta buckets de prazo (0-15,16-30,...,181+) em R$,
    olhando o estoque (agenda_df: Data, Valor).

    dia_zero = data que vamos considerar como D0.
       Normalmente: dt_ref (data-base do XML).
       Se vier None, cai pro menor Data do estoque.
    """
    if agenda_df is None or agenda_df.empty:
        return pd.DataFrame(columns=["Faixa", "ValorTotal"])

    df = agenda_df.copy()
    df["Data"] = pd.to_datetime(df["Data"]).dt.date

    if dia_zero is None:
        dia_zero = min(df["Data"])

    # diferença em dias corridos
    df["dias_para_liquidar"] = df["Data"].apply(lambda d: (d - dia_zero).days)

    # só olhamos fluxos futuros ou no D0
    df = df[df["dias_para_liquidar"] >= 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["Faixa", "ValorTotal"])

    # buckets fixos de 15 em 15 dias até 181+
    buckets = [
        (0, 15),
        (16, 30),
        (31, 45),
        (46, 60),
        (61, 75),
        (76, 90),
        (91, 105),
        (106, 120),
        (121, 135),
        (136, 150),
        (151, 165),
        (166, 180),
        (181, None),  # 181+
    ]

    faixas = []
    valores = []

    for (lo, hi) in buckets:
        if hi is None:
            label = f"{lo}+"
            mask = df["dias_para_liquidar"] >= lo
        else:
            label = f"{lo}-{hi}"
            mask = (df["dias_para_liquidar"] >= lo) & (df["dias_para_liquidar"] <= hi)

        soma_bucket = df.loc[mask, "Valor"].sum()
        faixas.append(label)
        valores.append(soma_bucket)

    bucket_df = pd.DataFrame({"Faixa": faixas, "ValorTotal": valores})
    return bucket_df

# ==== Função de leitura do ZIP SEMPRE com Polars (agora com TAB) ====

@st.cache_data(show_spinner=True)
def read_estoque_from_zip(zip_file) -> tuple[pd.DataFrame, dt.date | None]:
    """
    Lê o primeiro CSV do ZIP em modo streaming com Polars, tratando arquivos enormes.
    Retorna:
      - agenda_df: pandas DataFrame com colunas [Data (date), Valor (float)], agregada por Data
      - zip_base_date: date ou None (DATA_REFERENCIA / DATA_FUNDO do CSV)
    Regras importantes:
      - Lemos um header rápido (sem forçar dtype)
      - Descobrimos nomes de colunas
      - Depois relermos tudo via scan_csv forçando todas as colunas como texto (Utf8)
        usando schema_overrides dinâmico
    """
    if zip_file is None:
        return pd.DataFrame(columns=["Data", "Valor"]), None

    import tempfile
    from pathlib import Path

    # Abre o ZIP e pega o primeiro CSV
    zf = zipfile.ZipFile(zip_file)
    csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not csv_names:
        st.error("ZIP não contém CSV.")
        return pd.DataFrame(columns=["Data", "Valor"]), None

    inner = csv_names[0]

    # Lê alguns bytes iniciais para detectar separador
    with zf.open(inner) as f:
        head_bytes = f.read(8192)

    def _detect_sep(sample_bytes: bytes) -> str:
        head = sample_bytes.decode(errors="ignore")
        candidates = {
            "\t": head.count("\t"),
            ";":  head.count(";"),
            ",":  head.count(","),
            "|":  head.count("|"),
        }
        # escolhe o mais frequente; em empate, prioriza TAB
        return max(candidates, key=lambda k: (candidates[k], 1 if k == "\t" else 0))

    sep = _detect_sep(head_bytes)

    # Copia o CSV inteiro pra um arquivo temporário
    with zf.open(inner) as f_in, tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        for chunk in iter(lambda: f_in.read(1024 * 1024), b""):
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    # Helper pra normalizar nome de coluna
    def _norm_col(name: str) -> str:
        return (
            str(name)
            .replace("\ufeff", "")
            .strip()
            .replace("\t", " ")
            .replace("  ", " ")
            .upper()
        )

    # 1) PRIMEIRA LEITURA RÁPIDA SÓ PRA PEGAR AS COLUNAS (sem dtypes forçado!)
    try:
        head_tbl = pl.read_csv(
            tmp_path,
            has_header=True,
            separator=sep,
            n_rows=200,
            infer_schema_length=200,
            try_parse_dates=False,
            encoding="utf8-lossy",
            ignore_errors=True,
            truncate_ragged_lines=True,
            low_memory=True,
        )
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        st.error(f"Falha lendo cabeçalho CSV do ZIP com Polars: {e}")
        return pd.DataFrame(columns=["Data", "Valor"]), None

    if head_tbl.height == 0 or len(head_tbl.columns) == 0:
        tmp_path.unlink(missing_ok=True)
        return pd.DataFrame(columns=["Data", "Valor"]), None

    # mapeia nomes normalizados -> nome original
    name_map = {_norm_col(c): c for c in head_tbl.columns}

    wanted_date = ["DATA_VENCIMENTO_AJUSTADA", "DTA_VENCIMENTO", "DATA_VENCIMENTO"]
    wanted_val  = ["VALOR_PRESENTE", "VALOR_NOMINAL", "VALOR_AQUISICAO", "VALOR"]
    wanted_ref  = ["DATA_REFERENCIA", "DATA_FUNDO", "DT_REF"]

    def _pick(candidates):
        for logical in candidates:
            logical_norm = _norm_col(logical)
            for seen_norm, real in name_map.items():
                # match exato ou prefixo
                if seen_norm.startswith(logical_norm):
                    return real
        return None

    date_col = _pick(wanted_date)
    if not date_col:
        tmp_path.unlink(missing_ok=True)
        st.error("Coluna de Data ('DATA_VENCIMENTO_AJUSTADA'/'DTA_VENCIMENTO'/'DATA_VENCIMENTO') não encontrada.")
        return pd.DataFrame(columns=["Data", "Valor"]), None

    val_col = _pick(wanted_val)
    if not val_col:
        # fallback: última coluna se não achou valor típico
        val_col = head_tbl.columns[-1]

    ref_col = _pick(wanted_ref)  # pode ser None

    # Agora que conhecemos TODAS as colunas, criamos schema_overrides
    # => força TODAS como Utf8 (texto)
    schema_overrides = {col: pl.Utf8 for col in head_tbl.columns}

    # helpers de limpeza e conversão (iguais aos que usamos antes)

    def _strip_expr(e: pl.Expr) -> pl.Expr:
        try:
            return e.str.strip_chars()
        except Exception:
            return e.str.strip()

    def _pl_to_date(expr: pl.Expr, fmt: str = "%Y-%m-%d") -> pl.Expr:
        try:
            return expr.str.to_date(format=fmt, strict=False)
        except Exception:
            return expr.str.strptime(pl.Date, format=fmt, strict=False)

    def _pl_any_date(expr_utf8: pl.Expr) -> pl.Expr:
        e = expr_utf8
        d1 = _pl_to_date(e.str.slice(0, 10), fmt="%Y-%m-%d")   # 2025-09-30
        d2 = _pl_to_date(e, fmt="%d/%m/%Y")                    # 30/09/2025
        try:
            d3 = e.str.strptime(pl.Date, format="%Y%m%d", strict=False)
        except Exception:
            d3 = None
        try:
            d4 = e.str.to_datetime(strict=False).dt.date()
        except Exception:
            d4 = None

        cands = [d1, d2]
        if d3 is not None:
            cands.append(d3)
        if d4 is not None:
            cands.append(d4)
        return pl.coalesce(cands)

    def _pl_normalize_money(expr_utf8: pl.Expr) -> pl.Expr:
        e = expr_utf8.fill_null("").str.replace_all(r"\s+", "")
        # trata números com parênteses negativos
        e = e.str.replace_all(r"^\((.*)\)$", r"-\1")

        both = (e.str.contains(",")) & (e.str.contains(r"\."))
        only_comma = (e.str.contains(",")) & (~e.str.contains(r"\."))

        cleaned = (
            pl.when(both)
              .then(e.str.replace_all(r"\.", "").str.replace_all(",", "."))
              .when(only_comma)
              .then(e.str.replace_all(",", "."))
              .otherwise(e)
        )

        return cleaned.cast(pl.Float64, strict=False)

    # 2) LEITURA STREAMING DE FATO (scan_csv), AGORA FORÇANDO UTF8 EM TODAS AS COLUNAS
    lf_main = (
        pl.scan_csv(
            tmp_path,
            has_header=True,
            separator=sep,
            try_parse_dates=False,
            encoding="utf8-lossy",
            ignore_errors=True,
            truncate_ragged_lines=True,
            low_memory=True,
            infer_schema_length=0,
            schema_overrides=schema_overrides,  # <- aqui está a mágica
        )
        .select([
            _strip_expr(pl.col(date_col)).alias("Data_raw"),
            _strip_expr(pl.col(val_col)).alias("Valor_raw"),
            (
                _strip_expr(pl.col(ref_col)).alias("Ref_raw")
                if ref_col is not None
                else pl.lit(None).alias("Ref_raw")
            ),
        ])
        .with_columns([
            _pl_any_date(pl.col("Data_raw")).alias("Data"),
            _pl_normalize_money(pl.col("Valor_raw")).alias("Valor"),
            _pl_any_date(pl.col("Ref_raw")).alias("RefDate"),
        ])
        .drop_nulls(["Data", "Valor"])
        .group_by("Data")
        .agg(pl.col("Valor").sum())
        .sort("Data")
    )

    tbl = lf_main.collect(streaming=True)
    agenda_df = tbl.to_pandas()
    agenda_df["Data"] = pd.to_datetime(agenda_df["Data"]).dt.date

    st.caption(
        "✅ Agenda carregada: "
        f"{agenda_df.shape[0]:,} datas únicas; separador='{sep}'"
    )
    st.caption(
        "📊 Soma total do estoque: R$ "
        + f"{agenda_df['Valor'].sum():,.2f}"
          .replace(",", "X").replace(".", ",").replace("X", ".")
    )

    # 3) PEGAR A DATA BASE DO ZIP (DATA_REFERENCIA / DATA_FUNDO...):
    zip_base_date = None
    try:
        lf_ref = (
            pl.scan_csv(
                tmp_path,
                has_header=True,
                separator=sep,
                try_parse_dates=False,
                encoding="utf8-lossy",
                ignore_errors=True,
                truncate_ragged_lines=True,
                low_memory=True,
                infer_schema_length=0,
                schema_overrides=schema_overrides,
            )
            .select([
                _pl_any_date(
                    _strip_expr(pl.col(ref_col)) if ref_col is not None else pl.lit(None)
                ).alias("RefDate")
            ])
            .drop_nulls()
            .head(1)
        )
        ref_first = lf_ref.collect(streaming=True).to_pandas()
        if not ref_first.empty and pd.notna(ref_first.iloc[0, 0]):
            zip_base_date = pd.to_datetime(ref_first.iloc[0, 0]).date()
    except Exception:
        pass
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return agenda_df, zip_base_date


# =========================
# XML ANBIMA v4 (parse robusto)
# =========================
def _num_locale(s):
    """Converte string numérica pt-BR / en-US / científico / (negativo)."""
    if s is None:
        return 0.0
    t = str(s).strip().replace(" ", "")
    if t == "":
        return 0.0
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1]
    # se tem vírgula e ponto: milhar '.' e decimal ','
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".")
    elif "," in t and "." not in t:
        t = t.replace(",", ".")
    # senão, assume decimal com ponto
    try:
        v = float(t)
    except Exception:
        # last resort: remove milhar “não-decimal” (ex.: 1 234 567.89)
        import re
        t2 = re.sub(r"(?<=\d)[\s\.](?=\d{3}(\D|$))", "", t)
        try:
            v = float(t2)
        except Exception:
            v = 0.0
    return -v if neg else v

@st.cache_data(show_spinner=False)
def parse_anbima_xml_v4(file):
    """
    Extrai:
      - dtposicao, patliq
      - dc_fidc = soma de //fidc//valorfinanceiro
      - dc_debentures = soma de //debentures//valorfindisp + //debenture//valorfindisp
      - soberano0 = soma para nós que contenham simultaneamente (isin ∈ SOBERANO_ISINS, qtdisponivel, puposicao)
    """
    if file is None:
        return None
    tree = ET.parse(file)
    root = tree.getroot()
    f = root.find("fundo")
    if f is None:
        st.error("XML fora do layout v4 (nó <fundo> ausente).")
        return None

    out = {"dtposicao": None, "patliq": 0.0, "dc_fidc": 0.0, "dc_debentures": 0.0, "soberano0": 0.0}

    header = f.find("header")
    if header is not None:
        try:
            d = header.find("dtposicao").text
            out["dtposicao"] = dt.datetime.strptime(d, "%Y%m%d").date()
        except Exception:
            pass
        try:
            out["patliq"] = _num_locale(header.find("patliq").text)
        except Exception:
            pass

    # DC FIDC
    for node in f.findall(".//fidc//valorfinanceiro"):
        out["dc_fidc"] += _num_locale(node.text)

    # DC Debêntures
    for node in f.findall(".//debentures//valorfindisp"):
        out["dc_debentures"] += _num_locale(node.text)
    for node in f.findall(".//debenture//valorfindisp"):
        out["dc_debentures"] += _num_locale(node.text)

    # Soberano (apenas quando os 3 filhos estão no MESMO bloco)
    sob = 0.0
    for parent in f.iter():
        isin = parent.find("isin")
        qtd  = parent.find("qtdisponivel")
        pu   = parent.find("puposicao")
        if isin is not None and qtd is not None and pu is not None:
            isin_txt = (isin.text or "").strip()
            if isin_txt in SOBERANO_ISINS:
                sob += _num_locale(qtd.text) * _num_locale(pu.text)
    out["soberano0"] = sob

    out["dc_total"] = out["dc_fidc"] + out["dc_debentures"]
    return out

# =========================
# Simulação
# =========================
def simulate(pl0, dc0, sob0, agenda, orc_df, lim, months, start_date):
    """
    - PL: cresce só em dia útil → PL_t = PL_{t-1} * (1 + PL_DAILY_GROWTH)
    - DC: DC_t = DC_{t-1} + Aq_t − Liq_total
    - Soberano: SOB_t = SOB_{t-1} − Aq_t + Liq_total
    - Buckets: liquidações de aquisições são ajustadas para o próximo dia útil
    - Desenquadrado: DC/PL < lim
    """
    eixo = [start_date + timedelta(days=i) for i in range(months * 30 + 1)]

    # liquidações do estoque (agenda)
    liq_estoque = {}
    if agenda is not None and not agenda.empty:
        tmp = agenda.copy()
        tmp["Data"] = pd.to_datetime(tmp["Data"], errors="coerce").dt.date
        tmp = tmp.dropna(subset=["Data", "Valor"])
        for _, r in tmp.iterrows():
            add_to_map(liq_estoque, r["Data"], r["Valor"])

    # aquisições mensais ponderadas por dia da semana + liquidações por FAIXAS
    acq_increase, liq_aquis = {}, {}
    if orc_df is not None and not orc_df.empty:
        tmpo = orc_df.copy()
        tmpo["Mes"] = pd.to_datetime(tmpo["Mes"], errors="coerce").dt.date
        tmpo = tmpo.dropna(subset=["Mes"])

        RANGE_BUCKETS = pick_range_buckets(fundo)        # faixas de liquidação (já implementadas)
        WEEK_WEIGHTS = pick_week_weights(fundo)          # pesos por dia da semana

        for _, row in tmpo.iterrows():
            mes = row["Mes"]
            valor_mensal = float(row.get("Aquisicoes", 0.0) or 0.0)
            if valor_mensal <= 0:
                continue

            # distribui o valor mensal pelos dias úteis do mês, ponderado por weekday
            diario_map = distribute_monthly_acq_by_weekday(mes, valor_mensal, WEEK_WEIGHTS)
            if not diario_map:
                continue

            # registra o aumento de DC no dia de cada aquisição
            for d_acq, aq_val in diario_map.items():
                add_to_map(acq_increase, d_acq, aq_val)

                # gera as liquidações futuras conforme FAIXAS (cada dia terá seu próprio "aq_val")
                distribute_acq_by_ranges(
                    d_acq,
                    aq_val,
                    RANGE_BUCKETS,
                    lambda d_liq, v: add_to_map(liq_aquis, d_liq, v)
                )

    # iteração diária
    plv, dcv, sobv = float(pl0), float(dc0), float(sob0)
    rows = []
    for d in eixo:
        aq = acq_increase.get(d, 0.0)
        liq_e = liq_estoque.get(d, 0.0)
        liq_a = liq_aquis.get(d, 0.0)
        liq_total = liq_e + liq_a

        # Soberano
        sob_raw = sobv - aq + liq_total
        caixa_nec = max(0.0, -sob_raw)
        sobv = max(0.0, sob_raw)

        # DC
        dcv = max(0.0, dcv + aq - liq_total)

        # PL (apenas dia útil)
        if is_business_day_calendar(d):
            plv = plv * (1.0 + PL_DAILY_GROWTH)

        ratio = (dcv / plv) if plv > 0 else np.nan
        desenq = (False if np.isnan(ratio) else (ratio < lim))

        rows.append({
            "Data": d, "PL": plv, "DC": dcv, "DC/PL": ratio, "Soberano": sobv,
            "Aquisicao_diaria": aq, "Liq_Estoque": liq_e, "Liq_Aquisicoes": liq_a,
            "Liq_Total": liq_total, "Caixa_Necessario": caixa_nec,
            "Caixa Zerado": caixa_nec > 0.0, "Desenquadrado": desenq
        })

    return pd.DataFrame(rows)

# =========================
# 1) Estoque (ZIP)
# =========================
st.subheader("1) Estoque (zip)")
res, zip_base_date = read_estoque_from_zip(estoque_zip)

# compat: aceita retorno (df) ou (df, zip_base_date) ou (df, zip_base_date, csv_inner_name)
if isinstance(res, tuple):
    if len(res) == 3:
        agenda_df, zip_base_date, csv_inner_name = res
    elif len(res) == 2:
        agenda_df, zip_base_date = res
        csv_inner_name = None
    else:
        agenda_df, zip_base_date = res[0], None
        csv_inner_name = None
else:
    agenda_df = res
    zip_base_date = None
    csv_inner_name = None

if agenda_df is None or agenda_df.empty:
    st.info("Envie o **.zip** do estoque.")

# =========================
# 2) Carteira (XML v4)
# =========================
st.subheader("2) Carteira (xml v4)")
carteira = parse_anbima_xml_v4(carteira_xml)
if not carteira:
    st.info("Envie o **XML v4** da carteira.")
    st.stop()

dt_ref = carteira["dtposicao"]
pl0 = carteira["patliq"]
dc0 = carteira["dc_total"]
sob0 = carteira["soberano0"]
dcpl_pct = (dc0 / pl0 * 100) if pl0 > 0 else None

if zip_base_date is not None and isinstance(dt_ref, dt.date):
    if zip_base_date != dt_ref:
        st.error(
            f"Datas-base divergentes: ZIP = {zip_base_date} vs XML = {dt_ref}. "
            "Envie arquivos com a mesma data-base."
        )
        st.stop()
    else:
        st.caption(f"✅ Datas coerentes entre ZIP ({zip_base_date}) e XML ({dt_ref}).")

def _fmt_carteira(v):
    absv = abs(v)
    if absv >= 1e9:  txt = f"R$ {v/1e9:,.2f} bi"
    elif absv >= 1e6: txt = f"R$ {v/1e6:,.2f} mi"
    elif absv >= 1e3: txt = f"R$ {v/1e3:,.0f} mil"
    else:             txt = f"R$ {v:,.0f}"
    return txt.replace(",", "X").replace(".", ",").replace("X", ".")

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Data Base (XML)", value=str(dt_ref or "—"))
with c2: st.metric("PL", value=(f"R$ {_fmt_carteira(pl0)}" if pl0 else "—"))
with c3: st.metric("DC (Estoque + Debêntures)", value=(f"R$ {_fmt_carteira(dc0)}" if dc0 else "—"))
with c4: st.metric("Soberano (Caixa)", value=(f"R$ {_fmt_carteira(sob0)}" if sob0 else "—"))
with c5: st.metric("DC/PL (%)", value=(f"{dcpl_pct:,.2f}%" if dcpl_pct is not None else "—"))

# =========================
# 3) Orçamento — aquisições
# =========================
st.subheader("3) Orçamento de Aquisição (mês a mês)")
meses = pd.date_range(dt_ref.replace(day=1), periods=12, freq="MS").date
orc_default = pd.DataFrame({"Mes": meses, "Aquisicoes": 0.0})
orc_df = st.data_editor(orc_default, num_rows="dynamic", use_container_width=True)

if orc_df is not None and not orc_df.empty:
    orc_df = orc_df.copy()
    orc_df["Aquisicoes"] = pd.to_numeric(orc_df["Aquisicoes"], errors="coerce").fillna(0.0)

# =========================
# 4) Projeção e Gráficos
# =========================
st.divider()
st.subheader("4) Projeção e desenquadramento (linha diária)")

if st.button("Rodar projeção", type="primary"):
    start_date = dt_ref + timedelta(days=1)
    out = simulate(pl0, dc0, sob0, agenda_df, orc_df, limite, horizonte_meses, start_date)

    first_desenq = out.loc[out["Desenquadrado"] == True, "Data"]
    first_desenq_date = first_desenq.iloc[0] if len(first_desenq) > 0 else None

    first_cash_zero = out.loc[out["Caixa Zerado"] == True, "Data"]
    first_cash_zero_date = first_cash_zero.iloc[0] if len(first_cash_zero) > 0 else None

    # Filtra a série da projeção para só dias úteis (sem fds e feriados)
    out_bd = out[out["Data"].apply(is_business_day_calendar)].copy()

    # Ajusta as datas dos “eventos” para o próximo dia útil (visual)
    fd_plot = None
    if first_desenq_date is not None:
        fd_plot = next_business_day_calendar(pd.to_datetime(first_desenq_date).date())

    fc_plot = None
    if first_cash_zero_date is not None:
        fc_plot = next_business_day_calendar(pd.to_datetime(first_cash_zero_date).date())

    # ===== Gráficos (salvos antes de exibir) =====
    img_dcpl = BytesIO()
    img_pldc = BytesIO()
    img_faixa = BytesIO()

    col_g1, col_g2, col_g3 = st.columns(3)

    # ----- G1: DC/PL -----
    with col_g1:
        st.markdown("**DC/PL ao longo do tempo (DIAS ÚTEIS)**")
        fig, ax = plt.subplots(figsize=(9, 5), dpi=140)

        x = pd.to_datetime(out_bd["Data"])
        y = out_bd["DC/PL"]

        ok_mask = y >= limite
        bad_mask = y < limite
        ax.fill_between(x, y, limite, where=ok_mask, alpha=0.25, step="pre")
        ax.fill_between(x, y, limite, where=bad_mask, alpha=0.25, step="pre")

        ax.plot(x, y, linewidth=2.2, label="DC/PL")
        ax.axhline(limite, linestyle="--", linewidth=1.6, label="Limite mínimo")

        if fd_plot is not None:
            ax.axvline(pd.to_datetime(fd_plot), color="red", linestyle="-", linewidth=1.8, label="1º Desenq.")
        if fc_plot is not None:
            ax.axvline(pd.to_datetime(fc_plot), color="red", linestyle="--", linewidth=1.8, label="1º Caixa Zerado")

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, alpha=0.25); ax.set_xlabel("Data"); ax.set_ylabel("DC/PL")
        ax.margins(x=0.01, y=0.05); ax.legend(loc="upper left", frameon=False, ncol=3)
        fig.tight_layout()

        # salva imagem para o XLSX ANTES de exibir
        fig.savefig(img_dcpl, format="png", dpi=200, bbox_inches="tight")
        img_dcpl.seek(0)
        st.pyplot(fig, clear_figure=False)
        plt.close(fig)

    # ----- G2: PL / DC / Soberano -----
    with col_g2:
        st.markdown("**PL, DC e Soberano (DIAS ÚTEIS)**")
        fig2, ax2 = plt.subplots(figsize=(9, 5), dpi=140)

        x = pd.to_datetime(out_bd["Data"])
        ax2.plot(x, out_bd["PL"], linewidth=2.0, label="PL")
        ax2.plot(x, out_bd["DC"], linewidth=2.0, label="DC")
        ax2.plot(x, out_bd["Soberano"], linewidth=2.0, label="Soberano")

        if fd_plot is not None:
            ax2.axvline(pd.to_datetime(fd_plot), color="red", linestyle="-", linewidth=1.8, label="1º Desenq.")
        if fc_plot is not None:
            ax2.axvline(pd.to_datetime(fc_plot), color="red", linestyle="--", linewidth=1.8, label="1º Caixa Zerado")

        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))

        def _fmt_brl_compacto(v, pos):
            absv = abs(v)
            if absv >= 1e9:  txt = f"R$ {v/1e9:,.2f} bi"
            elif absv >= 1e6: txt = f"R$ {v/1e6:,.2f} mi"
            elif absv >= 1e3: txt = f"R$ {v/1e3:,.0f} mil"
            else:             txt = f"R$ {v:,.0f}"
            return txt.replace(",", "X").replace(".", ",").replace("X", ".")
        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl_compacto))

        ax2.grid(True, alpha=0.25); ax2.set_xlabel("Data"); ax2.set_ylabel("R$")
        ax2.margins(x=0.01, y=0.08); ax2.legend(loc="upper left", frameon=False, ncol=3)
        fig2.tight_layout()

        # salva imagem para o XLSX ANTES de exibir
        fig2.savefig(img_pldc, format="png", dpi=200, bbox_inches="tight")
        img_pldc.seek(0)
        st.pyplot(fig2, clear_figure=False)
        plt.close(fig2)

    # ----- 3) Gráfico de barras: liquidação prevista por faixa de prazo -----
    with col_g3:
        st.markdown("**Distribuição de Liquidações por Faixa de Prazo (Estoque da Data Importada)**")

        st.write("zip_base_date usado:", zip_base_date)
        if agenda_df is not None and not agenda_df.empty:
            debug_tmp = agenda_df.copy()
            debug_tmp["Data"] = pd.to_datetime(debug_tmp["Data"]).dt.date
            base_dbg = zip_base_date if zip_base_date is not None else min(debug_tmp["Data"])
            debug_tmp["dias_para_liquidar"] = debug_tmp["Data"].apply(lambda d: (d - base_dbg).days)
            debug_preview = debug_tmp[["Data", "Valor", "dias_para_liquidar"]].sort_values("dias_para_liquidar").head(20)
            st.write("preview diferença dias:", debug_preview)

        liq_bucket_df = build_liq_bucket_df(agenda_df, dt_ref)

        if liq_bucket_df.empty:
            st.info("Não foi possível montar a distribuição por faixas (estoque ou data-base ausentes).")
        else:
            # formatação BRL no eixo Y
            def _fmt_brl(v, pos):
                absv = abs(v)
                if absv >= 1e9:
                    txt = f"R$ {v/1e9:,.2f} bi"
                elif absv >= 1e6:
                    txt = f"R$ {v/1e6:,.2f} mi"
                elif absv >= 1e3:
                    txt = f"R$ {v/1e3:,.0f} mil"
                else:
                    txt = f"R$ {v:,.0f}"
                # troca , . pro padrão brasileiro
                return (
                    txt.replace(",", "X")
                    .replace(".", ",")
                    .replace("X", ".")
                )

            fig3, ax3 = plt.subplots(figsize=(9,4.5), dpi=140)

            x_pos = range(len(liq_bucket_df))
            ax3.bar(x_pos, liq_bucket_df["ValorTotal"], width=0.6)

            ax3.set_xticks(list(x_pos))
            ax3.set_xticklabels(liq_bucket_df["Faixa"], rotation=0)

            ax3.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl))

            ax3.set_xlabel("Faixa de Dias até a Liquidação")
            ax3.set_ylabel("Valor a Liquidar (R$)")
            ax3.set_title("Liquidações Futuras do Estoque por Bucket de Prazo")

            ax3.grid(axis="y", alpha=0.25)

            # coloca label em cima de cada barra, em R$ mi/bil etc.
            for xpos, val in zip(x_pos, liq_bucket_df["ValorTotal"]):
                if val and val != 0:
                    label_txt = _fmt_brl(val, None)
                    ax3.text(
                        xpos,
                        val,
                        label_txt,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=0,
                    )

            fig3.savefig(img_faixa, format="png", dpi=200, bbox_inches="tight")
            img_faixa.seek(0)
            st.pyplot(fig3, clear_figure=True)
            plt.close(fig3)

    # ===== Resultado =====
    st.markdown("### Resultado")
    bullets = []
    if first_desenq_date:
        bullets.append(f"Primeiro **desenquadramento** (DC/PL < {limite:.0%}) em **{pd.to_datetime(first_desenq_date).date()}**.")
    else:
        bullets.append("Nenhum desenquadramento detectado no horizonte configurado.")
    if first_cash_zero_date:
        bullets.append(f"Primeiro dia com **Caixa Zerado** em **{pd.to_datetime(first_cash_zero_date).date()}**.")
    else:
        bullets.append("Nenhum evento de **Caixa Zerado** no horizonte configurado.")
    st.write("- " + "\n- ".join(bullets))

    # ===== Tabela =====
    st.markdown("### Tabela da projeção (diária)")
    cols_fmt = ["PL", "DC", "Soberano", "Aquisicao_diaria", "Liq_Estoque", "Liq_Aquisicoes", "Liq_Total", "Caixa_Necessario"]
    for c in cols_fmt:
        if c in out_bd.columns:
            out_bd[c] = out_bd[c].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    out_bd["DC/PL"] = out_bd["DC/PL"].map(lambda x: f"{x:.2%}".replace(".", ","))
    st.dataframe(out_bd, use_container_width=True)

    # ===== Exportar XLSX com gráficos embutidos =====
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        out_bd.to_excel(writer, sheet_name="Projecao", index=False)
        if agenda_df is not None and not agenda_df.empty:
            agenda_df.to_excel(writer, sheet_name="Liquidacoes em Estoque", index=False)
        if orc_df is not None and not orc_df.empty:
            orc_df.to_excel(writer, sheet_name="Orcamento Informado", index=False)

        wb = writer.book
        ws = wb.add_worksheet("Graficos")
        title_fmt = wb.add_format({"bold": True, "font_size": 14})

        ws.write("A1",  "DC/PL ao longo do tempo (diário)", title_fmt)
        ws.insert_image("A3",  "dcpl.png", {"image_data": img_dcpl, "x_scale": 1.0, "y_scale": 1.0})
        ws.write("A27", "PL, DC e Soberano (diário)",        title_fmt)
        ws.insert_image("A29", "pldc.png", {"image_data": img_pldc, "x_scale": 1.0, "y_scale": 1.0})
        ws.write("A53", "Liquidações Futuras por Faixa de Prazo", title_fmt)
        ws.insert_image("A55", "faixas.png", {"image_data": img_faixa})

        ws_proj = writer.sheets["Projecao"]
        ws_proj.freeze_panes(1, 1)
        ws_proj.set_column(0, 0, 12)
        ws_proj.set_column(1, 10, 16)

    output.seek(0)
    st.download_button(
        "Baixar projeção (XLSX)",
        data=output.getvalue(),
        file_name=f"projecao_desenquadramento-{fundo}-{zip_base_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="secondary",
        help="Inclui abas de dados e as imagens dos gráficos"
    )