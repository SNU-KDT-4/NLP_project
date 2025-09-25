# app.py
# -*- coding: utf-8 -*-

import os
import re
import json
import math
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import plotly.express as px

# LangChain / Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# =========================
# 0) 설정
# =========================
EMB_MODEL   = "intfloat/multilingual-e5-large"
PERSIST_DIR = "./chroma_db"
COLLECTION  = "unified_data"

# =========================
# 1) 벡터스토어 & Retriever
# =========================
@st.cache_resource(show_spinner=False)
def get_vectorstore_and_retriever(k_docs: int = 300, score_threshold: float = 0.10):
    emb = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=emb,
    )
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k_docs, "score_threshold": score_threshold},
    )
    compressor = EmbeddingsFilter(embeddings=emb, similarity_threshold=0.70)
    comp = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)
    return db, comp, emb

# =========================
# 2) LLM
# =========================
@st.cache_resource(show_spinner=False)
def get_llm(temp: float = 0.2, max_tokens: int = 4500):
    return ChatOpenAI(
        model="gpt-4o",
        temperature=float(temp),
        max_tokens=int(max_tokens),
        api_key='' # secret
    )

# =========================
# 3) 컨텍스트 포맷
# =========================
def format_docs_for_context(docs: List[Any]) -> str:
    chunks = []
    for d in docs:
        m = d.metadata or {}
        tag = f"[year:{m.get('year')} | file:{m.get('filename')} | section:{m.get('section','N/A')}]"
        chunks.append(tag + "\n" + (d.page_content or "")[:1600])
    return "\n\n---\n\n".join(chunks)

# =========================
# 4) 대시보드 스펙 (LLM → JSON만 출력)
# =========================
DASHBOARD_SCHEMA = {
    "summary_md":  "핵심 요약 (마크다운, 5~8문장)",
    "analysis_md": "심화 분석 (마크다운, 500~1000자: 추세·변화요인·리스크·경영진 주장에 대한 검증/반박 포함)",
    "panels": [
        {
            "kind":  "table | bar | line | pie",
            "title": "패널 제목",
            "unit":  "단위(예: KRW_million, % 등) 또는 빈문자열",
            "notes": "간단 출처(연도/파일) 1~2개",
            "data":  [
                {"year": 2023, "value": 123.4, "series": "보통주", "label": "배당총액"}
            ],
            "columns": ["연도", "지표", "값", "단위"],
            "rows":    [{"연도": 2023, "지표": "EPS", "값": 3740, "단위": "KRW"}]
        }
    ]
}

DASHBOARD_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 삼성전자 재무/사업보고서 전문 애널리스트이자 데이터 시각화 디자이너다. "
     "주어진 컨텍스트만 근거로 사실을 서술하라. 숫자에는 단위/연도를 명확히 붙이고, "
     "분석 요청 시 전문적인 리포트를 작성하라. "
     "경영진의 의견/전망은 홍보적 수사일 수 있으므로 **숫자·주석 등 근거가 있는 경우에만 인용**하고, "
     "근거가 약하면 '경영진 주장'으로 명시하고 **반례·불확실성·리스크**를 함께 기술하라. "
     "대시보드는 **JSON 스펙으로만** 출력한다. "
     "반드시 아래 JSON 스키마를 따르며, **코드블록 없이 JSON 객체 하나만** 반환한다."),
    ("human",
     "사용자 질문: {question}\n\n"
     "컨텍스트 문서(요약):\n{context}\n\n"
     "요구사항:\n"
     "- 총 4개 패널: 표(table) 1개 + 차트 3개(막대/꺾은선/원형 중 적절히 선택).\n"
     "- **시계열 규칙**:\n"
     "  1) 질문이 특정 1개 연도만의 단순 비교가 아니라면, 관련 지표에 대해 **가능한 모든 연도**를 x축으로 하는 **추세(line) 패널**을 반드시 포함.\n"
     "  2) 위 추세 지표 중 핵심 1~2개에 대해 **직전 2개 연도(최근연도·그 직전연도) 값 bar 비교**를 추가. (x축: 연도, y축: 값)\n"
     "  3) 연도가 1개만 확보되면 line/bar는 생략하고 표(notes)에 해당 사실 명시.\n"
     "- 라인/막대는 x축에 year 사용을 우선. pie는 label/value 사용.\n"
     "- panel.title은 간결하게, panel.notes에는 (year, filename) 형태의 출처 1~2개.\n"
     "- **summary_md**: 5~8문장으로 핵심 결론(최근연도 수준, 전년 대비 변화, 다년 추세 요약 등).\n"
     "- **analysis_md**: 500~1000자 심화 분석(추세/드라이버, 정책변화, 리스크,재무상태표,현금흐름표,손익계산서,경영진 주장 검증/반박 포함).\n"
     "- 데이터 구성 규칙:\n"
     "  • 모든 패널의 data는 {{\"year\":정수, \"value\":숫자, \"series\":\"문자열\"}} 형태를 기본으로 하되,\n"
     "    pie는 {{\"label\":\"문자열\", \"value\":숫자}} 형태를 사용.\n"
     "  • 단위는 panel.unit에 명시하고, 서로 다른 단위를 같은 패널에 혼합하지 않는다.\n"
     "- 출력은 JSON 하나만. 그 외 텍스트 절대 금지.\n\n"
     "JSON 스키마:\n{schema}\n")
])

def generate_dashboard_spec(llm: ChatOpenAI, question: str, context_text: str) -> Dict[str, Any]:
    msg = DASHBOARD_PROMPT.format_messages(
        question=question,
        context=context_text,
        schema=json.dumps(DASHBOARD_SCHEMA, ensure_ascii=False, indent=2),
    )
    raw = llm.invoke(msg).content or ""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            fixed = raw.strip().strip("```").strip()
            return json.loads(fixed)
        except Exception:
            return {}

# =========================
# 5) 렌더 유틸
# =========================
def render_table_panel(panel: Dict[str, Any]):
    st.subheader(panel.get("title", "표"))
    if "columns" in panel and "rows" in panel and panel["columns"] and panel["rows"]:
        df = pd.DataFrame(panel["rows"], columns=panel["columns"])
    else:
        df = pd.DataFrame(panel.get("data", []))
        if df.empty:
            st.info("표 데이터가 없습니다.")
            return
    st.dataframe(df, use_container_width=True)
    if panel.get("unit"):  st.caption(f"단위: {panel['unit']}")
    if panel.get("notes"): st.caption(f"출처: {panel['notes']}")

def render_line_panel(panel: Dict[str, Any]):
    st.subheader(panel.get("title", "추이(꺾은선)"))
    df = pd.DataFrame(panel.get("data", []))
    if df.empty or "year" not in df or "value" not in df:
        st.info("year/value 필드가 필요합니다.")
        return
    df = df.sort_values(["series", "year"]) if "series" in df else df.sort_values("year")
    fig = px.line(df, x="year", y="value", color=("series" if "series" in df else None), markers=True)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    if panel.get("unit"):  st.caption(f"단위: {panel['unit']}")
    if panel.get("notes"): st.caption(f"출처: {panel['notes']}")

def render_bar_panel(panel: Dict[str, Any]):
    st.subheader(panel.get("title", "막대"))
    df = pd.DataFrame(panel.get("data", []))
    if df.empty:
        st.info("데이터가 없습니다.")
        return
    if {"year", "value"}.issubset(df.columns):
        x, y = "year", "value"
        color = "series" if "series" in df else None
    elif {"label", "value"}.issubset(df.columns):
        x, y = "label", "value"
        color = None
    else:
        st.info("bar는 (year,value) 또는 (label,value) 가 필요합니다.")
        return
    fig = px.bar(df, x=x, y=y, color=color, text_auto=True)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    if panel.get("unit"):  st.caption(f"단위: {panel['unit']}")
    if panel.get("notes"): st.caption(f"출처: {panel['notes']}")

def render_pie_panel(panel: Dict[str, Any]):
    st.subheader(panel.get("title", "원형"))
    df = pd.DataFrame(panel.get("data", []))
    if df.empty or not {"label", "value"}.issubset(df.columns):
        st.info("pie는 (label,value) 필드가 필요합니다.")
        return
    fig = px.pie(df, names="label", values="value", hole=0)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    if panel.get("unit"):  st.caption(f"단위: {panel['unit']}")
    if panel.get("notes"): st.caption(f"출처: {panel['notes']}")

def render_panel(panel: Dict[str, Any]):
    kind = (panel.get("kind") or "").lower()
    if kind == "table":   render_table_panel(panel)
    elif kind == "line":  render_line_panel(panel)
    elif kind == "bar":   render_bar_panel(panel)
    elif kind == "pie":   render_pie_panel(panel)
    else:
        st.info(f"지원하지 않는 패널 유형: {panel.get('kind')}")

def render_grid(panels: List[Dict[str, Any]]):
    n = len(panels)
    if n == 0:
        st.warning("표시할 패널이 없습니다.")
        return
    cols = 1 if n == 1 else (2 if n == 2 else 3)
    rows = math.ceil(n / cols)
    idx = 0
    for _ in range(rows):
        cols_list = st.columns(cols, gap="medium")
        for c in cols_list:
            if idx >= n: break
            with c:
                render_panel(panels[idx])
            idx += 1

# =========================
# 6) Streamlit UI
# =========================
st.set_page_config(page_title="Financial RAG Dashboard", page_icon="📈", layout="wide")

with st.sidebar:
    st.header("설정")
    k_docs = st.slider("LLM 컨텍스트에 넣을 문서 수", 10, 200, 72, step=2)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.20, step=0.05)
    st.caption("문서가 많을수록 근거가 풍부하지만 비용/시간이 늘어납니다.")

st.markdown(
    """
    <style>
    .center-box {max-width: 1024px; margin: 0 auto;}
    .stButton>button {width: 100%; height: 42px; font-weight: 700; background:#ff4b4b;}
    .stTextInput>div>div>input {text-align: center;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="center-box">', unsafe_allow_html=True)
st.title("📈 Financial RAG Dashboard")
default_q = "2023년과 2024년 삼성전자 실적 추이 변화를 분석해줘"
user_q = st.text_input("", value=default_q, placeholder="분석할 질문을 입력하세요")
run = st.button("분석 실행")
st.markdown('</div>', unsafe_allow_html=True)

if run:
    with st.spinner("🔍 문서 검색 및 컨텍스트 구성 중..."):
        _, comp_retriever, _ = get_vectorstore_and_retriever(k_docs=k_docs)
        docs = comp_retriever.get_relevant_documents(user_q)
        ctx = format_docs_for_context(docs)

    st.success(f"검색된 문서 수: {len(docs)}")
    with st.expander("컨텍스트 미리보기", expanded=False):
        st.text(ctx[:4000] + ("\n...\n" if len(ctx) > 4000 else ""))

    with st.spinner("🧠 LLM이 대시보드를 설계 중..."):
        llm = get_llm(temp=temperature, max_tokens=4500)
        spec = generate_dashboard_spec(llm, user_q, ctx)

    if not spec or "panels" not in spec:
        st.error("대시보드 스펙을 만들지 못했습니다. 질문을 조금 더 구체화해 보세요.")
    else:
        if spec.get("summary_md"):
            st.markdown("## 요약")
            st.markdown(spec["summary_md"])
        if spec.get("analysis_md"):
            st.markdown("## 심화 분석")
            st.markdown(spec["analysis_md"])

        st.markdown("---")
        st.markdown("## 📊 자동 대시보드")
        panels = spec.get("panels", [])
        render_grid(panels)
