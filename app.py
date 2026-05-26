import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📈 실시간 주식 분석기",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .signal-강력매수 { color: #00d26a; font-size: 1.6rem; font-weight: 900; }
    .signal-매수     { color: #4ade80; font-size: 1.6rem; font-weight: 900; }
    .signal-중립     { color: #facc15; font-size: 1.6rem; font-weight: 900; }
    .signal-매도     { color: #f87171; font-size: 1.6rem; font-weight: 900; }
    .signal-강력매도 { color: #ef4444; font-size: 1.6rem; font-weight: 900; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── 기술 지표 계산 ────────────────────────────────────────────────────────────
def calc_rsi(s, n=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(n).mean()
    l = (-d.where(d < 0, 0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l)

def calc_macd(s, f=12, sl=26, sig=9):
    m = s.ewm(span=f).mean() - s.ewm(span=sl).mean()
    return m, m.ewm(span=sig).mean()

def calc_bb(s, n=20, k=2):
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return ma + k * sd, ma, ma - k * sd

def enrich(df):
    c = df["Close"].squeeze()
    df["MA5"]  = c.rolling(5).mean()
    df["MA20"] = c.rolling(20).mean()
    df["MA60"] = c.rolling(60).mean()
    df["RSI"]  = calc_rsi(c)
    df["MACD"], df["MACD_Sig"] = calc_macd(c)
    df["BB_U"], df["BB_M"], df["BB_L"] = calc_bb(c)
    return df


# ── 매수/매도 신호 판정 ───────────────────────────────────────────────────────
def judge(df):
    r, p = df.iloc[-1], df.iloc[-2]
    score, logs = 50, []

    def chk(cond, pts, msg):
        nonlocal score
        score += pts if cond else -pts
        logs.append(("✅" if cond else "❌") + " " + msg)

    # 이평선 방향
    chk(r["MA5"] > r["MA20"], 8, f"MA5({r['MA5']:.0f}) > MA20({r['MA20']:.0f})")
    chk(r["MA20"] > r["MA60"], 6, f"MA20({r['MA20']:.0f}) > MA60({r['MA60']:.0f})")

    # 골든/데드 크로스
    golden = p["MA5"] <= p["MA20"] and r["MA5"] > r["MA20"]
    dead   = p["MA5"] >= p["MA20"] and r["MA5"] < r["MA20"]
    if golden: score += 20; logs.append("🌟 골든크로스 발생!")
    if dead:   score -= 20; logs.append("💀 데드크로스 발생!")

    # RSI
    rsi = r["RSI"]
    if   rsi < 30: score += 18; logs.append(f"🔥 RSI {rsi:.1f} — 과매도 (매수)")
    elif rsi > 70: score -= 18; logs.append(f"❄️  RSI {rsi:.1f} — 과매수 (매도)")
    else:          logs.append(f"➖ RSI {rsi:.1f} — 중립")

    # MACD
    macd_bull = r["MACD"] > r["MACD_Sig"]
    chk(macd_bull, 10, f"MACD({'강세' if macd_bull else '약세'})")
    if p["MACD"] <= p["MACD_Sig"] and r["MACD"] > r["MACD_Sig"]:
        score += 15; logs.append("🚀 MACD 골든크로스!")
    if p["MACD"] >= p["MACD_Sig"] and r["MACD"] < r["MACD_Sig"]:
        score -= 15; logs.append("⬇️  MACD 데드크로스!")

    # 볼린저밴드
    cp = float(r["Close"])
    if   cp <= r["BB_L"]: score += 14; logs.append("📍 볼린저 하단 터치 (매수)")
    elif cp >= r["BB_U"]: score -= 14; logs.append("🚧 볼린저 상단 터치 (매도)")
    else: logs.append(f"➖ 볼린저 중간대 ({(cp-r['BB_L'])/(r['BB_U']-r['BB_L'])*100:.0f}%)")

    score = max(0, min(100, score))
    sig = ("강력매수" if score >= 75 else
           "매수"     if score >= 60 else
           "중립"     if score >= 40 else
           "매도"     if score >= 25 else "강력매도")
    return sig, score, logs


# ── 목표가 계산 ───────────────────────────────────────────────────────────────
def targets(df, n=30):
    w = df.tail(n)
    sup = float(w["Low"].min())
    res = float(w["High"].max())
    cp  = float(df["Close"].iloc[-1])
    rng = res - sup
    return {
        "현재가":   cp,
        "매수목표": round(sup + rng * 0.382, 2),
        "매도목표": round(sup + rng * 0.618, 2),
        "지지선":   round(sup, 2),
        "저항선":   round(res, 2),
        "손절가":   round(sup * 0.97, 2),
    }


# ── 캔들 + 지표 차트 ─────────────────────────────────────────────────────────
def draw_chart(df, ticker):
    df2 = df.tail(120).copy()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} 가격", "MACD", "RSI"),
    )

    # 캔들
    fig.add_trace(go.Candlestick(
        x=df2.index, open=df2["Open"], high=df2["High"],
        low=df2["Low"], close=df2["Close"],
        increasing_line_color="#00d26a", decreasing_line_color="#ef4444",
        name="캔들"), row=1, col=1)

    colors = {"MA5": "#60a5fa", "MA20": "#f59e0b", "MA60": "#a78bfa"}
    for ma, col in colors.items():
        if ma in df2:
            fig.add_trace(go.Scatter(x=df2.index, y=df2[ma], name=ma,
                line=dict(color=col, width=1.2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df2.index, y=df2["BB_U"], name="BB상단",
        line=dict(color="#94a3b8", width=0.8, dash="dot"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df2.index, y=df2["BB_L"], name="BB하단",
        fill="tonexty", fillcolor="rgba(148,163,184,0.05)",
        line=dict(color="#94a3b8", width=0.8, dash="dot"), showlegend=False), row=1, col=1)

    # MACD
    hist_colors = ["#00d26a" if v >= 0 else "#ef4444" for v in df2["MACD"] - df2["MACD_Sig"]]
    fig.add_trace(go.Bar(x=df2.index, y=df2["MACD"] - df2["MACD_Sig"],
        marker_color=hist_colors, name="히스토그램", opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=df2.index, y=df2["MACD"], name="MACD",
        line=dict(color="#60a5fa", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df2.index, y=df2["MACD_Sig"], name="시그널",
        line=dict(color="#f59e0b", width=1.2)), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df2.index, y=df2["RSI"], name="RSI",
        line=dict(color="#a78bfa", width=1.5)), row=3, col=1)
    for lvl, col in [(70, "rgba(239,68,68,0.3)"), (30, "rgba(0,210,106,0.3)")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=col, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.05)",
        line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,210,106,0.05)",
        line_width=0, row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8"),
        height=620,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_xaxes(gridcolor="#1e293b", showgrid=True)
    fig.update_yaxes(gridcolor="#1e293b", showgrid=True)
    return fig


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 실시간 주식 분석기")
    st.caption("한국 · 미국 주식 실시간 분석")
    st.divider()

    st.markdown("#### 종목 입력")
    st.caption("한국: `005930.KS` (삼성전자), `035420.KS` (NAVER)\n미국: `AAPL`, `TSLA`, `NVDA`")

    raw = st.text_area("종목 코드 (줄바꿈으로 여러 개)",
        value="005930.KS\n000660.KS\nAAPL\nTSLA",
        height=120)
    tickers = [t.strip().upper() for t in raw.splitlines() if t.strip()]

    st.divider()
    period = st.selectbox("데이터 기간", ["3mo", "6mo", "1y", "2y"], index=1,
        format_func=lambda x: {"3mo":"3개월","6mo":"6개월","1y":"1년","2y":"2년"}[x])
    interval = st.selectbox("봉 단위", ["1d", "1h", "30m", "15m"], index=0,
        format_func=lambda x: {"1d":"일봉","1h":"1시간봉","30m":"30분봉","15m":"15분봉"}[x])

    st.divider()
    auto_refresh = st.toggle("🔄 자동 새로고침", value=False)
    refresh_sec  = st.slider("새로고침 간격 (초)", 30, 300, 60, step=30,
                              disabled=not auto_refresh)

    st.divider()
    st.markdown("#### ⚠️ 알림 조건")
    alert_buy  = st.multiselect("매수 알림", ["강력매수","매수"], default=["강력매수","매수"])
    alert_sell = st.multiselect("매도 알림", ["강력매도","매도"], default=["강력매도","매도"])

    run_btn = st.button("🔍 분석 시작", use_container_width=True, type="primary")


# ── 메인 화면 ─────────────────────────────────────────────────────────────────
st.markdown("# 📊 실시간 주식 분석 대시보드")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if "alerts" not in st.session_state:
    st.session_state.alerts = []

SIG_EMOJI = {"강력매수":"💚 강력매수","매수":"🟢 매수","중립":"🟡 중립",
             "매도":"🔴 매도","강력매도":"❤️‍🔥 강력매도"}
SIG_DELTA = {"강력매수":"normal","매수":"normal","중립":"off",
             "매도":"inverse","강력매도":"inverse"}

if run_btn or (auto_refresh and "last_run" in st.session_state and
               time.time() - st.session_state.last_run >= refresh_sec):

    st.session_state.last_run = time.time()

    if not tickers:
        st.warning("종목 코드를 입력하세요.")
        st.stop()

    tabs = st.tabs([f"📌 {t}" for t in tickers] + ["🔔 알림 기록"])

    for idx, ticker in enumerate(tickers):
        with tabs[idx]:
            with st.spinner(f"{ticker} 데이터 로딩 중..."):
                try:
                    raw_df = yf.download(ticker, period=period,
                                         interval=interval, progress=False,
                                         auto_adjust=True)
                    if raw_df.empty or len(raw_df) < 30:
                        st.error(f"❌ {ticker}: 데이터 부족 또는 잘못된 종목 코드")
                        continue

                    df = enrich(raw_df.copy())
                    sig, score, logs = judge(df)
                    tgt = targets(df)

                    info = yf.Ticker(ticker).info
                    name = info.get("shortName") or info.get("longName") or ticker
                    currency = info.get("currency", "")

                except Exception as e:
                    st.error(f"❌ {ticker} 오류: {e}")
                    continue

            # 알림 기록
            if sig in alert_buy or sig in alert_sell:
                alert_msg = {
                    "time":   datetime.now().strftime("%H:%M:%S"),
                    "ticker": ticker,
                    "name":   name,
                    "signal": sig,
                    "price":  tgt["현재가"],
                }
                if (not st.session_state.alerts or
                        st.session_state.alerts[-1]["ticker"] != ticker or
                        st.session_state.alerts[-1]["signal"] != sig):
                    st.session_state.alerts.insert(0, alert_msg)

            # ─ 헤더 ─
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"### {name} `{ticker}`")
            with c2:
                st.markdown(f"<div class='signal-{sig}'>{SIG_EMOJI[sig]}</div>",
                            unsafe_allow_html=True)

            # ─ 핵심 메트릭 ─
            cp   = tgt["현재가"]
            prev = float(df["Close"].iloc[-2])
            chg  = cp - prev
            chg_pct = chg / prev * 100

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("현재가", f"{cp:,.2f} {currency}",
                      f"{chg:+.2f} ({chg_pct:+.2f}%)")
            m2.metric("매수 목표가 🟢", f"{tgt['매수목표']:,.2f}")
            m3.metric("매도 목표가 🔴", f"{tgt['매도목표']:,.2f}")
            m4.metric("지지선 / 저항선",
                      f"{tgt['지지선']:,.2f} / {tgt['저항선']:,.2f}")
            m5.metric("손절가 🛑", f"{tgt['손절가']:,.2f}",
                      delta=f"{(tgt['손절가']-cp)/cp*100:.1f}%",
                      delta_color="inverse")

            st.divider()

            # ─ 차트 ─
            st.plotly_chart(draw_chart(df, ticker), use_container_width=True)

            # ─ 신호 점수 + 로그 ─
            st.divider()
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown("#### 📊 매수 강도 점수")
                bar_color = ("#00d26a" if score >= 60 else
                             "#ef4444" if score <= 40 else "#facc15")
                st.markdown(f"""
                <div style="background:#0f172a;border-radius:12px;padding:20px;text-align:center">
                  <div style="font-size:3rem;font-weight:900;color:{bar_color}">{score}</div>
                  <div style="color:#64748b;font-size:.8rem">/ 100점 (50 = 중립)</div>
                  <div style="background:#1e293b;border-radius:8px;height:10px;margin-top:12px">
                    <div style="background:{bar_color};width:{score}%;height:100%;border-radius:8px;transition:width .5s"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            with col_b:
                st.markdown("#### 🔍 세부 신호 분석")
                for log in logs:
                    st.markdown(f"- {log}")

            # ─ 현재 지표값 ─
            st.divider()
            last = df.iloc[-1]
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("RSI",  f"{last['RSI']:.1f}")
            i2.metric("MACD", f"{last['MACD']:.3f}")
            i3.metric("MA5",  f"{last['MA5']:,.2f}")
            i4.metric("MA20", f"{last['MA20']:,.2f}")

    # ─ 알림 탭 ─
    with tabs[-1]:
        st.markdown("### 🔔 신호 알림 기록")
        if not st.session_state.alerts:
            st.info("아직 알림이 없습니다. 분석을 실행하면 매수/매도 신호 발생 시 여기에 기록됩니다.")
        else:
            for a in st.session_state.alerts[:30]:
                color = "#00d26a" if "매수" in a["signal"] else "#ef4444"
                st.markdown(f"""
                <div style="background:#0f172a;border:1px solid #1e293b;border-left:4px solid {color};
                            border-radius:8px;padding:12px 16px;margin-bottom:8px;
                            display:flex;justify-content:space-between;align-items:center">
                  <div>
                    <span style="font-weight:700">{a['ticker']}</span>
                    <span style="color:#64748b;font-size:.85rem;margin-left:8px">{a['name']}</span>
                  </div>
                  <div style="color:{color};font-weight:700">{SIG_EMOJI[a['signal']]}</div>
                  <div style="color:#94a3b8">{a['price']:,.2f}</div>
                  <div style="color:#64748b;font-size:.8rem">{a['time']}</div>
                </div>""", unsafe_allow_html=True)

        if st.button("🗑️ 알림 기록 초기화"):
            st.session_state.alerts = []
            st.rerun()

else:
    # 초기 화면
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#475569">
      <div style="font-size:4rem;margin-bottom:16px">📈</div>
      <h2 style="color:#94a3b8">왼쪽 사이드바에서 종목을 입력하고<br>분석 시작 버튼을 누르세요</h2>
      <br>
      <div style="display:inline-grid;grid-template-columns:1fr 1fr;gap:16px;text-align:left;max-width:500px">
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:16px">
          <b>🇰🇷 한국 주식 예시</b><br>
          <code>005930.KS</code> 삼성전자<br>
          <code>000660.KS</code> SK하이닉스<br>
          <code>035420.KS</code> NAVER<br>
          <code>035720.KQ</code> 카카오
        </div>
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:16px">
          <b>🇺🇸 미국 주식 예시</b><br>
          <code>AAPL</code> 애플<br>
          <code>TSLA</code> 테슬라<br>
          <code>NVDA</code> 엔비디아<br>
          <code>MSFT</code> 마이크로소프트
        </div>
      </div>
      <br><br>
      <div style="color:#334155;font-size:.85rem">
        ⚠️ 본 분석은 기술적 지표 기반 참고 정보입니다. 투자 결정은 본인 책임입니다.
      </div>
    </div>
    """, unsafe_allow_html=True)

# 자동 새로고침
if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
