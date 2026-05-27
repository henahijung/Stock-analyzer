import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

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
</style>
""", unsafe_allow_html=True)


# ── yfinance MultiIndex 수정 ───────────────────────────────────────────────────
def fix_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


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
    df = fix_df(df)
    c = df["Close"].astype(float)
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

    chk(float(r["MA5"]) > float(r["MA20"]), 8,
        f"MA5({float(r['MA5']):.2f}) > MA20({float(r['MA20']):.2f})")
    chk(float(r["MA20"]) > float(r["MA60"]), 6,
        f"MA20({float(r['MA20']):.2f}) > MA60({float(r['MA60']):.2f})")

    if float(p["MA5"]) <= float(p["MA20"]) and float(r["MA5"]) > float(r["MA20"]):
        score += 20; logs.append("🌟 골든크로스 발생!")
    if float(p["MA5"]) >= float(p["MA20"]) and float(r["MA5"]) < float(r["MA20"]):
        score -= 20; logs.append("💀 데드크로스 발생!")

    rsi = float(r["RSI"])
    if   rsi < 30: score += 18; logs.append(f"🔥 RSI {rsi:.1f} — 과매도 (매수 신호)")
    elif rsi > 70: score -= 18; logs.append(f"❄️  RSI {rsi:.1f} — 과매수 (매도 신호)")
    else:          logs.append(f"➖ RSI {rsi:.1f} — 중립")

    macd_bull = float(r["MACD"]) > float(r["MACD_Sig"])
    chk(macd_bull, 10, f"MACD({'강세' if macd_bull else '약세'})")
    if float(p["MACD"]) <= float(p["MACD_Sig"]) and float(r["MACD"]) > float(r["MACD_Sig"]):
        score += 15; logs.append("🚀 MACD 골든크로스!")
    if float(p["MACD"]) >= float(p["MACD_Sig"]) and float(r["MACD"]) < float(r["MACD_Sig"]):
        score -= 15; logs.append("⬇️  MACD 데드크로스!")

    cp = float(r["Close"])
    if   cp <= float(r["BB_L"]): score += 14; logs.append("📍 볼린저 하단 터치 (매수 신호)")
    elif cp >= float(r["BB_U"]): score -= 14; logs.append("🚧 볼린저 상단 터치 (매도 신호)")
    else: logs.append(f"➖ 볼린저 중간대")

    score = max(0, min(100, score))
    sig = ("강력매수" if score >= 75 else
           "매수"     if score >= 60 else
           "중립"     if score >= 40 else
           "매도"     if score >= 25 else "강력매도")
    return sig, score, logs


# ── 목표가 계산 ───────────────────────────────────────────────────────────────
def targets(df, n=20):
    w   = df.tail(n)
    cp  = float(df["Close"].iloc[-1])

    # 현재가 기준 근접 지지선: 최근 n일 저가 중 현재가보다 낮은 것의 최대값
    recent_lows = w["Low"].astype(float)
    lows_below  = recent_lows[recent_lows < cp]
    sup = float(lows_below.max()) if not lows_below.empty else round(cp * 0.97, 2)

    # 현재가 기준 근접 저항선: 최근 n일 고가 중 현재가보다 높은 것의 최소값
    recent_highs = w["High"].astype(float)
    highs_above  = recent_highs[recent_highs > cp]
    res = float(highs_above.min()) if not highs_above.empty else round(cp * 1.05, 2)

    return {
        "현재가":   round(cp, 2),
        "매수목표": round(sup, 2),           # 현재가 아래 지지선 = 분할매수 기준
        "매도목표": round(cp * 1.07, 2),     # 현재가 +7% = 1차 매도 목표
        "지지선":   round(sup, 2),
        "저항선":   round(res, 2),
        "손절가":   round(cp * 0.95, 2),     # 현재가 -5% = 손절
    }


# ── 차트 ─────────────────────────────────────────────────────────────────────
def draw_chart(df, ticker):
    df2 = df.tail(120).copy()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} 가격", "MACD", "RSI"),
    )

    fig.add_trace(go.Candlestick(
        x=df2.index, open=df2["Open"].astype(float),
        high=df2["High"].astype(float), low=df2["Low"].astype(float),
        close=df2["Close"].astype(float),
        increasing_line_color="#00d26a", decreasing_line_color="#ef4444",
        name="캔들"), row=1, col=1)

    for ma, col in [("MA5","#60a5fa"),("MA20","#f59e0b"),("MA60","#a78bfa")]:
        if ma in df2:
            fig.add_trace(go.Scatter(x=df2.index, y=df2[ma].astype(float),
                name=ma, line=dict(color=col, width=1.2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df2.index, y=df2["BB_U"].astype(float),
        line=dict(color="#94a3b8", width=0.8, dash="dot"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df2.index, y=df2["BB_L"].astype(float),
        fill="tonexty", fillcolor="rgba(148,163,184,0.05)",
        line=dict(color="#94a3b8", width=0.8, dash="dot"), showlegend=False), row=1, col=1)

    hist = (df2["MACD"] - df2["MACD_Sig"]).astype(float)
    hist_colors = ["#00d26a" if v >= 0 else "#ef4444" for v in hist]
    fig.add_trace(go.Bar(x=df2.index, y=hist,
        marker_color=hist_colors, name="히스토그램", opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=df2.index, y=df2["MACD"].astype(float),
        name="MACD", line=dict(color="#60a5fa", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df2.index, y=df2["MACD_Sig"].astype(float),
        name="시그널", line=dict(color="#f59e0b", width=1.2)), row=2, col=1)

    fig.add_trace(go.Scatter(x=df2.index, y=df2["RSI"].astype(float),
        name="RSI", line=dict(color="#a78bfa", width=1.5)), row=3, col=1)
    for lvl, col in [(70,"rgba(239,68,68,0.3)"),(30,"rgba(0,210,106,0.3)")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=col, row=3, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8"), height=620,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


# ── 사이드바 ──────────────────────────────────────────────────────────────────
SIG_EMOJI = {"강력매수":"💚 강력매수","매수":"🟢 매수","중립":"🟡 중립",
             "매도":"🔴 매도","강력매도":"❤️‍🔥 강력매도"}
SIG_COLOR = {"강력매수":"#00d26a","매수":"#4ade80","중립":"#facc15",
             "매도":"#f87171","강력매도":"#ef4444"}

with st.sidebar:
    st.markdown("## 📈 실시간 주식 분석기")
    st.caption("미국 주식 분석 대시보드")
    st.divider()

    st.markdown("#### 종목 입력")
    st.caption("예: `AAPL`, `TSLA`, `NVDA`, `MSFT`")
    raw = st.text_area("종목 코드 (줄바꿈으로 여러 개)",
        value="AAPL\nTSLA\nNVDA\nMSFT", height=120)
    tickers = [t.strip().upper() for t in raw.splitlines() if t.strip()]

    st.divider()
    period = st.selectbox("데이터 기간",
        ["3mo","6mo","1y","2y"], index=1,
        format_func=lambda x: {"3mo":"3개월","6mo":"6개월","1y":"1년","2y":"2년"}[x])
    interval = st.selectbox("봉 단위",
        ["1d","1h","30m","15m"], index=0,
        format_func=lambda x: {"1d":"일봉","1h":"1시간봉","30m":"30분봉","15m":"15분봉"}[x])

    st.divider()
    auto_refresh = st.toggle("🔄 자동 새로고침", value=False)
    refresh_sec  = st.slider("새로고침 간격 (초)", 30, 300, 60,
                              step=30, disabled=not auto_refresh)

    st.divider()
    alert_buy  = st.multiselect("매수 알림", ["강력매수","매수"],
                                 default=["강력매수","매수"])
    alert_sell = st.multiselect("매도 알림", ["강력매도","매도"],
                                 default=["강력매도","매도"])

    run_btn = st.button("🔍 분석 시작", use_container_width=True, type="primary")

    st.divider()
    st.markdown("#### 🌟 오늘의 추천")
    scan_btn = st.button("📡 종목 스캔 시작", use_container_width=True)
    st.caption("주요 미국 주식 30개를 자동 분석해 매수 추천 순위를 보여줍니다")


# ── 메인 ─────────────────────────────────────────────────────────────────────
st.markdown("# 📊 실시간 주식 분석 대시보드")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if "alerts" not in st.session_state:
    st.session_state.alerts = []

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
                    raw_df = yf.download(
                        ticker, period=period, interval=interval,
                        progress=False, auto_adjust=True)

                    if raw_df.empty or len(raw_df) < 30:
                        st.error(f"❌ {ticker}: 데이터 부족 또는 잘못된 종목 코드")
                        continue

                    df  = enrich(raw_df.copy())
                    sig, score, logs = judge(df)
                    tgt = targets(df)

                    try:
                        info = yf.Ticker(ticker).info
                        name = info.get("shortName") or info.get("longName") or ticker
                        currency = info.get("currency", "USD")
                    except:
                        name, currency = ticker, "USD"

                except Exception as e:
                    st.error(f"❌ {ticker} 오류: {e}")
                    continue

            # 알림 기록
            if sig in alert_buy or sig in alert_sell:
                am = {"time": datetime.now().strftime("%H:%M:%S"),
                      "ticker": ticker, "name": name,
                      "signal": sig, "price": tgt["현재가"]}
                if (not st.session_state.alerts or
                        st.session_state.alerts[0].get("ticker") != ticker or
                        st.session_state.alerts[0].get("signal") != sig):
                    st.session_state.alerts.insert(0, am)

            col_h, col_s = st.columns([3, 1])
            with col_h:
                st.markdown(f"### {name} `{ticker}`")
            with col_s:
                sc = SIG_COLOR.get(sig, "#facc15")
                st.markdown(
                    f"<div style='font-size:1.4rem;font-weight:900;color:{sc};text-align:right'>"
                    f"{SIG_EMOJI[sig]}</div>", unsafe_allow_html=True)

            cp   = tgt["현재가"]
            prev = float(df["Close"].astype(float).iloc[-2])
            chg  = cp - prev
            pct  = chg / prev * 100

            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("현재가", f"${cp:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
            m2.metric("매수 목표가 🟢", f"${tgt['매수목표']:,.2f}")
            m3.metric("매도 목표가 🔴", f"${tgt['매도목표']:,.2f}")
            m4.metric("지지 / 저항", f"${tgt['지지선']:,.2f} / ${tgt['저항선']:,.2f}")
            m5.metric("손절가 🛑", f"${tgt['손절가']:,.2f}",
                      delta=f"{(tgt['손절가']-cp)/cp*100:.1f}%", delta_color="inverse")

            st.divider()
            st.plotly_chart(draw_chart(df, ticker), use_container_width=True)
            st.divider()

            ca, cb = st.columns([1, 2])
            with ca:
                st.markdown("#### 📊 매수 강도")
                bar_col = "#00d26a" if score>=60 else "#ef4444" if score<=40 else "#facc15"
                st.markdown(f"""
                <div style="background:#0f172a;border-radius:12px;padding:20px;text-align:center">
                  <div style="font-size:3rem;font-weight:900;color:{bar_col}">{score}</div>
                  <div style="color:#64748b;font-size:.8rem">/ 100점 (50 = 중립)</div>
                  <div style="background:#1e293b;border-radius:8px;height:10px;margin-top:12px">
                    <div style="background:{bar_col};width:{score}%;height:100%;border-radius:8px"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            with cb:
                st.markdown("#### 🔍 세부 신호")
                for log in logs:
                    st.markdown(f"- {log}")

            st.divider()
            last = df.iloc[-1]
            i1,i2,i3,i4 = st.columns(4)
            i1.metric("RSI",  f"{float(last['RSI']):.1f}")
            i2.metric("MACD", f"{float(last['MACD']):.3f}")
            i3.metric("MA5",  f"${float(last['MA5']):,.2f}")
            i4.metric("MA20", f"${float(last['MA20']):,.2f}")

    with tabs[-1]:
        st.markdown("### 🔔 알림 기록")
        if not st.session_state.alerts:
            st.info("분석 시작 후 매수/매도 신호 발생 시 여기에 기록됩니다.")
        else:
            for a in st.session_state.alerts[:30]:
                col = "#00d26a" if "매수" in a["signal"] else "#ef4444"
                st.markdown(f"""
                <div style="background:#0f172a;border:1px solid #1e293b;
                            border-left:4px solid {col};border-radius:8px;
                            padding:12px 16px;margin-bottom:8px;
                            display:flex;justify-content:space-between;align-items:center">
                  <div><b>{a['ticker']}</b>
                    <span style="color:#64748b;font-size:.85rem;margin-left:8px">{a['name']}</span>
                  </div>
                  <div style="color:{col};font-weight:700">{SIG_EMOJI[a['signal']]}</div>
                  <div style="color:#94a3b8">${a['price']:,.2f}</div>
                  <div style="color:#64748b;font-size:.8rem">{a['time']}</div>
                </div>""", unsafe_allow_html=True)
        if st.button("🗑️ 알림 초기화"):
            st.session_state.alerts = []
            st.rerun()

else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#475569">
      <div style="font-size:4rem;margin-bottom:16px">📈</div>
      <h2 style="color:#94a3b8">왼쪽 사이드바에서 종목을 입력하고<br>분석 시작을 눌러주세요</h2>
      <br>
      <div style="display:inline-block;background:#0f172a;border:1px solid #1e293b;
                  border-radius:12px;padding:20px 32px;text-align:left">
        <b>💡 미국 주식 예시</b><br><br>
        <code>AAPL</code> &nbsp; 애플<br>
        <code>TSLA</code> &nbsp; 테슬라<br>
        <code>NVDA</code> &nbsp; 엔비디아<br>
        <code>MSFT</code> &nbsp; 마이크로소프트<br>
        <code>AMZN</code> &nbsp; 아마존
      </div>
      <br><br>
      <div style="color:#334155;font-size:.85rem">
        ⚠️ 본 분석은 기술적 지표 기반 참고 정보입니다. 투자 결정은 본인 책임입니다.
      </div>
    </div>""", unsafe_allow_html=True)

if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()

# ── 오늘의 추천 종목 스캔 ────────────────────────────────────────────────────
WATCHLIST = {
    "AAPL":  "애플",       "MSFT":  "마이크로소프트", "GOOGL": "구글",
    "AMZN":  "아마존",     "NVDA":  "엔비디아",       "META":  "메타",
    "TSLA":  "테슬라",     "AMD":   "AMD",             "AVGO":  "브로드컴",
    "QCOM":  "퀄컴",       "INTC":  "인텔",            "TSM":   "TSMC",
    "JPM":   "JP모건",     "BAC":   "뱅크오브아메리카","GS":    "골드만삭스",
    "V":     "비자",       "MA":    "마스터카드",       "PYPL":  "페이팔",
    "JNJ":   "존슨앤존슨", "PFE":   "화이자",          "UNH":   "유나이티드헬스",
    "XOM":   "엑슨모빌",   "CVX":   "셰브론",          "NFLX":  "넷플릭스",
    "DIS":   "디즈니",     "UBER":  "우버",             "ABNB":  "에어비앤비",
    "SPY":   "S&P500 ETF", "QQQ":   "나스닥100 ETF",   "ARKK":  "아크이노베이션",
}

if scan_btn:
    st.markdown("---")
    st.markdown("## 🌟 오늘의 매수 추천 종목")
    st.caption(f"분석 기준: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 기술적 지표 기반")

    results = []
    prog = st.progress(0, text="종목 스캔 중...")

    for i, (ticker, name) in enumerate(WATCHLIST.items()):
        prog.progress((i + 1) / len(WATCHLIST), text=f"스캔 중... {ticker} ({i+1}/{len(WATCHLIST)})")
        try:
            raw_df = yf.download(ticker, period="3mo", interval="1d",
                                  progress=False, auto_adjust=True)
            if raw_df.empty or len(raw_df) < 30:
                continue
            df  = enrich(raw_df.copy())
            sig, score, _ = judge(df)
            tgt = targets(df)
            cp   = tgt["현재가"]
            prev = float(df["Close"].astype(float).iloc[-2])
            chg_pct = (cp - prev) / prev * 100
            results.append({
                "ticker":  ticker,
                "name":    name,
                "signal":  sig,
                "score":   score,
                "price":   cp,
                "chg_pct": chg_pct,
                "buy":     tgt["매수목표"],
                "sell":    tgt["매도목표"],
                "stop":    tgt["손절가"],
            })
        except:
            continue

    prog.empty()

    if not results:
        st.warning("스캔 결과가 없습니다. 잠시 후 다시 시도해주세요.")
    else:
        buy_results = [r for r in results if r["signal"] in ("강력매수", "매수")]
        buy_results.sort(key=lambda x: x["score"], reverse=True)
        other_results = [r for r in results if r["signal"] not in ("강력매수", "매수")]
        other_results.sort(key=lambda x: x["score"], reverse=True)

        st.markdown(f"### ✅ 매수 추천 종목 ({len(buy_results)}개)")

        if not buy_results:
            st.info("현재 매수 신호 종목이 없습니다. 시장 전체가 관망 구간일 수 있어요.")
        else:
            for rank, r in enumerate(buy_results, 1):
                sc = SIG_COLOR.get(r["signal"], "#facc15")
                chg_color = "#00d26a" if r["chg_pct"] >= 0 else "#ef4444"
                chg_arrow = "▲" if r["chg_pct"] >= 0 else "▼"
                st.markdown(f"""
                <div style="background:#0f172a;border:1px solid #1e293b;
                            border-left:4px solid {sc};border-radius:12px;
                            padding:16px 20px;margin-bottom:10px">
                  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
                    <div>
                      <span style="font-size:1.1rem;font-weight:900;color:#e2e8f0">#{rank} {r['ticker']}</span>
                      <span style="color:#64748b;margin-left:8px">{r['name']}</span>
                    </div>
                    <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap">
                      <div style="text-align:center">
                        <div style="font-size:.7rem;color:#64748b">현재가</div>
                        <div style="font-weight:700">${r['price']:,.2f}
                          <span style="color:{chg_color};font-size:.8rem">{chg_arrow}{abs(r['chg_pct']):.1f}%</span>
                        </div>
                      </div>
                      <div style="text-align:center">
                        <div style="font-size:.7rem;color:#64748b">매수목표</div>
                        <div style="color:#4ade80;font-weight:700">${r['buy']:,.2f}</div>
                      </div>
                      <div style="text-align:center">
                        <div style="font-size:.7rem;color:#64748b">매도목표</div>
                        <div style="color:#f87171;font-weight:700">${r['sell']:,.2f}</div>
                      </div>
                      <div style="text-align:center">
                        <div style="font-size:.7rem;color:#64748b">손절가</div>
                        <div style="color:#94a3b8;font-weight:700">${r['stop']:,.2f}</div>
                      </div>
                      <div style="text-align:center">
                        <div style="font-size:.7rem;color:#64748b">신호</div>
                        <div style="color:{sc};font-weight:900">{SIG_EMOJI[r['signal']]}</div>
                      </div>
                      <div style="text-align:center">
                        <div style="font-size:.7rem;color:#64748b">점수</div>
                        <div style="color:{sc};font-weight:900;font-size:1.2rem">{r['score']}</div>
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

        if other_results:
            with st.expander(f"📋 나머지 종목 보기 ({len(other_results)}개)"):
                for r in other_results:
                    sc = SIG_COLOR.get(r["signal"], "#facc15")
                    st.markdown(
                        f"**{r['ticker']}** {r['name']} &nbsp;|&nbsp; "
                        f"<span style='color:{sc}'>{SIG_EMOJI[r['signal']]}</span> &nbsp;|&nbsp; "
                        f"점수: {r['score']} &nbsp;|&nbsp; ${r['price']:,.2f}",
                        unsafe_allow_html=True)

        st.markdown("""
        <div style="color:#334155;font-size:.8rem;margin-top:16px;padding:12px;
                    background:#0f172a;border-radius:8px">
        ⚠️ 본 추천은 기술적 지표(이동평균, RSI, MACD, 볼린저밴드)만을 기반으로 한 참고 정보입니다.
        기업 실적·뉴스·거시경제는 반영되지 않습니다. 투자 결정은 반드시 본인이 판단하세요.
        </div>""", unsafe_allow_html=True)
