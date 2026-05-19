import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import time
from datetime import datetime, timedelta
from sklearn.ensemble import HistGradientBoostingRegressor
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 설정
# ==========================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '').strip()
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
MAX_WORKERS = 20
HISTORY_FILE = 'history.csv'
EPS_HISTORY_FILE = 'eps_history.csv'  # 🆕 리비전 자체 트래킹용

EXCLUDED_SECTORS = ['Financial Services', 'Utilities', 'Real Estate', 'Basic Materials', 'Healthcare']

def get_broad_universe():
    print("🚀 미국 전종목 티커 수집 시작...")
    try:
        url_a = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        res_a = requests.get(url_a, timeout=10)
        if res_a.status_code == 200:
            tickers = [t.strip().replace('.', '-') for t in res_a.text.split('\n') if t.strip() and len(t.strip()) <= 5]
            if len(tickers) > 1000:
                print(f"✅ [Plan A 성공] 총 {len(tickers)}개 티커 확보!")
                return tickers
    except Exception as e:
        print(f"⚠️ Plan A 실패: {e}")

    print("🔄 [Plan B 가동] 미국 SEC 공식 데이터베이스에서 직접 추출합니다...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        url_b = "https://www.sec.gov/files/company_tickers.json"
        res_b = requests.get(url_b, headers=headers, timeout=10)
        if res_b.status_code == 200:
            data = res_b.json()
            tickers = list(set([v['ticker'].replace('.', '-') for v in data.values()]))
            print(f"✅ [Plan B 성공] SEC 공식 티커 {len(tickers)}개 확보!")
            return tickers
    except Exception as e:
        print(f"❌ Plan B마저 실패: {e}")
    return []

def get_market_baseline():
    try:
        spy = yf.Ticker("SPY").history(period="6mo")
        if len(spy) > 63:
            return ((spy['Close'].iloc[-1] / spy['Close'].iloc[-63]) - 1) * 100
    except: pass
    return 0.0

# ==========================================
# 2. 증거 수집 (52주 신고가 근접도 + EPS 원본 저장)
# ==========================================
def fetch_evidence(ticker, start_date, end_date, spy_ret_3m):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if info.get('marketCap', 0) < 2_000_000_000: return None
        sector = info.get('sector', 'Unknown')
        if sector in EXCLUDED_SECTORS: return None

        name = info.get('shortName', ticker)
        hist = stock.history(start=start_date, end=end_date)
        if len(hist) < 65: return None

        close_px = hist['Close'].iloc[-1]

        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
        ma60 = hist['Close'].rolling(60).mean().iloc[-1]
        trend_ok = 1 if (close_px > ma20 > ma60) else 0

        eps_trl = info.get('trailingEps', 0.1)
        eps_fwd = info.get('forwardEps', 0)
        # 기존 스프레드 (폴백용으로 유지)
        eps_growth = ((eps_fwd - eps_trl) / abs(eps_trl)) * 100 if eps_trl != 0 else 0

        price_3m = hist['Close'].iloc[-63]
        mom_3m = ((close_px / price_3m) - 1) * 100

        ma20_disparity = (close_px / ma20) * 100 if ma20 > 0 else 100
        rs_rating = mom_3m - spy_ret_3m
        vol_5d = hist['Volume'].iloc[-5:].mean()
        vol_60d = hist['Volume'].iloc[-60:].mean()
        vol_breakout = (vol_5d / vol_60d) if vol_60d > 0 else 1.0

        # 🆕 52주 신고가 근접도 (100에 가까울수록 신고가 부근)
        high_52w = info.get('fiftyTwoWeekHigh', close_px)
        high_52w_pct = (close_px / high_52w) * 100 if high_52w > 0 else 0

        return {
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'Ticker': ticker, 'Name': name, 'Sector': sector, 'Close': close_px,
            'Trend_OK': trend_ok,
            'EPS_Growth': eps_growth,          # 기존 스프레드 (폴백)
            'ForwardEps_Raw': eps_fwd,         # 🆕 리비전 트래킹용 원본
            'Mom_3M': mom_3m,
            'MA20_Disparity': ma20_disparity,
            'RS_Rating': rs_rating,
            'Volume_Breakout': vol_breakout,
            'High_52W_Pct': high_52w_pct,      # 🆕 선행 지표
            'Revision_7D': 0.0,                # 🆕 아래에서 채워짐
            'Revision_30D': 0.0,               # 🆕 아래에서 채워짐
            'Target': np.nan
        }
    except: return None

# ==========================================
# 3. 🆕 자체 리비전 트래킹 (자동 누적, 자동 계산)
# ==========================================
def track_and_compute_revision(today_df):
    """
    매일 forwardEps를 eps_history.csv에 저장.
    7일/30일 전 값과 비교해서 진짜 애널리스트 추정치 변화율을 계산.
    데이터가 아직 없으면 0을 반환 → 자동으로 EPS_Growth(폴백)가 쓰임.
    """
    today_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))

    # --- eps_history 로드 ---
    if os.path.exists(EPS_HISTORY_FILE):
        try:
            eps_hist = pd.read_csv(EPS_HISTORY_FILE)
            eps_hist['Date'] = pd.to_datetime(eps_hist['Date'], format='mixed', errors='coerce')
            eps_hist = eps_hist.dropna(subset=['Date'])
        except:
            eps_hist = pd.DataFrame(columns=['Date', 'Ticker', 'ForwardEps'])
    else:
        eps_hist = pd.DataFrame(columns=['Date', 'Ticker', 'ForwardEps'])

    # --- 오늘 EPS 저장 ---
    today_eps = today_df[['Date', 'Ticker', 'ForwardEps_Raw']].copy()
    today_eps.columns = ['Date', 'Ticker', 'ForwardEps']
    today_eps['Date'] = pd.to_datetime(today_eps['Date'])
    eps_hist = pd.concat([eps_hist, today_eps]).drop_duplicates(subset=['Date', 'Ticker'], keep='last')

    # 90일 이상 된 데이터 정리 (파일 비대화 방지)
    cutoff = today_date - timedelta(days=90)
    eps_hist = eps_hist[eps_hist['Date'] >= cutoff]
    eps_hist.to_csv(EPS_HISTORY_FILE, index=False)

    # --- 리비전 계산 ---
    # 티커별로 과거 EPS를 한번에 조회 (성능)
    past_eps = eps_hist[eps_hist['Date'] < today_date].copy()

    if past_eps.empty:
        print("📝 EPS 트래킹 1일차 — 내일부터 리비전 데이터가 쌓입니다.")
        return today_df

    rev_7d_map = {}
    rev_30d_map = {}

    for ticker in today_df['Ticker'].unique():
        current_eps = today_df.loc[today_df['Ticker'] == ticker, 'ForwardEps_Raw'].iloc[0]
        ticker_hist = past_eps[past_eps['Ticker'] == ticker].sort_values('Date')

        if ticker_hist.empty or abs(current_eps) < 0.01:
            continue

        # 7일 전 기준 (7~14일 사이 가장 오래된 값)
        ref_7d = ticker_hist[(ticker_hist['Date'] >= today_date - timedelta(days=14)) &
                             (ticker_hist['Date'] <= today_date - timedelta(days=5))]
        if not ref_7d.empty and abs(ref_7d.iloc[0]['ForwardEps']) > 0.01:
            rev_7d_map[ticker] = ((current_eps - ref_7d.iloc[0]['ForwardEps']) / abs(ref_7d.iloc[0]['ForwardEps'])) * 100

        # 30일 전 기준 (25~40일 사이 가장 오래된 값)
        ref_30d = ticker_hist[(ticker_hist['Date'] >= today_date - timedelta(days=40)) &
                              (ticker_hist['Date'] <= today_date - timedelta(days=25))]
        if not ref_30d.empty and abs(ref_30d.iloc[0]['ForwardEps']) > 0.01:
            rev_30d_map[ticker] = ((current_eps - ref_30d.iloc[0]['ForwardEps']) / abs(ref_30d.iloc[0]['ForwardEps'])) * 100

    today_df['Revision_7D'] = today_df['Ticker'].map(rev_7d_map).fillna(0.0)
    today_df['Revision_30D'] = today_df['Ticker'].map(rev_30d_map).fillna(0.0)

    active_count = (today_df['Revision_7D'] != 0).sum() + (today_df['Revision_30D'] != 0).sum()
    print(f"📊 리비전 트래킹 활성: 7D {(today_df['Revision_7D'] != 0).sum()}개 / 30D {(today_df['Revision_30D'] != 0).sum()}개 종목 계산 완료")

    return today_df

# ==========================================
# 4. 복기 및 오답노트 (동태적 MDD 페널티 채점)
# ==========================================
def manage_historical_data(today_df):
    if os.path.exists(HISTORY_FILE):
        try:
            history_df = pd.read_csv(HISTORY_FILE)
            history_df['Date'] = pd.to_datetime(history_df['Date'], format='mixed', errors='coerce')
        except: history_df = pd.DataFrame()
    else: history_df = pd.DataFrame()

    if not history_df.empty:
        history_df = history_df.dropna(subset=['Date'])
        today_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
        tickers_to_update = history_df[history_df['Target'].isna() & ((today_date - history_df['Date']).dt.days >= 20)]

        if not tickers_to_update.empty:
            t_list = tickers_to_update['Ticker'].unique().tolist()
            print(f"📝 채점 대상: {len(t_list)}개 티커, {len(tickers_to_update)}행")
            
            # 배치 다운로드 (50개씩 끊어서 타임아웃 방지)
            all_close = {}
            for i in range(0, len(t_list), 50):
                batch = t_list[i:i+50]
                try:
                    hist_data = yf.download(batch, period="3mo", progress=False, show_errors=False)
                    
                    # MultiIndex 컬럼 대응
                    if isinstance(hist_data.columns, pd.MultiIndex):
                        close_data = hist_data['Close']
                    elif 'Close' in hist_data.columns:
                        close_data = hist_data[['Close']]
                        close_data.columns = [batch[0]] if len(batch) == 1 else close_data.columns
                    else:
                        continue
                    
                    if isinstance(close_data, pd.Series):
                        close_data = close_data.to_frame(name=batch[0])
                    
                    # timezone 제거 (핵심 버그 수정)
                    if close_data.index.tz is not None:
                        close_data.index = close_data.index.tz_localize(None)
                    
                    for col in close_data.columns:
                        all_close[col] = close_data[col].dropna()
                        
                except Exception as e:
                    print(f"  ⚠️ 배치 {i}~{i+len(batch)} 다운로드 실패: {e}")
            
            # 채점 실행
            scored = 0
            for idx, row in tickers_to_update.iterrows():
                ticker = row['Ticker']
                buy_date = pd.to_datetime(row['Date'])
                buy_price = row['Close']
                
                if ticker not in all_close:
                    continue
                    
                period_data = all_close[ticker]
                period_data = period_data[period_data.index >= buy_date]
                
                if not period_data.empty:
                    min_px = float(period_data.min())
                    last_px = float(period_data.iloc[-1])
                    mdd_pct = max(0, ((buy_price - min_px) / buy_price) * 100)
                    ret_pct = ((last_px - buy_price) / buy_price) * 100
                    risk_adjusted_score = ret_pct - (mdd_pct * 1.5)
                    history_df.at[idx, 'Target'] = risk_adjusted_score
                    scored += 1
                    
            print(f"✅ 채점 완료: {scored}행 / {len(tickers_to_update)}행")

    today_df['Date'] = pd.to_datetime(today_df['Date'])
    updated_history = pd.concat([history_df, today_df], ignore_index=True).drop_duplicates(subset=['Date', 'Ticker'], keep='last')
    updated_history.to_csv(HISTORY_FILE, index=False)
    return updated_history

# ==========================================
# 5. 동태적 학습 필터 (리비전 + 52주 신고가 반영)
# ==========================================
def dynamic_ml_filter(history_df, today_df):
    train_data = history_df.dropna(subset=['Target'])

    has_real_revision = (today_df['Revision_7D'] != 0).any() or (today_df['Revision_30D'] != 0).any()
    if has_real_revision:
        print("✅ 진짜 리비전 데이터 사용 중 (선행적)")
    else:
        print("⏳ 리비전 데이터 축적 중 — EPS 스프레드(폴백)로 대체")

    # --- 룰 기반 점수 ---
    features = ['Trend_OK', 'Revision_Strength', 'Mom_3M']

    rev_norm = np.clip(today_df['Revision_Strength'] / 100, -1, 1)
    mom_norm = np.clip(today_df['Mom_3M'] / 50, 0, 1)

    # 기존 가중치 원본 유지
    today_df['Rule_Prob'] = (
        today_df['Trend_OK'] * 0.50 +
        rev_norm * 0.25 +
        mom_norm * 0.25
    )
    today_df['Rule_Score'] = today_df['Rule_Prob'].rank(pct=True) * 100

    # --- ML 점수 ---
    if len(train_data) < 100:
        today_df['ML_Score'] = 0.0
    else:
        # history에 없을 수 있는 컬럼 대비
        ml_features = ['Trend_OK', 'Revision_Strength', 'Mom_3M']
        available_features = [f for f in ml_features if f in train_data.columns]

        if len(available_features) >= 2:
            clf = HistGradientBoostingRegressor(random_state=42).fit(
                train_data[available_features].fillna(0), train_data['Target']
            )
            today_df['ML_Pred'] = clf.predict(today_df[available_features].fillna(0))
            today_df['ML_Score'] = today_df['ML_Pred'].rank(pct=True) * 100
        else:
            today_df['ML_Score'] = 0.0

    return today_df

# ==========================================
# 6. 텔레그램 발송 (리비전 상태 표시 추가)
# ==========================================
def send_telegram(df, has_real_revision):
    if df.empty: return
    top_n = 10
    today_str = datetime.now().strftime("%Y-%m-%d")

    rev_status = "📡 실시간 리비전" if has_real_revision else "⏳ 리비전 축적중(폴백)"
    msg = f"💎 *{today_str} 미장 주도주 스캐너* 💎\n{rev_status}\n\n"

    # --- [트랙 1] 기본 룰 랭킹 ---
    rule_df = df.sort_values('Rule_Score', ascending=False).head(top_n)
    msg += "🏆 *[기본 룰 랭킹]*\n"

    for i, (_, row) in enumerate(rule_df.iterrows(), 1):
        is_target = (row['Rule_Score'] >= 90.0) and (row['Trend_OK'] == 1) and (99 <= row['MA20_Disparity'] <= 105)
        mark = "🎯" if is_target else "✅"

        msg += f"*{i}. {row['Name']}* ({row['Ticker']}) {mark}\n"
        msg += f"📊 북 점수: {row['Rule_Score']:.1f}점\n"

        trend_str = "정배열" if row['Trend_OK'] == 1 else "역배열"

        # 리비전 표시: 진짜 리비전이면 7D/30D, 아니면 기존 방식
        if has_real_revision and (row['Revision_7D'] != 0 or row['Revision_30D'] != 0):
            rev_str = f"리비전 7D {row['Revision_7D']:+.1f}% / 30D {row['Revision_30D']:+.1f}%"
        else:
            rev_str = f"EPS스프레드 {row['EPS_Growth']:.1f}%"

        msg += f"🧾 {rev_str} | RS {row['RS_Rating']:.1f}% | 수급 {row['Volume_Breakout']:.1f}x | 52W고 {row['High_52W_Pct']:.1f}% | 이격도 {row['MA20_Disparity']:.1f}% ({trend_str})\n\n"

    msg += "---\n\n"

    # --- [트랙 2] AI 동태적 학습 랭킹 ---
    ml_ready = df['ML_Score'].max() > 0
    msg += "🤖 *[AI 동태적 MDD 학습 랭킹]*\n"

    if ml_ready:
        ml_df = df.sort_values('ML_Score', ascending=False).head(top_n)
        for i, (_, row) in enumerate(ml_df.iterrows(), 1):
            msg += f"*{i}. {row['Name']}* ({row['Ticker']})\n"
            msg += f"📊 AI 점수: {row['ML_Score']:.1f}점 (룰: {row['Rule_Score']:.1f}점)\n"

            trend_str = "정배열" if row['Trend_OK'] == 1 else "역배열"
            if has_real_revision and (row['Revision_7D'] != 0 or row['Revision_30D'] != 0):
                rev_str = f"리비전 7D {row['Revision_7D']:+.1f}% / 30D {row['Revision_30D']:+.1f}%"
            else:
                rev_str = f"EPS스프레드 {row['EPS_Growth']:.1f}%"

            msg += f"🧾 {rev_str} | RS {row['RS_Rating']:.1f}% | 수급 {row['Volume_Breakout']:.1f}x | 52W고 {row['High_52W_Pct']:.1f}% | 이격도 {row['MA20_Disparity']:.1f}% ({trend_str})\n\n"
    else:
        msg += "⏳ _아직 과거 20일 복기 데이터(100개)를 수집 및 채점 중입니다. 며칠 후 AI 랭킹이 활성화됩니다._\n"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})

# ==========================================
# 7. 메인 실행
# ==========================================
if __name__ == "__main__":
    universe = get_broad_universe()

    if not universe:
        error_msg = "🚨 *시스템 알림*\n외부 티커 수집 서버에 일시적 문제가 발생하여 오늘 AI 스캔을 건너뜁니다."
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={'chat_id': TELEGRAM_CHAT_ID, 'text': error_msg, 'parse_mode': 'Markdown'})
        print("티커 수집 실패로 작업을 종료합니다.")
        exit()

    spy_ret_3m = get_market_baseline()

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        start_dt = datetime.now() - timedelta(days=180)
        end_dt = datetime.now()
        futures = {executor.submit(fetch_evidence, t, start_dt, end_dt, spy_ret_3m): t for t in universe}
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)

    today_df = pd.DataFrame(results)
    if not today_df.empty:
        # 🆕 리비전 자동 트래킹 & 계산
        today_df = track_and_compute_revision(today_df)

        has_real_revision = (today_df['Revision_7D'] != 0).any() or (today_df['Revision_30D'] != 0).any()

        # Revision_Strength를 저장 전에 확정 (ML 학습 데이터 일관성)
        if has_real_revision:
            today_df['Revision_Strength'] = today_df['Revision_7D'] * 0.6 + today_df['Revision_30D'] * 0.4
        else:
            today_df['Revision_Strength'] = today_df['EPS_Growth']

        history_df = manage_historical_data(today_df)
        ranked_df = dynamic_ml_filter(history_df, today_df)
        send_telegram(ranked_df, has_real_revision)
