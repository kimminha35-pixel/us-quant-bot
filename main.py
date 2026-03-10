import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import time
from datetime import datetime, timedelta
from sklearn.ensemble import HistGradientBoostingClassifier
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

# 제외 섹터: 금융, 유틸리티, 리츠, 소재 (주도주 성격이 약함)
EXCLUDED_SECTORS = ['Financial Services', 'Utilities', 'Real Estate', 'Basic Materials']

def get_broad_universe():
    print("🚀 미국 전종목 티커(NASDAQ/NYSE/AMEX) 수집 시작...")
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        response = requests.get(url)
        if response.status_code == 200:
            tickers = response.text.split('\n')
            # 티커 정제 (공백 제거, 점을 하이픈으로 변경)
            tickers = [t.strip().replace('.', '-') for t in tickers if t.strip() and len(t.strip()) <= 5]
            print(f"✅ 총 {len(tickers)}개 티커 확보!")
            return tickers
        else: raise Exception("티커 서버 응답 없음")
    except Exception as e:
        print(f"❌ 수집 실패: {e}")
        return ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "META", "AMZN", "PLTR", "AVGO", "SMCI"]

# ==========================================
# 2. 증거 수집 (시총 1조 이상 필터링)
# ==========================================
def fetch_evidence(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 🌟 [핵심] 시가총액 필터: 1조 원 이상 (약 7.5억 달러 기준)
        # 1,000,000,000,000 KRW ≒ 750,000,000 USD
        mkt_cap = info.get('marketCap', 0)
        if mkt_cap < 750_000_000: return None
        
        sector = info.get('sector', 'Unknown')
        if sector in EXCLUDED_SECTORS: return None
        
        name = info.get('shortName', ticker)
        hist = stock.history(start=start_date, end=end_date)
        if len(hist) < 60: return None
            
        close_px = hist['Close'].iloc[-1]
        
        # 추세 증거: 20일선 > 60일선 (정배열)
        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
        ma60 = hist['Close'].rolling(60).mean().iloc[-1]
        trend_ok = 1 if (close_px > ma20 > ma60) else 0
        
        # 실적 증거: 전년비 분기 성장률(서프) 및 가이던스(리비전)
        eps_surp = info.get('earningsQuarterlyGrowth', 0) * 100
        eps_trl = info.get('trailingEps', 0.1)
        eps_fwd = info.get('forwardEps', 0)
        revision = ((eps_fwd - eps_trl) / abs(eps_trl)) * 100 if eps_trl != 0 else 0
        
        # 수급 증거: 최근 3개월 수익률(모멘텀)
        price_3m = hist['Close'].iloc[-63]
        mom_3m = ((close_px / price_3m) - 1) * 100
        
        return {
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'Ticker': ticker, 'Name': name, 'Sector': sector, 'Close': close_px,
            'Trend_OK': trend_ok, 'EPS_Surprise': eps_surp, 'Revision_Strength': revision,
            'Mom_3M': mom_3m, 'Target': np.nan
        }
    except: return None

# ==========================================
# 3. 데이터 관리 및 학습 (History)
# ==========================================
def manage_historical_data(today_df):
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
    else:
        history_df = pd.DataFrame()
        
    if not history_df.empty:
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        today_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
        today_prices = today_df.set_index('Ticker')['Close'].to_dict()
        
        for idx, row in history_df.iterrows():
            if pd.isna(row.get('Target')):
                # 20거래일(약 한 달) 뒤 결과 채점
                if (today_date - row['Date']).days >= 20:
                    ticker = row['Ticker']
                    if ticker in today_prices:
                        ret = (today_prices[ticker] / row['Close']) - 1
                        history_df.at[idx, 'Target'] = 1 if ret > 0.05 else 0

    updated_history = pd.concat([history_df, today_df], ignore_index=True).drop_duplicates(subset=['Date', 'Ticker'], keep='last')
    updated_history.to_csv(HISTORY_FILE, index=False)
    return updated_history

# ==========================================
# 4. 동태적 분석 및 AI 스코어링
# ==========================================
def dynamic_ml_filter(history_df, today_df):
    train_data = history_df.dropna(subset=['Target'])
    features = ['Trend_OK', 'EPS_Surprise', 'Revision_Strength', 'Mom_3M']
    
    # 학습 데이터 부족 시(첫 한 달) 가중치 기반 우선 순위
    if len(train_data) < 100:
        rev_norm = np.clip(today_df['Revision_Strength'] / 100, 0, 1)
        mom_norm = np.clip(today_df['Mom_3M'] / 50, 0, 1)
        today_df['Raw_Prob'] = (today_df['Trend_OK'] * 0.5) + (rev_norm * 0.25) + (mom_norm * 0.25)
    else:
        # 데이터가 쌓이면 정답(Target) 기반 머신러닝 가동
        clf = HistGradientBoostingClassifier(random_state=42).fit(train_data[features].fillna(0), train_data['Target'].astype(int))
        today_df['Raw_Prob'] = clf.predict_proba(today_df[features].fillna(0))[:, 1]
    
    # 시장 내 상대 점수 (백분위 0~100점)
    today_df['AI_Score'] = today_df['Raw_Prob'].rank(pct=True) * 100
    return today_df.sort_values('AI_Score', ascending=False)

# ==========================================
# 5. 텔레그램 리포트 발송
# ==========================================
def send_telegram(df):
    if df.empty: return
    top_n = min(7, len(df))
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    msg = f"💎 *{today_str} 미장 주도주 AI 리포트* 💎\n"
    msg += "_(시총 1조 이상 전수조사 결과)_\n\n"
    
    for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        is_star = (row['AI_Score'] >= 90.0) and (row['Trend_OK'] == 1)
        mark = "🚀" if is_star else "⚠️"
        
        msg += f"*{i}. {row['Name']}* ({row['Ticker']}) {mark}\n"
        msg += f"🎯 *AI 점수:* {row['AI_Score']:.1f}점\n"
        
        evidences = []
        if row['Revision_Strength'] > 10: evidences.append(f"리비전(+{row['Revision_Strength']:.0f}%)")
        if row['EPS_Surprise'] > 10: evidences.append(f"어닝서프({row['EPS_Surprise']:.0f}%)")
        if row['Mom_3M'] > 15: evidences.append(f"강력시세(+{row['Mom_3M']:.0f}%)")
        evidences.append("정배열" if row['Trend_OK'] == 1 else "역배열(주의)")
        
        msg += f"🧾 *상태:* {', '.join(evidences)}\n\n"
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})

if __name__ == "__main__":
    universe = get_broad_universe()
    results = []
    # 4,000개 이상 스캔을 위해 멀티스레딩 활용
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_evidence, t, datetime.now()-timedelta(days=120), datetime.now()): t for t in universe}
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)
            
    today_df = pd.DataFrame(results)
    if not today_df.empty:
        history_df = manage_historical_data(today_df)
        ranked_df = dynamic_ml_filter(history_df, today_df)
        send_telegram(ranked_df)
