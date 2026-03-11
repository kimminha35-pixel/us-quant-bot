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

EXCLUDED_SECTORS = ['Financial Services', 'Utilities', 'Real Estate', 'Basic Materials']

def get_broad_universe():
    print("🚀 미국 전종목 티커 수집 시작...")
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        response = requests.get(url)
        if response.status_code == 200:
            tickers = response.text.split('\n')
            tickers = [t.strip().replace('.', '-') for t in tickers if t.strip() and len(t.strip()) <= 5]
            print(f"✅ 총 {len(tickers)}개 티커 확보!")
            return tickers
    except Exception as e:
        print(f"❌ 수집 실패: {e}")
        return ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "META", "AMZN", "PLTR", "AVGO", "SMCI"]

def get_market_baseline():
    try:
        spy = yf.Ticker("SPY").history(period="6mo")
        if len(spy) > 63:
            return ((spy['Close'].iloc[-1] / spy['Close'].iloc[-63]) - 1) * 100
    except: pass
    return 0.0

# ==========================================
# 2. 증거 수집
# ==========================================
def fetch_evidence(ticker, start_date, end_date, spy_ret_3m):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if info.get('marketCap', 0) < 750_000_000: return None
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
        revision = ((eps_fwd - eps_trl) / abs(eps_trl)) * 100 if eps_trl != 0 else 0
        
        price_3m = hist['Close'].iloc[-63]
        mom_3m = ((close_px / price_3m) - 1) * 100
        
        ma20_disparity = (close_px / ma20) * 100 if ma20 > 0 else 100
        rs_rating = mom_3m - spy_ret_3m
        vol_5d = hist['Volume'].iloc[-5:].mean()
        vol_60d = hist['Volume'].iloc[-60:].mean()
        vol_breakout = (vol_5d / vol_60d) if vol_60d > 0 else 1.0
        
        return {
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'Ticker': ticker, 'Name': name, 'Sector': sector, 'Close': close_px,
            'Trend_OK': trend_ok, 'Revision_Strength': revision, 'Mom_3M': mom_3m,
            'MA20_Disparity': ma20_disparity, 'RS_Rating': rs_rating, 'Volume_Breakout': vol_breakout,
            'Target': np.nan
        }
    except: return None

# ==========================================
# 3. 복기 및 오답노트 (동태적 MDD 페널티 채점)
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
            try:
                hist_data = yf.download(t_list, period="1mo", progress=False, show_errors=False)
                close_data = hist_data['Close'] if 'Close' in hist_data else hist_data
                if isinstance(close_data, pd.Series): close_data = close_data.to_frame(name=t_list[0])
                
                for idx, row in tickers_to_update.iterrows():
                    ticker = row['Ticker']
                    buy_date = pd.to_datetime(row['Date'])
                    buy_price = row['Close']
                    
                    if ticker in close_data.columns:
                        period_data = close_data[ticker].dropna()
                        period_data = period_data[period_data.index.tz_localize(None) >= buy_date]
                        
                        if not period_data.empty:
                            min_px = float(period_data.min())
                            last_px = float(period_data.iloc[-1])
                            
                            mdd_pct = max(0, ((buy_price - min_px) / buy_price) * 100) 
                            ret_pct = ((last_px - buy_price) / buy_price) * 100 
                            risk_adjusted_score = ret_pct - (mdd_pct * 1.5)
                            history_df.at[idx, 'Target'] = risk_adjusted_score
                            
            except Exception as e:
                print(f"채점 중 에러: {e}")

    today_df['Date'] = pd.to_datetime(today_df['Date'])
    updated_history = pd.concat([history_df, today_df], ignore_index=True).drop_duplicates(subset=['Date', 'Ticker'], keep='last')
    updated_history.to_csv(HISTORY_FILE, index=False)
    return updated_history

# ==========================================
# 4. 동태적 학습 필터 (독립적 점수 산출)
# ==========================================
def dynamic_ml_filter(history_df, today_df):
    train_data = history_df.dropna(subset=['Target'])
    features = ['Trend_OK', 'Revision_Strength', 'Mom_3M']
    
    rev_norm = np.clip(today_df['Revision_Strength'] / 100, 0, 1)
    mom_norm = np.clip(today_df['Mom_3M'] / 50, 0, 1)
    today_df['Rule_Prob'] = (today_df['Trend_OK'] * 0.5) + (rev_norm * 0.25) + (mom_norm * 0.25)
    today_df['Rule_Score'] = today_df['Rule_Prob'].rank(pct=True) * 100
    
    if len(train_data) < 100:
        today_df['ML_Score'] = 0.0 
    else:
        clf = HistGradientBoostingRegressor(random_state=42).fit(train_data[features].fillna(0), train_data['Target'])
        today_df['ML_Pred'] = clf.predict(today_df[features].fillna(0))
        today_df['ML_Score'] = today_df['ML_Pred'].rank(pct=True) * 100
        
    return today_df

# ==========================================
# 5. 텔레그램 투트랙 발송 (상위 7개씩 출력)
# ==========================================
def send_telegram(df):
    if df.empty: return
    top_n = 7 # 🌟 상위 7개씩 출력으로 변경
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    msg = f"💎 *{today_str} 미장 주도주 스캐너* 💎\n\n"
    
    # --- [트랙 1] 기본 룰 랭킹 ---
    rule_df = df.sort_values('Rule_Score', ascending=False).head(top_n)
    msg += "🏆 *[기본 룰 랭킹]* (회원님 알고리즘 100%)\n"
    
    for i, (_, row) in enumerate(rule_df.iterrows(), 1):
        is_target = (row['Rule_Score'] >= 90.0) and (row['Trend_OK'] == 1) and (99 <= row['MA20_Disparity'] <= 105)
        mark = "🎯" if is_target else "✅"
        
        msg += f"*{i}. {row['Name']}* ({row['Ticker']}) {mark}\n"
        msg += f"📊 룰 점수: {row['Rule_Score']:.1f}점\n"
        
        trend_str = "정배열" if row['Trend_OK'] == 1 else "역배열"
        msg += f"🧾 리비전 {row['Revision_Strength']:.1f}% | RS {row['RS_Rating']:.1f}% | 수급 {row['Volume_Breakout']:.1f}x | 이격도 {row['MA20_Disparity']:.1f}% ({trend_str})\n\n"
        
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
            msg += f"🧾 리비전 {row['Revision_Strength']:.1f}% | RS {row['RS_Rating']:.1f}% | 수급 {row['Volume_Breakout']:.1f}x | 이격도 {row['MA20_Disparity']:.1f}% ({trend_str})\n\n"
    else:
        msg += "⏳ _아직 과거 20일 복기 데이터(100개)를 수집 및 채점 중입니다. 며칠 후 AI 랭킹이 활성화됩니다._\n"
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})

if __name__ == "__main__":
    universe = get_broad_universe()
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
        history_df = manage_historical_data(today_df)
        ranked_df = dynamic_ml_filter(history_df, today_df)
        send_telegram(ranked_df)
