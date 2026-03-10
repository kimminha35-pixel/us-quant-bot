import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
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
MAX_WORKERS = 12 
HISTORY_FILE = 'history.csv'

EXCLUDED_SECTORS = ['Financial Services', 'Utilities', 'Real Estate', 'Basic Materials']

def get_broad_universe():
    """S&P 1500 전체 유니버스를 가져오고, 실패하면 그냥 빈 리스트 반환(쉬기)"""
    print("S&P 1500 유니버스 동적 스크래핑 중...")
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
        sp400 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]['Symbol']
        sp600 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0]['Symbol']
        universe = pd.concat([sp500, sp400, sp600]).dropna().unique().tolist()
        return [t.replace('.', '-') for t in universe]
    except:
        print("위키피디아 연결 실패. 오늘은 쉽니다.")
        return [] # 빈 리스트 반환 -> 분석 스킵

# [fetch_evidence, manage_historical_data 함수는 이전과 동일]
def fetch_evidence(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        if sector in EXCLUDED_SECTORS: return None
        
        name = info.get('shortName', ticker)
        hist = stock.history(start=start_date, end=end_date)
        if len(hist) < 60: return None
            
        close_px = hist['Close'].iloc[-1]
        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
        ma60 = hist['Close'].rolling(60).mean().iloc[-1]
        trend_ok = 1 if (close_px > ma20 > ma60) else 0
        
        eps_surp = info.get('earningsQuarterlyGrowth', 0) * 100
        eps_trl = info.get('trailingEps', 0.1)
        eps_fwd = info.get('forwardEps', 0)
        revision = ((eps_fwd - eps_trl) / abs(eps_trl)) * 100
        
        price_3m = hist['Close'].iloc[-63]
        mom_3m = ((close_px / price_3m) - 1) * 100
        
        return {
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'Ticker': ticker, 'Name': name, 'Sector': sector, 'Close': close_px,
            'Trend_OK': trend_ok, 'EPS_Surprise': eps_surp, 'Revision_Strength': revision,
            'Mom_3M': mom_3m, 'Target': np.nan
        }
    except: return None

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
                if (today_date - row['Date']).days >= 20:
                    ticker = row['Ticker']
                    if ticker in today_prices:
                        ret = (today_prices[ticker] / row['Close']) - 1
                        history_df.at[idx, 'Target'] = 1 if ret > 0.05 else 0

    updated_history = pd.concat([history_df, today_df], ignore_index=True).drop_duplicates(subset=['Date', 'Ticker'], keep='last')
    updated_history.to_csv(HISTORY_FILE, index=False)
    return updated_history

def dynamic_ml_filter(history_df, today_df):
    train_data = history_df.dropna(subset=['Target'])
    features = ['Trend_OK', 'EPS_Surprise', 'Revision_Strength', 'Mom_3M']
    
    if len(train_data) < 100:
        rev_norm = np.clip(today_df['Revision_Strength'] / 100, 0, 1)
        mom_norm = np.clip(today_df['Mom_3M'] / 50, 0, 1)
        today_df['Raw_Prob'] = (today_df['Trend_OK'] * 0.5) + (rev_norm * 0.25) + (mom_norm * 0.25)
    else:
        clf = HistGradientBoostingClassifier(random_state=42).fit(train_data[features].fillna(0), train_data['Target'].astype(int))
        today_df['Raw_Prob'] = clf.predict_proba(today_df[features].fillna(0))[:, 1]
    
    today_df['AI_Score'] = today_df['Raw_Prob'].rank(pct=True) * 100
    return today_df.sort_values('AI_Score', ascending=False)

def send_telegram(df):
    if df.empty: return
    top_n = min(7, len(df))
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    msg = f"🌟 *{today_str} AI 주도주 랭킹* 🌟\n\n"
    for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        is_perfect = (row['AI_Score'] >= 90.0) and (row['Trend_OK'] == 1)
        mark = "🚀" if is_perfect else "⚠️"
        msg += f"*{i}. {row['Name']}* ({row['Ticker']}) {mark}\n"
        msg += f"🎯 *AI 점수:* {row['AI_Score']:.1f}점\n"
        conds = []
        if row['Revision_Strength'] > 10: conds.append(f"리비전(+{row['Revision_Strength']:.0f}%)")
        if row['EPS_Surprise'] > 10: conds.append(f"어닝서프({row['EPS_Surprise']:.0f}%)")
        if row['Mom_3M'] > 15: conds.append(f"강력시세(+{row['Mom_3M']:.0f}%)")
        conds.append("정배열" if row['Trend_OK'] == 1 else "역배열(주의)")
        msg += f"🧾 *상태:* {', '.join(conds)}\n\n"
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})

if __name__ == "__main__":
    universe = get_broad_universe()
    if not universe:
        print("분석할 유니버스가 없습니다. 종료합니다.")
    else:
        results = []
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
