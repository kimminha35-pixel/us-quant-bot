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
# 1. 설정 및 필터링
# ==========================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '').strip()
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
MAX_WORKERS = 10  # 안정성을 위해 10으로 조정
HISTORY_FILE = 'history.csv'

EXCLUDED_SECTORS = ['Financial Services', 'Utilities', 'Real Estate', 'Basic Materials']
EXCLUDED_INDUSTRIES = ['Banks - Regional', 'Banks - Diversified']

def get_broad_universe():
    print("S&P 1500 유니버스 동적 스크래핑 중...")
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
        sp400 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]['Symbol']
        sp600 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0]['Symbol']
        universe = pd.concat([sp500, sp400, sp600]).dropna().unique().tolist()
        return [t.replace('.', '-') for t in universe]
    except:
        return ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "META", "AMZN", "PLTR"]

# ==========================================
# 2. 증거 수집 (이름 데이터 추가)
# ==========================================
def fetch_evidence(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        sector = info.get('sector', 'Unknown')
        if sector in EXCLUDED_SECTORS: return None
        
        # [핵심] 종목 이름(Name) 명시적 추출
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
            'Ticker': ticker,
            'Name': name,
            'Sector': sector,
            'Close': close_px,
            'Trend_OK': trend_ok,
            'EPS_Surprise': eps_surp,
            'Revision_Strength': revision,
            'Mom_3M': mom_3m,
            'Target': np.nan
        }
    except:
        return None

# ==========================================
# 3. 기억력 관리 (History 누적)
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
                if (today_date - row['Date']).days >= 20:
                    ticker = row['Ticker']
                    if ticker in today_prices:
                        ret = (today_prices[ticker] / row['Close']) - 1
                        history_df.at[idx, 'Target'] = 1 if ret > 0.05 else 0

    updated_history = pd.concat([history_df, today_df], ignore_index=True).drop_duplicates(subset=['Date', 'Ticker'], keep='last')
    updated_history.to_csv(HISTORY_FILE, index=False)
    return updated_history

# ==========================================
# 4. 동태적 학습 및 상대 평가
# ==========================================
def dynamic_ml_filter(history_df, today_df):
    train_data = history_df.dropna(subset=['Target'])
    features = ['Trend_OK', 'EPS_Surprise', 'Revision_Strength', 'Mom_3M']
    
    if len(train_data) < 100:
        # 초기 단계: 기본 가중치 + 로직 학습
        mom_threshold = today_df['Mom_3M'].quantile(0.7)
        y_temp = np.where((today_df['Trend_OK'] == 1) & (today_df['Revision_Strength'] > 0) & (today_df['Mom_3M'] > mom_threshold), 1, 0)
        clf = HistGradientBoostingClassifier(random_state=42).fit(today_df[features].fillna(0), y_temp)
    else:
        clf = HistGradientBoostingClassifier(random_state=42).fit(train_data[features].fillna(0), train_data['Target'].astype(int))
        
    today_df['Raw_Prob'] = clf.predict_proba(today_df[features].fillna(0))[:, 1]
    today_df['AI_Score'] = today_df['Raw_Prob'].rank(pct=True) * 100
    return today_df.sort_values('AI_Score', ascending=False)

# ==========================================
# 5. 텔레그램 발송 (가독성 개선)
# ==========================================
def send_telegram(df):
    if df.empty: return
    top_n = min(7, len(df))
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    msg = f"🔥 *{today_str} 미장 주도주 AI 랭킹* 🔥\n\n"
    for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        is_perfect = (row['AI_Score'] >= 95.0) and (row['Trend_OK'] == 1)
        mark = "✅" if is_perfect else "⚠️"
        
        msg += f"*{i}. {row['Name']}* ({row['Ticker']}) {mark}\n"
        msg += f"🎯 *상대 랭킹:* {row['AI_Score']:.1f}점 (확률: {row['Raw_Prob']*100:.0f}%)\n"
        
        evidences = []
        if row['Revision_Strength'] > 15: evidences.append(f"가이던스 대폭상향(+{row['Revision_Strength']:.0f}%)")
        elif row['Revision_Strength'] > 0: evidences.append("추정치 상향중")
        if row['EPS_Surprise'] > 10: evidences.append(f"어닝서프({row['EPS_Surprise']:.0f}%)")
        if row['Mom_3M'] > 20: evidences.append(f"강력모멘텀(+{row['Mom_3M']:.0f}%)")
        evidences.append("정배열" if row['Trend_OK'] == 1 else "역배열")
        
        msg += f"🧾 *상태:* {', '.join(evidences)}\n\n"
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})

if __name__ == "__main__":
    universe = get_broad_universe()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_evidence, t, datetime.now()-timedelta(days=120), datetime.now()): t for t in universe}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    today_df = pd.DataFrame(results)
    if not today_df.empty:
        history_df = manage_historical_data(today_df)
        ranked_df = dynamic_ml_filter(history_df, today_df)
        send_telegram(ranked_df)
