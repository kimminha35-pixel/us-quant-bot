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

# ==========================================
# 2. 증거 수집 (과거 Q4 데이터 배제, 미래 Revision 집중)
# ==========================================
def fetch_evidence(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        mkt_cap = info.get('marketCap', 0)
        if mkt_cap < 750_000_000: return None # 시총 1조 원 이상
        
        sector = info.get('sector', 'Unknown')
        if sector in EXCLUDED_SECTORS: return None
        
        name = info.get('shortName', ticker)
        hist = stock.history(start=start_date, end=end_date)
        if len(hist) < 60: return None
            
        close_px = hist['Close'].iloc[-1]
        
        # 추세: 20일선 > 60일선
        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
        ma60 = hist['Close'].rolling(60).mean().iloc[-1]
        trend_ok = 1 if (close_px > ma20 > ma60) else 0
        
        # [핵심] 1분기 서프라이즈 기대감은 '연간 Forward EPS'의 급격한 상향으로 나타납니다.
        eps_trl = info.get('trailingEps', 0.1)
        eps_fwd = info.get('forwardEps', 0)
        revision = ((eps_fwd - eps_trl) / abs(eps_trl)) * 100 if eps_trl != 0 else 0
        
        price_3m = hist['Close'].iloc[-63]
        mom_3m = ((close_px / price_3m) - 1) * 100
        
        return {
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'Ticker': ticker, 'Name': name, 'Sector': sector, 'Close': close_px,
            'Trend_OK': trend_ok, 'Revision_Strength': revision,
            'Mom_3M': mom_3m, 'Target': np.nan
        }
    except: return None

# ==========================================
# 3. 데이터 관리 (리스크 관리: -12% 이탈 시 오답 처리)
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
                            max_px = float(period_data.max())
                            last_px = float(period_data.iloc[-1])
                            
                            if min_px <= buy_price * 0.88:  # 추세 완전 이탈 (-12%)
                                history_df.at[idx, 'Target'] = 0
                            elif max_px >= buy_price * 1.15 or last_px >= buy_price * 1.08: 
                                history_df.at[idx, 'Target'] = 1
                            else: 
                                history_df.at[idx, 'Target'] = 0
            except Exception as e:
                print(f"채점 중 에러: {e}")

    today_df['Date'] = pd.to_datetime(today_df['Date'])
    updated_history = pd.concat([history_df, today_df], ignore_index=True).drop_duplicates(subset=['Date', 'Ticker'], keep='last')
    updated_history.to_csv(HISTORY_FILE, index=False)
    return updated_history

# ==========================================
# 4. 동태적 학습 필터
# ==========================================
def dynamic_ml_filter(history_df, today_df):
    train_data = history_df.dropna(subset=['Target'])
    features = ['Trend_OK', 'Revision_Strength', 'Mom_3M']
    
    if len(train_data) < 100:
        rev_norm = np.clip(today_df['Revision_Strength'] / 100, 0, 1)
        mom_norm = np.clip(today_df['Mom_3M'] / 50, 0, 1)
        # 과거 실적 변수 삭제, 순수하게 추세와 추정치 상향에만 집중
        today_df['Raw_Prob'] = (today_df['Trend_OK'] * 0.5) + (rev_norm * 0.25) + (mom_norm * 0.25)
    else:
        clf = HistGradientBoostingClassifier(random_state=42).fit(train_data[features].fillna(0), train_data['Target'].astype(int))
        today_df['Raw_Prob'] = clf.predict_proba(today_df[features].fillna(0))[:, 1]
    
    today_df['AI_Score'] = today_df['Raw_Prob'].rank(pct=True) * 100
    return today_df.sort_values('AI_Score', ascending=False)

# ==========================================
# 5. 텔레그램 발송
# ==========================================
def send_telegram(df):
    if df.empty: return
    top_n = min(7, len(df))
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    msg = f"💎 *{today_str} 미장 주도주 AI 리포트* 💎\n"
    msg += "_(시총 1조 이상 전수조사)_\n\n"
    
    for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        is_star = (row['AI_Score'] >= 90.0) and (row['Trend_OK'] == 1)
        mark = "🚀" if is_star else "⚠️"
        
        msg += f"*{i}. {row['Name']}* ({row['Ticker']}) {mark}\n"
        msg += f"🎯 *AI 랭킹:* {row['AI_Score']:.1f}점\n"
        
        evidences = []
        if row['Revision_Strength'] > 15: evidences.append(f"강력 연간리비전(+{row['Revision_Strength']:.0f}%)")
        elif row['Revision_Strength'] > 0: evidences.append(f"추정치상향(+{row['Revision_Strength']:.0f}%)")
        
        if row['Mom_3M'] > 20: evidences.append(f"강력모멘텀(+{row['Mom_3M']:.0f}%)")
        evidences.append("정배열" if row['Trend_OK'] == 1 else "역배열(주의)")
        
        msg += f"🧾 *상태:* {', '.join(evidences)}\n\n"
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})

if __name__ == "__main__":
    universe = get_broad_universe()
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
