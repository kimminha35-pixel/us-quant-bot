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
    
    # 🟢 [Plan A] 기존 고속 GitHub 리스트
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

    # 🟡 [Plan B] 미국 SEC(증권거래위원회) 공식 데이터베이스 (방어 로직)
    print("🔄 [Plan B 가동] 미국 SEC 공식 데이터베이스에서 직접 추출합니다...")
    try:
        # SEC 서버는 봇 접근을 막을 수 있으므로 브라우저인 척 위장
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        url_b = "https://www.sec.gov/files/company_tickers.json"
        res_b = requests.get(url_b, headers=headers, timeout=10)
        
        if res_b.status_code == 200:
            data = res_b.json()
            # SEC 데이터에서 티커만 추출 후 중복 제거
            tickers = [v['ticker'].replace('.', '-') for v in data.values()]
            tickers = list(set(tickers)) 
            print(f"✅ [Plan B 성공] SEC 공식 티커 {len(tickers)}개 확보!")
            return tickers
    except Exception as e:
        print(f"❌ Plan B마저 실패: {e}")
        
    return [] # 미국 정부 서버까지 터지는 초유의 사태에만 빈 리스트 반환

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
    top_n = 7
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    msg = f"💎 *{today_str} 미장 주도주 스캐너* 💎\n\n"
    
    # --- [트랙 1] 기본 룰 랭킹 ---
    rule_df = df.sort_values('Rule_Score', ascending=False).head(top_n)
    msg += "🏆 *[기본 룰 랭킹]*\n"
    
    for i, (_, row) in enumerate(rule_df.iterrows(), 1):
        is_target = (row['Rule_Score'] >= 90.0) and (row['Trend_OK'] == 1) and (99 <= row['MA20_Disparity'] <= 105)
        mark = "🎯" if is_target else "✅"
        
        msg += f"*{i}. {row['Name']}* ({row['Ticker']}) {mark}\n"
        msg += f"📊 북 점수: {row['Rule_Score']:.1f}점\n"
        
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
    
    # 🌟 [수정 완료] 수집 서버 1, 2안 모두 터졌을 때 깔끔하게 에러 뱉고 종료
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
        history_df = manage_historical_data(today_df)
        ranked_df = dynamic_ml_filter(history_df, today_df)
        send_telegram(ranked_df)
