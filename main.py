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

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

MAX_WORKERS = 10
HISTORY_FILE = 'history.csv'

EXCLUDED_SECTORS = ['Financial Services', 'Utilities', 'Real Estate', 'Basic Materials']
EXCLUDED_INDUSTRIES = ['Banks - Regional', 'Banks - Diversified']

def get_broad_universe():
    print("S&P 1500 유니버스 수집 중...")
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
        sp400 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]['Symbol']
        sp600 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0]['Symbol']
        universe = pd.concat([sp500, sp400, sp600]).dropna().unique().tolist()
        return [t.replace('.', '-') for t in universe]
    except:
        return ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "META", "AMZN", "SMCI", "ARM", "PLTR"]

def fetch_evidence(ticker, start_date, end_date):
    """오늘의 데이터 수집"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        if sector in EXCLUDED_SECTORS or industry in EXCLUDED_INDUSTRIES:
            return None
            
        hist = stock.history(start=start_date, end=end_date)
        if len(hist) < 60: return None
            
        close_px = hist['Close'].iloc[-1]
        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
        ma60 = hist['Close'].rolling(60).mean().iloc[-1]
        trend_score = 1 if (close_px > ma20 > ma60) else 0
        
        eps_surp = info.get('earningsQuarterlyGrowth', 0)
        eps_surp = eps_surp * 100 if pd.notna(eps_surp) else 0
        
        eps_trl = info.get('trailingEps', np.nan)
        eps_fwd = info.get('forwardEps', np.nan)
        revision_strength = 0
        if pd.notna(eps_trl) and pd.notna(eps_fwd) and eps_trl > 0:
            revision_strength = ((eps_fwd - eps_trl) / eps_trl) * 100
            
        price_3m = hist['Close'].iloc[-63]
        mom_3m = ((close_px / price_3m) - 1) * 100
        
        return {
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'Ticker': ticker,
            'Sector': sector,
            'Close': close_px,
            'Trend_OK': trend_score,
            'EPS_Surprise': eps_surp,
            'Revision_Strength': revision_strength,
            'Mom_3M': mom_3m,
            'Target': np.nan # 미래 수익률 정답지 (초기엔 비워둠)
        }
    except:
        return None

def build_today_dataset(tickers):
    print("오늘의 증거 데이터 병렬 수집 중...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_evidence, t, start_date, end_date): t for t in tickers}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    df = pd.DataFrame(results)
    return df

def manage_historical_data(today_df):
    """과거 데이터를 불러와 채점하고, 오늘 데이터를 누적시킴"""
    print("AI 메모리(CSV) 연동 및 정답지 자동 채점 중...")
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
    else:
        history_df = pd.DataFrame(columns=today_df.columns)
        
    # 과거 데이터 중 Target이 아직 비어있는 항목들 채점 (20일 경과 기준)
    if not history_df.empty:
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        today_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
        
        # 오늘 날짜 종가 딕셔너리 생성 (채점용)
        today_prices = today_df.set_index('Ticker')['Close'].to_dict()
        
        for idx, row in history_df.iterrows():
            if pd.isna(row['Target']):
                days_passed = (today_date - row['Date']).days
                if days_passed >= 20: # 20일(약 한 달)이 지났다면 채점 시작!
                    ticker = row['Ticker']
                    if ticker in today_prices:
                        past_price = row['Close']
                        current_price = today_prices[ticker]
                        return_rate = (current_price / past_price) - 1
                        
                        # 20일 동안 5% 이상 올랐으면 성공(1), 아니면 실패(0)
                        history_df.at[idx, 'Target'] = 1 if return_rate > 0.05 else 0

    # 오늘 데이터 합치기
    today_df['Date'] = pd.to_datetime(today_df['Date'])
    updated_history = pd.concat([history_df, today_df], ignore_index=True)
    
    # 중복 제거 (같은 날짜, 같은 티커)
    updated_history = updated_history.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
    
    # CSV로 다시 저장 (깃허브 액션이 이 파일을 커밋할 예정)
    updated_history.to_csv(HISTORY_FILE, index=False)
    
    return updated_history

def dynamic_ml_filter(history_df, today_df):
    """동태적 롤링 학습 및 오늘 종목 추천"""
    print("동태적 AI 학습 진행 중...")
    
    # Target(정답)이 채워진 과거 데이터만 추출하여 훈련 세트로 사용
    train_data = history_df.dropna(subset=['Target'])
    
    features = ['Trend_OK', 'EPS_Surprise', 'Revision_Strength', 'Mom_3M']
    
    if len(train_data) < 100:
        print("아직 훈련 데이터(정답지)가 충분하지 않아 기본 로직으로 대체합니다.")
        # 초기 한 달간 데이터가 쌓이기 전에는 기본 휴리스틱 정답지 사용
        mom_threshold = today_df['Mom_3M'].quantile(0.7)
        y_temp = np.where((today_df['Trend_OK'] == 1) & (today_df['Revision_Strength'] > 0) & (today_df['Mom_3M'] > mom_threshold), 1, 0)
        if sum(y_temp) == 0: return pd.DataFrame() # 예외처리
        X_train = today_df[features]
        y_train = y_temp
    else:
        # 한 달 뒤부터는 온전히 자기가 채점한 진짜 데이터로만 학습!
        X_train = train_data[features]
        y_train = train_data['Target'].astype(int)
        
    clf = HistGradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # 오늘 데이터 예측
    X_today = today_df[features].fillna(0)
    today_df['AI_Prob'] = clf.predict_proba(X_today)[:, 1] * 100
    
    # 확률 70% 이상 + 정배열 종목만 컷오프
    filtered_df = today_df[(today_df['AI_Prob'] >= 70) & (today_df['Trend_OK'] == 1)].copy()
    return filtered_df.sort_values('AI_Prob', ascending=False)

def send_telegram(df):
    if df.empty:
        msg = "📉 *오늘의 리포트*\n\n현재 시장에 AI의 엄격한 기준을 통과한 주도주가 없습니다.\n(하락장 우려. 현금 보유 권장)"
    else:
        top_n = min(7, len(df))
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        msg = f"🔥 *{today_str} 진화형 AI 주도주 픽* 🔥\n"
        msg += f"_(과거 데이터를 스스로 학습해 확률을 도출합니다)_\n\n"
        
        for i, (ticker, row) in enumerate(df.head(top_n).iterrows(), 1):
            msg += f"*{i}. {ticker}* ({row['Sector']})\n"
            msg += f"🎯 *AI 합격률:* {row['AI_Prob']:.1f}%\n"
            
            evidences = []
            if row['Revision_Strength'] > 15: evidences.append(f"가이던스 대폭상향(+{row['Revision_Strength']:.0f}%)")
            elif row['Revision_Strength'] > 0: evidences.append("추정치 상향중")
            if row['EPS_Surprise'] > 10: evidences.append(f"어닝서프({row['EPS_Surprise']:.0f}%)")
            if row['Mom_3M'] > 20: evidences.append(f"강력한 모멘텀(+{row['Mom_3M']:.0f}%)")
            
            msg += f"🧾 *증거:* {', '.join(evidences)}\n\n"
            
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})
    print("텔레그램 발송 완료!")

if __name__ == "__main__":
    universe = get_broad_universe()
    if universe:
        today_data = build_today_dataset(universe)
        if not today_data.empty:
            # 과거 데이터와 연동 및 채점
            history_data = manage_historical_data(today_data)
            
            # 동태적 기계학습 필터링
            best_stocks = dynamic_ml_filter(history_data, today_data)
            
            # 텔레그램 발송
            send_telegram(best_stocks)
