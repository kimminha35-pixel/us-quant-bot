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
# 1. 기본 설정 및 필터링
# ==========================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '').strip()
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
MAX_WORKERS = 15
HISTORY_FILE = 'history.csv'

# 노이즈를 발생시키는 금융, 유틸리티, 리츠 섹터 제외
EXCLUDED_SECTORS = ['Financial Services', 'Utilities', 'Real Estate', 'Basic Materials']
EXCLUDED_INDUSTRIES = ['Banks - Regional', 'Banks - Diversified']

def get_broad_universe():
    print("S&P 1500 전체 유니버스 동적 스크래핑 중...")
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
        sp400 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]['Symbol']
        sp600 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0]['Symbol']
        universe = pd.concat([sp500, sp400, sp600]).dropna().unique().tolist()
        return [t.replace('.', '-') for t in universe]
    except Exception as e:
        print(f"스크래핑 실패. 기본 유니버스로 대체합니다. ({e})")
        return ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "META", "AMZN", "SMCI", "ARM", "PLTR"]

# ==========================================
# 2. 오늘의 증거 데이터 수집
# ==========================================
def fetch_evidence(ticker, start_date, end_date):
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
        
        # 1. 추세: 20일선 > 60일선
        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
        ma60 = hist['Close'].rolling(60).mean().iloc[-1]
        trend_score = 1 if (close_px > ma20 > ma60) else 0
        
        # 2. 어닝 서프라이즈 (Proxy: 전년동기대비 분기 성장률)
        eps_surp = info.get('earningsQuarterlyGrowth', 0)
        eps_surp = eps_surp * 100 if pd.notna(eps_surp) else 0
        
        # 3. 리비전 / 가이던스 상향 (Proxy: Trailing vs Forward EPS)
        eps_trl = info.get('trailingEps', np.nan)
        eps_fwd = info.get('forwardEps', np.nan)
        revision_strength = 0
        if pd.notna(eps_trl) and pd.notna(eps_fwd) and eps_trl > 0:
            revision_strength = ((eps_fwd - eps_trl) / eps_trl) * 100
            
        # 4. 모멘텀: 최근 3개월 (63거래일) 수익률
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
            'Target': np.nan # AI가 20일 뒤에 채점할 빈칸
        }
    except:
        return None

def build_today_dataset(tickers):
    print("오늘의 시장 데이터 병렬 수집 중...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_evidence, t, start_date, end_date): t for t in tickers}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    return pd.DataFrame(results)

# ==========================================
# 3. 기억력 장착 & 정답지 자동 채점
# ==========================================
def manage_historical_data(today_df):
    print("메모리 연동 및 과거 데이터 자동 채점 중...")
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
    else:
        history_df = pd.DataFrame(columns=today_df.columns)
        
    if not history_df.empty:
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        today_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
        today_prices = today_df.set_index('Ticker')['Close'].to_dict()
        
        # 20일 전 데이터 채점 (수익률 5% 이상이면 성공=1)
        for idx, row in history_df.iterrows():
            if pd.isna(row['Target']):
                days_passed = (today_date - row['Date']).days
                if days_passed >= 20: 
                    ticker = row['Ticker']
                    if ticker in today_prices:
                        return_rate = (today_prices[ticker] / row['Close']) - 1
                        history_df.at[idx, 'Target'] = 1 if return_rate > 0.05 else 0

    today_df['Date'] = pd.to_datetime(today_df['Date'])
    updated_history = pd.concat([history_df, today_df], ignore_index=True)
    updated_history = updated_history.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
    
    # 깃허브가 이 파일을 커밋하게 됨
    updated_history.to_csv(HISTORY_FILE, index=False)
    return updated_history

# ==========================================
# 4. 동태적 롤링 학습 및 상대 평가 (크로스섹셔널)
# ==========================================
def dynamic_ml_filter(history_df, today_df):
    print("동태적 AI 학습 및 상대 점수 환산 중...")
    train_data = history_df.dropna(subset=['Target'])
    features = ['Trend_OK', 'EPS_Surprise', 'Revision_Strength', 'Mom_3M']
    
    # 정답지가 100개 미만일 때 (초기 한 달) 작동하는 기본 지능
    if len(train_data) < 100:
        mom_threshold = today_df['Mom_3M'].quantile(0.7)
        y_temp = np.where((today_df['Trend_OK'] == 1) & (today_df['Revision_Strength'] > 0) & (today_df['Mom_3M'] > mom_threshold), 1, 0)
        
        if sum(y_temp) == 0: 
            today_df['Raw_Prob'] = 0.0
        else:
            clf = HistGradientBoostingClassifier(random_state=42)
            clf.fit(today_df[features], y_temp)
            today_df['Raw_Prob'] = clf.predict_proba(today_df[features].fillna(0))[:, 1]
    
    # 한 달 뒤, 진짜 자기가 채점한 정답지로 학습(진화)하는 지능
    else:
        X_train = train_data[features]
        y_train = train_data['Target'].astype(int)
        clf = HistGradientBoostingClassifier(random_state=42)
        clf.fit(X_train, y_train)
        today_df['Raw_Prob'] = clf.predict_proba(today_df[features].fillna(0))[:, 1]
    
    # 시장 전체 대비 오늘 내가 상위 몇 % 인가? (동태적 상대 점수)
    today_df['AI_Score'] = today_df['Raw_Prob'].rank(pct=True) * 100
    
    return today_df.sort_values('AI_Score', ascending=False)

# ==========================================
# 5. 텔레그램 발송
# ==========================================
def send_telegram(df):
    if df.empty:
        msg = "오늘 분석할 수 있는 데이터가 없습니다."
    else:
        top_n = min(7, len(df))
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        msg = f"🔥 *{today_str} 주도주 AI 랭킹 브리핑* 🔥\n"
        msg += f"_(전체 시장 기준 동태적 상대평가 상위 7개)_\n\n"
        
        for i, (ticker, row) in enumerate(df.head(top_n).iterrows(), 1):
            # 상대 점수 상위 5% 이내 + 정배열 = 진짜 주도주
            is_perfect = (row['AI_Score'] >= 95.0) and (row['Trend_OK'] == 1)
            pass_mark = "✅" if is_perfect else "⚠️"
            
            msg += f"*{i}. {ticker}* ({row['Sector']}) {pass_mark}\n"
            msg += f"🎯 *상대 랭킹:* {row['AI_Score']:.1f}점 (절대승률: {row['Raw_Prob']*100:.0f}%)\n"
            
            evidences = []
            if row['Revision_Strength'] > 15: evidences.append(f"가이던스상향(+{row['Revision_Strength']:.0f}%)")
            elif row['Revision_Strength'] > 0: evidences.append("추정치 상향중")
            if row['EPS_Surprise'] > 10: evidences.append(f"어닝서프({row['EPS_Surprise']:.0f}%)")
            if row['Mom_3M'] > 20: evidences.append(f"강력모멘텀(+{row['Mom_3M']:.0f}%)")
            
            if row['Trend_OK'] == 1: evidences.append("정배열 추세")
            else: evidences.append("역배열")
            
            msg += f"🧾 *상태:* {', '.join(evidences) if evidences else '특이사항 없음'}\n\n"
            
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    res = requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})
    
    if res.status_code == 200:
        print("텔레그램 발송 완료!")
    else:
        print(f"텔레그램 전송 실패: {res.text}")

if __name__ == "__main__":
    universe = get_broad_universe()
    if universe:
        today_data = build_today_dataset(universe)
        if not today_data.empty:
            history_data = manage_historical_data(today_data)
            best_stocks = dynamic_ml_filter(history_data, today_data)
            send_telegram(best_stocks)
