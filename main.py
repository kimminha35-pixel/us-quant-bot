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
# 1. 기본 세팅 및 필터링 설정
# ==========================================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

MAX_WORKERS = 10  # 야후 파이낸스 IP 차단 방지를 위해 10~15 정도로 세팅

# 🛑 분석에서 제외할 섹터 및 산업 (금융, 유틸리티, 리츠 등 노이즈 제거)
EXCLUDED_SECTORS = ['Financial Services', 'Utilities', 'Real Estate', 'Basic Materials']
EXCLUDED_INDUSTRIES = ['Banks - Regional', 'Banks - Diversified']

def get_broad_universe():
    """S&P 1500 유니버스 스크래핑"""
    print("S&P 1500 유니버스 수집 중...")
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
        sp400 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]['Symbol']
        sp600 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0]['Symbol']
        universe = pd.concat([sp500, sp400, sp600]).dropna().unique().tolist()
        return [t.replace('.', '-') for t in universe]
    except Exception as e:
        print(f"유니버스 스크래핑 에러: {e}")
        # 실패 시 예비 유니버스
        return ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "META", "AMZN", "SMCI", "ARM", "PLTR"]

# ==========================================
# 2. 증거 수집기 (섹터 필터링 탑재)
# ==========================================
def fetch_evidence(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 1차 필터링: 원치 않는 섹터/산업은 데이터 다운로드 전에 바로 버림
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        if sector in EXCLUDED_SECTORS or industry in EXCLUDED_INDUSTRIES:
            return None
            
        hist = stock.history(start=start_date, end=end_date)
        if len(hist) < 60: 
            return None
            
        close_px = hist['Close'].iloc[-1]
        
        # [증거 1] 정배열 추세 (20일선 > 60일선)
        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
        ma60 = hist['Close'].rolling(60).mean().iloc[-1]
        trend_score = 1 if (close_px > ma20 > ma60) else 0
        
        # [증거 2] 실적 서프라이즈
        eps_surp = info.get('earningsQuarterlyGrowth', 0)
        eps_surp = eps_surp * 100 if pd.notna(eps_surp) else 0
        
        # [증거 3] 리비전 (추정치 상향)
        eps_trl = info.get('trailingEps', np.nan)
        eps_fwd = info.get('forwardEps', np.nan)
        revision_strength = 0
        if pd.notna(eps_trl) and pd.notna(eps_fwd) and eps_trl > 0:
            revision_strength = ((eps_fwd - eps_trl) / eps_trl) * 100
            
        # [증거 4] 3개월 모멘텀
        price_3m = hist['Close'].iloc[-63]
        mom_3m = ((close_px / price_3m) - 1) * 100
        
        return {
            'Ticker': ticker,
            'Sector': sector,
            'Trend_OK': trend_score,
            'EPS_Surprise': eps_surp,
            'Revision_Strength': revision_strength,
            'Mom_3M': mom_3m
        }
    except:
        return None

def build_dataset(tickers):
    print(f"[{len(tickers)}개 종목] 필터링 및 증거 데이터 병렬 수집 중...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_evidence, t, start_date, end_date): t for t in tickers}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    df = pd.DataFrame(results).set_index('Ticker')
    return df.fillna(0)

# ==========================================
# 3. ML 학습 및 압축 필터링
# ==========================================
def apply_ml_filter(df):
    print("AI 학습 및 쓰레기 종목 컷오프 중...")
    X = df[['Trend_OK', 'EPS_Surprise', 'Revision_Strength', 'Mom_3M']]
    
    # 정배열 + 추정치 상향 + 상위권 모멘텀 = 성공 패턴(1)
    mom_threshold = df['Mom_3M'].quantile(0.7)
    y = np.where((df['Trend_OK'] == 1) & 
                 (df['Revision_Strength'] > 0) & 
                 (df['Mom_3M'] > mom_threshold), 1, 0)
    
    # 예외 처리: 조건에 맞는 종목이 하나도 없을 경우
    if sum(y) == 0:
        df['AI_Prob'] = 0
        return df[(df['AI_Prob'] >= 70) & (df['Trend_OK'] == 1)]
    
    clf = HistGradientBoostingClassifier(random_state=42)
    clf.fit(X, y)
    
    df['AI_Prob'] = clf.predict_proba(X)[:, 1] * 100
    
    # 필터링: AI 합격률 70% 이상 + 정배열(추세) 통과 종목만 남김
    filtered_df = df[(df['AI_Prob'] >= 70) & (df['Trend_OK'] == 1)].copy()
    return filtered_df.sort_values('AI_Prob', ascending=False)

# ==========================================
# 4. 텔레그램 알림 발송
# ==========================================
def send_telegram(df):
    if df.empty:
        msg = "📉 *오늘의 리포트*\n\n현재 시장에 AI의 엄격한 기준(정배열+실적+리비전+모멘텀)을 완벽히 통과한 주도주가 없습니다.\n(하락장 혹은 횡보장 우려. 현금 보유 권장)"
    else:
        top_n = min(7, len(df))
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        msg = f"🔥 *{today_str} 미장 주도주 AI 필터링* 🔥\n"
        msg += f"_(엄격한 다중 증거를 통과한 최상위 종목)_\n\n"
        
        for i, (ticker, row) in enumerate(df.head(top_n).iterrows(), 1):
            msg += f"*{i}. {ticker}* ({row['Sector']})\n"
            msg += f"🎯 *AI 합격률:* {row['AI_Prob']:.1f}%\n"
            
            evidences = []
            if row['Revision_Strength'] > 15: evidences.append(f"가이던스 대폭상향(+{row['Revision_Strength']:.0f}%)")
            elif row['Revision_Strength'] > 0: evidences.append("추정치 상향중")
            if row['EPS_Surprise'] > 10: evidences.append(f"어닝서프({row['EPS_Surprise']:.0f}%)")
            if row['Mom_3M'] > 20: evidences.append(f"강력한 모멘텀(+{row['Mom_3M']:.0f}%)")
            evidences.append("정배열 추세")
            
            msg += f"🧾 *확보된 증거:* {', '.join(evidences)}\n\n"
            
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})
    print("텔레그램 발송 완료!")

if __name__ == "__main__":
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("경고: 텔레그램 토큰 또는 챗 ID가 없습니다. 텔레그램 발송이 실패할 수 있습니다.")
        
    universe = get_broad_universe()
    if universe:
        raw_data = build_dataset(universe)
        if not raw_data.empty:
            best_stocks = apply_ml_filter(raw_data)
            send_telegram(best_stocks)
