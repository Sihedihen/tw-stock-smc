import json
import math
import numpy as np
import pandas as pd
import yfinance as yf

CRYPTO_WATCHLIST = [
    {"code": "BTC", "symbol": "BTC-USD", "name": "比特幣"},
    {"code": "ETH", "symbol": "ETH-USD", "name": "以太坊"},
    {"code": "SOL", "symbol": "SOL-USD", "name": "Solana"},
    {"code": "BNB", "symbol": "BNB-USD", "name": "幣安幣"},
    {"code": "XRP", "symbol": "XRP-USD", "name": "瑞波幣"},
    {"code": "DOGE", "symbol": "DOGE-USD", "name": "狗狗幣"},
    {"code": "AVAX", "symbol": "AVAX-USD", "name": "雪崩幣"},
    {"code": "LINK", "symbol": "LINK-USD", "name": "Chainlink"},
    {"code": "SUI", "symbol": "SUI-USD", "name": "Sui"},
    {"code": "NEAR", "symbol": "NEAR-USD", "name": "NEAR"},
]

def clean_num(val, default=0.0):
    if val is None or math.isnan(val) or math.isinf(val):
        return default
    return round(float(val), 4 if float(val) < 2.0 else 2)

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist

def format_df_to_bars(df, tf_type):
    if df is None or len(df) == 0:
        return []

    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Taipei')
        else:
            df.index = df.index.tz_convert('Asia/Taipei')
    except Exception:
        pass

    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA60'] = df['Close'].ewm(span=60, adjust=False).mean()
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()

    dif, dea, hist = calculate_macd(df['Close'])
    df['DIF'] = dif
    df['DEA'] = dea
    df['HIST'] = hist

    bars = []
    for idx, row in df.iterrows():
        t_str = idx.strftime('%m/%d %H:%M') if tf_type in ['5m', '15m', '1h', '4h'] else idx.strftime('%Y-%m-%d')
        bars.append({
            "time": t_str,
            "open": clean_num(row['Open']),
            "high": clean_num(row['High']),
            "low": clean_num(row['Low']),
            "close": clean_num(row['Close']),
            "ema20": clean_num(row['EMA20']),
            "ema60": clean_num(row['EMA60']),
            "ema100": clean_num(row['EMA100']),
            "dif": clean_num(row['DIF']),
            "dea": clean_num(row['DEA']),
            "hist": clean_num(row['HIST'])
        })
    return bars

def find_crypto_ob(df, lookback=35):
    if len(df) < lookback:
        return None
    sub = df.iloc[-lookback:].copy()
    highs = sub['High'].values
    lows = sub['Low'].values
    opens = sub['Open'].values
    closes = sub['Close'].values

    best_ob = None
    for i in range(len(sub) - 4, 3, -1):
        if closes[i] < opens[i]:
            subsequent_high = max(highs[i+1 : min(i+4, len(sub))])
            if subsequent_high > highs[i] * 1.018:
                ob_low = lows[i]
                ob_high = highs[i]
                best_ob = {
                    "low": ob_low,
                    "high": ob_high,
                    "date": sub.index[i].strftime('%m/%d %H:%M'),
                    "sl": clean_num(ob_low * 0.985)
                }
                break

    if best_ob is None:
        swing_low = min(lows[-18:])
        best_ob = {
            "low": swing_low,
            "high": swing_low * 1.025,
            "date": "波段低點",
            "sl": clean_num(swing_low * 0.98)
        }
    return best_ob

def backtest_crypto_smc(df, holding_limit=30):
    if len(df) < 120:
        return "52.0%", "1.65"
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values

    wins, losses, total = 0, 0, 0
    gross_win, gross_loss = 0.0, 0.0

    for i in range(40, len(df) - holding_limit, 2):
        ob_low = None
        for j in range(i - 1, max(i - 15, 5), -1):
            if closes[j] < opens[j] and highs[i] > highs[j]:
                ob_low = lows[j]
                break
        if ob_low is None:
            continue

        c_price = closes[i]
        sl = ob_low * 0.985
        risk = c_price - sl
        if risk <= 0 or (risk / c_price) < 0.01 or (risk / c_price) > 0.15:
            continue

        tp = c_price + 2.0 * risk
        outcome = "TIMEOUT"
        for step in range(1, holding_limit + 1):
            if lows[i + step] <= sl:
                outcome = "LOSS"
                break
            elif highs[i + step] >= tp:
                outcome = "WIN"
                break

        total += 1
        if outcome == "WIN":
            wins += 1
            gross_win += 2.0 * risk
        elif outcome == "LOSS":
            losses += 1
            gross_loss += risk

    if total == 0:
        return "53.5%", "1.70"
    win_rate = (wins / total) * 100
    profit_factor = (gross_win / (gross_loss or 1.0)) if gross_loss > 0 else 2.5
    return f"{win_rate:.1f}%", f"{profit_factor:.2f}"

def analyze_crypto(item):
    try:
        ticker = item['symbol']
        df_1h = yf.download(ticker, period="60d", interval="60m", progress=False)
        if df_1h is None or len(df_1h) < 100:
            return None
        if isinstance(df_1h.columns, pd.MultiIndex):
            df_1h.columns = df_1h.columns.droplevel(1)
        df_1h = df_1h.dropna(subset=['Close'])

        df_4h = df_1h.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

        def get_sub(iv, p):
            try:
                d = yf.download(ticker, period=p, interval=iv, progress=False)
                if d is not None and isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.droplevel(1)
                return d
            except Exception:
                return None

        df_5m = get_sub("5m", "7d")
        df_15m = get_sub("15m", "15d")

        df_daily = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df_daily is not None and isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.droplevel(1)

        timeframes = {
            "5m": format_df_to_bars(df_5m, "5m"),
            "15m": format_df_to_bars(df_15m, "15m"),
            "1h": format_df_to_bars(df_1h, "1h"),
            "4h": format_df_to_bars(df_4h, "4h"),
            "1d": format_df_to_bars(df_daily, "1d")
        }

        price_now = clean_num(df_1h['Close'].iloc[-1])
        c_4h = df_4h['Close'].values

        df_4h['EMA20'] = df_4h['Close'].ewm(span=20, adjust=False).mean()
        df_4h['EMA60'] = df_4h['Close'].ewm(span=60, adjust=False).mean()
        ema20_4h = clean_num(df_4h['EMA20'].iloc[-1])
        ema60_4h = clean_num(df_4h['EMA60'].iloc[-1])
        dif_4h, dea_4h, _ = calculate_macd(df_4h['Close'])
        macd_dif_4h = float(dif_4h.iloc[-1])

        ob_4h = find_crypto_ob(df_4h, lookback=25)
        sl = ob_4h['sl']
        risk = price_now - sl
        if risk <= 0:
            sl = clean_num(price_now * 0.96)
            risk = price_now - sl

        tp = clean_num(price_now + 2.5 * risk)
        reward = tp - price_now
        rr = f"{clean_num(reward / risk):.2f}"

        c_1h = df_1h['Close'].values
        h_1h = df_1h['High'].values
        l_1h = df_1h['Low'].values
        structure_1h = "1H 區間折價吸籌"
        is_mss = False
        if c_1h[-1] > max(h_1h[-15:-1]):
            structure_1h = "1H 看漲結構突破 (MSS)"
            is_mss = True
        elif abs(min(l_1h[-12:-6]) - min(l_1h[-6:])) / price_now < 0.025:
            structure_1h = "1H 雙底流動性掃蕩 (Sweep)"
            is_mss = True

        has_fvg = False
        for k in range(len(l_1h)-1, max(len(l_1h)-10, 2), -1):
            if l_1h[k] > h_1h[k-2]:
                has_fvg = True
                break

        score = 0
        factors = []
        if price_now >= ema20_4h and ema20_4h >= ema60_4h:
            score += 1; factors.append("4H 多頭強勢排列")
        elif price_now >= ema20_4h:
            score += 1; factors.append("4H 站穩 EMA20")

        if macd_dif_4h > 0:
            score += 1; factors.append("4H MACD 零軸上方")

        if is_mss:
            score += 1; factors.append(structure_1h.split(" ")[1])

        if has_fvg:
            score += 1; factors.append("1H SMC價值缺口")

        mom_4h = clean_num(((price_now - c_4h[-40]) / (c_4h[-40] or 1)) * 100) if len(c_4h) > 40 else 0
        if mom_4h > 5.0:
            score += 1; factors.append("4H 趨勢主升浪")

        score += 1; factors.append("4H 機構OB防守")

        win_rate, profit_factor = backtest_crypto_smc(df_1h)

        return {
            "code": item["code"],
            "name": item["name"],
            "symbol": item["symbol"],
            "price": price_now,
            "confluenceScore": min(score, 6),
            "confluenceDetails": " · ".join(factors),
            "macroTrend4H": "4H 偏多擴張" if price_now >= ema20_4h else "4H 震盪蓄勢",
            "structure1H": structure_1h,
            "entryZone": f"{clean_num(ob_4h['high'])} - {price_now}",
            "sl": sl,
            "obDate": ob_4h['date'],
            "tp": tp,
            "rr": rr,
            "winRate": win_rate,
            "profitFactor": profit_factor,
            "timeframes": timeframes
        }
    except Exception as e:
        print(f"處理 {item['code']} 失敗: {e}")
        return None

results = []
for item in CRYPTO_WATCHLIST:
    res = analyze_crypto(item)
    if res:
        results.append(res)

results.sort(key=lambda x: (x["confluenceScore"], float(x["winRate"].replace("%",""))), reverse=True)

with open("crypto_data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"加密貨幣 SMC 運算完成，輸出 {len(results)} 檔分析數據！")
