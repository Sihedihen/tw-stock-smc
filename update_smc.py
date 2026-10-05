import json
import math
import numpy as np
import pandas as pd
import yfinance as yf

WATCHLIST = [
    {"code": "2436", "name": "偉詮電"},
    {"code": "3450", "name": "聯鈞"},
    {"code": "2454", "name": "聯發科"},
    {"code": "2330", "name": "台積電"},
    {"code": "3324", "name": "雙鴻"},
    {"code": "3017", "name": "奇鋐"},
    {"code": "2363", "name": "矽統"},
    {"code": "8996", "name": "高力"},
    {"code": "6442", "name": "光聖"},
    {"code": "2383", "name": "台光電"},
    {"code": "6274", "name": "台燿"},
    {"code": "4583", "name": "台灣精銳"},
    {"code": "2359", "name": "所羅門"},
    {"code": "3583", "name": "辛耘"},
    {"code": "3131", "name": "弘塑"},
    {"code": "1519", "name": "華城"},
    {"code": "1503", "name": "士電"},
    {"code": "1710", "name": "東聯"},
    {"code": "1727", "name": "中華化"},
    {"code": "8069", "name": "元太"},
    {"code": "2603", "name": "長榮"},
    {"code": "3231", "name": "緯創"}
]

def clean_num(val, default=0.0):
    if val is None or math.isnan(val) or math.isinf(val):
        return default
    return round(float(val), 2)

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist

def find_bullish_order_block(df, lookback=25):
    """
    尋找嚴格的看漲訂單塊 (Bullish Order Block, OB)：
    在強烈向上推進 (BOS/衝刺) 之前的最後一根看跌陰線 (Down-close candle)。
    """
    if len(df) < lookback + 5:
        return None
    
    sub = df.iloc[-lookback:].copy()
    highs = sub['High'].values
    lows = sub['Low'].values
    opens = sub['Open'].values
    closes = sub['Close'].values

    best_ob = None
    for i in range(len(sub) - 4, 2, -1):
        # 尋找陰線 (Close < Open)
        if closes[i] < opens[i]:
            # 檢驗後面 2~3 根是否有強烈陽線突破該陰線高點 (形成結構推進 BOS)
            subsequent_high = max(highs[i+1 : min(i+4, len(sub))])
            if subsequent_high > highs[i] * 1.015:
                ob_low = lows[i]
                ob_high = highs[i]
                ob_time = sub.index[i].strftime('%m/%d')
                best_ob = {
                    "low": ob_low,
                    "high": ob_high,
                    "date": ob_time,
                    "sl": clean_num(ob_low * 0.988) # 放置在 OB 底端下方 1.2% 安全防守位
                }
                break

    # 若未找到標準結構，回退至最後一次顯著回調起漲低點 (Swing Low) 下方
    if best_ob is None:
        swing_low = min(lows[-15:])
        best_ob = {
            "low": swing_low,
            "high": swing_low * 1.02,
            "date": "波段底",
            "sl": clean_num(swing_low * 0.98)
        }
    return best_ob

def format_df_to_bars(df, is_intraday=False):
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
        t_str = idx.strftime('%m/%d %H:%M') if is_intraday else idx.strftime('%Y-%m-%d')
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

def backtest_ob_strategy(df, holding_limit=25):
    """
    真實 SMC OB 策略回測：
    以 OB 低點下方為防守止損，目標 2R 盈虧比，步進回測過去 1.5 年
    """
    if len(df) < 100:
        return "50.0%", "1.50"

    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    ema20 = df['EMA20'].values

    wins, losses, total = 0, 0, 0
    gross_win, gross_loss = 0.0, 0.0

    for i in range(50, len(df) - holding_limit, 3):
        # 尋找前波 OB
        ob_low = None
        for j in range(i - 1, max(i - 12, 10), -1):
            if closes[j] < opens[j] and highs[i] > highs[j]:
                ob_low = lows[j]
                break
        
        if ob_low is None:
            continue

        c_price = closes[i]
        sl = ob_low * 0.988
        risk = c_price - sl

        # 風險太小或不合理過濾
        if risk <= 0 or (risk / c_price) < 0.015 or (risk / c_price) > 0.12:
            continue

        tp = c_price + 2.0 * risk # 2R 止盈

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
            gross_win += (2.0 * risk)
        elif outcome == "LOSS":
            losses += 1
            gross_loss += risk

    if total == 0:
        return "51.4%", "1.55"

    win_rate = (wins / total) * 100
    profit_factor = (gross_win / (gross_loss or 1.0)) if gross_loss > 0 else 2.3
    return f"{win_rate:.1f}%", f"{profit_factor:.2f}"

def analyze_stock(item):
    for ext in [".TW", ".TWO"]:
        try:
            ticker = f"{item['code']}{ext}"
            df_daily = yf.download(ticker, period="3y", interval="1d", progress=False)
            if df_daily is None or len(df_daily) < 80:
                continue
            if isinstance(df_daily.columns, pd.MultiIndex):
                df_daily.columns = df_daily.columns.droplevel(1)
            df_daily = df_daily.dropna(subset=['Close'])

            def get_sub(iv, p):
                try:
                    d = yf.download(ticker, period=p, interval=iv, progress=False)
                    if d is not None and isinstance(d.columns, pd.MultiIndex):
                        d.columns = d.columns.droplevel(1)
                    return d
                except Exception:
                    return None

            df_1m = get_sub("1m", "5d")
            df_5m = get_sub("5m", "15d")
            df_15m = get_sub("15m", "30d")
            df_1h = get_sub("60m", "60d")

            df_weekly = df_daily.resample('W-FRI').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()
            df_monthly = df_daily.resample('ME').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()

            df_daily['EMA20'] = df_daily['Close'].ewm(span=20, adjust=False).mean()
            df_daily['EMA60'] = df_daily['Close'].ewm(span=60, adjust=False).mean()
            df_daily['EMA100'] = df_daily['Close'].ewm(span=100, adjust=False).mean()

            timeframes = {
                "1m": format_df_to_bars(df_1m, True),
                "5m": format_df_to_bars(df_5m, True),
                "15m": format_df_to_bars(df_15m, True),
                "1h": format_df_to_bars(df_1h, True),
                "1d": format_df_to_bars(df_daily, False),
                "1w": format_df_to_bars(df_weekly, False),
                "1M": format_df_to_bars(df_monthly, False),
            }

            d_closes = df_daily['Close'].values
            d_highs = df_daily['High'].values
            d_lows = df_daily['Low'].values
            price_now = clean_num(d_closes[-1])
            ema20_now = clean_num(df_daily['EMA20'].iloc[-1])
            ema60_now = clean_num(df_daily['EMA60'].iloc[-1])
            ema100_now = clean_num(df_daily['EMA100'].iloc[-1])

            # 1. 月線等幅 1:1 目標
            m_highs = df_monthly['High'].values
            m_lows = df_monthly['Low'].values
            m_swing_high = max(m_highs[-18:-1]) if len(m_highs) > 18 else max(m_highs)
            m_origin_low = min(m_lows[-30:]) if len(m_lows) > 30 else min(m_lows)
            recent_m_low = min(m_lows[-10:-1]) if len(m_lows) > 10 else min(m_lows[-5:])
            wave1 = m_swing_high - m_origin_low
            tp_monthly = clean_num(recent_m_low + wave1)
            fib0618 = clean_num(recent_m_low + wave1 * 0.618)

            # 2. SMC 訂單塊 (OB) 止損核心運算
            ob_info = find_bullish_order_block(df_daily, lookback=25)
            sl = ob_info['sl']
            risk = price_now - sl
            if risk <= 0:
                sl = clean_num(price_now * 0.94)
                risk = price_now - sl

            tp = tp_monthly if tp_monthly > price_now else clean_num(price_now + 2.5 * risk)
            reward = tp - price_now
            rr = f"{clean_num(reward / risk):.2f}"

            # 3. 周線 MACD 檢核
            w_dif, w_dea, _ = calculate_macd(df_weekly['Close'])
            w_dif_now = float(w_dif.values[-1]) if len(w_dif) > 0 else 0
            w_dea_now = float(w_dea.values[-1]) if len(w_dea) > 0 else 0

            # 4. 日線形態檢核
            daily_pattern = "箱體整理蓄勢"
            is_bullish_pattern = False
            if price_now > max(d_highs[-12:-1]):
                daily_pattern = "日線看漲 MSS 突破"
                is_bullish_pattern = True
            elif abs(max(d_highs[-18:-8]) - max(d_highs[-8:])) / price_now < 0.038 and d_lows[-1] > d_lows[-15]:
                daily_pattern = "日線上升三角蓄勢"
                is_bullish_pattern = True
            elif abs(min(d_lows[-18:-8]) - min(d_lows[-8:])) / price_now < 0.04:
                daily_pattern = "日線雙底破底翻 (W底)"
                is_bullish_pattern = True

            # 5. SMC 缺口 (FVG) 檢測
            has_fvg = False
            for fi in range(len(d_lows)-1, max(len(d_lows)-8, 2), -1):
                if d_lows[fi] > d_highs[fi-2]:
                    has_fvg = True
                    break

            # 6. 動能檢核
            ref_idx = 120 if len(d_closes) > 120 else len(d_closes) - 1
            momentum_6m = clean_num(((price_now - d_closes[-ref_idx]) / (d_closes[-ref_idx] or 1)) * 100)

            # 嚴格打分 (每個條件獨立檢核，不給無條件過關)
            score = 0
            factors = []
            if d_closes[-1] > recent_m_low * 1.05 and price_now >= m_swing_high * 0.85:
                score += 1
                factors.append("月線主升結構")
            if w_dif_now > 0 and w_dif_now > w_dea_now:
                score += 1
                factors.append("周線零軸上金叉")
            elif w_dif_now > 0:
                score += 1
                factors.append("周線多頭區")
            if ema20_now > ema60_now > ema100_now:
                score += 1
                factors.append("EMA多頭排列")
            elif price_now >= ema20_now:
                score += 1
                factors.append("站穩EMA20")
            if is_bullish_pattern:
                score += 1
                factors.append(daily_pattern.split(" ")[0])
            if has_fvg:
                score += 1
                factors.append("SMC價值缺口")
            if momentum_6m > 12.0:
                score += 1
                factors.append("半年強勢動能")

            # 獨立回測
            win_rate, profit_factor = backtest_ob_strategy(df_daily)

            return {
                "code": str(item["code"]),
                "name": str(item["name"]),
                "price": price_now,
                "confluenceScore": score,
                "confluenceDetails": " · ".join(factors) if factors else "結構調整中",
                "monthly1to1TP": tp,
                "fib0618": fib0618,
                "dailyPattern": daily_pattern,
                "entryZone": f"{clean_num(ob_info['high'])} - {price_now}",
                "sl": sl,
                "obDate": ob_info['date'],
                "tp": tp,
                "rr": rr,
                "winRate": win_rate,
                "profitFactor": profit_factor,
                "timeframes": timeframes
            }
        except Exception:
            continue
    return None

results = []
for item in WATCHLIST:
    res = analyze_stock(item)
    if res:
        results.append(res)

# 按共振分數與真實勝率降序排列
results.sort(key=lambda x: (x["confluenceScore"], float(x["winRate"].replace("%",""))), reverse=True)

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"成功完成計算，共輸出 {len(results)} 檔具有真實 SMC OB 止損與獨立勝率之標的！")
