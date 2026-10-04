import json
import yfinance as yf
import pandas as pd
import numpy as np

# 挑選流動性佳、具題材性的台股標的池
WATCHLIST = [
    {"symbol": "2330.TW", "code": "2330", "name": "台積電"},
    {"symbol": "3017.TW", "code": "3017", "name": "奇鋐"},
    {"symbol": "3324.TW", "code": "3324", "name": "雙鴻"},
    {"symbol": "2383.TW", "code": "2383", "name": "台光電"},
    {"symbol": "4583.TW", "code": "4583", "name": "台灣精銳"},
    {"symbol": "6442.TW", "code": "6442", "name": "光聖"},
    {"symbol": "3583.TW", "code": "3583", "name": "辛耘"},
    {"symbol": "2359.TW", "code": "2359", "name": "所羅門"},
    {"symbol": "1519.TW", "code": "1519", "name": "華城"},
    {"symbol": "6274.TW", "code": "6274", "name": "台燿"}
]

def get_stock_data(item):
    for ext in [".TW", ".TWO"]:
        ticker = f"{item['code']}{ext}"
        try:
            df = yf.download(ticker, period="8mo", interval="1d", progress=False)
            if len(df) >= 30:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                return df
        except Exception:
            continue
    return None

results = []

for item in WATCHLIST:
    try:
        df_daily = get_stock_data(item)
        if df_daily is None or len(df_daily) < 30:
            continue
        
        # 扁平數據
        closes = [float(x) for x in df_daily['Close'].dropna().values]
        highs = [float(x) for x in df_daily['High'].dropna().values]
        lows = [float(x) for x in df_daily['Low'].dropna().values]
        dates = [d.strftime('%m/%d') for d in df_daily.index]

        price_now = round(closes[-1], 1)
        
        # 1. 近半年動能計算 (若不足120天則用最早一天比較)
        ref_idx = 120 if len(closes) > 120 else len(closes) - 1
        price_past = closes[-ref_idx]
        half_year_return = round(((price_now - price_past) / price_past) * 100, 1)

        # 2. 周線 HTF 計算
        df_weekly = df_daily.resample('W-FRI').agg({
            'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
        
        w_highs = [float(x) for x in df_weekly['High'].values]
        w_lows = [float(x) for x in df_weekly['Low'].values]
        
        w_swing_high = max(w_highs[-10:-1]) if len(w_highs) > 10 else max(w_highs)
        weekly_mss = "周線多頭破位 (MSS)" if price_now >= w_swing_high * 0.98 else "周線多頭回撤"
        
        weekly_fvg = "無明顯缺口"
        for i in range(len(df_weekly)-1, max(len(df_weekly)-5, 2), -1):
            if w_lows[i] > w_highs[i-2]:
                weekly_fvg = f"{round(w_highs[i-2], 1)} ~ {round(w_lows[i], 1)}"
                break

        # 3. 日線 LTF FVG 與折價區
        daily_fvg = "無回測缺口"
        for i in range(len(df_daily)-1, len(df_daily)-6, -1):
            if lows[i] > highs[i-2]:
                daily_fvg = f"{round(highs[i-2], 1)} ~ {round(lows[i], 1)}"
                break

        d_swing_low = min(lows[-15:-1])
        d_swing_high = max(highs[-15:-1])
        eq = d_swing_low + (d_swing_high - d_swing_low) * 0.5
        is_discount = price_now <= eq

        # 4. 風盈比計算
        sl = round(d_swing_low * 0.985, 1)
        risk = price_now - sl
        tp = round(d_swing_high * 1.08, 1)
        reward = tp - price_now
        rr = round(reward / risk, 2) if risk > 0 else 0

        history = [{"date": d, "close": round(c, 1)} for d, c in zip(dates[-30:], closes[-30:])]

        results.append({
            "code": item["code"],
            "name": item["name"],
            "price": price_now,
            "momentum_6m": half_year_return,
            "weeklyTrend": weekly_mss,
            "weeklyFVG": weekly_fvg,
            "dailyFVG": daily_fvg,
            "isDiscount": is_discount,
            "sl": sl,
            "tp": tp,
            "rr": f"{rr:.2f}",
            "history": history
        })
    except Exception as e:
        print(f"Skipping {item['code']}: {e}")
        continue

# 依半年漲幅排序
results.sort(key=lambda x: x["momentum_6m"], reverse=True)

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"成功輸出 {len(results)} 檔股票資料！")
