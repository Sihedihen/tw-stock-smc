import yfinance as yf
import pandas as pd
import numpy as np
import json
import math

# 包含 2436 偉詮電與強勢中小型股
WATCHLIST = [
    {"code": "2436", "name": "偉詮電"},
    {"code": "2330", "name": "台積電"},
    {"code": "3017", "name": "奇鋐"},
    {"code": "3324", "name": "雙鴻"},
    {"code": "2383", "name": "台光電"},
    {"code": "6274", "name": "台燿"},
    {"code": "4583", "name": "台灣精銳"},
    {"code": "6442", "name": "光聖"},
    {"code": "3583", "name": "辛耘"},
    {"code": "2359", "name": "所羅門"},
    {"code": "1519", "name": "華城"},
    {"code": "8996", "name": "高力"}
]

def clean_float(val, default=0.0):
    """防止 NaN 或 Infinity 破壞 JSON 格式"""
    if val is None or math.isnan(val) or math.isinf(val):
        return default
    return round(float(val), 2)

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist

def detect_chart_pattern(highs, lows, closes):
    if len(closes) < 20:
        return {"name": "高位整理", "desc": "歷史K線不足"}
    
    h = highs[-20:]
    l = lows[-20:]
    c = closes[-20:]
    
    # 1. 上升三角：上方平頂阻力，低點抬高
    h_max1, h_max2 = max(h[:10]), max(h[10:])
    l_min1, l_min2 = min(l[:10]), min(l[10:])
    
    if abs(h_max1 - h_max2) / (h_max1 or 1) < 0.03 and l_min2 > l_min1 * 1.015:
        return {
            "name": "上升三角 (Ascending Triangle)",
            "desc": f"壓制頸線約 {clean_float(h_max2)}，低點抬高蓄勢向上"
        }
    
    # 2. 雙底結構
    if abs(l_min1 - l_min2) / (l_min1 or 1) < 0.035 and c[-1] > min(l_min1, l_min2) * 1.03:
        return {
            "name": "雙底結構 (Double Bottom)",
            "desc": f"雙重支撐 {clean_float(min(l_min1, l_min2))} 回踩確認"
        }

    return {
        "name": "箱體震盪收斂",
        "desc": f"區間 {clean_float(min(l))} ~ {clean_float(max(h))} 蓄勢整理"
    }

def analyze_stock(item):
    for ext in [".TW", ".TWO"]:
        try:
            ticker = f"{item['code']}{ext}"
            df = yf.download(ticker, period="8mo", interval="1d", progress=False)
            if df is None or len(df) < 30:
                continue
            
            # 清理多層欄位
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            df = df.dropna(subset=['Close'])
            if len(df) < 30:
                continue

            closes = [float(x) for x in df['Close'].values]
            highs = [float(x) for x in df['High'].values]
            lows = [float(x) for x in df['Low'].values]
            dates = [d.strftime('%m/%d') for d in df.index]

            price_now = clean_float(closes[-1])
            
            # 半年動能
            ref_idx = 120 if len(closes) > 120 else len(closes) - 1
            momentum_6m = clean_float(((price_now - closes[-ref_idx]) / (closes[-ref_idx] or 1)) * 100)

            # 形態學
            pattern_info = detect_chart_pattern(highs, lows, closes)

            # MACD
            dif, dea, hist = calculate_macd(df['Close'])
            
            # 日線背離檢驗
            d_min_close = np.argmin(closes[-20:])
            d_min_dif = np.argmin(dif.values[-20:])
            if closes[-1] > closes[-20 + d_min_close] and dif.values[-1] > dif.values[-20 + d_min_dif]:
                macd_status = "日線底背離 (Bullish Div)"
            else:
                macd_status = "日線動能共振"

            # 周線 MACD 金叉
            df_weekly = df.resample('W-FRI').agg({'Close': 'last'}).dropna()
            w_dif, w_dea, _ = calculate_macd(df_weekly['Close'])
            if len(w_dif) >= 2 and w_dif.iloc[-1] > w_dea.iloc[-1] and w_dif.iloc[-2] <= w_dea.iloc[-2]:
                weekly_macd = "周線金叉突破"
            elif len(w_dif) >= 1 and w_dif.iloc[-1] > 0:
                weekly_macd = "周線多頭零軸上"
            else:
                weekly_macd = "周線低位蓄勢"

            # 日線 FVG
            daily_fvg = "無回測缺口"
            fvg_low, fvg_high = 0.0, 0.0
            for i in range(len(df)-1, len(df)-6, -1):
                if lows[i] > highs[i-2]:
                    fvg_low = clean_float(highs[i-2])
                    fvg_high = clean_float(lows[i])
                    daily_fvg = f"{fvg_low} ~ {fvg_high}"
                    break
            
            d_swing_low = min(lows[-15:-1])
            d_swing_high = max(highs[-15:-1])

            # 建議入場點位
            if daily_fvg != "無回測缺口":
                entry_zone = f"{fvg_low} - {fvg_high}"
            else:
                entry_zone = f"{clean_float(price_now * 0.97)} - {clean_float(price_now * 0.99)}"

            sl = clean_float(d_swing_low * 0.985)
            risk = price_now - sl
            tp = clean_float(d_swing_high * 1.08)
            reward = tp - price_now
            rr = f"{clean_float(reward / risk if risk > 0 else 0):.2f}"

            history = [{"date": str(d), "close": clean_float(c)} for d, c in zip(dates[-30:], closes[-30:])]

            return {
                "code": str(item["code"]),
                "name": str(item["name"]),
                "price": price_now,
                "momentum_6m": momentum_6m,
                "pattern": str(pattern_info["name"]),
                "patternDesc": str(pattern_info["desc"]),
                "macdStatus": f"{macd_status} / {weekly_macd}",
                "dailyFVG": daily_fvg,
                "entryZone": entry_zone,
                "sl": sl,
                "tp": tp,
                "rr": rr,
                "history": history
            }
        except Exception as e:
            continue
    return None

results = []
for item in WATCHLIST:
    res = analyze_stock(item)
    if res:
        results.append(res)

# 依半年動能排序
results.sort(key=lambda x: x["momentum_6m"], reverse=True)

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"成功輸出 {len(results)} 檔合法 JSON 標的！")
