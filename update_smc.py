import yfinance as yf
import pandas as pd
import numpy as np
import json

# 股票池：加入 2436 偉詮電與近期形態明確的題材標的
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

def calculate_macd(series, fast=12, slow=26, signal=9):
    """計算 MACD 快線、慢線與柱狀圖"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist

def detect_chart_pattern(highs, lows, closes):
    """日線形態學識別"""
    h = highs[-25:]
    l = lows[-25:]
    c = closes[-25:]
    
    # 1. 上升三角：上方水平阻力，下方低點不斷墊高
    h_max1 = max(h[5:14])
    h_max2 = max(h[15:])
    l_min1 = min(l[5:14])
    l_min2 = min(l[15:])
    
    if abs(h_max1 - h_max2) / h_max1 < 0.025 and l_min2 > l_min1 * 1.015:
        return {
            "name": "上升三角 (Ascending Triangle)",
            "desc": f"上方承壓 {round(h_max2, 1)}，低點持續墊高，強勢蓄勢突破"
        }
        
    # 2. 雙底 W 底：兩次回踩不破反彈
    l_b1 = min(l[:12])
    l_b2 = min(l[12:22])
    if abs(l_b1 - l_b2) / l_b1 < 0.03 and c[-1] > max(l_b1, l_b2) * 1.05:
        return {
            "name": "雙底結構 (Double Bottom)",
            "desc": f"雙重支撐驗證 {round(min(l_b1, l_b2), 1)}，頸線回踩確認"
        }

    # 3. 箱體強勢整理
    return {
        "name": "多頭箱體收斂",
        "desc": f"區間 {round(min(l), 1)} ~ {round(max(h), 1)} 縮量消化套牢盤"
    }

def detect_macd_divergence(closes, dif):
    """偵測日線 MACD 背離"""
    if len(closes) < 30:
        return "MACD 無顯著背離"
    
    # 檢查近 20 天是否有價格破低但 DIF 抬高（底背離）
    p_min_idx = np.argmin(closes[-25:])
    d_min_idx = np.argmin(dif[-25:])
    
    if closes[-1] > closes[-25 + p_min_idx] and dif[-1] > dif[-25 + d_min_idx]:
        # 柱狀體翻揚
        return "日線底背離確認 (Bullish Divergence)"
    return "日線動能共振"

def analyze_stock(item):
    for ext in [".TW", ".TWO"]:
        try:
            ticker = f"{item['code']}{ext}"
            df = yf.download(ticker, period="8mo", interval="1d", progress=False)
            if len(df) < 40:
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            closes = df['Close'].dropna().values
            highs = df['High'].dropna().values
            lows = df['Low'].dropna().values
            dates = [d.strftime('%m/%d') for d in df.index]
            
            price_now = round(float(closes[-1]), 1)
            
            # 近半年漲幅
            ref_idx = 120 if len(closes) > 120 else len(closes) - 1
            momentum_6m = round(((price_now - closes[-ref_idx]) / closes[-ref_idx]) * 100, 1)

            # 1. 形態學
            pattern_info = detect_chart_pattern(highs, lows, closes)

            # 2. 日線 MACD 與背離分析
            dif, dea, hist = calculate_macd(df['Close'])
            macd_divergence = detect_macd_divergence(closes, dif.values)

            # 3. 周線級別 MACD 與金叉分析
            df_weekly = df.resample('W-FRI').agg({'Close': 'last', 'High': 'max', 'Low': 'min'}).dropna()
            w_dif, w_dea, w_hist = calculate_macd(df_weekly['Close'])
            
            # 周線金叉檢查 (DIF 向上突破 DEA)
            is_weekly_cross = (w_dif.iloc[-1] > w_dea.iloc[-1]) and (w_dif.iloc[-2] <= w_dea.iloc[-2] or w_dif.iloc[-3] <= w_dea.iloc[-3])
            weekly_macd_status = "周線金叉突破 (HTF Golden Cross)" if is_weekly_cross else ("周線多頭零軸上" if w_dif.iloc[-1] > 0 else "周線低位蓄勢")

            # 4. 日線 FVG 缺口與進場點
            daily_fvg = "無回測缺口"
            fvg_low, fvg_high = 0, 0
            for i in range(len(df)-1, len(df)-6, -1):
                if lows[i] > highs[i-2]:
                    fvg_low = round(float(highs[i-2]), 1)
                    fvg_high = round(float(lows[i]), 1)
                    daily_fvg = f"{fvg_low} ~ {fvg_high}"
                    break
            
            d_swing_low = float(min(lows[-15:-1]))
            d_swing_high = float(max(highs[-15:-1]))
            
            # 建議入場位
            if daily_fvg != "無回測缺口":
                entry_zone = f"{fvg_low} - {fvg_high}"
            else:
                entry_zone = f"{round(price_now * 0.97, 1)} - {round(price_now * 0.99, 1)}"
                
            sl = round(d_swing_low * 0.985, 1)
            risk = price_now - sl
            tp = round(d_swing_high * 1.08, 1)
            reward = tp - price_now
            rr = round(reward / risk, 2) if risk > 0 else 0

            history = [{"date": d, "close": round(float(c), 1)} for d, c in zip(dates[-30:], closes[-30:])]

            return {
                "code": item["code"],
                "name": item["name"],
                "price": price_now,
                "momentum_6m": momentum_6m,
                "pattern": pattern_info["name"],
                "patternDesc": pattern_info["desc"],
                "macdDivergence": macd_divergence,
                "weeklyMacd": weekly_macd_status,
                "dailyFVG": daily_fvg,
                "entryZone": entry_zone,
                "sl": sl,
                "tp": tp,
                "rr": f"{rr:.2f}",
                "history": history
            }
        except Exception as e:
            print(f"Error {item['code']}: {e}")
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

print(f"成功輸出 {len(results)} 檔整合形態與 MACD 標的！")
