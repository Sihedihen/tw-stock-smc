import json
import math
import numpy as np
import pandas as pd
import yfinance as yf

# 包含 2436 偉詮電與近期題材強勢標的
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
    {"code": "8996", "name": "高力"},
]


def clean_num(val, default=0.0):
    """徹底防止 NaN / Inf 破壞 JSON 格式"""
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

    # 上升三角
    h_max1, h_max2 = max(h[:10]), max(h[10:])
    l_min1, l_min2 = min(l[:10]), min(l[10:])
    if (
        abs(h_max1 - h_max2) / (h_max1 or 1) < 0.03
        and l_min2 > l_min1 * 1.015
        and c[-1] >= h_max2 * 0.96
    ):
        return {
            "name": "上升三角 (Ascending Triangle)",
            "desc": f"壓制頸線約 {clean_num(h_max2)}，低點墊高蓄勢向上突破",
        }

    # 雙底結構
    if (
        abs(l_min1 - l_min2) / (l_min1 or 1) < 0.035
        and c[-1] > min(l_min1, l_min2) * 1.03
    ):
        return {
            "name": "雙底結構 (Double Bottom)",
            "desc": f"雙重支撐 {clean_num(min(l_min1, l_min2))} 築底回測確認",
        }

    return {
        "name": "多頭箱體整理",
        "desc": f"區間 {clean_num(min(l))} ~ {clean_num(max(h))} 縮量換手",
    }


def analyze_stock(item):
    for ext in [".TW", ".TWO"]:
        try:
            ticker = f"{item['code']}{ext}"
            df = yf.download(
                ticker, period="8mo", interval="1d", progress=False
            )
            if df is None or len(df) < 50:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df = df.dropna(subset=["Close"])
            if len(df) < 50:
                continue

            # 計算 EMA 20, 60, 100
            df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMA60"] = df["Close"].ewm(span=60, adjust=False).mean()
            df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()

            closes = [float(x) for x in df["Close"].values]
            highs = [float(x) for x in df["High"].values]
            lows = [float(x) for x in df["Low"].values]
            dates = [d.strftime("%m/%d") for d in df.index]

            price_now = clean_num(closes[-1])
            ema20_now = clean_num(df["EMA20"].iloc[-1])
            ema60_now = clean_num(df["EMA60"].iloc[-1])
            ema100_now = clean_num(df["EMA100"].iloc[-1])

            # 1. EMA 趨勢形態判斷
            if ema20_now > ema60_now > ema100_now:
                ema_status = "多頭排列 (Bullish)"
            elif price_now > ema20_now and ema20_now > ema60_now:
                ema_status = "突破站上 EMA20"
            else:
                ema_status = "均線震盪收斂"

            # 2. 近半年動能
            ref_idx = 120 if len(closes) > 120 else len(closes) - 1
            momentum_6m = clean_num(
                ((price_now - closes[-ref_idx]) / (closes[-ref_idx] or 1)) * 100
            )

            # 3. 形態學
            pattern_info = detect_chart_pattern(highs, lows, closes)

            # 4. MACD 分析 (日線背離 + 周線金叉)
            dif, dea, hist = calculate_macd(df["Close"])
            d_min_close = np.argmin(closes[-20:])
            d_min_dif = np.argmin(dif.values[-20:])
            if (
                closes[-1] > closes[-20 + d_min_close]
                and dif.values[-1] > dif.values[-20 + d_min_dif]
            ):
                macd_status = "底背離確認"
            else:
                macd_status = "動能共振"

            df_weekly = (
                df.resample("W-FRI").agg({"Close": "last"}).dropna()
            )
            w_dif, w_dea, _ = calculate_macd(df_weekly["Close"])
            if (
                len(w_dif) >= 2
                and w_dif.iloc[-1] > w_dea.iloc[-1]
                and w_dif.iloc[-2] <= w_dea.iloc[-2]
            ):
                weekly_macd = "周線金叉突破"
            elif len(w_dif) >= 1 and w_dif.iloc[-1] > 0:
                weekly_macd = "周線零軸上多頭"
            else:
                weekly_macd = "周線低位蓄勢"

            # 5. 日線 FVG 與入場點
            daily_fvg = "無缺口"
            fvg_low, fvg_high = 0.0, 0.0
            for i in range(len(df) - 1, len(df) - 6, -1):
                if lows[i] > highs[i - 2]:
                    fvg_low = clean_num(highs[i - 2])
                    fvg_high = clean_num(lows[i])
                    daily_fvg = f"{fvg_low} ~ {fvg_high}"
                    break

            d_swing_low = min(lows[-15:-1])
            d_swing_high = max(highs[-15:-1])

            # 建議入場位優先看 FVG，次看 EMA20 或頸線回踩
            if daily_fvg != "無缺口":
                entry_zone = f"{fvg_low} - {fvg_high}"
            elif price_now >= ema20_now:
                entry_zone = f"{clean_num(ema20_now)} - {price_now}"
            else:
                entry_zone = (
                    f"{clean_num(price_now * 0.98)} - {clean_num(price_now)}"
                )

            sl = clean_num(d_swing_low * 0.985)
            risk = price_now - sl
            tp = clean_num(d_swing_high * 1.08)
            reward = tp - price_now
            rr = f"{clean_num(reward / risk if risk > 0 else 0):.2f}"

            # 保留最近 30 天數據供繪圖（包含 EMA）
            history = [
                {
                    "date": str(d),
                    "close": clean_num(c),
                    "ema20": clean_num(e20),
                    "ema60": clean_num(e60),
                    "ema100": clean_num(e100),
                }
                for d, c, e20, e60, e100 in zip(
                    dates[-30:],
                    closes[-30:],
                    df["EMA20"].values[-30:],
                    df["EMA60"].values[-30:],
                    df["EMA100"].values[-30:],
                )
            ]

            return {
                "code": str(item["code"]),
                "name": str(item["name"]),
                "price": price_now,
                "momentum_6m": momentum_6m,
                "pattern": str(pattern_info["name"]),
                "patternDesc": str(pattern_info["desc"]),
                "emaStatus": ema_status,
                "ema20": ema20_now,
                "ema60": ema60_now,
                "ema100": ema100_now,
                "macdStatus": f"{macd_status} / {weekly_macd}",
                "dailyFVG": daily_fvg,
                "entryZone": entry_zone,
                "sl": sl,
                "tp": tp,
                "rr": rr,
                "history": history,
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

results.sort(key=lambda x: x["momentum_6m"], reverse=True)

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"成功輸出 {len(results)} 檔整合 EMA、形態學與 MACD 的標的！")
