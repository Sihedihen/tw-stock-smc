import json
import numpy as np
import pandas as pd
import yfinance as yf

# 監控近半年活躍之台股強勢題材標的池
WATCHLIST = [
    {"code": "2330", "name": "台積電"},
    {"code": "3017", "name": "奇鋐"},
    {"code": "3324", "name": "雙鴻"},
    {"code": "2383", "name": "台光電"},
    {"code": "4583", "name": "台灣精銳"},
    {"code": "6442", "name": "光聖"},
    {"code": "3583", "name": "辛耘"},
    {"code": "3131", "name": "弘塑"},
    {"code": "2359", "name": "所羅門"},
    {"code": "1519", "name": "華城"},
    {"code": "6274", "name": "台燿"},
    {"code": "8996", "name": "高力"},
]


def detect_chart_pattern(highs, lows, closes):
    """日線經典形態學識別核心"""
    h = highs[-20:]
    l = lows[-20:]
    c = closes[-20:]

    # 1. 偵測上升三角 (Ascending Triangle)：高點水平受阻，低點逐步抬高 (Higher Lows)
    h_max1 = max(h[5:12])
    h_max2 = max(h[13:])
    l_min1 = min(l[5:12])
    l_min2 = min(l[13:])

    if (
        abs(h_max1 - h_max2) / h_max1 < 0.02
        and l_min2 > l_min1 * 1.015
        and c[-1] >= h_max2 * 0.98
    ):
        return {
            "name": "上升三角 (Ascending Triangle)",
            "type": "bullish",
            "neckline": round(float(h_max2), 1),
            "desc": f"壓制頸線 {round(float(h_max2), 1)}，低點持續抬高，蓄勢向上破位",
        }

    # 2. 偵測雙底 / W 底 (Double Bottom)：兩次回踩相似支撐位後反彈
    l_bottom1 = min(l[:10])
    l_bottom2 = min(l[10:18])
    w_peak = max(h[7:14])
    if (
        abs(l_bottom1 - l_bottom2) / l_bottom1 < 0.02
        and c[-1] > (l_bottom1 + l_bottom2) / 2 * 1.03
    ):
        return {
            "name": "雙底結構 (Double Bottom)",
            "type": "bullish",
            "neckline": round(float(w_peak), 1),
            "desc": f"雙重支撐驗證 {round(float(min(l_bottom1, l_bottom2)), 1)}，回測頸線不破",
        }

    # 3. 偵測牛旗 / 矩形通道收斂 (Bull Flag Consolidation)
    recent_range = (max(h[-10:]) - min(l[-10:])) / min(l[-10:])
    prior_trend = (c[-10] - c[0]) / c[0]
    if prior_trend > 0.06 and recent_range < 0.045:
        return {
            "name": "牛旗整理 (Bull Flag)",
            "type": "bullish",
            "neckline": round(float(max(h[-10:])), 1),
            "desc": "前期強動能推升後進行高位旗形縮量消化，隨時再次突破",
        }

    return {
        "name": "多頭箱體震盪 (Box Consolidation)",
        "type": "neutral",
        "neckline": round(float(max(h)), 1),
        "desc": f"區間 {round(float(min(l)), 1)} ~ {round(float(max(h)), 1)} 內換手整理",
    }


def analyze_stock(item):
    for ext in [".TW", ".TWO"]:
        try:
            ticker = f"{item['code']}{ext}"
            df_daily = yf.download(
                ticker, period="8mo", interval="1d", progress=False
            )
            if len(df_daily) < 30:
                continue

            if isinstance(df_daily.columns, pd.MultiIndex):
                df_daily.columns = df_daily.columns.droplevel(1)

            closes = [float(x) for x in df_daily["Close"].dropna().values]
            highs = [float(x) for x in df_daily["High"].dropna().values]
            lows = [float(x) for x in df_daily["Low"].dropna().values]
            dates = [d.strftime("%m/%d") for d in df_daily.index]

            price_now = round(closes[-1], 1)

            # 近半年漲幅
            ref_idx = 120 if len(closes) > 120 else len(closes) - 1
            momentum_6m = round(
                ((price_now - closes[-ref_idx]) / closes[-ref_idx]) * 100, 1
            )

            # 形態學檢驗
            pattern = detect_chart_pattern(highs, lows, closes)

            # 周線 HTF 大趨勢
            df_weekly = (
                df_daily.resample("W-FRI")
                .agg({"High": "max", "Low": "min", "Close": "last"})
                .dropna()
            )
            w_highs = [float(x) for x in df_weekly["High"].values]
            w_swing_high = (
                max(w_highs[-10:-1]) if len(w_highs) > 10 else max(w_highs)
            )
            weekly_mss = (
                "周線強多頭 (MSS)"
                if price_now >= w_swing_high * 0.98
                else "周線多頭回撤"
            )

            # 日線 LTF FVG
            daily_fvg = "無回測缺口"
            fvg_low = 0
            fvg_high = 0
            for i in range(len(df_daily) - 1, len(df_daily) - 6, -1):
                if lows[i] > highs[i - 2]:
                    fvg_low = round(highs[i - 2], 1)
                    fvg_high = round(lows[i], 1)
                    daily_fvg = f"{fvg_low} ~ {fvg_high}"
                    break

            # 日線折價區與點位計算
            d_swing_low = min(lows[-15:-1])
            d_swing_high = max(highs[-15:-1])

            # 建議進場點 (Suggested Entry Zone)：若有 FVG 則取 FVG，若無則取形態頸線附近或折價區間
            if daily_fvg != "無回測缺口":
                entry_zone = f"{fvg_low} - {fvg_high}"
            elif pattern["name"].startswith("上升三角"):
                entry_zone = (
                    f"{round(pattern['neckline'] * 0.985, 1)} - {price_now}"
                )
            else:
                entry_zone = f"{round(price_now * 0.985, 1)} - {price_now}"

            # 嚴格止損 (SL) 與 目標止盈 (TP)
            sl = round(d_swing_low * 0.985, 1)
            risk = price_now - sl
            tp = round(d_swing_high * 1.08, 1)
            reward = tp - price_now
            rr = round(reward / risk, 2) if risk > 0 else 0

            history = [
                {"date": d, "close": round(c, 1)}
                for d, c in zip(dates[-30:], closes[-30:])
            ]

            return {
                "code": item["code"],
                "name": item["name"],
                "price": price_now,
                "momentum_6m": momentum_6m,
                "pattern": pattern["name"],
                "patternDesc": pattern["desc"],
                "weeklyTrend": weekly_mss,
                "dailyFVG": daily_fvg,
                "entryZone": entry_zone,
                "sl": sl,
                "tp": tp,
                "rr": f"{rr:.2f}",
                "history": history,
            }
        except Exception:
            continue
    return None


results = []
for item in WATCHLIST:
    res = analyze_stock(item)
    if res:
        results.append(res)

# 依半年動能排序
results.sort(key=lambda x: x["momentum_6m"], reverse=True)

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"成功輸出 {len(results)} 檔整合形態學與進場點的標的！")
