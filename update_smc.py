import json
import numpy as np
import pandas as pd
import yfinance as yf

# 監控的台股標的池（可自由增減中小型股）
WATCHLIST = [
    {"symbol": "2330.TW", "code": "2330", "name": "台積電"},
    {"symbol": "3017.TW", "code": "3017", "name": "奇鋐"},
    {"symbol": "3324.TWO", "code": "3324", "name": "雙鴻"},
    {"symbol": "2383.TW", "code": "2383", "name": "台光電"},
    {"symbol": "4763.TW", "code": "4763", "name": "材料-KY"},
    {"symbol": "1519.TW", "code": "1519", "name": "華城"},
    {"symbol": "2359.TW", "code": "2359", "name": "所羅門"},
    {"symbol": "4583.TW", "code": "4583", "name": "台灣精銳"},
    {"symbol": "3583.TW", "code": "3583", "name": "辛耘"},
    {"symbol": "3131.TWO", "code": "3131", "name": "弘塑"},
]


def analyze_smc(item):
    try:
        df = yf.download(
            item["symbol"], period="2mo", interval="1d", progress=False
        )
        if len(df) < 15:
            return None

        # 整理數值
        closes = df["Close"].values.flatten()
        highs = df["High"].values.flatten()
        lows = df["Low"].values.flatten()
        dates = [d.strftime("%m/%d") for d in df.index]

        current_price = round(float(closes[-1]), 1)

        # 1. 偵測多頭 FVG (Fair Value Gap)
        has_fvg = False
        fvg_range = "無顯著缺口"
        for i in range(len(df) - 1, len(df) - 5, -1):
            if lows[i] > highs[i - 2]:
                has_fvg = True
                fvg_range = f"{round(float(highs[i-2]), 1)} ~ {round(float(lows[i]), 1)}"
                break

        # 2. 結構突破 MSS
        swing_high = float(np.max(highs[-15:-1]))
        swing_low = float(np.min(lows[-15:-1]))
        has_mss = current_price >= swing_high * 0.99

        # 3. 均衡位與折價區判斷
        eq = swing_low + (swing_high - swing_low) * 0.5
        is_discount = current_price <= eq

        # 4. 風盈比 (止損放波段低點下 1.5%)
        sl = round(swing_low * 0.985, 1)
        risk = current_price - sl
        tp = round(swing_high * 1.08, 1)
        reward = tp - current_price
        rr = round(reward / risk, 2) if risk > 0 else 0

        # 保留最近 30 天走勢供前端畫圖
        history = [
            {"date": d, "close": round(float(c), 1)}
            for d, c in zip(dates[-30:], closes[-30:])
        ]

        return {
            "code": item["code"],
            "name": item["name"],
            "price": current_price,
            "hasMSS": has_mss,
            "structure": (
                "MSS 向上破位"
                if has_mss
                else ("FVG 回測吸籌" if has_fvg else "區間整理")
            ),
            "fvgRange": fvg_range,
            "hasFVG": has_fvg,
            "isDiscount": is_discount,
            "sl": sl,
            "tp": tp,
            "rr": f"{rr:.2f}",
            "history": history,
        }
    except Exception as e:
        print(f"Error fetching {item['code']}: {e}")
        return None


results = []
for item in WATCHLIST:
    res = analyze_smc(item)
    if res:
        results.append(res)

# 依風盈比排序
results.sort(key=lambda x: float(x["rr"]), reverse=True)

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"成功更新 {len(results)} 檔標的至 data.json")
