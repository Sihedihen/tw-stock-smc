import json
import numpy as np
import pandas as pd
import yfinance as yf

# 監控池：涵蓋台股近半年活耀的強勢族群（AI伺服器、散熱、機器人、光通訊、特化、重電）
WATCHLIST = [
    {"symbol": "2330.TW", "code": "2330", "name": "台積電"},
    {"symbol": "3017.TW", "code": "3017", "name": "奇鋐"},
    {"symbol": "3324.TWO", "code": "3324", "name": "雙鴻"},
    {"symbol": "2383.TW", "code": "2383", "name": "台光電"},
    {"symbol": "4583.TW", "code": "4583", "name": "台灣精銳"},
    {"symbol": "6442.TW", "code": "6442", "name": "光聖"},
    {"symbol": "3583.TW", "code": "3583", "name": "辛耘"},
    {"symbol": "3131.TWO", "code": "3131", "name": "弘塑"},
    {"symbol": "2359.TW", "code": "2359", "name": "所羅門"},
    {"symbol": "1519.TW", "code": "1519", "name": "華城"},
    {"symbol": "8996.TWO", "code": "8996", "name": "高力"},
    {"symbol": "6274.TWO", "code": "6274", "name": "台燿"},
]


def analyze_multitimeframe_smc(item):
    try:
        # 下載過去 8 個月日線數據（足夠計算近半年動能與周線轉換）
        df_daily = yf.download(
            item["symbol"], period="8mo", interval="1d", progress=False
        )
        if len(df_daily) < 120:
            return None

        # 扁平化多層欄位
        df_daily = df_daily[["Open", "High", "Low", "Close", "Volume"]]
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.droplevel(1)

        # 1. 計算近半年強勢動能 (半年約 120 個交易日)
        price_now = float(df_daily["Close"].iloc[-1])
        price_6m_ago = float(df_daily["Close"].iloc[-120])
        half_year_return = ((price_now - price_6m_ago) / price_6m_ago) * 100

        # 2. 轉換為「周線級別 (Weekly HTF)」定趨勢
        df_weekly = (
            df_daily.resample("W-FRI")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna()
        )

        w_closes = df_weekly["Close"].values
        w_highs = df_weekly["High"].values
        w_lows = df_weekly["Low"].values

        # 周線 MSS：收盤價是否站穩近 12 周波段高點以上，或處於強多頭移位 (Displacement)
        w_swing_high = float(np.max(w_highs[-12:-1]))
        w_swing_low = float(np.min(w_lows[-12:-1]))
        weekly_trend_mss = (
            "周線強多頭 (HTF Bullish MSS)"
            if price_now >= w_swing_high * 0.97
            else "周線多頭回撤 (HTF Retracement)"
        )

        # 周線 FVG 偵測
        has_weekly_fvg = False
        weekly_fvg_range = "無缺口"
        for i in range(len(df_weekly) - 1, max(len(df_weekly) - 6, 2), -1):
            if w_lows[i] > w_highs[i - 2]:
                has_weekly_fvg = True
                weekly_fvg_range = f"{round(float(w_highs[i-2]), 1)} ~ {round(float(w_lows[i]), 1)}"
                break

        # 3. 「日線級別 (Daily LTF)」尋找進場架構
        d_closes = df_daily["Close"].values
        d_highs = df_daily["High"].values
        d_lows = df_daily["Low"].values
        d_dates = [d.strftime("%m/%d") for d in df_daily.index]

        # 日線 FVG 缺口回踩
        has_daily_fvg = False
        daily_fvg_zone = "無回踩缺口"
        for i in range(len(df_daily) - 1, len(df_daily) - 6, -1):
            if d_lows[i] > d_highs[i - 2]:
                has_daily_fvg = True
                daily_fvg_zone = f"{round(float(d_highs[i-2]), 1)} ~ {round(float(d_lows[i]), 1)}"
                break

        # 日線折價區 (Discount Zone) 判斷：以近 20 日波段為基準
        d_swing_high = float(np.max(d_highs[-20:-1]))
        d_swing_low = float(np.min(d_lows[-20:-1]))
        d_eq = d_swing_low + (d_swing_high - d_swing_low) * 0.5
        is_discount = price_now <= d_eq

        # 4. SMC 風盈比 (SL 錨定在日線波段低點或前低之下 1.5%)
        sl = round(d_swing_low * 0.985, 1)
        risk = price_now - sl
        tp = round(d_swing_high * 1.08, 1)
        reward = tp - price_now
        rr = round(reward / risk, 2) if risk > 0 else 0

        # 保留近 30 個交易日的日線歷史數據
        history = [
            {"date": d, "close": round(float(c), 1)}
            for d, c in zip(d_dates[-30:], d_closes[-30:])
        ]

        return {
            "code": item["code"],
            "name": item["name"],
            "price": round(price_now, 1),
            "momentum_6m": round(half_year_return, 1),  # 近半年漲幅
            "weeklyTrend": weekly_trend_mss,  # 周線大級別結構
            "weeklyFVG": weekly_fvg_range,  # 周線大級別缺口
            "dailyFVG": daily_fvg_zone,  # 日線小級別缺口
            "isDiscount": is_discount,  # 是否在日線折價區
            "sl": sl,
            "tp": tp,
            "rr": f"{rr:.2f}",
            "history": history,
        }
    except Exception as e:
        print(f"Error analyzing {item['code']}: {e}")
        return None


# 執行掃描並排序
results = []
for item in WATCHLIST:
    res = analyze_multitimeframe_smc(item)
    if res:
        results.append(res)

# 優先條件：近半年漲幅由高到低（強勢股優先），次看風盈比
results.sort(key=lambda x: (x["momentum_6m"], float(x["rr"])), reverse=True)

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"已完成多時框 SMC 運算，輸出 {len(results)} 檔標的")
