import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from scipy.interpolate import PchipInterpolator
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from matplotlib.lines import Line2D
import tcmarkers

# ==================== 全局設定 ====================
# 優先讀取本地同目錄的 CSV，如果沒有則讀取 GitHub 遠端
SITE_BASE_URL = "https://raw.githubusercontent.com/zax41006-bot/TC-Track/main"

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

INTENSITY_CONFIG = {
    "TD":   {"cn": "熱帶低氣壓", "color": "#FAD02C"},
    "TS":   {"cn": "熱帶風暴",   "color": "#0072C6"},
    "STS":  {"cn": "強烈熱帶風暴", "color": "#00A651"},
    "TY":   {"cn": "颱風",       "color": "#FF9900"},
    "STY":  {"cn": "強颱風",     "color": "#E60012"},
    "SUTY": {"cn": "超強颱風",   "color": "#7B1FA2"},
    "LPA":  {"cn": "低壓區",     "color": "#A0A0A0"},
    "MD":   {"cn": "低壓區",     "color": "#A0A0A0"},
    "EX":   {"cn": "溫帶氣旋",   "color": "#8D8D8D"}
}

def get_intensity_color(intensity_str):
    if intensity_str in INTENSITY_CONFIG:
        return INTENSITY_CONFIG[intensity_str]["color"]
    for k, v in INTENSITY_CONFIG.items():
        if v["cn"] == intensity_str:
            return v["color"]
    return "#A0A0A0"

def draw_chart(tc_data):
    tc_id = tc_data['id']
    tc_name = tc_data['name']
    
    lon_min = tc_data.get('lon_min', 100.0)
    lon_max = tc_data.get('lon_max', 180.0)
    lat_min = tc_data.get('lat_min', 0.0)
    lat_max = tc_data.get('lat_max', 60.0)
    target_hours = sorted(list(set(tc_data.get('target_hours', [24, 48, 72, 96, 120]))))
    
    # 優先嘗試本地檔案，若無則讀取遠端
    past_csv = f"past_track_{tc_id}.csv"
    forecast_csv = f"forecast_track_{tc_id}.csv"
    past_url = past_csv if os.path.exists(past_csv) else f"{SITE_BASE_URL}/past_track_{tc_id}.csv"
    forecast_url = forecast_csv if os.path.exists(forecast_csv) else f"{SITE_BASE_URL}/forecast_track_{tc_id}.csv"
    
    try:
        df_past = pd.read_csv(past_url)
        df_past.columns = df_past.columns.str.strip()
        past_data = df_past[['datetime', 'lng', 'lat', 'wind', 'minimum central pressure', 'intensity']].values.tolist()
        curr = past_data[-1]
        base_time = pd.to_datetime(curr[0], dayfirst=True)
        curr_intensity_code = curr[5]
        
        forecast_data = []
        try:
            df_forecast = pd.read_csv(forecast_url)
            df_forecast.columns = df_forecast.columns.str.strip()
            for _, row in df_forecast.iterrows():
                h = int(str(row['f_time']).replace('hr', '').strip())
                forecast_data.append([f"{h}hr", row['lng'], row['lat'], row['wind'], h, row['minimum central pressure'], row['intensity']])
        except Exception as e:
            print(f"⚠️ 無法讀取預報數據 ({e})，僅繪製過去路徑。")

        fig, ax = plt.subplots(figsize=(12, 11), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # ==================== 高精度底圖 ====================
        ocean_f = NaturalEarthFeature('physical', 'ocean', '10m', facecolor="#D9ECF7", edgecolor="none")
        ax.add_feature(ocean_f, zorder=0)
        
        land_f = NaturalEarthFeature('physical', 'land', '10m', facecolor="#FCFCFC", edgecolor="#CCCCCC", linewidth=0.4)
        ax.add_feature(land_f, zorder=1)
        
        coast_f = NaturalEarthFeature('physical', 'coastline', '10m', facecolor='none', edgecolor='#666666', linewidth=0.7)
        ax.add_feature(coast_f, zorder=2)
        
        borders_f = NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m', facecolor='none', edgecolor='#888888', linewidth=0.6, linestyle='--')
        ax.add_feature(borders_f, zorder=2)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='#888888', alpha=0.3, linestyle='--', zorder=3)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {'size': 9, 'weight':'bold'}
        gl.ylabel_style = {'size': 9, 'weight':'bold'}
        gl.xlocator = mticker.MultipleLocator(5)
        gl.ylocator = mticker.MultipleLocator(5)
        
        # ==================== 過去路徑 ====================
        past_lngs = [d[1] for d in past_data]
        past_lats = [d[2] for d in past_data]
        ax.plot(past_lngs, past_lats, color="black", lw=1.8, zorder=5)
        
        # 【修改點】過去路徑：黑環圓點，內填強度顏色
        for d in past_data[:-1]:
            p_color = get_intensity_color(d[5])
            ax.plot(d[1], d[2], marker='o', ms=4.5, 
                    markerfacecolor=p_color,  # 填充強度顏色
                    markeredgecolor='black',  # 黑色邊框
                    markeredgewidth=0.8,      # 邊框粗細
                    zorder=6)
            
        # ==================== 預報路徑與誤差圈 ====================
        if forecast_data:
            f_hs = [d[4] for d in forecast_data]
            all_h = [0] + f_hs
            all_ln = [curr[1]] + [d[1] for d in forecast_data]
            all_lt = [curr[2]] + [d[2] for d in forecast_data]
            all_er = [0] + [((h // 24) * 100 + (h % 24) * (100 / 24)) * (1 / 111) for h in f_hs]
            
            ih = np.linspace(0, max(all_h), 100)
            xi = PchipInterpolator(all_h, all_ln)(ih)
            yi = PchipInterpolator(all_h, all_lt)(ih)
            ri = PchipInterpolator(all_h, all_er)(ih)
            
            ps = [Polygon(np.dstack((xi[i] + ri[i] * np.cos(np.linspace(0, 2 * np.pi, 360)), yi[i] + ri[i] * np.sin(np.linspace(0, 2 * np.pi, 360))))[0]) for i in range(len(ih))]
            ax.add_geometries([unary_union([MultiPolygon([ps[i], ps[i + 1]]).convex_hull for i in range(len(ps) - 1)])], 
                              ccrs.PlateCarree(), fc="#d1d1d1", alpha=0.5, ec="#707070", lw=0.7, zorder=4)
            ax.plot(xi, yi, color="#002a7d", lw=2.2, ls='--', zorder=5)
            
        # ==================== 預報節點與實時位置（颱風圖標） ====================
        for d in forecast_data:
            if d[4] not in target_hours: 
                continue
            col = get_intensity_color(d[6])
            ax.plot(d[1], d[2], marker=tcmarkers.HU, ms=5.5, color=col, mec='k', mew=0.4, zorder=10)
            
        ax.plot(curr[1], curr[2], marker=tcmarkers.HU, ms=6.5, color=get_intensity_color(curr_intensity_code), mec='k', mew=0.5, zorder=11)
        
        # ==================== 資訊文字方塊 ====================
        info_lines = [
            "現時資訊：",
            f"{base_time.strftime('%d/%m/%Y %H:%M')}",
            f"{curr_intensity_code}, {curr[3]}kph, {curr[4]}hPa",
            "──────────────────",
            "預報資訊："
        ]
        
        for d in forecast_data:
            if d[4] not in target_hours: 
                continue
            f_time = base_time + pd.Timedelta(hours=d[4])
            info_lines.append(f"{f_time.strftime('%d/%m/%Y %H:%M')}  {d[6]}")
            
        ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes, ha='left', va='top',
                fontsize=10.0, fontweight='bold', color='#111111', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#B0BEC5", alpha=0.92, lw=0.8), zorder=25)
                
        max_forecast_hr = max([d[4] for d in forecast_data]) if forecast_data else 0
        
        # ==================== 標題與機構 ====================
        fig.text(0.5, 0.950, f"熱帶氣旋「{tc_name}」路徑預報圖", ha='center', va='top', 
                 fontsize=20, fontweight='bold', color='#111111')
        fig.text(0.5, 0.910, f"預報時效：{max_forecast_hr}小時", ha='center', va='top', 
                 fontsize=13, fontweight='bold', color='#222222')
        
        ax.text(0.98, 0.98, "港澳天氣站 HKMO_MET 發佈", transform=ax.transAxes, ha='right', va='top', fontsize=14, fontweight='bold',
                color='#003399', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="none"), zorder=20)
                
        # ==================== 圖例 ====================
        leg_core = [
            Line2D([0], [0], color="black", lw=2.3, label='過去路徑'),
            Line2D([0], [0], color="#003399", lw=2.5, ls='--', label='預報路徑'),
            plt.Rectangle((0, 0), 1, 1, fc="#C8C8C8", alpha=0.45, ec="#707070", lw=1.2, label='預報誤差範圍')
        ]
        
        leg_int = [
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["LPA"]["color"], label='低壓區', ms=9.0, mec='k', mew=0.7, ls=''),
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["TD"]["color"], label='熱帶低氣壓', ms=9.0, mec='k', mew=0.7, ls=''),
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["TS"]["color"], label='熱帶風暴', ms=9.0, mec='k', mew=0.7, ls=''),
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["STS"]["color"], label='強烈熱帶風暴', ms=9.0, mec='k', mew=0.7, ls=''),
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["TY"]["color"], label='颱風', ms=9.0, mec='k', mew=0.7, ls=''),
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["STY"]["color"], label='強颱風', ms=9.0, mec='k', mew=0.7, ls=''),
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["SUTY"]["color"], label='超強颱風', ms=9.0, mec='k', mew=0.7, ls='')
        ]
        
        leg_params = dict(loc='lower center', frameon=True, edgecolor='#CFD8DC', facecolor='white', framealpha=0.95)
        fig.legend(handles=leg_core, ncol=3, bbox_to_anchor=(0.5, 0.15), fontsize=12.0, prop={'weight':'bold'}, **leg_params)
        fig.legend(handles=leg_int, ncol=8, bbox_to_anchor=(0.5, 0.10), fontsize=12.0, prop={'weight':'bold'}, **leg_params)
        
        plt.subplots_adjust(bottom=0.24, top=0.86)
        os.makedirs("output", exist_ok=True)
        out_path = f"output/forecast_{tc_id}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ {tc_name} ({tc_id}) 預報圖已生成至 {out_path}")
        
    except Exception as e:
        print(f"❌ {tc_name} ({tc_id}) 繪圖失敗: {e}")

if __name__ == "__main__":
    if not os.path.exists('tc_config.json'):
        print("⚠️ 找不到 tc_config.json")
        sys.exit(1)
        
    with open('tc_config.json', 'r', encoding='utf-8-sig') as f:
        tc_list = json.load(f)
    
    if len(sys.argv) > 1:
        target_id = sys.argv[1].upper()
        for tc in tc_list:
            if tc['id'] == target_id:
                draw_chart(tc)
                break
    else:
        print(f"🌐 準備繪製所有氣旋，共 {len(tc_list)} 個...")
        for tc in tc_list:
            draw_chart(tc)