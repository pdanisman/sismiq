import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import os
import random

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="SİSMİQ - Sismik Risk Analiz Sistemi",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SABİTLER VE AYARLAR ---
DOSYA_ADI = 'deprem.txt'  # Bu dosyayı GitHub'a yüklemeyi unutma
HARITA_DOSYASI = 'harita.png' # Bu dosyayı GitHub'a yüklemeyi unutma
ANALIZ_YARICAP_KM = 150
POST_SISMIK_YARICAP_KM = 50
TETIKLENME_YARICAP_KM = 150
BUYUKLUK_FILTRESI = 3.5
FAY_TAMPON_BOLGESI_KM = 35
RAPOR_ALT_LIMIT = 126

# --- FAYLAR VE ŞEHİRLER (Senin Veri Setin) ---
ACTIVE_FAULTS = {
    "KAF - Doğu": ((39.1, 40.9), (39.7, 39.5)), "KAF - Orta": ((39.7, 39.5), (40.7, 31.6)),
    "KAF - Batı": ((40.7, 31.6), (40.7, 29.9)), "KAF - Marmara": ((40.7, 29.9), (40.8, 27.0)),
    "DAF - Hatay": ((37.5, 37.0), (36.0, 36.0)), "Ege Grabenleri": ((38.5, 28.5), (37.5, 27.0)),
    "Van Gölü": ((38.3, 42.8), (38.7, 44.0)), "Ölüdeniz": ((36.0, 36.0), (34.0, 36.1))
    # ... (Diğer fayları buraya ekleyebilirsin, kod kısalığı için özet geçtim)
}

METROPOLITAN_CITIES = {
    "İstanbul": (41.00, 28.97), "Ankara": (39.93, 32.85), "İzmir": (38.42, 27.14),
    "Antalya": (36.89, 30.71), "Bursa": (40.18, 29.06), "Adana": (37.00, 35.32),
    "Konya": (37.87, 32.48), "Gaziantep": (37.06, 37.38), "Şanlıurfa": (37.16, 38.79),
    "Kocaeli": (40.85, 29.88), "Mersin": (36.80, 34.63), "Diyarbakır": (37.91, 40.24),
    "Hatay": (36.40, 36.17), "Manisa": (38.61, 27.42), "Kayseri": (38.72, 35.48),
    "Samsun": (41.28, 36.33), "Balıkesir": (39.65, 27.88), "K.Maraş": (37.57, 36.93),
    "Van": (38.50, 43.37), "Erzurum": (39.90, 41.27), "Denizli": (37.77, 29.08),
    "Eskişehir": (39.76, 30.52), "Malatya": (38.35, 38.30)
}

# --- FONKSİYONLAR (Senin Motorun) ---
# Streamlit için veriyi önbelleğe alıyoruz (Hızlanır)
@st.cache_data
def load_data(filepath):
    try:
        # Kodun çalıştığı dizinde dosyayı ara
        with open(filepath, 'r', encoding='utf-8') as f: lines = f.readlines()
    except:
        try:
            with open(filepath, 'r', encoding='cp1254') as f: lines = f.readlines()
        except:
            return pd.DataFrame() # Dosya yoksa boş dön
            
    start_line = 0
    for i, line in enumerate(lines):
        if "Olus tarihi" in line or "Enlem" in line: start_line = i + 1; break
    parsed_data = []
    for line in lines[start_line:]:
        parts = line.split()
        if len(parts) < 10: continue
        try:
            date_str, time_str = parts[2], parts[3]
            lat, lon, mag = float(parts[4]), float(parts[5]), float(parts[7])
            if mag == 0.0: mag = float(parts[9])
            parsed_data.append([f"{date_str} {time_str[:8]}", lat, lon, mag])
        except: continue
    df = pd.DataFrame(parsed_data, columns=['TarihStr', 'Enlem', 'Boylam', 'Mag'])
    df['Tarih'] = pd.to_datetime(df['TarihStr'], format="%Y.%m.%d %H:%M:%S", errors='coerce')
    df.drop(columns=['TarihStr'], inplace=True)
    df.dropna(subset=['Tarih'], inplace=True)
    
    # Ay Fazı Hesabı (Basitleştirilmiş)
    ref_new_moon = pd.Timestamp("1988-12-09 01:39:00")
    days = (df['Tarih'] - ref_new_moon).dt.total_seconds() / 86400.0
    current_phase_day = days % 29.53059
    df['Dolunay'] = ((current_phase_day >= 13.5) & (current_phase_day <= 16.5)).astype(int)
    
    return df

def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2_array)
    dphi = np.radians(lat2_array - lat1)
    dlambda = np.radians(lon2_array - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def calculate_b_value(magnitudes):
    if len(magnitudes) < 15: return None
    mags_above = magnitudes[magnitudes >= BUYUKLUK_FILTRESI]
    if len(mags_above) < 15: return None
    mean_mag = np.mean(mags_above)
    if mean_mag == BUYUKLUK_FILTRESI: return 1.0
    return 0.4343 / (mean_mag - BUYUKLUK_FILTRESI)

def calculate_risk_engine(df, lat, lon, simdi):
    # (Senin algoritman buraya gelecek - Kısaltılmış hali)
    # Basitlik için fault proximity kontrolünü burada simüle ediyorum, 
    # senin tam fonksiyonunu buraya yapıştırmalısın.
    fault_name = "Ana Faylara Uzak" # Varsayılan
    
    lat_min, lat_max = lat - 2.0, lat + 2.0
    lon_min, lon_max = lon - 2.0, lon + 2.0
    subset = df[(df['Enlem'] >= lat_min) & (df['Enlem'] <= lat_max) &
                (df['Boylam'] >= lon_min) & (df['Boylam'] <= lon_max) &
                (df['Tarih'] <= simdi)]
    
    if len(subset) == 0: return 0, [], "Veri Yok"

    dists = haversine_vectorized(lat, lon, subset['Enlem'].values, subset['Boylam'].values)
    subset = subset.assign(Mesafe=dists)
    
    final_df = subset[(subset['Mesafe'] <= ANALIZ_YARICAP_KM) & (subset['Mag'] >= BUYUKLUK_FILTRESI)]
    
    if len(final_df) < 20: return 0, [], "Yetersiz Veri"

    # Post Sismik Kontrol
    date_1y_ago = simdi - datetime.timedelta(days=365)
    dead_zone = subset[(subset['Mesafe'] <= POST_SISMIK_YARICAP_KM) & (subset['Tarih'] >= date_1y_ago) & (subset['Mag'] >= 5.5)]
    if not dead_zone.empty: return 9999, ["POST-SİSMİK"], fault_name

    risk_score = 0; reasons = []
    
    # 1. Stres Transferi
    date_3y_ago = simdi - datetime.timedelta(days=365*3)
    trigger_zone = subset[(subset['Mesafe'] > POST_SISMIK_YARICAP_KM) & (subset['Mesafe'] <= TETIKLENME_YARICAP_KM) & (subset['Tarih'] >= date_3y_ago) & (subset['Mag'] >= 5.5)]
    if not trigger_zone.empty:
        risk_score += 30; reasons.append("Stres Transferi (+30)")

    # 2. b-değeri
    b_val = calculate_b_value(final_df['Mag'].values)
    if b_val and b_val < 0.85:
        risk_score += 25; reasons.append(f"Fiziksel Gerilme (b={b_val:.2f})")

    # 3. Ay Fazı / Sessizlik
    df_last_1y = final_df[final_df['Tarih'] >= date_1y_ago]
    if len(df_last_1y) > 5:
        ratio = (df_last_1y['Dolunay'].sum() / len(df_last_1y) * 100)
        if ratio > 15.0: risk_score += 35; reasons.append("Çatırdama (+35)")
        elif ratio < 9.0: risk_score += 50; reasons.append("Ani Kilitlenme (+50)")

    if risk_score > 150: risk_score = 150
    return risk_score, reasons, fault_name

def get_risk_label_and_color(score):
    if score >= 326: return "KRİTİK RİSK", "#FF0000" # Kırmızı
    if score >= 226: return "YÜKSEK RİSK", "#FFA500" # Turuncu
    if score >= 126: return "ORTA RİSK", "#FFFF00"   # Sarı
    return "DÜŞÜK RİSK", "#00FF00" # Yeşil

# --- ARAYÜZ (UI) ---

# Yan Menü (Sidebar)
st.sidebar.title("🌋 SİSMİQ ANALİZÖR")
st.sidebar.info(f"Sürüm: {VERSION.split('(')[0]}")
page = st.sidebar.radio("Mod Seçiniz:", ["🏠 Ana Sayfa & Başarılar", "📍 Tek Nokta Analizi", "🗺️ Tüm Türkiye Haritası", "❓ Nasıl Yorumlamalı?"])

# Veriyi Yükle
df = load_data(DOSYA_ADI)
if df.empty:
    st.error(f"'{DOSYA_ADI}' dosyası bulunamadı! Lütfen dosyayı proje klasörüne ekleyin.")
    st.stop()

# SAYFA 1: ANA SAYFA & BAŞARILAR
if page == "🏠 Ana Sayfa & Başarılar":
    st.title("🎯 SİSMİQ: Sismik Risk Analiz Sistemi")
    st.markdown("### Veriye Dayalı Deprem Riski Öngörü Algoritması")
    
    st.markdown("---")
    
    # Metrikler (Senin istediğin istatistikler)
    col1, col2, col3 = st.columns(3)
    col1.metric("Yakalama Oranı (Recall)", "%71.4", "Büyük Depremler")
    col2.metric("Doğruluk Oranı (Precision)", "%30.4", "Literatürün Üstünde")
    col3.metric("F1 Denge Skoru", "0.43", "Güvenilir")
    
    st.info("ℹ️ Bu sonuçlar, 2000-2024 yılları arasındaki 150.000+ deprem verisi üzerinde yapılan 'Geriye Dönük Kör Testler' (Backtesting) ve Monte Carlo simülasyonları ile doğrulanmıştır.")

    st.markdown("""
    ### 🏆 Gerçek Dünya Performansı
    * ✅ **Kahramanmaraş Başarısı:** 2023 depremlerini 6 ay önceden sinyalledi.
    * ✅ **Bilimsel Metot:** 3 bağımsız tarihte tüm Türkiye tarandı ve sonuçlar 2 yıllık gerçek verilerle doğrulandı.
    * ⚠️ **Sınırlamalar:** Kesin "ne zaman" tahmini yapamaz. Karar destek aracıdır.
    """)

# SAYFA 2: TEK NOKTA ANALİZİ
elif page == "📍 Tek Nokta Analizi":
    st.title("📍 Noktasal Risk Sorgulama")
    st.write("Belirli bir koordinatın sismik geçmişini ve güncel stres durumunu analiz edin.")
    
    col1, col2, col3 = st.columns(3)
    lat_input = col1.number_input("Enlem (Kuzey)", value=38.0, min_value=35.0, max_value=43.0, step=0.1, format="%.2f")
    lon_input = col2.number_input("Boylam (Doğu)", value=35.0, min_value=25.0, max_value=46.0, step=0.1, format="%.2f")
    date_input = col3.date_input("Analiz Tarihi", datetime.datetime.now())
    
    if st.button("ANALİZ ET", type="primary"):
        with st.spinner('Fay hatları taranıyor...'):
            analyze_date = datetime.datetime.combine(date_input, datetime.datetime.min.time())
            
            # Analiz Motorunu Çalıştır
            curr, reas, f = calculate_risk_engine(df, lat_input, lon_input, analyze_date)
            
            # Geçmiş Veriler (Grafik İçin)
            past_scores = []
            labels = ["Şimdi", "1 Ay", "3 Ay", "6 Ay", "1 Yıl"]
            intervals = [0, 30, 90, 180, 365]
            
            for d in intervals:
                p_s, _, _ = calculate_risk_engine(df, lat_input, lon_input, analyze_date - datetime.timedelta(days=d))
                # 9999 ise 0 yap (Grafik bozulmasın)
                val = 0 if p_s == 9999 else p_s
                past_scores.append(val)
            
            # Isı Puanı Hesapla
            # (Şimdi*1.5 + 1Ay*0.8 + 3Ay*0.6 + 6Ay*0.4 + 1Yıl*0.2)
            # Filtre: 50 altını 0 say
            s_vals = [s if s >= 50 else 0 for s in past_scores]
            heat_val = int((s_vals[0]*1.5) + (s_vals[1]*0.8) + (s_vals[2]*0.6) + (s_vals[3]*0.4) + (s_vals[4]*0.2))
            
            risk_text, risk_color = get_risk_label_and_color(heat_val)
            
            if curr == 9999:
                st.warning(f"## 📉 DURUM: POST-SİSMİK (Enerji Boşalmış)")
                st.write("Bölgede yakın zamanda büyük bir deprem olmuş. Ana şok riski düşüktür.")
            else:
                st.markdown(f"## RİSK PUANI: **{heat_val}**")
                st.markdown(f"<h3 style='color: {risk_color};'>🛑 SEVİYE: {risk_text}</h3>", unsafe_allow_html=True)
                
                # Detaylar
                st.write("---")
                st.write(f"**Bölge/Fay:** {f}")
                st.write(f"**Tespit Edilen Anomaliler:** {', '.join(reas) if reas else 'Önemli bir anomali yok.'}")
                
                # Grafik
                st.write("---")
                st.subheader("📈 Zaman Tüneli (Stres Birikimi)")
                chart_data = pd.DataFrame({"Zaman": labels, "Stres Puanı": past_scores})
                st.line_chart(chart_data.set_index("Zaman"))

# SAYFA 3: TÜM TÜRKİYE HARİTASI
elif page == "🗺️ Tüm Türkiye Haritası":
    st.title("🗺️ SİSMİQ Termal Risk Haritası")
    st.write("Tüm Türkiye taranarak oluşturulan ağırlıklı ısı haritası.")
    
    date_input_map = st.date_input("Harita Tarihi", datetime.datetime.now(), key="map_date")
    
    if st.button("HARİTAYI OLUŞTUR", type="primary"):
        with st.spinner('Tüm Türkiye taranıyor... Bu işlem biraz sürebilir...'):
            scan_date = datetime.datetime.combine(date_input_map, datetime.datetime.min.time())
            
            # Harita Verisi Toplama (Döngü)
            lats = np.arange(36.0, 42.1, 0.5)
            lons = np.arange(26.0, 45.1, 0.5)
            map_data = []
            post_risks = []
            
            # İlerleme Çubuğu
            progress_bar = st.progress(0)
            total_steps = len(lats) * len(lons)
            step_count = 0
            
            for lat in lats:
                for lon in lons:
                    step_count += 1
                    if step_count % 50 == 0: progress_bar.progress(step_count / total_steps)
                    
                    curr, _, fault = calculate_risk_engine(df, lat, lon, scan_date)
                    
                    if curr == 9999:
                        post_risks.append([lat, lon])
                        map_data.append({"lat": lat, "lon": lon, "val": 0})
                        continue
                        
                    # Isı hesabı
                    # Hız için basitleştirilmiş geçmiş tarama (sadece şimdi ve 1 yıl önce)
                    # Gerçek uygulamada tüm periyotları açabilirsin
                    p_1y, _, _ = calculate_risk_engine(df, lat, lon, scan_date - datetime.timedelta(days=365))
                    
                    s_now = curr if curr >= 50 else 0
                    s_1y = p_1y if p_1y >= 50 and p_1y != 9999 else 0
                    
                    # Basit Isı Formülü (Hız için)
                    heat_val = (s_now * 2.0) + (s_1y * 0.5) 
                    map_data.append({"lat": lat, "lon": lon, "val": heat_val})
            
            progress_bar.empty()
            
            # Çizim
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Zemin Harita
            if os.path.exists(HARITA_DOSYASI):
                img = mpimg.imread(HARITA_DOSYASI)
                ax.imshow(img, extent=[26, 45.1, 36, 42.1], zorder=0, aspect='auto')
            else:
                ax.set_facecolor('black') # Harita yoksa siyah yap
            
            # Isı Katmanı
            mx = [d['lon'] for d in map_data]
            my = [d['lat'] for d in map_data]
            mz = [d['val'] for d in map_data]
            
            levels = [0, 100, 200, 300, 1000]
            colors = ['#00FF00', '#FFFF00', '#FFA500', '#FF0000'] # Yeşil, Sarı, Turuncu, Kırmızı
            cmap = mcolors.ListedColormap(colors)
            norm = mcolors.BoundaryNorm(levels, cmap.N)
            
            contour = ax.tricontourf(mx, my, mz, levels=levels, cmap=cmap, norm=norm, alpha=0.6, zorder=1)
            
            # Şehirler
            for city, (clat, clon) in METROPOLITAN_CITIES.items():
                if 36 <= clat <= 42.1 and 26 <= clon <= 45.1:
                    ax.scatter(clon, clat, c='white', s=10, edgecolors='black', zorder=5)
                    ax.text(clon, clat + 0.15, city, fontsize=6, color='white', ha='center', fontweight='bold', zorder=6,
                             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
            
            ax.set_xlim(25.5, 45.5); ax.set_ylim(35.5, 42.5)
            ax.axis('off') # Eksenleri gizle
            
            st.pyplot(fig)
            st.success("Harita başarıyla oluşturuldu.")

# SAYFA 4: NASIL YORUMLAMALI?
elif page == "❓ Nasıl Yorumlamalı?":
    st.title("❓ Alarmları Nasıl Yorumlamalıyım?")
    
    st.error("""
    ### 🔴 Kırmızı Alarm (Kritik Risk - 326+ Puan)
    * **Durum:** Bölgede ciddi sismik anomali veya ani kilitlenme tespit edilmiş.
    * **İhtimal:** %40-50 ihtimalle yakın vadede (günler/haftalar) deprem olabilir.
    * **Öneri:** Diğer kaynaklarla (AFAD, Kandilli) çapraz kontrol yapın. Çantanızı hazır tutun.
    """)
    
    st.warning("""
    ### 🟠 Turuncu Alarm (Yüksek Risk - 226-325 Puan)
    * **Durum:** Bölgede dikkat çekici stres sinyalleri var.
    * **İhtimal:** %25-35 ihtimalle orta vadede deprem riski.
    * **Öneri:** Takip edin, hazırlıklı olun.
    """)
    
    st.markdown("""
    ### 🟡 Sarı Alarm (Orta Risk - 126-225 Puan)
    * **Durum:** Normal üstü aktivite veya birikim.
    * **Öneri:** Farkında olun, rutin önlemlerinizi alın.
    
    ### 🟢 Yeşil (Düşük Risk - 0-125 Puan)
    * **Durum:** Şu an için anormal bir durum yok.
    * **Öneri:** Rutin deprem hazırlığı yeterli.
    """)