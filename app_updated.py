"""
Prediksi Harga Mobil Bekas - Streamlit
Tiga cara prediksi sekaligus : Decision Tree, Random Forest, Gradient Boosting.

Menjalankan lokal :
    pip install -r requirements.txt
    streamlit run app.py

Dataset (quikr_car.csv) dicari dengan urutan :
    1. file quikr_car.csv di folder yang sama dengan app.py (atau folder data/)
    2. unduh otomatis dari Kaggle lewat kagglehub
    3. upload manual lewat halaman utama aplikasi
"""
import importlib
import importlib.util
import gc
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Harga Mobil Bekas", page_icon="🚗", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f9fc;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 249, 252, 0.9);
    }

    .hero {
        background: #ffffff;
        border: 1px solid #dfe6ef;
        border-left: 5px solid #2563eb;
        border-radius: 16px;
        padding: 1.25rem 1.45rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .hero h1 {
        margin: 0;
        color: #0f172a;
        font-size: 2rem;
        letter-spacing: -0.02em;
    }

    .hero p {
        margin: 0.4rem 0 0;
        color: #64748b;
        font-size: 0.95rem;
    }

    .section-label {
        color: #2563eb;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #dfe6ef;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        min-height: 100px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
    }

    .result-label {
        color: #64748b;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .result-value {
        color: #0f172a;
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 0.25rem;
    }

    .result-note {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 0.15rem;
    }

    .winner-card {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin: 0.55rem 0 1rem;
        color: #1e3a5f;
    }

    .formula-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        color: #334155;
        font-family: monospace;
        font-size: 0.84rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dfe6ef;
        border-radius: 14px;
        padding: 0.65rem 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def pastikan_paket(modul, paket_pip):
    """Pasang paket secara otomatis bila belum ada di server.

    Biasanya paket dipasang lewat requirements.txt. Fungsi ini hanya jaring pengaman
    kalau file tersebut tidak terbaca saat deploy (ModuleNotFoundError).
    """
    if importlib.util.find_spec(modul) is not None:
        return
    with st.spinner(f"Memasang {paket_pip} (hanya sekali, sekitar 1 menit)..."):
        hasil = subprocess.run(
            [sys.executable, "-m", "pip", "install", paket_pip],
            capture_output=True, text=True,
        )
    if hasil.returncode != 0:
        st.error(
            f"Paket `{paket_pip}` belum terpasang dan pemasangan otomatis gagal. "
            "Pastikan file `requirements.txt` ada di folder yang sama dengan `app.py` di repo GitHub, "
            "lalu reboot app."
        )
        st.code(hasil.stderr[-1500:])
        st.stop()
    importlib.invalidate_caches()


pastikan_paket("sklearn", "scikit-learn>=1.4")

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

FITUR_KATEGORI = ["company", "model", "fuel_type"]
FITUR_ANGKA = ["year", "kms"]
TARGET = "Price"
MODEL_LAIN = "(model lain, ketik sendiri)"
NAMA_MODEL = ["Decision Tree", "Random Forest", "Gradient Boosting"]

LOKASI_CSV = ["quikr_car.csv", os.path.join("data", "quikr_car.csv")]


# ----------------------------------------------------------------------------
# Memuat & membersihkan data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def baca_csv(sumber):
    return pd.read_csv(sumber, dtype=str)


@st.cache_data(show_spinner="Mengunduh dataset dari Kaggle...")
def unduh_kaggle():
    import kagglehub

    path = kagglehub.dataset_download("muhammadmusharraf444/used-car-price-prediction-dataset")
    return pd.read_csv(os.path.join(path, "quikr_car.csv"), dtype=str)


def muat_data_mentah():
    for lokasi in LOKASI_CSV:
        if os.path.exists(lokasi):
            return baca_csv(lokasi)
    try:
        return unduh_kaggle()
    except Exception as error:
        st.warning(f"Dataset belum berhasil diunduh otomatis dari Kaggle: {error}")

    st.warning(
        "Dataset `quikr_car.csv` tidak ditemukan dan tidak bisa diunduh otomatis. "
        "Upload file CSV-nya di bawah ini."
    )
    unggahan = st.file_uploader("Upload quikr_car.csv", type="csv")
    if unggahan is None:
        st.stop()
    return baca_csv(unggahan)


@st.cache_data(show_spinner=False)
def bersihkan_data(mentah):
    """Pembersihan sama persis dengan notebook (bagian persiapan data)."""
    data = mentah.copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"].str.replace(",", "", regex=False), errors="coerce")
    data["kms"] = pd.to_numeric(
        data["kms_driven"].str.replace(",", "", regex=False).str.extract(r"(\d+)")[0],
        errors="coerce",
    )

    data = data.dropna(subset=["name", "company", "year", "Price", "kms", "fuel_type"]).drop_duplicates()
    data["model"] = data["name"].str.split().str[:3].str.join(" ")

    q1, q3 = data["Price"].quantile([0.25, 0.75])
    batas_atas = q3 + 1.5 * (q3 - q1)
    data = data[(data["Price"] <= batas_atas) & (data["kms"] <= 200_000)].reset_index(drop=True)
    return data, float(batas_atas)


# ----------------------------------------------------------------------------
# Training 3 model
# ----------------------------------------------------------------------------
def buat_pipeline(estimator):
    pre = ColumnTransformer(
        [("kategori", OneHotEncoder(handle_unknown="ignore"), FITUR_KATEGORI)],
        remainder="passthrough",
    )
    return Pipeline([("preprocess", pre), ("model", estimator)])


@st.cache_resource(show_spinner=False)
def latih_semua_model(data):
    X = data[FITUR_KATEGORI + FITUR_ANGKA]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kandidat = {
        "Decision Tree": (
            DecisionTreeRegressor(random_state=42),
            {"max_depth": [8, 12, None], "min_samples_leaf": [2, 5, 10]},
        ),
        "Random Forest": (
            # Satu proses worker agar penggunaan RAM Streamlit Cloud lebih terkendali.
            RandomForestRegressor(random_state=42, n_estimators=80, n_jobs=1),
            {
                "max_depth": [10, None],
                "min_samples_leaf": [2, 5],
                "max_features": [0.5],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [80, 120], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        ),
    }

    model, metrik, param_terbaik, pred_test, penting = {}, [], {}, {}, {}
    for nama, (estimator, grid) in kandidat.items():
        pencarian = GridSearchCV(
            buat_pipeline(estimator),
            {f"model__{k}": v for k, v in grid.items()},
            # CV 5 lipatan sesuai ketentuan tugas/dosen.
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
            pre_dispatch=1,
            refit=True,
        )
        pencarian.fit(X_train, y_train)
        terbaik = pencarian.best_estimator_

        p_train, p_test = terbaik.predict(X_train), terbaik.predict(X_test)
        metrik.append(
            {
                "Model": nama,
                "R2 Training": r2_score(y_train, p_train),
                "R2 Testing": r2_score(y_test, p_test),
                "MAE Training": mean_absolute_error(y_train, p_train),
                "MAE Testing": mean_absolute_error(y_test, p_test),
                "RMSE Testing": float(np.sqrt(mean_squared_error(y_test, p_test))),
            }
        )
        model[nama] = terbaik
        param_terbaik[nama] = {k.replace("model__", ""): v for k, v in pencarian.best_params_.items()}
        pred_test[nama] = p_test

        nama_fitur = terbaik.named_steps["preprocess"].get_feature_names_out()
        nilai = pd.Series(terbaik.named_steps["model"].feature_importances_, index=nama_fitur)
        penting[nama] = nilai.groupby(_kolom_asal).sum()

    metrik = pd.DataFrame(metrik).set_index("Model")
    metrik["Selisih R2 (Train - Test)"] = metrik["R2 Training"] - metrik["R2 Testing"]

    return {
        "model": model,
        "metrik": metrik,
        "param": param_terbaik,
        "penting": pd.DataFrame(penting).fillna(0),
        "X_train": X_train,
        "y_train": y_train,
        "y_test": y_test,
        "pred_test": pred_test,
    }


def _kolom_asal(nama):
    """Gabungkan hasil One-Hot (mis. company_Hyundai) kembali ke kolom asalnya."""
    nama = nama.replace("kategori__", "").replace("remainder__", "")
    for kolom in FITUR_KATEGORI:
        if nama.startswith(kolom + "_"):
            return kolom
    return nama


# ----------------------------------------------------------------------------
# Prediksi
# ----------------------------------------------------------------------------
def prediksi_semua(hasil, merk, nama_mobil, tahun, kms, bbm):
    model_mobil = " ".join(nama_mobil.split()[:3])
    baris = pd.DataFrame(
        [{"company": merk, "model": model_mobil, "fuel_type": bbm, "year": float(tahun), "kms": float(kms)}]
    )

    keluaran = {}
    for nama, pipeline in hasil["model"].items():
        harga = max(float(pipeline.predict(baris)[0]), 0.0)
        rendah = tinggi = None
        pre, estimator = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]

        if nama == "Random Forest":
            # Sebaran prediksi dari tiap pohon
            angka = pre.transform(baris)
            per_pohon = np.array([p.predict(angka)[0] for p in estimator.estimators_])
            rendah, tinggi = np.percentile(per_pohon, [10, 90])
            keterangan = "Rentang prediksi antar pohon (P10 - P90)"
        elif nama == "Decision Tree":
            # Harga data training yang jatuh di daun yang sama
            daun = estimator.apply(pre.transform(baris))[0]
            daun_train = estimator.apply(pre.transform(hasil["X_train"]))
            mirip = hasil["y_train"][daun_train == daun]
            if len(mirip) > 1:
                rendah, tinggi = mirip.min(), mirip.max()
            keterangan = f"Rentang mobil serupa di data training ({len(mirip)} data)"
        else:
            keterangan = "Boosting tidak punya rentang bawaan"

        keluaran[nama] = {"harga": harga, "rendah": rendah, "tinggi": tinggi, "keterangan": keterangan}
    return keluaran


def rp(nilai):
    return "Rp " + f"{nilai:,.0f}".replace(",", ".")


def inr(nilai):
    return "INR " + f"{nilai:,.0f}".replace(",", ".")


# ----------------------------------------------------------------------------
# Teks penjelasan (diambil dari notebook)
# ----------------------------------------------------------------------------
PENJELASAN = {
    "Decision Tree": {
        "ikon": "🌳",
        "ringkas": "Satu pohon keputusan",
        "cara": (
            "Membuat satu pohon pertanyaan Ya/Tidak (mis. \"dibuat tahun 2012 atau lebih lama?\", "
            "\"merk Maruti?\") sampai tiba di satu daun. Prediksinya adalah rata-rata harga mobil "
            "pada data training yang jatuh di daun yang sama."
        ),
        "plus": "Mudah dipahami dan dijelaskan, alurnya bisa digambar.",
        "minus": "Mudah menghafal data (overfitting) dan hasilnya bisa berubah jauh bila data sedikit berubah.",
    },
    "Random Forest": {
        "ikon": "🌲",
        "ringkas": "Banyak pohon, hasilnya dirata-rata",
        "cara": (
            "Membangun banyak pohon keputusan. Tiap pohon dilatih dari sampel data dan subset fitur "
            "yang berbeda, lalu hasil semua pohon dirata-rata. Sebaran hasil antar pohon dipakai "
            "sebagai rentang prediksi."
        ),
        "plus": "Biasanya lebih stabil daripada satu pohon.",
        "minus": "Lebih lambat dilatih dan sulit dijelaskan, karena tidak ada satu pohon yang bisa digambar sebagai alur keputusan.",
    },
    "Gradient Boosting": {
        "ikon": "🚀",
        "ringkas": "Pohon kecil yang saling memperbaiki",
        "cara": (
            "Membangun pohon-pohon kecil secara berurutan. Setiap pohon baru khusus memperbaiki "
            "kesalahan pohon sebelumnya, lalu semua hasilnya dijumlahkan."
        ),
        "plus": "Sering paling akurat untuk data tabel seperti ini.",
        "minus": "Paling sensitif terhadap pengaturan parameter dan bisa overfitting bila pohonnya terlalu banyak.",
    },
}


def kekuatan_korelasi(r):
    r = abs(r)
    for batas, nama in [(0.2, "sangat lemah"), (0.4, "lemah"), (0.6, "sedang"), (0.8, "kuat")]:
        if r < batas:
            return nama
    return "sangat kuat"


def kalimat_korelasi(label, r, teks_positif, teks_negatif):
    arah = "positif" if r >= 0 else "negatif"
    if abs(r) < 0.2:
        tafsir = "Hubungan linearnya hampir tidak terlihat."
    else:
        tafsir = teks_positif if r >= 0 else teks_negatif
    return f"- **{label}** : korelasi {r:+.2f} ({kekuatan_korelasi(r)}, {arah}). {tafsir}"


def analisis_overfitting(selisih):
    if selisih > 0.2:
        return "indikasi overfitting cukup kuat"
    if selisih > 0.1:
        return "ada indikasi overfitting"
    return "tidak terlihat overfitting yang besar"


# ----------------------------------------------------------------------------
# Penjelasan hasil prediksi
# ----------------------------------------------------------------------------
def angka(nilai):
    """Format angka dengan pemisah titik (gaya Indonesia)."""
    return f"{nilai:,.0f}".replace(",", ".")


def rp_selisih(nilai):
    tanda = "+" if nilai > 0 else "-" if nilai < 0 else ""
    return tanda + rp(abs(nilai))


def harga_rata_banyak(hasil, daftar_baris):
    """Rata-rata prediksi ketiga model untuk banyak baris sekaligus."""
    masukan = pd.DataFrame(daftar_baris)
    semua = np.column_stack([np.maximum(p.predict(masukan), 0) for p in hasil["model"].values()])
    return semua.mean(axis=1)


def simulasi_perubahan(hasil, data, merk, model_mobil, tahun, kms, bbm, kurs):
    """Ubah satu hal pada satu waktu (lainnya tetap), lalu lihat perubahan harga rata-rata."""
    tahun_min, tahun_max = int(data["year"].min()), int(data["year"].max())

    skenario = []  # (faktor, deskripsi, tahun, kms, bbm)
    if min(tahun + 3, tahun_max) != tahun:
        t = min(tahun + 3, tahun_max)
        skenario.append(("Tahun pembuatan", f"Dibuat tahun {t} (lebih baru)", t, kms, bbm))
    if max(tahun - 3, tahun_min) != tahun:
        t = max(tahun - 3, tahun_min)
        skenario.append(("Tahun pembuatan", f"Dibuat tahun {t} (lebih lama)", t, kms, bbm))
    skenario.append(("Jarak tempuh", f"Jarak tempuh {angka(kms + 25_000)} km (+25.000 km)", tahun, kms + 25_000, bbm))
    if kms > 0:
        k = max(kms - 25_000, 0)
        skenario.append(("Jarak tempuh", f"Jarak tempuh {angka(k)} km (-25.000 km)", tahun, k, bbm))
    for b in sorted(data["fuel_type"].unique()):
        if b != bbm:
            skenario.append(("Bahan bakar", f"Bahan bakar {b}", tahun, kms, b))

    def baris(t, k, b):
        return {"company": merk, "model": model_mobil, "fuel_type": b, "year": float(t), "kms": float(k)}

    harga = harga_rata_banyak(hasil, [baris(tahun, kms, bbm)] + [baris(t, k, b) for _, _, t, k, b in skenario])
    dasar, lainnya = harga[0], harga[1:]

    tabel = pd.DataFrame(
        {
            "Faktor": [s[0] for s in skenario],
            "Skenario": [s[1] for s in skenario],
            "selisih": (lainnya - dasar) * kurs,
            "persen": (lainnya - dasar) / max(dasar, 1) * 100,
            "harga": lainnya * kurs,
        }
    )
    return tabel


def jalur_decision_tree(hasil, baris):
    """Telusuri pertanyaan Ya/Tidak yang dilewati mobil ini di Decision Tree."""
    pipeline = hasil["model"]["Decision Tree"]
    pre, pohon = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    x = pre.transform(baris)
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    nama = pre.get_feature_names_out()
    t = pohon.tree_

    langkah, node = [], 0
    while t.children_left[node] != t.children_right[node]:
        f, batas, nilai = t.feature[node], t.threshold[node], x[0, t.feature[node]]
        n = nama[f].replace("kategori__", "").replace("remainder__", "")
        if n == "year":
            tanya, ya = f"Dibuat tahun {int(np.floor(batas))} atau lebih lama?", nilai <= batas
        elif n == "kms":
            tanya, ya = f"Jarak tempuh {angka(batas)} km atau kurang?", nilai <= batas
        elif n.startswith("company_"):
            tanya, ya = f"Merk mobil adalah {n[len('company_'):]}?", nilai > batas
        elif n.startswith("model_"):
            tanya, ya = f"Model mobil adalah {n[len('model_'):]}?", nilai > batas
        elif n.startswith("fuel_type_"):
            tanya, ya = f"Bahan bakar adalah {n[len('fuel_type_'):]}?", nilai > batas
        else:
            tanya, ya = f"{n} <= {batas:.2f}?", nilai <= batas
        langkah.append(f"{tanya} **{'Ya' if ya else 'Tidak'}**")
        node = t.children_left[node] if nilai <= batas else t.children_right[node]

    return langkah, int(t.n_node_samples[node]), float(t.value[node][0][0])


def mobil_serupa(data, merk, model_mobil, tahun, kms, bbm, jumlah=5):
    """Mobil di dataset yang paling mirip (model sama dulu, lalu merk sama)."""
    kandidat = data[data["model"] == model_mobil]
    if len(kandidat) < 3:
        kandidat = data[data["company"] == merk]
    jarak = (
        (kandidat["year"] - tahun).abs() / 5
        + (kandidat["kms"] - kms).abs() / 50_000
        + (kandidat["fuel_type"] != bbm).astype(float)
    )
    return kandidat.loc[jarak.sort_values().index[:jumlah]]


def tampilkan_penjelasan(hasil, data, merk, nama_mobil, tahun, kms, bbm, prediksi, kurs):
    model_mobil = " ".join(nama_mobil.split()[:3])
    baris = pd.DataFrame(
        [{"company": merk, "model": model_mobil, "fuel_type": bbm, "year": float(tahun), "kms": float(kms)}]
    )
    rata = float(np.mean([prediksi[n]["harga"] for n in NAMA_MODEL]))

    sim = simulasi_perubahan(hasil, data, merk, model_mobil, tahun, kms, bbm, kurs)
    serupa = mobil_serupa(data, merk, model_mobil, tahun, kms, bbm)

    st.divider()
    st.subheader("Penjelasan hasil")

    # ---- Ringkasan dalam bahasa sehari-hari
    kalimat = [f"Ketiga model memperkirakan harga mobil ini sekitar **{rp(rata * kurs)}** ({inr(rata)})."]
    pengaruh = sim.assign(abs_persen=sim["persen"].abs()).groupby("Faktor")["abs_persen"].max()
    if len(pengaruh) and pengaruh.max() >= 1:
        kalimat.append(
            f"Pada mobil ini, perubahan **{pengaruh.idxmax().lower()}** paling besar pengaruhnya terhadap "
            f"harga (sekitar {pengaruh.max():.0f}% pada simulasi di bawah)."
        )
    if len(serupa):
        kalimat.append(
            f"Sebagai pembanding, {len(serupa)} mobil paling mirip di dataset berharga "
            f"{inr(serupa['Price'].min())} - {inr(serupa['Price'].max())} "
            f"(median {inr(serupa['Price'].median())})."
        )
    st.write(" ".join(kalimat))

    # ---- 1. Simulasi
    st.markdown("#### 1. Apa yang membuat harga naik atau turun?")
    st.caption(
        "Simulasi memakai rata-rata ketiga model. Setiap baris hanya mengubah satu hal, "
        "data mobil lainnya tetap sama."
    )
    tampil = pd.DataFrame(
        {
            "Skenario": sim["Skenario"],
            "Perkiraan harga": sim["harga"].map(rp),
            "Selisih dari prediksi": sim["selisih"].map(rp_selisih),
            "Persen": sim["persen"].map(lambda v: f"{v:+.1f}%"),
        }
    )
    st.dataframe(tampil, hide_index=True, width="stretch")

    # ---- 2. Cara tiap model
    st.markdown("#### 2. Bagaimana tiap model sampai pada angkanya?")

    langkah, jumlah_daun, harga_daun = jalur_decision_tree(hasil, baris)
    with st.container(border=True):
        st.markdown(f"**{PENJELASAN['Decision Tree']['ikon']} Decision Tree : {rp(prediksi['Decision Tree']['harga'] * kurs)}**")
        st.write("Mobil ini menjawab pertanyaan berikut, dari atas ke bawah :")
        st.markdown("\n".join(f"{i}. {l}" for i, l in enumerate(langkah, 1)))
        st.write(
            f"Jawaban-jawaban itu membawa mobil ini ke kelompok akhir (daun) yang berisi **{jumlah_daun}** mobil "
            f"pada data training. Rata-rata harga kelompok itu, **{inr(harga_daun)}**, menjadi prediksinya."
        )

    pipeline_rf = hasil["model"]["Random Forest"]
    pohon_rf = pipeline_rf.named_steps["model"]
    angka_rf = pipeline_rf.named_steps["preprocess"].transform(baris)
    per_pohon = np.array([p.predict(angka_rf)[0] for p in pohon_rf.estimators_])
    harga_rf = prediksi["Random Forest"]["harga"]
    sepakat = float(np.mean(np.abs(per_pohon - harga_rf) <= 0.2 * max(harga_rf, 1)))
    p10, p90 = np.percentile(per_pohon, [10, 90])
    if sepakat >= 0.7:
        arti = "Pohon-pohon cukup sepakat, jadi perkiraannya relatif mantap."
    elif sepakat >= 0.4:
        arti = "Pendapat antar pohon cukup beragam, jadi perkiraannya sedang."
    else:
        arti = "Pendapat antar pohon sangat beragam, jadi perkiraannya kurang pasti."
    with st.container(border=True):
        st.markdown(f"**{PENJELASAN['Random Forest']['ikon']} Random Forest : {rp(harga_rf * kurs)}**")
        st.write(
            f"Hutan ini terdiri dari **{len(per_pohon)}** pohon. Setiap pohon memberi harga sendiri, lalu semuanya "
            f"dirata-rata menjadi {inr(harga_rf)}. Harga dari tiap pohon berkisar {inr(per_pohon.min())} - "
            f"{inr(per_pohon.max())} (80% pohon ada di {inr(p10)} - {inr(p90)}), dan **{sepakat * 100:.0f}%** pohon "
            f"memberi harga dalam batas 20% dari hasil akhir. {arti}"
        )

    pipeline_gb = hasil["model"]["Gradient Boosting"]
    gb = pipeline_gb.named_steps["model"]
    tahap = list(gb.staged_predict(pipeline_gb.named_steps["preprocess"].transform(baris)))
    titik = sorted({max(1, len(tahap) // 10), max(1, len(tahap) // 2), len(tahap)})
    baris_gb = [("Titik awal : rata-rata harga data training", float(hasil["y_train"].mean()))]
    baris_gb += [(f"Setelah {k} pohon koreksi", float(tahap[k - 1][0])) for k in titik]
    with st.container(border=True):
        st.markdown(f"**{PENJELASAN['Gradient Boosting']['ikon']} Gradient Boosting : {rp(prediksi['Gradient Boosting']['harga'] * kurs)}**")
        st.write(
            "Model mulai dari harga rata-rata, lalu setiap pohon kecil menambahkan koreksi untuk memperbaiki "
            "kesalahan pohon sebelumnya. Berikut perkembangan prediksi untuk mobil ini :"
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Tahap": [b[0] for b in baris_gb],
                    "Prediksi (INR)": [inr(b[1]) for b in baris_gb],
                    "Prediksi (Rp)": [rp(b[1] * kurs) for b in baris_gb],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    # ---- 3. Mobil serupa
    st.markdown("#### 3. Mobil serupa di dataset")
    st.caption("Mobil dengan model (atau merk) sama yang tahun, jarak tempuh, dan bahan bakarnya paling mirip. Bisa dipakai untuk mengecek apakah prediksi masuk akal.")
    st.dataframe(
        pd.DataFrame(
            {
                "Mobil": serupa["name"].values,
                "Tahun": serupa["year"].astype(int).values,
                "Jarak tempuh (km)": serupa["kms"].map(angka).values,
                "Bahan bakar": serupa["fuel_type"].values,
                "Harga (Rp)": (serupa["Price"] * kurs).map(rp).values,
                "Harga (INR)": serupa["Price"].map(angka).values,
            }
        ),
        hide_index=True,
        width="stretch",
    )



# ----------------------------------------------------------------------------
# Ringkasan performa model
# ----------------------------------------------------------------------------
def tampilkan_performa_model(metrik, y_test):
    """Menjelaskan model terbaik berdasarkan R² Testing, MAE, dan RMSE."""
    rata_harga_test = float(y_test.mean())
    evaluasi = metrik.copy()

    # MAE relatif membantu memberi konteks terhadap besar harga pada dataset.
    evaluasi["MAE Relatif (%)"] = (
        evaluasi["MAE Testing"] / max(rata_harga_test, 1) * 100
    )
    evaluasi["Indikator Error (%)"] = evaluasi["MAE Relatif (%)"]

    terbaik_r2 = evaluasi["R2 Testing"].idxmax()
    terbaik_mae = evaluasi["MAE Testing"].idxmin()
    terbaik_rmse = evaluasi["RMSE Testing"].idxmin()

    # R² Testing menjadi indikator utama; MAE/RMSE menjadi pengecekan pendukung.
    terbaik = terbaik_r2
    r2 = float(evaluasi.loc[terbaik, "R2 Testing"])
    mae = float(evaluasi.loc[terbaik, "MAE Testing"])
    rmse = float(evaluasi.loc[terbaik, "RMSE Testing"])
    mae_relatif = float(evaluasi.loc[terbaik, "MAE Relatif (%)"])

    # Ini hanya indikator interpretatif berbasis MAE, bukan accuracy score klasifikasi.
    akurasi_relatif = max(0.0, 100.0 - mae_relatif)

    st.markdown("### 🏆 Performa model")
    st.markdown(
        f"""
        <div class="winner-card">
            <strong>{terbaik}</strong> memiliki performa terbaik berdasarkan <strong>R² Testing</strong>,
            yaitu <strong>{r2:.3f}</strong>. Artinya, model menjelaskan sekitar
            <strong>{r2 * 100:.1f}%</strong> variasi harga pada data testing.
            MAE-nya sebesar <strong>{angka(mae)} INR</strong>, sehingga rata-rata kesalahan absolut
            berada di sekitar <strong>{mae_relatif:.1f}%</strong> dari rata-rata harga pada data testing.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    kartu = [
        ("R² Testing", f"{r2:.3f}", "semakin tinggi semakin baik"),
        ("MAE Testing", f"{angka(mae)} INR", "semakin rendah semakin baik"),
        ("RMSE Testing", f"{angka(rmse)} INR", "semakin rendah semakin baik"),
        ("MAE relatif", f"{mae_relatif:.1f}%", "error terhadap rata-rata harga"),
    ]
    for col, (label, value, note) in zip((c1, c2, c3, c4), kartu):
        with col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">{label}</div>
                    <div class="result-value">{value}</div>
                    <div class="result-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("**Kenapa model ini dianggap lebih akurat?**")
    st.write(
        f"{terbaik} dipilih karena mempunyai nilai R² Testing paling tinggi, yaitu "
        f"{r2:.3f}. R² menunjukkan seberapa baik model menjelaskan variasi harga pada "
        f"data yang tidak digunakan untuk melatih model. Sebagai pengecekan tambahan, "
        f"MAE dan RMSE digunakan karena keduanya mengukur besar kesalahan prediksi; "
        f"nilai yang lebih kecil berarti kesalahan prediksi lebih kecil."
    )

    st.markdown("**Perhitungan error pada data testing**")
    st.markdown(
        f"""
        <div class="formula-card">
        MAE relatif = MAE / rata-rata harga testing × 100%<br>
        = {angka(mae)} / {angka(rata_harga_test)} × 100%<br>
        ≈ <strong>{mae_relatif:.1f}%</strong><br><br>
        Indikator sederhana = 100% − MAE relatif<br>
        = 100% − {mae_relatif:.1f}%<br>
        ≈ <strong>{akurasi_relatif:.1f}%</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Catatan: indikator terakhir adalah ukuran interpretatif berbasis MAE, bukan "
        "accuracy score klasifikasi. Untuk prediksi harga (regresi), R², MAE, dan RMSE "
        "tetap menjadi metrik utama."
    )

    tabel_performa = evaluasi[
        ["R2 Testing", "MAE Testing", "RMSE Testing", "MAE Relatif (%)"]
    ].copy()
    tabel_performa.columns = [
        "R² Testing", "MAE Testing", "RMSE Testing", "MAE Relatif (%)"
    ]

    st.markdown("**Perbandingan ketiga model**")
    st.dataframe(
        tabel_performa.style.format(
            {
                "R² Testing": "{:.3f}",
                "MAE Testing": angka,
                "RMSE Testing": angka,
                "MAE Relatif (%)": "{:.1f}%",
            }
        ),
        width="stretch",
    )

    if terbaik_mae == terbaik_r2:
        st.success(
            f"{terbaik} konsisten menjadi yang terbaik pada R² Testing dan MAE Testing."
        )
    else:
        st.info(
            f"R² tertinggi adalah **{terbaik_r2}**, sedangkan MAE terkecil adalah "
            f"**{terbaik_mae}**. Perbedaan ini wajar karena R² menilai kemampuan "
            f"menjelaskan variasi data, sedangkan MAE langsung mengukur rata-rata besar error."
        )

    if terbaik_rmse != terbaik_r2:
        st.caption(
            f"RMSE terkecil dimiliki {terbaik_rmse}. RMSE memberi penalti lebih besar "
            "pada kesalahan prediksi yang besar."
        )


# ----------------------------------------------------------------------------
# Tampilan
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="section-label">Used Car Price Prediction</div>
        <h1>🚗 Prediksi Harga Mobil Bekas</h1>
        <p>Bandingkan tiga model regresi untuk mendapatkan perkiraan harga dan melihat performa model pada data testing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

mentah = muat_data_mentah()
data, batas_harga = bersihkan_data(mentah)

with st.spinner("Melatih 3 model (hanya sekali saat pertama dibuka)..."):
    hasil = latih_semua_model(data)

metrik = hasil["metrik"]

tab_prediksi, tab_model, tab_data = st.tabs(["🔮 Prediksi", "🧠 Model & Performa", "📊 Data & EDA"])

# ============================ Tab 1 : Prediksi ================================
with tab_prediksi:
    st.markdown(
        "**Cara pakai** : isi data mobil di bawah, lalu klik **Prediksi Harga**. "
        "Hasilnya muncul dari **Decision Tree**, **Random Forest**, dan **Gradient Boosting** sekaligus."
    )

    kiri, kanan = st.columns(2)
    with kiri:
        daftar_merk = sorted(data["company"].unique())
        merk = st.selectbox("Merk", daftar_merk, index=daftar_merk.index("Maruti") if "Maruti" in daftar_merk else 0)

        model_merk = data.loc[data["company"] == merk, "model"].value_counts().index.tolist()
        pilihan_model = st.selectbox("Model mobil", model_merk + [MODEL_LAIN])
        if pilihan_model == MODEL_LAIN:
            nama_mobil = st.text_input("Ketik nama/model mobil", placeholder=f"mis. {model_merk[0]}").strip()
            if nama_mobil and not nama_mobil.lower().startswith(merk.lower()):
                nama_mobil = f"{merk} {nama_mobil}"
        else:
            nama_mobil = pilihan_model

        daftar_bbm = sorted(data["fuel_type"].unique())
        bbm = st.selectbox("Jenis bahan bakar", daftar_bbm)

    with kanan:
        tahun_min, tahun_max = int(data["year"].min()), int(data["year"].max())
        tahun = st.number_input(
            "Tahun pembuatan", min_value=tahun_min, max_value=tahun_max,
            value=min(max(2012, tahun_min), tahun_max), step=1,
        )
        kms = st.number_input("Jarak tempuh (km)", min_value=0, max_value=1_000_000, value=40_000, step=1_000)
        kurs = st.number_input(
            "Kurs 1 INR ke Rupiah", min_value=1.0, value=195.0, step=1.0,
            help="Harga pada dataset dalam Rupee India (INR). Konversi bersifat perkiraan, sesuaikan dengan kurs terbaru.",
        )

    if st.button("Prediksi Harga", type="primary", width="stretch"):
        if not nama_mobil:
            st.error("Nama/model mobil belum diisi.")
        else:
            prediksi = prediksi_semua(hasil, merk, nama_mobil, tahun, kms, bbm)
            model_dikenal = " ".join(nama_mobil.split()[:3]) in set(data["model"])

            st.divider()
            st.markdown(f"#### {nama_mobil} · {tahun} · {angka(kms)} km · {bbm}")
            if not model_dikenal:
                st.warning(
                    "Model mobil ini tidak ada di dataset, jadi prediksi hanya memakai merk, tahun, "
                    "jarak tempuh, dan bahan bakar. Akurasinya lebih rendah."
                )

            for kolom, nama in zip(st.columns(3), NAMA_MODEL):
                p, info = prediksi[nama], PENJELASAN[nama]
                with kolom.container(border=True):
                    st.markdown(f"**{info['ikon']} {nama}**")
                    st.caption(info["ringkas"])
                    st.markdown(f"## {rp(p['harga'] * kurs)}")
                    st.caption(inr(p["harga"]))
                    if p["rendah"] is not None:
                        st.write(f"Rentang : {inr(p['rendah'])} - {inr(p['tinggi'])}")
                        st.caption(p["keterangan"])
                    else:
                        st.caption("Tidak ada rentang bawaan")

            semua = np.array([prediksi[n]["harga"] for n in NAMA_MODEL])
            rata, selisih = semua.mean(), semua.max() - semua.min()
            st.info(
                f"**Rata-rata ketiga model : {rp(rata * kurs)}** ({inr(rata)}). "
                f"Selisih antara model tertinggi dan terendah sekitar {selisih / max(rata, 1) * 100:.0f}% dari rata-rata."
            )
            if selisih / max(rata, 1) > 0.3:
                st.caption("Ketiga model cukup berbeda pendapat untuk mobil ini, jadi anggap hasilnya sebagai perkiraan kasar.")
            else:
                st.caption("Makin dekat hasil ketiga model, makin kuat perkiraannya.")

            tampilkan_penjelasan(hasil, data, merk, nama_mobil, tahun, kms, bbm, prediksi, kurs)

    st.caption(
        "Hasil prediksi adalah **perkiraan awal**, bukan harga pasti. Dataset tidak memuat kondisi mobil, "
        "tipe transmisi, atau kapasitas mesin."
    )

# ============================ Tab 2 : Tentang Model ===========================
with tab_model:
    st.subheader("Tiga cara prediksi")
    st.write(
        "Ketiganya memakai fitur dan data yang sama (merk, model, bahan bakar, tahun, dan jarak tempuh), "
        "sehingga perbedaan hasil murni berasal dari cara kerja algoritmanya."
    )
    for kolom, nama in zip(st.columns(3), NAMA_MODEL):
        info = PENJELASAN[nama]
        with kolom.container(border=True):
            st.markdown(f"**{info['ikon']} {nama}**")
            st.write(info["cara"])
            st.markdown(f"✅ {info['plus']}")
            st.markdown(f"⚠️ {info['minus']}")

    st.subheader("Alur pembuatan model")
    st.markdown(
        "1. **Persiapan data** : data dibersihkan, fitur `model` dibuat dari 3 kata pertama nama mobil, outlier dibuang.\n"
        "2. **Split data** : 80% training dan 20% testing.\n"
        "3. **Training + tuning** : fitur kategori diubah dengan *One-Hot Encoding*, lalu `GridSearchCV` "
        "(cross-validation 5 lipatan, kriteria MAE terkecil) mencari parameter terbaik dengan proses serial agar lebih hemat RAM.\n"
        "4. **Evaluasi** : model diuji pada data testing yang belum pernah dilihat.\n"
        "5. **Prediksi** : data mobil dari user diprediksi oleh ketiga model."
    )

    st.subheader("Evaluasi model")
    tabel = metrik[["R2 Training", "R2 Testing", "Selisih R2 (Train - Test)", "MAE Testing", "RMSE Testing"]]
    st.dataframe(
        tabel.style.format(
            {
                "R2 Training": "{:.3f}",
                "R2 Testing": "{:.3f}",
                "Selisih R2 (Train - Test)": "{:.3f}",
                "MAE Testing": angka,
                "RMSE Testing": angka,
            }
        ),
        width="stretch",
    )
    st.caption(
        "R2 makin mendekati 1 makin baik. MAE dan RMSE dalam INR, makin kecil makin baik. "
        "Selisih R2 training dan testing yang besar menandakan model menghafal data (overfitting)."
    )

    tampilkan_performa_model(metrik, hasil["y_test"])

    st.subheader("Kesimpulan model")
    poin = [
        f"Dengan data yang sudah dibersihkan, model terbaik pada data test adalah **{terbaik}** "
        f"(R2 {metrik.loc[terbaik, 'R2 Testing']:.3f}, MAE {angka(metrik.loc[terbaik, 'MAE Testing'])} INR).",
        "Pengecekan overfitting : "
        + "; ".join(f"{n} {analisis_overfitting(metrik.loc[n, 'Selisih R2 (Train - Test)'])}" for n in NAMA_MODEL)
        + ".",
        (
            "Nilai R2 belum tinggi, sesuai kesimpulan EDA : data ini terbatas (tidak ada fitur seperti "
            "kondisi mobil, tipe transmisi, atau kapasitas mesin)."
            if metrik.loc[terbaik, "R2 Testing"] < 0.7
            else "Meski R2 sudah cukup baik, data ini tetap terbatas (tidak ada fitur seperti kondisi mobil, "
            "tipe transmisi, atau kapasitas mesin), sehingga masih ada selisih dengan harga sebenarnya."
        ),
        "Hasil prediksi sebaiknya dipakai sebagai **perkiraan awal**, bukan harga pasti.",
        "Untuk memperbaiki : kumpulkan data dan fitur lebih banyak.",
    ]
    st.markdown("\n".join(f"- {t}" for t in poin))

    with st.expander("Fitur yang paling berpengaruh terhadap harga"):
        st.bar_chart(hasil["penting"])
    with st.expander("Parameter terbaik hasil GridSearchCV"):
        for nama in NAMA_MODEL:
            st.write(f"**{nama}**", hasil["param"][nama])

# ============================ Tab 3 : Data & EDA ==============================
with tab_data:
    st.subheader("Tujuan analisis")
    st.markdown(
        "- Mencari faktor utama yang memengaruhi harga mobil bekas.\n"
        "- Mencari pola dan hubungan antara harga, tahun pembuatan, jarak tempuh, jenis bahan bakar, dan model."
    )

    st.subheader("Informasi yang tersedia")
    st.markdown(
        "- Nama / model mobil\n- Merk (company)\n- Tahun pembuatan\n- Jenis bahan bakar\n- Jarak tempuh (km)\n- Harga (INR)"
    )

    st.subheader("Kondisi data mentah")
    st.markdown(
        "Dataset awalnya kotor :\n"
        "- `year` mengandung nilai tidak valid (mis. \"zire\").\n"
        "- `Price` mengandung nilai tidak valid (mis. \"Ask For Price\").\n"
        "- `kms_driven` mengandung nilai tidak valid (mis. \"Petrol\") dan nilai kosong, begitu juga `fuel_type`."
    )
    st.write(
        f"Dari **{len(mentah):,}** baris data mentah, tersisa **{len(data):,}** baris setelah membuang data "
        f"kosong/tidak valid, duplikat, harga di atas {angka(batas_harga)} INR (outlier IQR), dan jarak tempuh di atas 200.000 km."
    )
    with st.expander("Lihat contoh data bersih"):
        st.dataframe(data[["name", "company", "year", "Price", "kms", "fuel_type"]].head(30), width="stretch")

    st.subheader("Hubungan antar fitur numerik")
    korelasi = data[["year", "Price", "kms"]].corr()
    r_harga_tahun = korelasi.loc["year", "Price"]
    r_harga_kms = korelasi.loc["Price", "kms"]
    r_tahun_kms = korelasi.loc["year", "kms"]

    tampil = korelasi.rename(index={"year": "Tahun", "Price": "Harga", "kms": "Jarak tempuh"},
                             columns={"year": "Tahun", "Price": "Harga", "kms": "Jarak tempuh"})
    st.dataframe(tampil.style.format("{:.2f}"), width="stretch")

    st.markdown(
        "\n".join(
            [
                kalimat_korelasi("Harga vs Tahun", r_harga_tahun,
                                 "Mobil yang lebih baru cenderung lebih mahal.",
                                 "Mobil yang lebih baru cenderung lebih murah."),
                kalimat_korelasi("Harga vs Jarak tempuh", r_harga_kms,
                                 "Jarak tempuh lebih besar cenderung diikuti harga lebih tinggi.",
                                 "Jarak tempuh lebih besar cenderung diikuti harga lebih rendah."),
                kalimat_korelasi("Tahun vs Jarak tempuh", r_tahun_kms,
                                 "Mobil yang lebih baru cenderung punya jarak tempuh lebih besar.",
                                 "Mobil yang lebih baru cenderung punya jarak tempuh lebih kecil."),
            ]
        )
    )

    st.subheader("Sebaran merk dan bahan bakar")
    kiri, kanan = st.columns(2)
    with kiri:
        st.markdown("**Jumlah mobil per merk**")
        st.bar_chart(data["company"].value_counts())
    with kanan:
        st.markdown("**Jumlah mobil per jenis bahan bakar**")
        st.bar_chart(data["fuel_type"].value_counts())

    merk_teratas = ", ".join(data["company"].value_counts().head(4).index)
    st.markdown(f"Merk yang paling banyak muncul di dataset : **{merk_teratas}**.")

    st.subheader("Kesimpulan EDA")
    korelasi_terkuat = max(abs(r_harga_tahun), abs(r_harga_kms))
    if korelasi_terkuat < 0.6:
        st.write(
            "Terlihat ada **hubungan** antar fitur, tetapi secara keseluruhan korelasinya belum cukup kuat. "
            "Model yang dilatih hanya dari fitur yang tersedia tidak akan sangat akurat."
        )
    else:
        st.write(
            "Terlihat ada **hubungan** antar fitur dengan korelasi yang cukup kuat, "
            "sehingga fitur yang tersedia sudah memberi sinyal yang berguna untuk memprediksi harga."
        )
