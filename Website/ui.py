import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# import joblib  # Pour charger votre modèle pré-entraîné

data_path = Path("data")

def validate_dataset(df: pd.DataFrame, required_features: tuple[str]):
    """
    Vérifie l'adéquation entre le dataset chargé et les besoins du modèle.
    Not implemented yet
    """
    columns_present = set(df.columns)
    columns_missing = [col for col in required_features if col not in columns_present]
    return columns_missing


def evaluate_model(df: pd.DataFrame):
    """
    Fonction d'évaluation du modèle.
    Retourne un dictionnaire de métriques.
    """

    with open(data_path.joinpath("model.pkl"), "rb") as f:
        model = pickle.load(f)

    X_train, X_test, y_train, y_test = get_training_data(df)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

def get_training_data(df):
    """
    Transform user inputs into training data format for the model.
    """
    cols_to_fill = ['Malware Indicators', 'Alerts/Warnings', 'Firewall Logs', 'IDS/IPS Alerts', 'Proxy Information']
    for col in cols_to_fill:
        df[col] = df[col].fillna('None')
    num_features = ['Source Port', 'Destination Port', 'Packet Length', 'Anomaly Scores']
    cat_features = ['Timestamp', 'Protocol', 'Packet Type', 'Traffic Type', 'Malware Indicators', 'Alerts/Warnings', 
                    'Attack Signature', 'Action Taken', 'Severity Level', 'Network Segment', 'Log Source']
    text_feature = 'Payload Data'

    X = df[num_features + cat_features + [text_feature]]
    y = df['Attack Type']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# TODO: put our necessary features here
FEATURES_REQUIRED = ("Device Information", "Packet Length")

# ── Métadonnées des onglets ──────────────────────────────────────────────────
tab_names = [
    "🧪 Model evaluation",
    "📖 Data dictionary",
    "📊 Data visualisation",
]
logo_url = "https://tse4.mm.bing.net/th/id/OIP.AUMTBNwndjiBkCicWtjNrgHaEd?pid=Api&P=0&h=180"

# ── Création des onglets ─────────────────────────────────────────────────────
tabs = st.tabs(tab_names)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – Model evaluation
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col1, col2 = st.columns([5, 1])
    col1.title("🧪 Model evaluation")
    col2.image(logo_url, width=100)
    st.divider()

    st.header("File Upload")
    uploaded_file = st.file_uploader("Choose your evaluation dataset", type="csv")

    if uploaded_file is not None:
        df_eval = pd.read_csv(uploaded_file)
        st.success(f"✅ File loaded — {df_eval.shape[0]:,} rows × {df_eval.shape[1]} columns")
        st.dataframe(df_eval.head(), width="stretch")

        st.divider()
        st.header("Evaluation")

        if st.button("🚀 Évaluer le modèle", type="primary"):
            with st.spinner("Evaluation in progress…"):
                missing = validate_dataset(df_eval, FEATURES_REQUIRED)
                if missing:
                    st.warning(f"⚠️ Missing required feature(s): {missing}")
                else:
                    results = evaluate_model(df_eval)
                    st.subheader("📈 Results")
                    r_cols = st.columns(len(results))
                    for i, (metric, value) in enumerate(results.items()):
                        r_cols[i].metric(metric, value)
                    st.info("ℹ️ The trained model was not provided in the `evaluate_model()` function to display the result and real metrics.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – Data dictionary
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    col1, col2 = st.columns([5, 1])
    col1.title("📖 Data dictionary")
    col2.image(logo_url, width=100)
    st.divider()

    # ── Descriptions détaillées des colonnes ──────────────────────────────
    column_descriptions = {
        "Source IP Address":   "IP address of the attacking machine",
        "Source Port":         "Port number used by the attacker",
        "Destination IP Address": "IP address of the targeted machine",
        "Destination Port":    "Port number on the target machine",
        "User Information":    "Details about the targeted user account",
        "Device Information":  "Information about the targeted device",
        "Packet Length":       "Size of the network packet in bytes",
        "Packet Type":         "Type/protocol of the network packet",
        "Traffic Type":        "Classification of network traffic flow",
        "Payload Data":        "Content carried within the packet",
        "Network Segment":     "Network zone or subnet location",
        "Timestamp":           "Date and time of the event",
        "Malware Indicators":  "Signs of malicious software presence",
        "Anomaly Scores":      "Statistical deviation from normal behavior",
        "Alerts/Warnings":     "Security alerts triggered by the event",
        "Attack Signature":    "Pattern matching known attack types",
        "Action Taken":        "Response action executed by security systems",
        "Severity Level":      "Criticality rating of the threat",
        "Geo-location Data":   "Geographic origin of the traffic",
        "Proxy Information":   "Intermediate server details if used",
        "Firewall Logs":       "Firewall activity records",
        "IDS/IPS Alerts":      "Intrusion detection/prevention system warnings",
        "Log Source":          "Origin system that generated the log",
        "Attack Type":         "Category of cyber attack (target variable)",
    }

    # ── Catégories ────────────────────────────────────────────────────────
    dictionary = {
        "🏴‍☠️ Pirate": {
            "subtitle":  "Information about the sender/attacker",
            "columns":   ["Source IP Address", "Source Port"],
            "bg":        "#2563eb",   # blue
            "text":      "#dbeafe",
        },
        "🎯 Target": {
            "subtitle":  "Information about the receiver/target of the attack",
            "columns":   ["Destination IP Address", "Destination Port",
                          "User Information", "Device Information"],
            "bg":        "#16a34a",   # green
            "text":      "#dcfce7",
        },
        "📦 Message": {
            "subtitle":  "Information about message content and structure",
            "columns":   ["Packet Length", "Packet Type", "Traffic Type",
                          "Payload Data", "Network Segment"],
            "bg":        "#64748b",   # slate
            "text":      "#f1f5f9",
        },
        "🌐 Contextual": {
            "subtitle":  "Contextual information and attack metadata",
            "columns":   ["Timestamp", "Malware Indicators", "Anomaly Scores",
                          "Alerts/Warnings", "Attack Signature", "Action Taken",
                          "Severity Level", "Geo-location Data", "Proxy Information",
                          "Firewall Logs", "IDS/IPS Alerts", "Log Source"],
            "bg":        "#d97706",   # amber
            "text":      "#fefce8",
        },
        "🎲 To_predict": {
            "subtitle":  "Target variable to predict",
            "columns":   ["Attack Type"],
            "bg":        "#dc2626",   # red
            "text":      "#fee2e2",
        },
    }

    # ── CSS compact grid ──────────────────────────────────────────────────
    st.markdown("""
    <style>
    .dd-card {
        border-radius: 12px;
        padding: 14px 16px 10px 16px;
        margin-bottom: 12px;
        box-shadow: 0 3px 8px rgba(0,0,0,.18);
    }
    .dd-card-title {
        font-size: 18px;
        font-weight: 700;
        color: #fff;
        margin: 0 0 2px 0;
    }
    .dd-card-sub {
        font-size: 12px;
        font-style: italic;
        color: rgba(255,255,255,.82);
        margin: 0 0 10px 0;
    }
    .dd-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }
    .dd-tag {
        border-radius: 6px;
        padding: 3px 9px;
        font-size: 12px;
        font-weight: 600;
        cursor: default;
        white-space: nowrap;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Rendu en grille 2 colonnes ────────────────────────────────────────
    grid_cols = st.columns(2)
    for i, (cat, data) in enumerate(dictionary.items()):
        tags_html = "".join(
            f'<span class="dd-tag" style="background:{data["bg"]};'
            f'color:{data["text"]};border:1px solid rgba(255,255,255,.3)" '
            f'title="{column_descriptions.get(c, "")}">{c}</span>'
            for c in data["columns"]
        )
        card = f"""
        <div class="dd-card" style="background:{data['bg']}">
            <p class="dd-card-title">{cat}</p>
            <p class="dd-card-sub">{data['subtitle']}</p>
            <div class="dd-tags">{tags_html}</div>
        </div>
        """
        with grid_cols[i % 2]:
            st.markdown(card, unsafe_allow_html=True)

    # ── Métriques récapitulatives ─────────────────────────────────────────
    st.divider()
    m_cols = st.columns(len(dictionary))
    for col, (cat, data) in zip(m_cols, dictionary.items()):
        col.metric(cat, len(data["columns"]))

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Data visualisation
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    col1, col2 = st.columns([5, 1])
    col1.title("📊 Data visualisation")
    col2.image(logo_url, width=100)
    st.divider()

    st.header("Explore a variable")

    # ── Chargement direct du dataset ──────────────────────────────────────
    DATASET_PATH = "data/cyber_attacks_full.csv"

    @st.cache_data
    def load_data(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    try:
        df_vis = load_data(DATASET_PATH)
    except FileNotFoundError:
        st.error(f"❌ File `{DATASET_PATH}` not found. Please place it in the same directory as `ui.py`.")
        st.stop()

    st.caption(f"Dataset : **{DATASET_PATH}** — {df_vis.shape[0]:,} rows × {df_vis.shape[1]} columns")

    selected_col = st.selectbox(
        "Select a column to visualise",
        options=df_vis.columns.tolist(),
    )

    TARGET = "Attack Type"
    ATTACK_COLORS = px.colors.qualitative.Set1

    if st.button("📈 Show chart", type="primary"):
        st.subheader(f"Distribution of **{selected_col}**  ·  coloured by {TARGET}")
         # Si l'utilisateur sélectionne la variable cible elle-même
        if selected_col == TARGET:
            st.info(f"ℹ️ **{TARGET}** is the target variable — showing its own distribution.")
            vc = df_vis[TARGET].value_counts().reset_index()
            vc.columns = [TARGET, "Count"]
            fig_target = px.bar(
                vc, x=TARGET, y="Count", color=TARGET,
                title=f"Distribution of {TARGET}",
                color_discrete_sequence=ATTACK_COLORS,
                template="plotly_white",
                text="Count",
            )
            fig_target.update_traces(textposition="outside")
            st.plotly_chart(fig_target, width="stretch")
            st.stop()
        cols_needed = [selected_col, TARGET]
        df_clean = df_vis[cols_needed].dropna()
        series   = df_clean[selected_col]
        n_unique = series.nunique()
        attack_types = sorted(df_clean[TARGET].unique())

        # ── Detect column kind ────────────────────────────────────────────
        is_datetime = selected_col in ("Timestamp",) or pd.api.types.is_datetime64_any_dtype(series)
        is_numeric  = pd.api.types.is_numeric_dtype(series) and not is_datetime

        if is_datetime:
            # ── Timestamp → line chart (events per day per attack type) ──
            df_ts = df_vis[["Timestamp", TARGET]].dropna().copy()
            df_ts["Date"] = pd.to_datetime(df_ts["Timestamp"], errors="coerce").dt.date
            daily = (
                df_ts.groupby(["Date", TARGET])
                .size()
                .reset_index(name="Count")
            )
            fig = px.line(
                daily, x="Date", y="Count", color=TARGET,
                title=f"Events per day by {TARGET}",
                color_discrete_sequence=ATTACK_COLORS,
                template="plotly_white",
                markers=False,
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Number of events",
                              legend_title=TARGET)

        elif is_numeric:
            # ── Numeric → overlaid histogram per attack type ──────────────
            fig = px.histogram(
                df_clean, x=selected_col, color=TARGET,
                nbins=min(60, n_unique),
                barmode="overlay",
                opacity=0.70,
                title=f"Distribution of {selected_col} by {TARGET}",
                color_discrete_sequence=ATTACK_COLORS,
                template="plotly_white",
            )
            fig.update_layout(bargap=0.02, legend_title=TARGET)

        elif n_unique <= 30:
            # ── Categorical (few) → 100% stacked bar ──────────────────────
            counts = (
                df_clean.groupby([selected_col, TARGET])
                .size()
                .reset_index(name="Count")
            )
            totals = counts.groupby(selected_col)["Count"].transform("sum")
            counts["Pct"] = (counts["Count"] / totals * 100).round(1)

            fig = px.bar(
                counts, x=selected_col, y="Pct", color=TARGET,
                barmode="stack",
                text=counts["Pct"].apply(lambda v: f"{v:.0f}%" if v >= 5 else ""),
                title=f"Share of {TARGET} within each {selected_col} value",
                color_discrete_sequence=ATTACK_COLORS,
                template="plotly_white",
                labels={"Pct": "Share (%)"},
            )
            fig.update_traces(textposition="inside")
            fig.update_layout(yaxis_title="Share (%)", legend_title=TARGET,
                              uniformtext_minsize=9, uniformtext_mode="hide")

        else:
            # ── Categorical (many) → stacked horizontal bar, top 20 ───────
            top20 = series.value_counts().head(20).index.tolist()
            df_top = df_clean[df_clean[selected_col].isin(top20)]
            counts = (
                df_top.groupby([selected_col, TARGET])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                counts, y=selected_col, x="Count", color=TARGET,
                barmode="stack",
                orientation="h",
                title=f"Top 20 values of {selected_col}  –  split by {TARGET}",
                color_discrete_sequence=ATTACK_COLORS,
                template="plotly_white",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              legend_title=TARGET)

        st.plotly_chart(fig, width='stretch')

        # ── Stats de base liées à la cible ────────────────────────────────
        st.markdown(f"#### 📋 Base statistics by *{TARGET}*")

        stat_cols = st.columns(len(attack_types))
        for col_st, atk in zip(stat_cols, attack_types):
            subset = df_clean[df_clean[TARGET] == atk][selected_col]
            col_st.markdown(f"**{atk}**  `n={len(subset):,}`")
            if is_numeric:
                col_st.dataframe(subset.describe().round(2).to_frame(), width="stretch")
            else:
                vc = subset.value_counts().head(5).reset_index()
                vc.columns = [selected_col, "Count"]
                col_st.dataframe(vc, width="stretch", hide_index=True)

        # ── Répartition globale de la cible ───────────────────────────────
        st.markdown(f"#### 🎯 Overall *{TARGET}* distribution")
        target_counts = df_vis[TARGET].value_counts().reset_index()
        target_counts.columns = [TARGET, "Count"]
        target_counts["Share"] = (target_counts["Count"] / target_counts["Count"].sum() * 100).round(1)
        fig_pie = px.pie(
            target_counts, names=TARGET, values="Count",
            color_discrete_sequence=ATTACK_COLORS,
            template="plotly_white",
            hole=0.3,
        )
        fig_pie.update_traces(textinfo="label+percent")
        st.plotly_chart(fig_pie, width="stretch")