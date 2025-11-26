# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ===========================
# Page config
# ===========================
st.set_page_config(page_title="Football Intelligence Suite", page_icon="⚽", layout="centered")
st.sidebar.title("Football Intelligence Suite")

# ===========================
# Sidebar menu
# ===========================
app_choice = st.sidebar.radio(
    "Choose an App:",
    [
        "🧤 Goalkeeper Value Predictor",
        "⚽ Attacker Value Predictor",
        "🎯 Midfielder Recommender + Value Predictor",
        "🛡️ Defender Market Value Predictor",
        "🧩 National Team Formation Viewer"
    ]
)

# ============================================================
# 🧤 Goalkeeper Market Value Predictor
# ============================================================
if app_choice == "🧤 Goalkeeper Value Predictor":

    @st.cache_resource
    def load_gk_artifacts():
        preprocessor = joblib.load("preprocessor.pkl")
        model = joblib.load("stacked_model_gk_value (1).pkl")
        return preprocessor, model

    preprocessor, model = load_gk_artifacts()

    st.title("🧤 Goalkeeper Market Value Predictor")
    st.markdown("Enter goalkeeper stats below to estimate their **market value (EUR)**.")

    col1, col2 = st.columns(2)

    with col1:
        height_in_cm = st.number_input("Height (cm)", min_value=150, max_value=210, value=190)
        age = st.number_input("Age", min_value=15, max_value=45, value=28)
        minutes_played = st.number_input("Minutes Played", min_value=0, max_value=6000, value=3000)
        foot = st.selectbox("Preferred Foot", ["right", "left"])

    with col2:
        saves = st.number_input("Total Saves", min_value=0, max_value=400, value=150)
        goals_conceded = st.number_input("Goals Conceded", min_value=0, max_value=200, value=30)
        yellow_cards = st.number_input("Yellow Cards", min_value=0, max_value=20, value=1)
        red_cards = st.number_input("Red Cards", min_value=0, max_value=10, value=0)

    if st.button("💰 Predict Market Value"):
        # Prepare input DataFrame
        sample_input = pd.DataFrame([{
            "height_in_cm": height_in_cm,
            "yellow_cards": yellow_cards,
            "red_cards": red_cards,
            "minutes_played": minutes_played,
            "age": age,
            "saves": saves,
            "goals_conceded": goals_conceded,
            "foot": foot
        }])

        # Compute per-90 stats
        if minutes_played > 0:
            sample_input["goals_conceded_per_90"] = sample_input["goals_conceded"] / (minutes_played / 90)
            sample_input["saves_per_90"] = sample_input["saves"] / (minutes_played / 90)
            sample_input["yellow_cards_per_90"] = sample_input["yellow_cards"] / (minutes_played / 90)
            sample_input["red_cards_per_90"] = sample_input["red_cards"] / (minutes_played / 90)
        else:
            sample_input["goals_conceded_per_90"] = 0
            sample_input["saves_per_90"] = 0
            sample_input["yellow_cards_per_90"] = 0
            sample_input["red_cards_per_90"] = 0

        X_processed = preprocessor.transform(sample_input)
        encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(['foot'])
        numeric_features = [
            'height_in_cm', 'yellow_cards', 'red_cards', 'minutes_played', 'age', 'saves',
            'goals_conceded', 'goals_conceded_per_90', 'saves_per_90',
            'yellow_cards_per_90', 'red_cards_per_90'
        ]
        all_feature_names = numeric_features + list(encoded_cat_names)
        X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
        X_processed_df = X_processed_df.drop(['goals_conceded', 'saves', 'yellow_cards', 'red_cards',"foot_both"], axis=1)

        predicted_value = model.predict(X_processed_df)[0]
        st.success(f"🏷️ **Estimated Market Value:** €{predicted_value:,.0f}")

        with st.expander("Show processed features"):
            st.dataframe(sample_input)

# ============================================================
# ⚽ Attacker Market Value Predictor
# ============================================================
elif app_choice == "⚽ Attacker Value Predictor":

    @st.cache_resource
    def load_attacker_artifacts():
        preprocessor = joblib.load("preprocessor_attacker.pkl")
        model = joblib.load("attackers_value_stack_pipeline.joblib")
        cluster_artifacts = joblib.load("attackers_cluster_artifacts.joblib")
        return (
            preprocessor,
            model,
            cluster_artifacts["scaler_cluster"],
            cluster_artifacts["pca"],
            cluster_artifacts["kmeans"]
        )

    preprocessor, model, scaler_cluster, pca, kmeans = load_attacker_artifacts()

    st.title("⚽ Attacker Market Value Predictor")
    st.markdown("Enter attacker stats below to predict **market value (EUR)**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=16, max_value=45, value=25)
        goals = st.number_input("Goals", min_value=0, max_value=100, value=15)
        assists = st.number_input("Assists", min_value=0, max_value=100, value=8)
    with col2:
        height_in_cm = st.number_input("Height (cm)", min_value=150, max_value=210, value=182)
        minutes_played = st.number_input("Minutes Played", min_value=0, max_value=6000, value=2700)
    with col3:
        yellow_cards = st.number_input("Yellow Cards", min_value=0, max_value=20, value=2)
        red_cards = st.number_input("Red Cards", min_value=0, max_value=5, value=0)
        foot = st.selectbox("Preferred Foot", ["Right", "Left"])

    league_norm = st.selectbox(
        "League",
        ["premier league", "la liga", "bundesliga", "serie a", "ligue 1", "other"]
    )

    def safe_per90(numer, minutes):
        return np.where(minutes > 0, (numer / np.clip(minutes, 1e-9, None)) * 90.0, 0.0)

    if st.button("🔮 Predict Market Value"):
        sample_input = pd.DataFrame([{
            'age': age,
            'height_in_cm': height_in_cm,
            'goals': goals,
            'assists': assists,
            'minutes_played': minutes_played,
            'yellow_cards': yellow_cards,
            'red_cards': red_cards,
            'foot': foot,
            'league_norm': league_norm,
            'team_quality': 0.35
        }])

        sample_input["goals_per_90"] = safe_per90(sample_input["goals"], sample_input["minutes_played"])
        sample_input["assists_per_90"] = safe_per90(sample_input["assists"], sample_input["minutes_played"])
        sample_input["yellow_cards_per_90"] = safe_per90(sample_input["yellow_cards"], sample_input["minutes_played"])
        sample_input["red_cards_per_90"] = safe_per90(sample_input["red_cards"], sample_input["minutes_played"])

        sample_input["age_x_goals"] = sample_input["age"] * sample_input["goals"]
        sample_input["height_x_age"] = sample_input["height_in_cm"] * sample_input["age"]

        league_strength = {
            "premier league": 1.00, "la liga": 0.95, "bundesliga": 0.95,
            "serie a": 0.90, "ligue 1": 0.85
        }
        sample_input["league_strength_coefficient"] = sample_input["league_norm"].map(league_strength).fillna(0.8)

        cluster_features = ["goals_per_90", "assists_per_90"]
        Xc = scaler_cluster.transform(sample_input[cluster_features])
        Xc_pca = pca.transform(Xc)
        sample_input["attacker_cluster"] = kmeans.predict(Xc_pca)

        sample_input.drop(columns=["minutes_played", "yellow_cards", "red_cards"], inplace=True, errors="ignore")

        pred_log = model.predict(sample_input)
        pred_value = np.expm1(pred_log)

        st.success(f"💰 **Predicted Market Value:** €{pred_value[0]:,.0f}")

# ============================================================
# 🎯 Midfielder Recommender + Market Value Predictor
# ============================================================
elif app_choice == "🎯 Midfielder Recommender + Value Predictor":

    @st.cache_resource
    def load_mid_files():
        with open("player_database.pkl", "rb") as f:
            df = pickle.load(f)

        with open("feature_columns.pkl", "rb") as f:
            feature_cols = pickle.load(f)

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("kmeans.pkl", "rb") as f:
            kmeans = pickle.load(f)

        try:
            with open("mid_value_model.pkl", "rb") as f:
                value_model = pickle.load(f)
        except:
            value_model = None

        return df, feature_cols, scaler, kmeans, value_model

    df, feature_cols, scaler, kmeans, value_model = load_mid_files()
    PLAYER_COL = "name"
    VALUE_COL = "market_value_in_eur"

    st.title("🎯 Midfielder Recommender & Market Value Predictor")
    tab1, tab2, tab3 = st.tabs(["🔍 Recommend by Player Name", "📊 Recommend by Stats", "💰 Market Value Predictor"])

    # ---------- HELPER FUNCTIONS ----------
    def prepare_input(stats):
        return pd.DataFrame([{col: stats.get(col, 0) for col in feature_cols}])

    def predict_cluster(stats):
        X = scaler.transform(prepare_input(stats))
        return int(kmeans.predict(X)[0])

    def predict_market_value(stats):
        if value_model is None:
            return "❌ Model not found"
        X = scaler.transform(prepare_input(stats))
        return float(value_model.predict(X)[0])

    def recommend_midfielder(player, budget, top_n):
        if player not in df[PLAYER_COL].values:
            return pd.DataFrame({"Message": ["Player not found"]})

        idx = df.index[df[PLAYER_COL] == player][0]
        cluster = df.loc[idx, "cluster"]
        cluster_players = df[df["cluster"] == cluster].copy()

        if budget:
            cluster_players = cluster_players[cluster_players[VALUE_COL] <= budget]

        if cluster_players.empty:
            return pd.DataFrame({"Message": ["No players found"]})

        target_vec = scaler.transform(df.loc[[idx], feature_cols])
        cluster_scaled = scaler.transform(cluster_players[feature_cols])
        sims = cosine_similarity(target_vec, cluster_scaled)[0]

        cluster_players["Similarity"] = sims
        cluster_players = cluster_players[cluster_players[PLAYER_COL] != player]

        display_cols = [PLAYER_COL, VALUE_COL, "Similarity"]
        if "club_name" in cluster_players.columns:
            display_cols.insert(1, "club_name")

        return cluster_players.sort_values("Similarity", ascending=False).head(top_n)[display_cols]

    def recommend_by_stats(stats, budget, top_n):
        cluster = predict_cluster(stats)
        cluster_players = df[df["cluster"] == cluster].copy()

        if budget:
            cluster_players = cluster_players[cluster_players[VALUE_COL] <= budget]

        if cluster_players.empty:
            return pd.DataFrame({"Message": ["No players found"]})

        X = scaler.transform(prepare_input(stats))
        cluster_scaled = scaler.transform(cluster_players[feature_cols])
        sims = cosine_similarity(X, cluster_scaled)[0]

        cluster_players["Similarity"] = sims

        display_cols = [PLAYER_COL, VALUE_COL, "Similarity"]
        if "club_name" in cluster_players.columns:
            display_cols.insert(1, "club_name")

        return cluster_players.sort_values("Similarity", ascending=False).head(top_n)[display_cols]

    # ---------- TAB 1 ----------
    with tab1:
        player_list = sorted(df[PLAYER_COL].unique())
        selected_player = st.selectbox("Select Player", player_list)
        budget = st.number_input("Max Budget (€)", min_value=0, value=30000000)
        top_n = st.slider("Top N Recommendations", 1, 20, 5)

        if st.button("Recommend"):
            st.dataframe(recommend_midfielder(selected_player, budget, top_n))

    # ---------- TAB 2 ----------
    with tab2:
        st.write("Enter stats:")

        inputs = {col: st.number_input(col, 0, 5000, 0) for col in feature_cols}
        budget2 = st.number_input("Budget (€)", 0, value=30000000)
        top_n2 = st.slider("Top Recommendations", 1, 20, 5)

        if st.button("Get Recommendations"):
            st.dataframe(recommend_by_stats(inputs, budget2, top_n2))

    # ---------- TAB 3 ----------
    with tab3:
        stats_input = {col: st.number_input(col, 0, 5000, 0, key="mv"+col) for col in feature_cols}

        if st.button("Predict Value"):
            value = predict_market_value(stats_input)
            if isinstance(value, float):
                st.success(f"Estimated Market Value: €{value:,.2f}")
            else:
                st.error(value)

# ============================================================
# 🧩 National Team Formation Viewer
# ============================================================
elif app_choice == "🧩 National Team Formation Viewer":
    st.title("🏆 National Team Formation Viewer")

    FORMATIONS = {
        "4-3-3": [1, 4, 3, 3],
        "4-4-2": [1, 4, 4, 2],
        "3-5-2": [1, 3, 5, 2],
        "4-2-3-1": [1, 4, 2, 3, 1],
        "4-1-4-1": [1, 4, 1, 4, 1],
        "5-3-2": [1, 5, 3, 2],
        "3-4-3": [1, 3, 4, 3],
        "4-5-1": [1, 4, 5, 1]
    }

    df = pd.read_csv(r"C:\Users\Manish.Khurana\Downloads\formation\best_11_ai_by_country.csv")
    df["country"] = df["country"].str.strip()

    country = st.selectbox("Select a Country", df["country"].unique())

    team = df[df["country"].str.lower() == country.lower()]
    if team.empty:
        st.warning(f"No data for {country}")
    else:
        formation_name = team["chosen_formation"].iloc[0]
        st.subheader(f"{country} - {formation_name}")

        if formation_name not in FORMATIONS:
            st.warning(f"Unknown formation: {formation_name}")
        else:
            formation = FORMATIONS[formation_name]
            players = team["name"].tolist()

            index = 0
            st.markdown("<div style='background-color:green; padding:20px;'>", unsafe_allow_html=True)

            for row_players in formation:
                cols = st.columns(row_players)
                for i in range(row_players):
                    if index < len(players):
                        with cols[i]:
                            st.markdown(
                                f"<div style='background-color:darkgreen; color:white; padding:10px; text-align:center; border-radius:5px;'>{players[index]}</div>",
                                unsafe_allow_html=True
                            )
                        index += 1

            st.markdown("</div>", unsafe_allow_html=True)
elif app_choice == "🛡️ Defender Market Value Predictor":
    @st.cache_resource
    def load_defender_artifacts():
        lgb_model = joblib.load("lgb_defender_model.pkl")
        xgb_model = joblib.load("xgb_defender_model.pkl")
        encoders = joblib.load("defender_encoders.pkl")
        return lgb_model, xgb_model, encoders
    lgb_model, xgb_model, encoders = load_defender_artifacts()

    st.title("🛡️ Defender Market Value Predictor")
    st.write("Enter defender details to predict market value (EUR)")

    with st.form("def_form"):
        name = st.text_input("Player Name","John Doe")
        country = st.text_input("Country of Citizenship","England")
        position = st.selectbox("Position",["Centre-Back","Full-Back","Wing-Back"])
        foot = st.selectbox("Preferred Foot",["Right","Left"])
        height = st.number_input("Height (cm)",140,220,185)
        club_comp = st.text_input("Club Domestic Competition ID","GB1")
        club_name = st.text_input("Club Name","Chelsea FC")
        yellow_cards = st.number_input("Yellow Cards",0)
        red_cards = st.number_input("Red Cards",0)
        goals = st.number_input("Goals",0)
        assists = st.number_input("Assists",0)
        minutes_played = st.number_input("Minutes Played",0)
        age = st.number_input("Age",16,45,25)
        submitted = st.form_submit_button("Predict Market Value")

    if submitted:
        player_df = pd.DataFrame([{
            'name': name,
            'country_of_citizenship': country,
            'position': position,
            'foot': foot,
            'height_in_cm': height,
            'current_club_domestic_competition_id': club_comp,
            'current_club_name': club_name,
            'yellow_cards': yellow_cards,
            'red_cards': red_cards,
            'goals': goals,
            'assists': assists,
            'minutes_played': minutes_played,
            'age': age
        }])
        categorical_cols = ['name','country_of_citizenship','position','foot','current_club_domestic_competition_id','current_club_name']
        for col in categorical_cols:
            le = encoders[col]
            if 'Unknown' not in le.classes_: le.classes_ = np.append(le.classes_,'Unknown')
            player_df[col] = player_df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            player_df[col] = le.transform(player_df[col])

        pred_lgb = lgb_model.predict(player_df)[0]
        pred_xgb = xgb_model.predict(player_df)[0]
        predicted_value = 0.6*pred_lgb + 0.4*pred_xgb
        predicted_value = max(predicted_value,0)
        st.success(f"💰 Predicted Market Value for {name}: €{predicted_value:,.0f}")