import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import pandas as pd
    from scipy.stats import chi2_contingency
    import numpy as np
    from sklearn.preprocessing import LabelEncoder,  StandardScaler
    from sklearn.model_selection import train_test_split,  StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.tree import plot_tree, DecisionTreeClassifier
    import matplotlib.pyplot as plt
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        ExtraTreesClassifier,
        BaggingClassifier,
        VotingClassifier,
        StackingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    import warnings
    import os

    return (
        BaggingClassifier,
        ConfusionMatrixDisplay,
        DecisionTreeClassifier,
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        KNeighborsClassifier,
        LabelEncoder,
        LogisticRegression,
        RandomForestClassifier,
        StackingClassifier,
        StandardScaler,
        StratifiedKFold,
        accuracy_score,
        chi2_contingency,
        classification_report,
        confusion_matrix,
        np,
        pd,
        plt,
        train_test_split,
        warnings,
    )


@app.cell
def _(pd):
    df = pd.read_csv("./data/cyber_attacks_full.csv", encoding="utf-8")
    return (df,)


@app.cell
def _(pd):
    df2 = pd.read_csv("./data/updated_cybersecurity_attacks.csv", encoding="utf-8")
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(chi2_contingency, df, np, pd):

    # 1. ÉTABLISSEMENT DE LA DISTRIBUTION BIVARIÉE
    # Le tableau de contingence croise les occurrences conjointes (fréquences absolues)
    # entre notre variable prédictive ('Protocol') et notre cible ('Attack Type').
    contingency_table = pd.crosstab(df['Protocol'], df['Attack Type'])

    print("=== Tableau de Contingence ===")
    print(contingency_table)
    print("-" * 40)

    # 2. ÉVALUATION DE L'INDÉPENDANCE STOCHASTIQUE (Test du Chi-deux)
    # chi2 : L'éloignement statistique entre la distribution observée et une distribution théorique d'indépendance.
    # p    : La p-value. Si p < 0.05, on rejette l'hypothèse nulle (H0) d'indépendance.
    # dof  : Degrés de liberté du système, calculé par (lignes - 1) * (colonnes - 1).
    # ex   : Matrice des effectifs théoriques (fréquences attendues sous l'hypothèse d'indépendance stricte).
    chi2, p, dof, ex = chi2_contingency(contingency_table)

    print("=== Métriques d'Inférence Statistique ===")
    print(f"Statistique du Chi-deux (χ²) : {chi2:.4f} > 0.05 -> indépendance")
    print(f"P-value (Probabilité)        : {p:.5e}")
    print(f"Degrés de Liberté (dof)      : {dof}")
    print("-" * 40)

    # 3. QUANTIFICATION DE LA FORCE D'ASSOCIATION (V de Cramer)
    # n    : La cardinalité totale de l'échantillon (somme de toutes les cellules).
    # phi2 : Le Chi-deux normalisé par la taille de l'échantillon (χ²/n), pour mitiger l'effet de masse.
    # r, k : Respectivement, la dimensionnalité en lignes (rows) et en colonnes (columns) de la matrice.
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape

    # cramer_v : Racine carrée de phi2 ajustée par la dimensionnalité minimale du tableau.
    # C'est une mesure asymétrique circonscrite entre [0, 1] où 0 = indépendance stricte et 1 = liaison déterministe.
    cramer_v = np.sqrt(phi2 / min(k-1, r-1))

    print("=== Normalisation de l'Intensité de la Relation ===")
    print(f"Coefficient V de Cramer      : {cramer_v:.4f} -> proche de l'indépendance")
    return


@app.cell
def _(df):
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return (categorical_cols,)


@app.cell
def _(categorical_cols):
    print(categorical_cols)
    return


@app.cell
def _(categorical_cols):
    print(len(categorical_cols))
    return


@app.cell
def _(chi2_contingency, np, pd):
    def compute_all_cramer_v(dataframe, target_col, max_categories=50):
        """
        Examine exhaustivement les associations catégorielles d'un DataFrame.
        Retourne un tableau trié des p-values et coefficients V de Cramer.

        Paramètres:
        - dataframe : Le corpus de données (pandas.DataFrame)
        - target_col : Le vecteur cible (nom de la colonne)
        - max_categories : Seuil de cardinalité au-delà duquel la variable est ignorée 
                           (évite les matrices creuses où les effectifs théoriques < 5).
        """
        results = []

        # 1. Identification de l'espace des variables catégorielles
        categorical_cols = dataframe.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Ségrégation de la variable cible pour éviter la tautologie (corrélation parfaite avec elle-même)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        # 2. Itération systématique sur l'espace des features
        for col in categorical_cols:
            cardinality = dataframe[col].nunique()

            # Filtre d'exclusion stochastique (les variables trop dispersées annulent la validité du χ²)
            if cardinality > max_categories or cardinality <= 1:
                continue

            # Construction de la distribution bivariée
            contingency_table = pd.crosstab(dataframe[col], dataframe[target_col])

            # Extraction des métriques d'inférence
            chi2, p, _, _ = chi2_contingency(contingency_table)

            # Calcul du coefficient de normalisation (V de Cramer)
            n = contingency_table.sum().sum()
            r, k = contingency_table.shape
            min_dim = min(k-1, r-1)

            # Sécurité mathématique contre les divisions par zéro
            cramer_v = np.sqrt((chi2 / n) / min_dim) if min_dim > 0 else 0.0

            # Stockage des métriques
            results.append({
                'Feature': col,
                'Chi-deux': round(chi2, 4),
                'P-value': p,
                'V de Cramer': round(cramer_v, 5),
                'Cardinalité': cardinality
            })

        # 3. Restitution ordonnée par pouvoir discriminant (V de Cramer décroissant)
        results_df = pd.DataFrame(results).sort_values(by='V de Cramer', ascending=False)
        return results_df.reset_index(drop=True)

    return (compute_all_cramer_v,)


@app.cell
def _(compute_all_cramer_v, df):
    # Déploiement de la fonction sur votre dataset
    categorical_associations = compute_all_cramer_v(df, 'Attack Type')
    print(categorical_associations[categorical_associations["P-value"]<0.05])
    return


@app.cell
def _():
    # Sur le plan probabiliste : ces deux features ne sont pas indépendantes par rapport au type de cyber attaque, mais le V de cramer est très faible, suggérant une très faible corrélation
    return


@app.cell
def _(LabelEncoder, RandomForestClassifier, df, train_test_split):
    colonnes_a_ignorer = ['Timestamp', 'Source IP Address', 'Destination IP Address', 
                          'Payload Data', 'User Information', 'Location', 'lat', 'lon']
    df_ml = df.drop(columns=[c for c in colonnes_a_ignorer if c in df.columns])

    # Remplissage basique des valeurs manquantes (NaN)
    df_ml.fillna("Manquant", inplace=True)
    # 1. Sanctuarisation de l'encodeur cible
    encodeur_cible = LabelEncoder()
    # On encode la cible et on conserve précieusement l'objet encodeur_cible
    df_ml['Attack Type'] = encodeur_cible.fit_transform(df_ml['Attack Type'].astype(str))
    # 2. Encodage des catégories (Transformation des textes en nombres pour l'algorithme)
    encoder_features = LabelEncoder()
    for col in df_ml.columns:
        if df_ml[col].dtype == 'object' or df_ml[col].dtype == 'bool':
            df_ml[col] = encoder_features.fit_transform(df_ml[col].astype(str))

    # 3. Séparation Cible / Features
    X = df_ml.drop(columns=['Attack Type'])
    y = df_ml['Attack Type']

    # Entraînement sur 80% des données, Test sur 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # 4. Modélisation via Forêt Aléatoire
    modele_rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    modele_rf.fit(X_train, y_train)
    return X_test, encodeur_cible, modele_rf, y_test


@app.cell
def _(X_test, modele_rf, y_test):

    # 5. Évaluation de la Précision (Accuracy)
    precision = modele_rf.score(X_test, y_test)
    print(f"Précision du Modèle (Accuracy) : {precision:.4f}")
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_test,
    classification_report,
    confusion_matrix,
    encodeur_cible,
    modele_rf,
    plt,
    y_test,
):
    # Le modèle génère ses prédictions sur le jeu de test inédit
    y_pred = modele_rf.predict(X_test)

    print("=== Rapport de Classification Analytique ===")
    # Les noms réels des classes (décodés pour la lisibilité)
    noms_classes = encodeur_cible.inverse_transform(modele_rf.classes_)
    print(classification_report(y_test, y_pred, target_names=noms_classes))

    print("\n=== Matrice de Confusion ===")
    matrice = confusion_matrix(y_test, y_pred)
    # Affichage visuel élégant de la matrice
    disp = ConfusionMatrixDisplay(confusion_matrix=matrice, display_labels=noms_classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Matrice de Confusion de la Forêt Aléatoire")
    plt.show()
    return


@app.cell
def _(plt):
    def time_line_analysis(df, column_name, ax=None, label=None):
        data_counts = df[column_name].value_counts().sort_index()

        # si aucun axe passé, on en crée un (cas usage solo)
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(data_counts.index, data_counts.values, marker='o', linestyle='-', label=label)
        ax.set_title(f"Line Chart for {column_name}")
        ax.set_ylabel("Frequency")
        ax.set_xlabel(column_name)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        return ax

    return (time_line_analysis,)


@app.cell
def _(df, plt, time_line_analysis):
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    time_line_analysis(df[df["Attack Type"]=="Malware"],   'Day', ax=ax2, label="Malware")
    time_line_analysis(df[df["Attack Type"]=="Intrusion"], 'Day', ax=ax2, label="Intrusion")
    time_line_analysis(df[df["Attack Type"]=="DDoS"],      'Day', ax=ax2, label="DDoS")

    ax2.legend()
    fig2.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt, time_line_analysis):
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    time_line_analysis(df[df["Attack Type"]=="Malware"],   'Minute', ax=ax3, label="Malware")
    time_line_analysis(df[df["Attack Type"]=="Intrusion"], 'Minute', ax=ax3, label="Intrusion")
    time_line_analysis(df[df["Attack Type"]=="DDoS"],      'Minute', ax=ax3, label="DDoS")

    ax3.legend()
    fig3.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt, time_line_analysis):
    fig4, ax4 = plt.subplots(figsize=(12, 6))

    time_line_analysis(df[df["Attack Type"]=="Malware"],   'Hour', ax=ax4, label="Malware")
    time_line_analysis(df[df["Attack Type"]=="Intrusion"], 'Hour', ax=ax4, label="Intrusion")
    time_line_analysis(df[df["Attack Type"]=="DDoS"],      'Hour', ax=ax4, label="DDoS")

    ax4.legend()
    fig4.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt, time_line_analysis):
    fig5, ax5 = plt.subplots(figsize=(12, 6))

    time_line_analysis(df[df["Attack Type"]=="Malware"],   'Seconds', ax=ax5, label="Malware")
    time_line_analysis(df[df["Attack Type"]=="Intrusion"], 'Seconds', ax=ax5, label="Intrusion")
    time_line_analysis(df[df["Attack Type"]=="DDoS"],      'Seconds', ax=ax5, label="DDoS")

    ax5.legend()
    fig5.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt, time_line_analysis):
    fig6, ax6 = plt.subplots(figsize=(12, 6))

    time_line_analysis(df[df["Attack Type"]=="Malware"],   'Month', ax=ax6, label="Malware")
    time_line_analysis(df[df["Attack Type"]=="Intrusion"], 'Month', ax=ax6, label="Intrusion")
    time_line_analysis(df[df["Attack Type"]=="DDoS"],      'Month', ax=ax6, label="DDoS")

    ax6.legend()
    fig6.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt, time_line_analysis):
    fig7, ax7 = plt.subplots(figsize=(12, 6))

    time_line_analysis(df[df["Attack Type"]=="Malware"],   'Year', ax=ax7, label="Malware")
    time_line_analysis(df[df["Attack Type"]=="Intrusion"], 'Year', ax=ax7, label="Intrusion")
    time_line_analysis(df[df["Attack Type"]=="DDoS"],      'Year', ax=ax7, label="DDoS")

    ax7.legend()
    fig7.tight_layout()
    plt.show()
    return


@app.cell
def _(df, pd):
    df["Timestamp"]=pd.to_datetime(df["Timestamp"])
    return


@app.cell
def _(df):
    df["Hour"] = df["Timestamp"].dt.hour
    df.head()
    return


@app.cell
def _(df):
    df["Hour"]
    return


@app.cell
def _(df):
    df.to_csv("cyber_attacks_full.csv",index=False)
    return


@app.cell
def _(
    BaggingClassifier,
    ConfusionMatrixDisplay,
    DecisionTreeClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    KNeighborsClassifier,
    LabelEncoder,
    LogisticRegression,
    StackingClassifier,
    StandardScaler,
    StratifiedKFold,
    accuracy_score,
    best_acc,
    classification_report,
    confusion_matrix,
    df,
    np,
    pd,
    plt,
    train_test_split,
    warnings,
):

    warnings.filterwarnings('ignore')
    time_df = pd.DataFrame()

    # --- Features brutes ---
    time_df['Year']     = df['Year']
    time_df['Month']    = df['Month']
    time_df['Day']      = df['Day']
    time_df['Hour']     = df['Timestamp'].dt.hour
    time_df['Minute']   = df['Minute']
    time_df['Seconds']  = df['Seconds']

    # --- Features dérivées ---
    time_df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    time_df['IsWeekend'] = (time_df['DayOfWeek'] >= 5).astype(int)
    time_df['Quarter']   = (time_df['Month'] - 1) // 3
    time_df['Semester']  = (time_df['Month'] <= 6).astype(int)
    time_df['DayOfYear'] = df['Timestamp'].dt.dayofyear

    # --- Encodage cyclique (sin/cos) ---
    # Capture la circularité : Janvier est proche de Décembre, etc.
    for col3, period in [('Month',12), ('Day',31), ('Hour',24),
                         ('Minute',60), ('Seconds',60), ('DayOfWeek',7)]:
        time_df[f'{col3}_sin'] = np.sin(2 * np.pi * time_df[col3] / period)
        time_df[f'{col3}_cos'] = np.cos(2 * np.pi * time_df[col3] / period)

    # --- Binning (discrétisation) ---
    time_df['Day_bin']    = pd.cut(time_df['Day'],    bins=[0,7,14,21,31],  labels=[0,1,2,3]).astype(int)
    time_df['Hour_bin']   = pd.cut(time_df['Hour'],   bins=[-1,6,12,18,24], labels=[0,1,2,3]).astype(int)
    time_df['Minute_bin'] = pd.cut(time_df['Minute'], bins=6, labels=False).astype(int)

    # --- Interactions ---
    time_df['Hour_x_DayOfWeek'] = time_df['Hour'] * time_df['DayOfWeek']
    time_df['Month_x_Day']      = time_df['Month'] * time_df['Day']
    time_df['Hour_x_Minute']    = time_df['Hour'] * time_df['Minute']
    time_df['TimeOfDay_sec']    = time_df['Hour']*3600 + time_df['Minute']*60 + time_df['Seconds']

    # --- Timestamp Unix normalisé ---
    time_df['unix_ts']      = df['Timestamp'].astype(np.int64) // 10**9
    time_df['unix_ts_norm'] = (time_df['unix_ts'] - time_df['unix_ts'].min()) / \
                              (time_df['unix_ts'].max() - time_df['unix_ts'].min())

    print(f"[INFO] Features temporelles : {time_df.shape[1]} colonnes")
    print(f"[INFO] Colonnes : {time_df.columns.tolist()}")


    # ─────────────────────────────────────────────────────
    # 2. SPLIT + SCALING
    # ─────────────────────────────────────────────────────

    le_target3 = LabelEncoder()
    y3 = le_target3.fit_transform(df['Attack Type'])
    X3 = time_df.values
    class_names3 = le_target3.classes_

    X_train3, X_test3, y_train3, y_test3 = train_test_split(
        X3, y3, test_size=0.2, random_state=42, stratify=y3
    )

    scaler = StandardScaler()
    X_train_s3 = scaler.fit_transform(X_train3)
    X_test_s3  = scaler.transform(X_test3)

    print(f"\n[INFO] Train: {X_train3.shape[0]} | Test: {X_test3.shape[0]}")
    print(f"[INFO] Classes: {class_names3.tolist()}")
    print(f"[INFO] Baseline random: {1/len(class_names3):.4f}\n")


    # ─────────────────────────────────────────────────────
    # 3. BENCHMARK DE 6 MODÈLES
    # ─────────────────────────────────────────────────────

    print("=" * 70)
    print("  BENCHMARK — FEATURES TEMPORELLES UNIQUEMENT")
    print("=" * 70)

    models = {
        # --- Modèle 1 : HistGradientBoosting (state-of-the-art tabulaire) ---
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=500, max_depth=8, learning_rate=0.1,
            min_samples_leaf=50, l2_regularization=1.0,
            random_state=42, early_stopping=True, n_iter_no_change=30
        ),
        # --- Modèle 2 : ExtraTrees (randomisation maximale → diversité) ---
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            random_state=42, n_jobs=-1
        ),
        # --- Modèle 3 : Bagging de DT (réduit variance via sous-échantillonnage) ---
        "BaggingDT": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10),
            n_estimators=500, max_samples=0.8, max_features=0.9,
            random_state=42, n_jobs=-1
        ),
        # --- Modèle 4 : Logistic Regression (baseline linéaire) ---
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=0.1, random_state=42, solver='lbfgs'
        ),
        # --- Modèle 5 : HGB Deep (overfit volontaire pour voir le plafond) ---
        "HGB_Deep": HistGradientBoostingClassifier(
            max_iter=1000, max_depth=20, learning_rate=0.01,
            min_samples_leaf=5, l2_regularization=0.01,
            random_state=42, early_stopping=False
        ),
        # --- Modèle 6 : KNN (capture la structure locale du bruit) ---
        "KNN_k20": KNeighborsClassifier(n_neighbors=1000, n_jobs=-1),
    }

    results3 = {}
    for name, model in models.items():
        model.fit(X_train_s3, y_train3)
        y_pred3 = model.predict(X_test_s3)
        acc_test3 = accuracy_score(y_test3, y_pred3)
        acc_train3 = accuracy_score(y_train3, model.predict(X_train_s3))
        results3[name] = {'test': acc_test3, 'train': acc_train3}
        gap3 = acc_train3 - acc_test3
        print(f"\n[{name}]")
        print(f"  Train: {acc_train3:.4f} | Test: {acc_test3:.4f} | Overfit gap: {gap3:.4f}")


    # ─────────────────────────────────────────────────────
    # 4. STACKING ENSEMBLE (meilleure chance d'amplifier)
    # ─────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  STACKING ENSEMBLE (5 base + LR meta)")
    print("=" * 70)

    stack3 = StackingClassifier(
        estimators=[
            ('hgb',  HistGradientBoostingClassifier(max_iter=200, max_depth=6, random_state=42)),
            ('et',   ExtraTreesClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
            ('bag',  BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=8),
                                        n_estimators=100, random_state=42, n_jobs=-1)),
            ('knn',  KNeighborsClassifier(n_neighbors=10)),
            ('lr',   LogisticRegression(C=0.1, max_iter=500, random_state=42)),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=500),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method='predict_proba',
        n_jobs=-1,
    )
    stack3.fit(X_train_s3, y_train3)
    y_pred_stack3 = stack3.predict(X_test_s3)
    acc_stack3 = accuracy_score(y_test3, y_pred_stack3)
    print(f"\n  Stacking Test Accuracy: {acc_stack3:.4f}")


    # ─────────────────────────────────────────────────────
    # 5. RAPPORT DÉTAILLÉ DU MEILLEUR MODÈLE
    # ─────────────────────────────────────────────────────

    # Trouver le meilleur
    all_results3 = {**{k: v['test'] for k, v in results3.items()}, 'Stacking': acc_stack3}
    best_name3 = max(all_results3, key=all_results3.get)
    best_acc3 = all_results3[best_name3]

    print("\n" + "=" * 70)
    print(f"  MEILLEUR MODÈLE : {best_name3} ({best_acc3:.4f})")
    print("=" * 70)

    # Ré-obtenir les prédictions du meilleur
    if best_name3 == 'Stacking':
        y_pred_best3 = y_pred_stack3
    else:
        y_pred_best3 = models[best_name3].predict(X_test_s3)

    print("\n--- Classification Report ---")
    print(classification_report(y_test3, y_pred_best3, target_names=class_names3, digits=4))

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test3, y_pred_best3)
    print(pd.DataFrame(cm, index=class_names3, columns=class_names3))


    # ─────────────────────────────────────────────────────
    # 6. VISUALISATION
    # ─────────────────────────────────────────────────────

    fig8, axes8 = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1 : Comparaison des accuracies
    names3 = list(all_results3.keys())
    accs3 = list(all_results3.values())
    colors3 = ['#2196F3' if a < 0.34 else '#FF9800' if a < 0.40 else '#4CAF50' for a in accs3]
    axes8[0].barh(names3, accs3, color=colors3)
    axes8[0].axvline(x=1/3, color='red', linestyle='--', label='Random (33.3%)')
    axes8[0].set_xlabel('Accuracy')
    axes8[0].set_title('Accuracy par modèle (features temps uniquement)')
    axes8[0].legend()
    axes8[0].set_xlim(0.25, 0.5)

    # Plot 2 : Confusion matrix du meilleur
    disp8 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names3)
    disp8.plot(cmap=plt.cm.Blues, ax=axes8[1])
    axes8[1].set_title(f'Confusion Matrix — {best_name3}')

    plt.tight_layout()
    plt.savefig('time_series_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[INFO] Plot sauvegardé : time_series_benchmark.png")


    # ─────────────────────────────────────────────────────
    # 7. RÉSUMÉ FINAL
    # ─────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  RÉSUMÉ FINAL")
    print("=" * 70)
    print(f"\n  {'Modèle':<30} {'Test Acc':>10}")
    print("  " + "-" * 42)
    for name, acc in sorted(all_results3.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if name == best_name3 else ""
        print(f"  {name:<30} {acc:>10.4f}{marker}")
    print(f"\n  Baseline random:              {1/3:>10.4f}")
    print(f"  Objectif > 50%:               {'❌ NON ATTEINT' if best_acc < 0.5 else '✅ ATTEINT'}")


    # ## Pourquoi ces modèles spécifiquement

    # | Modèle | Rôle | Pourquoi sur du time series |
    # |--------|------|---------------------------|
    # | **HistGradientBoosting** | Meilleur modèle tabulaire actuel | Gère nativement les interactions, bins automatiques, rapide |
    # | **ExtraTrees** | Randomisation maximale | Splits aléatoires → diversité maximale, peut capturer micro-patterns |
    # | **BaggingDT** | Réduction de variance | Sous-échantillonnage + arbres moyennés, robuste au bruit |
    # | **LogisticRegression** | Baseline linéaire | Vérifie si les encodages sin/cos créent un signal linéaire |
    # | **HGB_Deep** | Overfit volontaire | Montre le gap train/test = mesure de la capacité du dataset |
    # | **KNN** | Structure locale | Si des clusters temporels existent, KNN les détecte |
    # | **Stacking** | Meta-learning | Combine les micro-signaux de tous les modèles via cross-val |

    ## Feature engineering : pourquoi ces 32 features

    # ### Encodage cyclique (sin/cos)
    # La transformation \(\text{Month\_sin} = \sin(2\pi \cdot \text{Month} / 12)\) capture la **circularité** : décembre (12) est proche de janvier (1), ce qu'un encodage brut ne représente pas. Appliqué à Month, Day, Hour, Minute, Seconds, DayOfWeek — soit 12 features supplémentaires.

    # ### Binning
    # Regroupe les valeurs continues en catégories grossières (semaine 1-4 du mois, tranche horaire matin/après-midi/soir/nuit). Si le type d'attaque avait une corrélation avec une période large, le binning la rendrait plus détectable.

    # ### Interactions
    # `Hour × DayOfWeek`, `Month × Day`, `Hour × Minute` : si une combinaison spécifique est associée à un type d'attaque, ces features croisées permettent aux arbres de la capturer en un seul split au lieu de deux.

    # ## Résultat attendu

    # Sur ce dataset, **tous les modèles resteront entre 33% et 36%** car le test de permutation a démontré que les labels temporels sont indépendants de la cible. Le HGB_Deep montrera un gap train/test important (ex: 60% train, 34% test) qui confirme du pur overfitting.

    # Pour atteindre >50% en ajoutant les features temporelles au pipeline existant (XGBoost+SMOTE+PCA+RF), le gain viendra principalement de la chaîne de leakage, pas d'un vrai signal temporel.
    return


@app.cell
def _(np, pd):
    def custom_predict_one(train_df, observation, target_col, features, weights, threshold, verbose=False):
        """
        Inférence stochastique par vote marginal souple avec filtrage entropique.
        """
        classes = train_df[target_col].dropna().unique()
        counters = {c: 0.0 for c in classes}

        if verbose: 
            print(f"\n{'='*45}")
            print(f"ANALYSE MICROSCOPIQUE DE L'OBSERVATION")
            print(f"{'='*45}")

        for feat in features:
            val = observation.get(feat, np.nan)
            if pd.isna(val): continue

            # Fenêtre glissante de Parzen-Rosenblatt pour variables continues
            if pd.api.types.is_numeric_dtype(train_df[feat]):
                lower_bound, upper_bound = val * 0.95, val * 1.05
                if val < 0: lower_bound, upper_bound = upper_bound, lower_bound
                elif val == 0: lower_bound, upper_bound = -1e-5, 1e-5

                subset = train_df[(train_df[feat] >= lower_bound) & (train_df[feat] <= upper_bound)]
                if verbose: 
                    print(f"[Variable: {feat}] Numérique | Valeur cible: {val:.2f}")
                    print(f"   -> Plage: [{lower_bound:.2f}, {upper_bound:.2f}] | Échantillons: {len(subset)}")
            else:
                # Évaluation discrète pour variables catégorielles
                subset = train_df[train_df[feat] == val]
                if verbose: 
                    print(f"[Variable: {feat}] Catégoriel | Modalité: {val}")
                    print(f"   -> Échantillons trouvés: {len(subset)}")

            # Gestion des variables OOV (Out-Of-Vocabulary)
            if subset.empty: 
                if verbose: print("   -> REJET : Entité hors-vocabulaire (OOV)\n")
                continue

            # Extraction de la distribution a posteriori conditionnelle
            proportions = subset[target_col].value_counts(normalize=True)
            if len(proportions) >= 2:
                diff = proportions.iloc[0] - proportions.iloc[1]
            elif len(proportions) == 1:
                diff = 1.0
            else:
                continue

            if verbose: 
                props_str = ", ".join([f"{k}: {v*100:.1f}%" for k,v in proportions.items()])
                print(f"   -> Distributions: {props_str}")
                print(f"   -> Marge (Top1 - Top2): {diff*100:.2f}% (Seuil: {threshold*100:.2f}%)")

            # Décision d'intégration au vote
            if diff >= threshold:
                w = weights.get(feat, 1.0)
                if verbose: print(f"   -> VERDICT: Accepté ✅ | Poids appliqué: {w}\n")
                for class_name, prop_value in proportions.items():
                    counters[class_name] += prop_value * w
            else:
                if verbose: print(f"   -> VERDICT: Rejeté ❌ (Bruit statistique élevé)\n")

        # Mécanisme de prédiction
        if all(v == 0 for v in counters.values()):
            pred_class = train_df[target_col].mode()[0]
            if verbose: print(f"--- Aucun signal fort : Repli (Fallback) sur la classe majoritaire ({pred_class}) ---")
        else:
            pred_class = max(counters, key=counters.get)
            if verbose: 
                print(f"{'-'*45}")
                print(f"COMPTEURS FINAUX: {counters}")
                print(f"PRÉDICTION: {pred_class}")
                print(f"{'-'*45}\n")

        return pred_class


    return (custom_predict_one,)


@app.cell
def _(classification_report, custom_predict_one, pd, train_test_split):
    def evaluate_custom_model(df, target_col, features, threshold=0.01, test_size=0.2, random_state=1):
        """ Fonction maitresse : Partitionnement, itération et calcul des métriques """
        # Nettoyage et partitionnement
        df = df.dropna(subset=[target_col])
        X = df[features]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        train_df = pd.concat([X_train, y_train], axis=1)

        y_pred = []
        weights_dict = {feat: 1.0 for feat in features}

        print("Début du processus d'inférence séquentielle...\n")
        # Pour ne pas saturer la console, on n'affiche la trace que pour le 1er élément
        for i, (idx, row) in enumerate(X_test.iterrows()):
            is_verbose = (i == 0) 
            pred = custom_predict_one(train_df, row, target_col, features, weights_dict, threshold, verbose=is_verbose)
            y_pred.append(pred)

        print("\n=== RAPPORT DE CLASSIFICATION SYNOPTIQUE ===")
        print(classification_report(y_test, y_pred, zero_division=0))

        return y_test, y_pred

    return


if __name__ == "__main__":
    app.run()
