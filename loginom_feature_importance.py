import builtin_data
from builtin_data import InputTable, OutputTable
import pandas as pd
import numpy as np
import re
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from builtin_pandas_utils import to_data_frame, prepare_compatible_table, fill_table

# Normalization helpers (from 01_prepare_data.py)
SPACE_COLLAPSE_RE = re.compile(r"\s+")
NBSP_RE = re.compile(r"[\u00A0\u202F]")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\ufeff]")

def normalize_name(name: str) -> str:
    """Normalize column name."""
    if not isinstance(name, str):
        name = str(name)
    name = NBSP_RE.sub(" ", name)
    name = ZERO_WIDTH_RE.sub("", name)
    name = SPACE_COLLAPSE_RE.sub(" ", name).strip()
    return name

def pick_header_name(l0, l1, l2) -> str:
    """Pick name: prefer l2, else l1, else l0."""
    for level in [l2, l1, l0]:
        level_str = str(level)
        if 'Unnamed:' in level_str or level_str.lower() in ('nan', 'none', ''):
            continue
        norm = normalize_name(level)
        if norm:
            return norm
    return ""

# Static hints from column_analysis.txt (processed)
hints = {
    "categorical": {
        "группа наблюдения (1-здоровые, 2-гаг, 3-ум.пэ, 4-тяж.пэ,0-хрАГ)",
        "кровь (да-1, нет-2)",
        "местность (1 - город; 2 - село)",
        "брак (1 - зарегистрирован, 2 - не состоит в браке)",
        "образование (1 - высшее, 2 - неполное высшее, 3 - среднее)",
        "вид родоразрешения (1 - ер, 2 - кс, 0 - не родоразрешена)",
        "вредные привычки",
        "алкоголь (частота, количество) (1 - нет, 2 - да)",
        "наркотики (название, частота) (1 - нет, 2 - да)",
        "профессиональные вредности (1 - нет, 2 - да)",
        "многоплодие в анамнезе (1 - нет, 2- да)",
        "наследственные заболевания (1 - нет, 2 - да)",
        "переливания крови (1 - нет, 2 - да)",
        "регулярные (1 - нет, 2 - да)",
        "безболезненные (1 - нет, 2 - да)",
        "умеренные (1 - нет, 2 - да)",
        "контрацепция: (1 - нет, 2 - да)",
        "гинекологические заболевания (1 - нет, 2 - да)",
        "аборт",
        "самопроизвольный (1 - нет, 2 - да)",
        "неразвивающаяся беременность (1 - нет, 2 - да)",
        "преждевременные (1 - нет, 2 - да)",
        "операции (1 - нет, 2- да)",
        "сведения о детях",
        "патология плода (1 - нет, 2 -да)",
        "мертворождения (1 - нет, 2 - да)",
        "патология (1 - нет, 2 - орви, 3 - анемия, 4 - угроза прерывания, 5 - отеки, гипертензия)",
        "патология (1 - нет, 2 - орви, 3 - анемия, 4 - угроза прерывания, 5 -отеки, гипертензия, 6 - плацентарные нарушения)",
        "патология (1 - нет, 2 - орви, 3 - анемия, 4 - угроза прерывания, 5 - отеки, гипертензия, протеинурия, 6 - плацентарные нарушения)",
        "лекарственные препараты в i триместре беременности (1 - нет, 2 - вмк, 3 - микронизированный прогестерон, 4 - антигипертензивные препараты, 5 - препараты железа, 6 - антикоагулянты)",
        "ожирение (1 - нет, 2 - ожирение 1 степени, 3 - ожирение 2 степени, 4 - ожирение 3 степени)",
        "гирсутизм (1 - нет, 2 - да)",
        "стрии (1 - нет, 2 - да)",
        "послеоперационный рубец (1 - нет, 2 - да)",
        "отеки (1 - нет, 2 - да)",
        "периферические вены (1 - норма, 2 - варикозное расширение)",
        "матка (1 - в нормальном тонусе, 2 - возбудима)",
        "положение плода (1 - продольное, 2 - поперечное)",
        "предлежащая часть (1 - головка, 2 - тазовый конец)",
        "сердцебиение плода (1 - ясное ритмичное)",
        "околоплодные воды (1 - не изливались, 2 - подтекание оклоплодных вод)",
        "патологические выделения из половых путей (1 - нет, 2 - да)",
        "выделения (1 - слизистые, 2 - бели, 3 - творожистые)",
        "вход во влагалище (1 - рожавшей, 2 -нерожавшей)",
        "степень зрелости шейки (1 - зрелая, 2 -созревающая, 3 - незрелая, 4 - не проводилось в виду предлежания)",
        "деформации малого таза (1 - нет, 2 - да)",
        "риск (1 - низкий, 2 - средний, 3 - высокий)",
        "структура плаценты (1 - однородная, 2 - неоднородная)",
        "пуповина имеет (1 - 3 сосуда, 2 - 2 сосуда)"
    },
    "numerical": {
        "менструация с, лет",
        "начало половой жизни, лет",
        "номер настоящей беременности",
        "роды",
        "количество преждевременных родов",
        "общая прибавка в весе +, кг",
        "i триместр",
        "ii триместр",
        "iii триместр",
        "шкала глазго",
        "t",
        "сад",
        "дад",
        "ож, см",
        "вдм",
        "чсс",
        "биохимический анализ крови",
        "общий белок",
        "билирубин общий",
        "мочевина",
        "креатинин",
        "активность аст",
        "активность алт",
        "лейкоциты",
        "эпителий плоский",
        "эритроциты неизмененные",
        "общий анализ крови",
        "wbc-лейкоциты",
        "соэ",
        "rbc-эритроциты",
        "hct-гематокрит",
        "plt-тромбоциты",
        "mcv-средний объём эритроцита",
        "мчн-среднее объемное содержание гемоглобина в эритроците",
        "мчнс-средняя конц.гемоглобина в эритроците",
        "гемостаз",
        "протромбиновый индекс",
        "мно",
        "протромбиновое отношение",
        "ачтв",
        "свертываемость крови",
        "кровоточивость",
        "ог, см",
        "ож, см",
        "бедро, см",
        "плацента",
        "степень зрелости (0 - 0, 1 - 1, 2 - 2, 3 - 3, 4 - 0/1, 5- 1/2, 6 -2/3)",
        "количество околоплодных вод",
        "чсс",
        "апгар 1 минута",
        "апгар 5 минута",
        "первичная реанимация (1 - нет, 2 - да)",
        "длина",
        "пороки (1 - нет, 2 - упущен)",
        "ктр, мм",
        "бпр, мм",
        "твп, мм",
        "кровоток в венозном протоке (1 - норма)",
        "пи",
        "пи в маточных артериях справа",
        "пи в маточных артериях слева"
    }
}

def classify_features(X: pd.DataFrame, target_name: str) -> tuple[list[str], list[str]]:
    """Classify features using hints and heuristics."""
    categorical_features = []
    numerical_features = []
    for col in X.columns:
        if col == target_name:
            continue
        col_key = normalize_name(col).casefold()
        if col_key in hints["categorical"]:
            categorical_features.append(col)
            continue
        if col_key in hints["numerical"]:
            numerical_features.append(col)
            continue
        name_cues = ["(да-1", "(1 -", "группа", "тип", "категория", "степень", "стадия"]
        if any(cue in col_key for cue in name_cues):
            categorical_features.append(col)
            continue
        s = X[col]
        try:
            s_num = pd.to_numeric(s, errors="coerce")
            if s_num.notna().mean() >= 0.9:
                nuniq = s_num.dropna().nunique()
                if nuniq <= 10:
                    nearest_int = np.rint(s_num.dropna())
                    if np.all(np.abs(s_num.dropna() - nearest_int) <= 1e-9):
                        categorical_features.append(col)
                        continue
                numerical_features.append(col)
                continue
        except:
            pass
        non_null = s.dropna()
        nuniq = non_null.nunique()
        if nuniq <= 10:
            categorical_features.append(col)
        else:
            categorical_features.append(col)  # Treat as categorical if high cardinality
    return categorical_features, numerical_features

# Main logic
if InputTable:
    df = to_data_frame(InputTable)

    # Process column names as in 01_prepare_data.py
    # Check if columns are generic COL1, COL2, etc. (Loginom assigns these)
    expected_cols = [f'COL{i}' for i in range(1, len(df.columns) + 1)]
    if list(df.columns) == expected_cols and len(df) >= 3:
        # Create MultiIndex from first 3 rows as header
        df.columns = pd.MultiIndex.from_arrays([df.iloc[0], df.iloc[1], df.iloc[2]])
        # Remove header rows, keep data from row 3
        df = df.iloc[3:].reset_index(drop=True)
        # Now df.columns is MultiIndex, proceed as in original
        column_names = []
        for col in df.columns:
            l0, l1, l2 = col[:3] if len(col) >= 3 else (col[0], '', '')
            name = pick_header_name(l0, l1, l2)
            column_names.append(name)
    elif hasattr(df.columns, 'levels') and len(df.columns.levels) >= 3:  # MultiIndex
        column_names = []
        for col in df.columns:
            l0, l1, l2 = col[:3] if len(col) >= 3 else (col[0], '', '')
            name = pick_header_name(l0, l1, l2)
            column_names.append(name)
    else:
        # Assume simple columns, use as is
        column_names = list(df.columns)

    # Normalize column names
    column_names = [normalize_name(n) for n in column_names]

    # Handle duplicates
    seen = {}
    final_names = []
    for name in column_names:
        if name in seen:
            seen[name] += 1
            final_names.append(f"{name}_dup{seen[name]}")
        else:
            seen[name] = 0
            final_names.append(name)
    df.columns = final_names

    # Remove identifier
    identifier_raw = '№ пациента (пробирки) '
    identifier_key = normalize_name(identifier_raw).replace(" ", "").casefold()
    identifier_col = None
    for col in df.columns:
        if normalize_name(col).replace(" ", "").casefold() == identifier_key:
            identifier_col = col
            break
    if identifier_col:
        df = df.drop(columns=[identifier_col])

    # Find y
    y_prefix = "Группа наблюдения"
    y_key = normalize_name(y_prefix).casefold()
    y_col = None
    for col in df.columns:
        if normalize_name(col).casefold().startswith(y_key):
            y_col = col
            break
    if not y_col:
        raise ValueError(f"Target column not found. Available columns: {list(df.columns)}")
    y = df[y_col]
    X = df.drop(columns=[y_col])

    # Classify features
    categorical_features, numerical_features = classify_features(X, y_col)

    # Convert numerical features to float, handling comma as decimal separator
    for col in numerical_features:
        X[col] = pd.to_numeric(X[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Train model
    model = CatBoostClassifier(verbose=0, random_state=42)
    model.fit(X_train, y_train)

    # Feature importances
    importances = model.feature_importances_
    df_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Output to OutputTable
    if isinstance(OutputTable, builtin_data.ConfigurableOutputTableClass):
        prepare_compatible_table(OutputTable, df_importances, with_index=False)
    fill_table(OutputTable, df_importances, with_index=False)