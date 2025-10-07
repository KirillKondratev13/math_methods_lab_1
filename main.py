import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """Загрузка и предобработка медицинских данных"""
    
    # Загрузка данных
    df = pd.read_excel(file_path, sheet_name='Все данные')
    
    print(f"Размер данных: {df.shape}")
    print(f"Колонки: {len(df.columns)}")
    
    return df

def analyze_missing_values(df):
    """Анализ пропущенных значений"""
    
    missing_info = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df)) * 100,
        'data_type': df.dtypes
    }).sort_values('missing_percentage', ascending=False)
    
    print("Топ-20 колонок с пропусками:")
    print(missing_info.head(20))
    
    return missing_info

def impute_missing_values(df, missing_info):
    """Заполнение пропущенных значений с разными стратегиями"""
    
    df_imputed = df.copy()
    
    # 1. Медицинские лабораторные данные - заполнение медианой
    lab_columns = [
        'Глюкоза', 'Общий белок', 'Билирубин общий', 'Билирубин прямой',
        'Мочевина', 'Креатинин', 'Активность АСТ', 'Активность АЛТ',
        'Фосфатаза щелочная', 'HGB-гемоглобин', 'WBC-лейкоциты', 'СОЭ',
        'RBC-Эритроциты', 'HCT-гематокрит', 'PLT-тромбоциты'
    ]
    
    # Фильтруем только существующие колонки
    existing_lab_cols = [col for col in lab_columns if col in df.columns]
    
    for col in existing_lab_cols:
        if missing_info[missing_info['column'] == col]['missing_percentage'].values[0] < 50:
            median_val = df[col].median()
            df_imputed[col].fillna(median_val, inplace=True)
            print(f"Заполнено {col} медианой: {median_val}")
    
    # 2. Категориальные медицинские переменные - мода
    categorical_medical = [
        'Вид родоразрешения', 'Курение', 'Алкоголь', 'Ожирение',
        'Отеки', 'Положение плода', 'Степень зрелости шейки'
    ]
    
    existing_cat_cols = [col for col in categorical_medical if col in df.columns]
    
    for col in existing_cat_cols:
        if missing_info[missing_info['column'] == col]['missing_percentage'].values[0] < 30:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
            df_imputed[col].fillna(mode_val, inplace=True)
            print(f"Заполнено {col} модой: {mode_val}")
    
    # 3. УЗИ параметры - KNN импутация для связанных параметров
    ultrasound_columns = ['БПР, см', 'ОГ, см', 'ОЖ, см', 'Бедро, см', 'ПМП, гр']
    existing_us_cols = [col for col in ultrasound_columns if col in df.columns]
    
    if len(existing_us_cols) > 1:
        # Используем KNN только если достаточно данных
        us_data = df_imputed[existing_us_cols]
        if us_data.notna().sum().min() > len(us_data) * 0.7:
            knn_imputer = KNNImputer(n_neighbors=3)
            df_imputed[existing_us_cols] = knn_imputer.fit_transform(us_data)
            print("Применена KNN импутация для УЗИ параметров")
    
    # 4. Для остальных числовых колонок - медиана
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_imputed[col].isnull().sum() > 0 and df_imputed[col].isnull().sum() < len(df_imputed) * 0.3:
            df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
    
    return df_imputed

def standardize_data(df):
    """Стандартизация числовых переменных"""
    
    df_standardized = df.copy()
    
    # Выбираем только числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Исключаем бинарные и категориальные числовые переменные
    exclude_cols = [
        'Группа наблюдения', 'Местность', 'Брак', 'Образование',
        'Курение', 'Алкоголь', 'Вид родоразрешения'
    ]
    
    cols_to_standardize = [col for col in numeric_cols if col not in exclude_cols]
    
    # Проверяем, есть ли колонки для стандартизации
    if len(cols_to_standardize) == 0:
        print("Нет числовых переменных для стандартизации")
        # Возвращаем оригинальный датафрейм и None для скалера
        return df_standardized, None
    
    # Стандартизация
    scaler = StandardScaler()
    df_standardized[cols_to_standardize] = scaler.fit_transform(df_standardized[cols_to_standardize])
    
    print(f"Стандартизировано {len(cols_to_standardize)} числовых переменных")
    
    return df_standardized, scaler

def save_processed_data(df, output_path):
    """Сохранение обработанных данных"""
    
    df.to_excel(output_path, index=False)
    print(f"Обработанные данные сохранены в: {output_path}")

# Основной пайплайн обработки
def process_medical_data(input_file, output_file):
    """Полный пайплайн обработки медицинских данных"""
    
    # 1. Загрузка данных
    print("1. Загрузка данных...")
    df = load_and_preprocess_data(input_file)
    
    # 2. Анализ пропусков
    print("\n2. Анализ пропущенных значений...")
    missing_info = analyze_missing_values(df)
    
    # 3. Заполнение пропусков
    print("\n3. Заполнение пропущенных значений...")
    df_imputed = impute_missing_values(df, missing_info)
    
    # 4. Стандартизация
    print("\n4. Стандартизация данных...")
    df_final, scaler = standardize_data(df_imputed)
    
    # 5. Сохранение
    print("\n5. Сохранение результатов...")
    save_processed_data(df_final, output_file)
    
    return df_final, scaler

# Запуск обработки
if __name__ == "__main__":
    input_file = "med_stats.xlsx"  # Укажите ваш путь
    output_file = "result.xlsx"
    
    processed_df, scaler = process_medical_data(input_file, output_file)
    
    print("\nОбработка завершена!")
    print(f"Исходные размеры: {pd.read_excel(input_file).shape}")
    print(f"Обработанные размеры: {processed_df.shape}")
    
    if scaler is not None:
        print("Стандартизация выполнена успешно")
    else:
        print("Стандартизация не выполнена (нет числовых переменных)")
