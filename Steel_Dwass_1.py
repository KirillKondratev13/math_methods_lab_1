import builtin_data
from builtin_data import InputTable, OutputTable
import pandas as pd
import scikit_posthocs as sp
from builtin_pandas_utils import to_data_frame, prepare_compatible_table, fill_table

# === ПАРАМЕТР ДЛЯ РУЧНОГО ВЫБОРА КАТЕГОРИАЛЬНОГО ПРИЗНАКА ===
# Укажи здесь название категориального столбца:
selected_categorical = "_Obschaya_informatsiya___Gruppa_nablyudeniya__1_zdorovye__2_GAG__3_um_PE__4_tyazh_PE_0_khrAG__Zamena"  # <-- замени на нужное имя

# Преобразуем входную таблицу в pandas DataFrame
if InputTable:
    df = to_data_frame(InputTable)

# Проверяем, что указанный признак существует в таблице
if selected_categorical not in df.columns:
    raise ValueError(f"Категориальный признак '{selected_categorical}' не найден. "
                     f"Доступные столбцы: {list(df.columns)}")

# Категориальный и количественные признаки
cat_col = selected_categorical
num_cols = [c for c in df.columns if c != cat_col]

results = []

# Проходим по каждому количественному признаку
for col in num_cols:
    # Пост-хок тест Steel-Dwass
    posthoc = sp.posthoc_dscf(df, val_col=col, group_col=cat_col)
    
    # Преобразуем в длинный формат
    posthoc_long = posthoc.stack().reset_index()
    posthoc_long.columns = ['Группа 1', 'Группа 2', 'p-value']
    posthoc_long['Признак'] = col
    results.append(posthoc_long)

# Объединяем результаты
output_frame = pd.concat(results, ignore_index=True)

# Добавляем колонку с интерпретацией
output_frame['Значимая разница'] = output_frame['p-value'].apply(lambda x: 'Да' if x < 0.05 else 'Нет')

# Подготовка структуры выходной таблицы и заполнение
if isinstance(OutputTable, builtin_data.ConfigurableOutputTableClass):
    prepare_compatible_table(OutputTable, output_frame, with_index=False)
fill_table(OutputTable, output_frame, with_index=False)