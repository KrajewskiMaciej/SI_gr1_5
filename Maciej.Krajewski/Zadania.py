import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

dane = pd.read_csv('australian.txt', sep = " ")

def find_decision_classes(df, decision_col):
    return df[decision_col].unique()

def size_of_decision_classes(df, decision_col):
    return df[decision_col].value_counts()

def min_max_values(df):
    return df.describe().loc[["min", "max"]]

def count_unique_values(df):
    return df.nunique()

def list_unique_values(df):
    return {col: df[col].unique().tolist() for col in df.columns}

def standard_deviation(df, decision_col):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    std_all = df[numeric_cols].std()
    std_by_class = df.groupby(decision_col)[numeric_cols].std()
    return std_all, std_by_class

# def plot_standard_deviation(std_all, std_by_class):
#     plt.figure(figsize=(10, 5))
#     std_all.plot(kind='bar', color='blue', alpha=0.7)
#     plt.title("Odchylenie standardowe dla całego systemu")
#     plt.ylabel("Odchylenie standardowe")
#     plt.xlabel("Atrybut")
#     plt.xticks(rotation=45)
#     plt.show()
    
#     std_by_class.plot(kind='bar', figsize=(10, 5))
#     plt.title("Odchylenie standardowe w klasach decyzyjnych")
#     plt.ylabel("Odchylenie standardowe")
#     plt.xlabel("Atrybut")
#     plt.xticks(rotation=45)
#     plt.legend(title="Klasy decyzyjne")
#     plt.show()


# # Zadanie 3
# decision_col = dane.columns[-1]

# print("A: ",find_decision_classes(dane, decision_col),"\n")
# print("B: ",size_of_decision_classes(dane, decision_col), "\n")
# print("C: ",min_max_values(dane), "\n")
# print("D: ",count_unique_values(dane), "\n")
# print("E: ",list_unique_values(dane), "\n")
# print("F: \n")
# std_all, std_by_class = standard_deviation(dane, decision_col)
# print("Odchylenie standardowe (cały system):\n", std_all)
# print("Odchylenie standardowe (w klasach decyzyjnych):\n", std_by_class)
# #plot_standard_deviation(std_all, std_by_class)


# Zadanie 4

# a) Wprowadzenie 10% brakujących wartości i ich uzupełnienie
num_missing = int(0.1 * dane.size)
missing_indices = [(np.random.randint(0, dane.shape[0]), np.random.randint(0, dane.shape[1])) for _ in range(num_missing)]

for row, col in missing_indices:
    dane.iat[row, col] = np.nan

def fill_missing_values(df):
    for col in df.columns:
        if df[col].dtype == "O":  # Atrybuty symboliczne
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:  # Atrybuty numeryczne
            df[col].fillna(df[col].mean(), inplace=True)
    return df

data = fill_missing_values(dane)

# b) Normalizacja wartości numerycznych do przedziałów
def min_max_normalize(df, a, b):
    return df.apply(lambda col: (col - col.min()) * (b - a) / (col.max() - col.min()) + a if col.dtype != "O" else col)

normalized_data = {
    "[-1,1]": min_max_normalize(data, -1, 1),
    "[0,1]": min_max_normalize(data, 0, 1),
    "[-10,10]": min_max_normalize(data, -10, 10)
}

# c) Standaryzacja wartości numerycznych
def standardize(df):
    return df.apply(lambda col: (col - col.mean()) / col.std() if col.dtype != "O" else col)

standardized_data = standardize(data)

# d) 

# Wczytanie danych z pliku CSV
df = pd.read_csv('Churn_Modelling.csv')

# Wyświetlenie pierwszych kilku wierszy danych, aby zobaczyć strukturę
print(df.head())

# Przekształcenie atrybutu 'Geography' na zmienne dummy
df_dummies = pd.get_dummies(df['Geography'], prefix='Geography')

# Dodanie zmiennych dummy do oryginalnego DataFrame
df = pd.concat([df, df_dummies], axis=1)

# Usunięcie jednej z nowych kolumn (np. 'Geography_Spain'), aby uniknąć pułapki dummy variables
df = df.drop('Geography_Spain', axis=1)

# Wyświetlenie wynikowego DataFrame
print(df.head())

# Zapisanie zmienionego DataFrame do nowego pliku CSV (opcjonalnie)
df.to_csv('Churn_Modelling_with_dummies.csv', index=False)