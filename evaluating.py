import pandas as pd
import pingouin as pg
from statsmodels.stats.anova import AnovaRM

# Datos proporcionados
data_length = list(range(1, 28))

data_contour = pd.DataFrame({
    'subject': data_length,
    'JuanCarlos': [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
    'Mauricio': [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,1],
    'MoCASystem': [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
})

data_numbers = pd.DataFrame({
    'subject': data_length,
    'JuanCarlos': [0,3,0,0,1,3,2,2,4,4,4,4,4,2,4,4,2,4,4,2,4,2,4,4,4,4,2],
    'Mauricio': [0,3,0,0,2,2,1,3,4,4,4,3,4,4,3,3,3,3,3,3,3,3,3,4,4,3,2],
    'MoCASystem': [0,2,0,0,2,4,2,0,4,4,4,3,4,2,2,4,3,3,4,4,3,2,4,4,3,3,1]
})

data_handclocks = pd.DataFrame({
    'subject': data_length,
    'JuanCarlos': [0,3,1,0,0,3,1,3,4,4,3,3,4,3,3,4,4,4,3,4,4,3,4,3,4,3,2],
    'Mauricio': [0,1,0,0,0,0,1,2,4,4,3,3,3,3,3,4,4,3,2,3,3,3,3,3,3,3,1],
    'MoCASystem': [0,1,1,1,1,1,2,1,4,4,3,2,2,4,4,2,4,1,1,1,2,3,4,3,3,3,1]
})

data_algorithm = pd.DataFrame({
    'subject': data_length,
    'JuanCarlos': [1,8,3,2,3,8,5,7,10,10,9,9,10,7,9,10,8,10,9,8,10,7,10,9,10,9,6],
    'Mauricio': [1,6,2,2,4,4,4,7,10,10,9,8,9,9,8,9,9,8,7,7,8,8,8,9,9,8,4],
    'MoCASystem': [2,5,3,3,5,7,6,3,10,10,9,7,8,8,8,8,9,6,7,7,7,7,10,9,8,8,4]
})

# Función para calcular ICC y ANOVA
def analyze_data(data):
    # Calcular ICC
    data_melt = data.melt(id_vars=['subject'], var_name='Evaluator', value_name='Score')
    icc = pg.intraclass_corr(data=data_melt, targets='subject', raters='Evaluator', ratings='Score')
    print("ICC Results:\n", icc)
    
    # Realizar ANOVA de medidas repetidas
    aov = AnovaRM(data_melt, 'Score', 'subject', within=['Evaluator']).fit()
    print("ANOVA Results:\n", aov.summary())
    
    return icc, aov

# Analizar cada conjunto de datos
print("Contour Analysis")
icc_contour, aov_contour = analyze_data(data_contour)

print("\nNumbers Analysis")
icc_numbers, aov_numbers = analyze_data(data_numbers)

print("\nHand Clocks Analysis")
icc_handclocks, aov_handclocks = analyze_data(data_handclocks)

print("\nAlgorithm Analysis")
icc_algorithm, aov_algorithm = analyze_data(data_algorithm)
