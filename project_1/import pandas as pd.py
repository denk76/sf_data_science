import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Загрузка данных
churn_data = pd.read_csv(r'C:\Users\Denis\Documents\datasi\7\churn.csv')

# Удаляем лишний столбец
churn_data.drop('RowNumber', axis=1, inplace=True)

# Визуализация и выводы по данным

# 1. Соотношение ушедших и лояльных клиентов
counts = churn_data['Exited'].value_counts()
plt.figure(figsize=(6,4))
sns.barplot(x=counts.index, y=counts.values)
plt.xticks([0,1], ['Лояльные', 'Ушедшие'])
plt.ylabel('Количество')
plt.title('Соотношение ушедших и лояльных клиентов')
plt.show()

# Вывод
print("Лояльных клиентов значительно больше, их примерно 8000.")
print("Ушедших клиентов примерно 2000.")
print("Можно сделать вывод о том, что большинство клиентов остаются лояльными, однако есть заметная часть, которая покидает организацию.\n")

# 2. Распределение баланса для клиентов с балансом > 2500
high_balance = churn_data[churn_data['Balance'] > 2500]
plt.figure(figsize=(8,5))
sns.histplot(high_balance['Balance'], bins=30)
plt.title('Распределение баланса для клиентов с балансом > 2500')
plt.xlabel('Баланс')
plt.ylabel('Количество клиентов')
plt.show()

# Вывод
print("Основная часть клиентов имеет баланс в диапазоне от примерно 80,000 до 160,000, с пиком около 120,000.")
print("Большинство клиентов имеют средний уровень баланса, очень низкие и очень высокие встречаются реже.\n")

# 3. Распределение баланса по статусу оттока
plt.figure(figsize=(8,5))
sns.boxplot(x='Exited', y='Balance', data=churn_data)
plt.xticks([0,1], ['Лояльные', 'Ушедшие'])
plt.title('Распределение баланса по статусу оттока')
plt.xlabel('Статус оттока')
plt.ylabel('Баланс')
plt.show()

# Вывод
print("У лояльных клиентов баланс в основном в диапазоне 0-150,000, у ушедших — более широкий разброс.")
print("Оба типа имеют балансы в схожих пределах, но у лояльных меньшая вариативность.\n")
print("Возможные причины ухода: низкие процентные ставки, высокие комиссии, неудобные условия, недостаток персонализации.\n")

# 4. Распределение возраста по статусу оттока
plt.figure(figsize=(8,5))
sns.histplot(data=churn_data, x='Age', hue='Exited', multiple='stack', bins=30)
plt.title('Распределение возраста по статусу оттока')
plt.xlabel('Возраст')
plt.ylabel('Количество')
plt.show()

# Вывод
print("Большинство клиентов в возрасте 30-50 лет.")
print("Ушедшие — более равномерное распределение; лояльные — пиковая концентрация 35-45 лет.")
print("Для удержания клиентов в этих группах стоит рассматривать персонализированные подходы.\n")

# 5. Взаимосвязь кредитного рейтинга и зарплаты
plt.figure(figsize=(8,5))
sns.scatterplot(x='CreditScore', y='EstimatedSalary', hue='Exited', data=churn_data)
plt.title('Взаимосвязь кредитного рейтинга и зарплаты')
plt.xlabel('Кредитный рейтинг')
plt.ylabel('Предполагаемая зарплата')
plt.legend(title='Отток')
plt.show()

# Вывод
print("Большинство клиентов имеют высокие кредитные рейтинги и зарплаты.")
print("Низкий рейтинг и доход связаны с большим оттоком — важно фокусироваться на этих группах.\n")

# 6. Доля ушедших клиентов по полу
plt.figure(figsize=(6,4))
sns.barplot(x='Gender', y='Exited', data=churn_data, estimator=np.mean)
plt.title('Доля ушедших клиентов по полу')
plt.ylabel('Доля ушедших')
plt.show()

# Вывод
print("Доля ушедших выше среди женщин.")
print("Рекомендуется анализировать причины и разрабатывать меры по удержанию женщин-клиентов.\n")

# 7. Отток по количеству услуг
counts = churn_data.groupby(['NumOfProducts', 'Exited']).size().unstack()
counts.plot(kind='bar', stacked=True, figsize=(8,5))
plt.title('Отток по количеству услуг')
plt.xlabel('Количество услуг')
plt.ylabel('Количество клиентов')
plt.show()

# Вывод
print("Клиенты с меньшим количеством услуг чаще уходят.")
print("Расширение спектра услуг может снизить отток.\n")

# 8. Отток в зависимости от активности
plt.figure(figsize=(6,4))
sns.barplot(x='IsActiveMember', y='Exited', data=churn_data)
plt.xticks([0,1], ['Неактивные', 'Активные'])
plt.ylabel('Доля ушедших')
plt.title('Отток в зависимости от статуса активности')
plt.show()

# Вывод
print("Доля ушедших выше среди неактивных.")
print("Активность способствует удержанию клиентов.\n")

# 9. Доля ушедших по странам
country_exit_rate = churn_data.groupby('Geography')['Exited'].mean()
plt.figure(figsize=(8,4))
sns.heatmap(country_exit_rate.to_frame().T, annot=True, cmap='Reds')
plt.title('Доля ушедших клиентов по странам')
plt.show()

# Вывод
print("Наибольший уход — в Германии (32%), ниже — во Франции и Испании (~16-17%).\n")

# 10. Анализ по категориям кредитного рейтинга и Tenure
def get_credit_score_cat(credit_score):
    if credit_score >= 300 and credit_score < 500:
        return "Very_Poor"
    elif credit_score >= 500 and credit_score < 601:
        return "Poor"
    elif credit_score >= 601 and credit_score < 661:
        return "Fair"
    elif credit_score >= 661 and credit_score < 781:
        return "Good"
    elif credit_score >= 781 and credit_score < 851:
        return "Excellent"
    elif credit_score >= 851:
        return "Top"
    elif credit_score < 300:
        return "Deep"

# Применение функции
churn_data['CreditScoreCat'] = churn_data['CreditScore'].apply(get_credit_score_cat)

# Создание сводной таблицы
pivot_table = churn_data.pivot_table(
    index='CreditScoreCat',
    columns='Tenure',
    values='Exited',
    aggfunc='mean'
)

# Визуализация тепловой карты
plt.figure(figsize=(10,6))
sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlGnBu')
plt.title('Доля ушедших по категориям кредитного рейтинга и Tenure')
plt.xlabel('Класс Tenure')
plt.ylabel('Класс кредитного рейтинга')
plt.show()

# Вывод
print("Наиболее уязвимы клиенты с низким кредитным рейтингом и коротким сроком обслуживания.")
print("Клиенты с высоким рейтингом и большим Tenure менее склонны к уходу.\n")
