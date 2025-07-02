"""Игра угадай число"""
"""Компьютер сам угадывает число с минимальным количеством попыток"""

import numpy as np

def smart_predict(number: int = 1) -> int:
    
    """Угадываем число с помощью бинарного поиска, минимизируя количество попыток"""
    
    low = 1
    high = 100
    count = 0

    while low <= high:
        count += 1
        predict_number = (low + high) // 2
        if predict_number == number:
            break
        elif predict_number < number:
            low = predict_number + 1
        else:
            high = predict_number - 1
    return count

def score_game(predict_function) -> int:
    
    """Измеряем среднее количество попыток за 1000 игр"""
    
    count_ls = []  # список для количества попыток
    np.random.seed(1)  # фиксируем сид для воспроизводимости
    random_array = np.random.randint(1, 101, size=1000)  # список загаданных чисел

    for number in random_array:
        count_ls.append(predict_function(number))

    score = int(np.mean(count_ls))
    print(f'Ваш алгоритм угадывает число в среднем за: {score} попыток')
    return score

if __name__ == '__main__':
# RUN
    score_game(smart_predict)