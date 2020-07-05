# -*- coding: utf-8 -*-
"""homework_classification-2_les-2_ddmitry.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IUtXrJbc7dTcOnk-xWRpLPtuhysMQk4y

# Урок 2. Многоклассовая классификация.

Посмотрим на примере алгоритма логистической регрессии и метода опорных векторов, как работать с различными методами многоклассовой классификации.

### 1.
Вспомните датасет Wine. Загрузите его, разделите на тренировочную и тестовую выборки (random_state=17), используя только [9, 11, 12] признаки.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

### YOUR CODE HERE ###
wine_data = load_wine()
x_train, x_test, y_train, y_test = train_test_split(wine_data.data[:, [9, 11, 12]], wine_data.target, 
                                                    random_state=17)

print(wine_data.DESCR[:600])

"""**Задайте тип кросс-валидации с помощью StratifiedKFold: 5-кратная, random_state=17.**"""

from sklearn.model_selection import StratifiedKFold, cross_val_score

# skf = ### YOUR CODE HERE ###
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
# skf.get_n_splits(x_train, y_train)

"""### 2.
Обучите логистическую регрессию (LogisticRegression) с параметром C по умолчанию и random_state=17. Укажите гиперпараметр multi_class='ovr' - по умолчанию многие классификаторы используют именно его. С помощью cross_val_score сделайте кросс-валидацию (используйте объект skf) и выведите среднюю долю правильных ответов на ней (используйте функцию mean). Отдельно выведите долю правильных ответов на тестовой выборке.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

### YOUR CODE HERE ###
lr = LogisticRegression(random_state=17, multi_class='ovr')
cvs = cross_val_score(lr, X=x_train, y=y_train, cv=skf, )
print('Predicted accuracy:', cvs.mean())
lr.fit(x_train, y_train)
print('accuracy_score:',
      accuracy_score(y_test, lr.predict(x_test)) )



"""### 3.
Обучите метод опорных векторов (SVC) с random_state=17 и остальными параметрами по умолчанию. Этот метод при мультиклассовой классификации также использует метод "ovr". Сделайте кросс-валидацию (используйте skf) и, как и в предыдущем пункте, выведите среднюю долю правильных ответов на ней. Отдельно выведите долю правильных ответов на тестовой выборке.
"""

from sklearn.svm import SVC

### YOUR CODE HERE ###
svc = SVC(random_state=17)
cvs = cross_val_score(svc, X=x_train, y=y_train, cv=skf, )
print('Predicted accuracy:', cvs.mean())
svc.fit(x_train, y_train)
print('accuracy_score:',
      accuracy_score(y_test, svc.predict(x_test)) )

"""Как видно из полученной метрики, на тестовой выборке метод с гиперпараметрами по умолчанию работает явно намного хуже логистической регрессии. В целом, SVM достаточно плохо масштабируется на размер обучающего набора данных (как видно, даже с тремя признаками он работает не очень хорошо), но благодаря возможности выбора различных ядер (функций близости, которые помогают разделять данные) и другим гиперпараметрам SVM можно достаточно точно настроить под определенный вид данных. Подробнее на этом останавливаться в контексте данного урока не будем.

### 4.
Для предсказаний обеих моделей постройте матрицу ошибок (confusion matrix) и напишите, какие классы каждая из моделей путает больше всего между собой.
"""

from sklearn.metrics import classification_report, confusion_matrix

### YOUR CODE HERE ###
print('Алгоритм LogisticRegression, confusion_matrix:')
print(confusion_matrix(y_test, lr.predict(x_test)) )

print('\nАлгоритм SVC, confusion_matrix:')
print(confusion_matrix(y_test, svc.predict(x_test)) )

"""Для LogisticRegression, алгоритм принял 4 экземпляра вина третьего типа за второй
Для SVC все хуже. Он не смог распознать третий тип вина, и отнес данные к первому и второму.

### 5.
Для каждой модели выведите classification report.
"""

### YOUR CODE HERE ###
print('Алгоритм LogisticRegression, classification_report:')
print(classification_report(y_test, lr.predict(x_test)) )

print('\n\nАлгоритм SVC, classification_report:')
print(classification_report(y_test, svc.predict(x_test)) )

