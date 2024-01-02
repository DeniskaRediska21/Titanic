# Анализ данных
Состав данных:
+ ID пассажира
+ Выжил ли
+ Класс билета
+ Имя
+ Пол
+ Возраст 
+ Сколько родственников + муж/жена
+ Сколько роителей + детей
+ Номер билета
+ Сколько заплачено за билет??
+ Номер каюты
+ Где погрузился

## Пробелы в данных

Почти во всех данных присутствуют пробелы, их надо как-то устранить

+ Возраст: предлагается когда возрасть неизвестен брать средний возраст для пассажиров с таким-же классом билета и полом
+ В номерах кают очень много пропусков так что данные нужно либо опустить, либо обрабатывать отдельной моделью, либо учитывать только известность номера каюты
+ В номерах билетов есть класс LINE, не уверен, является ли это пропуском


## Feature engeneering

+ Номера билетов можно формально разделить на несколько классов, может это важно
+ Нам даны имена пассажиров, возможно разделить пассажиров по предпологаемым рассам(хотя бы на 2 рассы) по имени/фамилии, возможно выживаемость зависила еще и от рассы
+ Полнота данных тоже может являться показателем, т.к. выживший человек скорее всего передаст о себе полную информацию
 
## Подготовка данных

+ Приведение данных в численную форму
+ Нормализация и центровка всех данных
+ Декорреляция данных с помощью метода главных компонент
+ One-hot-encoding класса пассажиров и места посадки

## Предпологаемое решение

### Нейросетевой алгоритм

+ Полносвязная сеть
+ Использование раздельных моделей для разных типов людей (М/Ж, старые/молодые/дети) *может не хватить обучающих данных*
+ Подход к обучению основанный на сиамских сетях (увеличение расстояния между классами в многомерном пространстве)