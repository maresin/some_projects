# импорт библиотек
import sys
import os
import re
import datetime
import pandas as pd
import numpy as np
import Levenshtein as lev
import easyocr
os.environ["OMP_NUM_THREADS"] = '1'
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Form405:
    '''
    Класс для распознавания отметки о взятии крови из "Учетной карточки донора" (Форма № 405-05/у).
    Атрибуты класса:    
        folder                         директория с файлами
        image_file                     название исходного файла изображения с расширением
        image_path                     путь к исходному файлу изображения           
        person_id                      префикс названия файлов, соответствующих определенному пользователю
        csv_recognised_path            путь к сгенерированному csv-файлу
        image_original                 исходный изображенное в обработке
        image_processed                преобразованное изображение в обработке
        n_rows                         число строк в таблице
        n_cols                         число столбцов в таблице
        num_field                      число основных полей в таблице
        ocr_result                     неообработанные результаты распознавания
        data                           датафрейм с результаты распознавания и дополнительными полями
        df_general                     обработанные результаты распознаавания в виде таблицы с с изображения                   
        df_recognised                  сгенерированный датафрейм из трех полей
    
    Последовательность операций для запуска распознавания:
       1. Создание экземпляра класса:
            obj = Form405('245365 .jpg', '405/')
       2. Распознавание и сохранение файла с результатами
            obj.recognize_image()
    '''
    # инициализация класса и назначение пути к файлу изображения
    def __init__(self, file, folder="resources/"):
        '''
        Предполагаем, что программа будет работать с конкретной, заранее назначенной папкой. 
        Названия файлов соответствуют ID пользователя и имеют целочисленный формат.
        Новые файлы создаются поддиректории "resources/".
        '''
        self.folder, self.image_file, self.image_path = None, None, None
        self.image_original, self.image_processed = None, None        
        self.ocr_result, self.data = None, None
        self.n_cols, self.n_rows, self.num_field = None, None, None
        self.df_general, self.df_recognised = None, None
        self.person_id =  None
        self.csv_recognised_path = None
        
        # проверка наличия указанного файла
        image_path = folder + file
        if os.path.exists(image_path) == False:
            print(f'Warning:1. Line:{sys._getframe().f_lineno}. Файл {file} отсутствует в рабочей директории!')
            return None
        # проверка считывания файла
        try:
            image_original = np.array(Image.open(image_path))
            # инициализация путей
            self.folder = folder
            self.image_file = file
            self.image_path = image_path
            self.image_original = image_original
        except:
            print(f'Warning:2. Line:{sys._getframe().f_lineno}. Файл {file} не является файлом изображения!')
            return None
        # получения префикса названия файла, соответствующего id пользователя
        try:
            self.person_id = re.match(r'(\D*)(\d+)(\D*)(.*)(\.)(\D*)', self.image_file).group(2)
        except:
            # id пользователя не является названием файля
            self.person_id = re.match(r'(.*)(\.)(\D*)', self.image_file).group(1)
        self.csv_recognised_path = self.folder + self.person_id + '_recognised.csv'
    
    # функция распознавания изображения (основная)
    def recognize_image(self):
        '''
        Стадии распознавания:
        (1) Предобработка изображения в виде обрезки его верхней части;
        (2) Получение первичных данных распознавания с помощью EasyOCR;
        (3) Сведение полученных первичных данных в одну таблицу;
        (4) Отсев мусорных данных, не относящихся к записям о донациях;
        (5) Парсинг горизонтальных пространственных меток через кластеризацию столбцов;
        (6) Парсинг вертикальных пространственных меток через наращивание строки с близкими параметрами;
        (7) Создание таблицы данных по результатам парсинга;
        (8) Коррекция записей в таблице данных;
        (9) Формирование выходной таблицы данных. 
        '''        
        
        # проверка успешности считывания файла
        if self.image_original is None:
                     return
        # (1) предобработка изображения
        # обрезка верхней части изображения
        self.image_processed = self.image_original[int(self.image_original.shape[0]/2):self.image_original.shape[0], :]
        
        # (2) первичной распознавания изображения без разметки таблицы
        # абсолютный путь к файлам медели
        model_path = os.path.abspath('model/')
        # файлы модели были скопированы вручную в заданную директороию 
        reader = easyocr.Reader(lang_list=['ru'],
                                gpu=False, 
                                download_enabled=False, 
                                model_storage_directory=model_path)
        self.ocr_result = reader.readtext(image=self.image_processed, width_ths=0.7)
        # проверка результатов первичного распознавания 
        if len(self.ocr_result) == 0:
            print(f'Warning:3. Line:{sys._getframe().f_lineno}. Файл {self.image_file} не распознан!')
            return
        
        # (3) подготовка данных распознавания и их запись в датафрейм
        # запись первичных результатов распознавания в словарь
        data_dict = self.__fill_data_dict()        
        # формирования датафрейма из записей словаря
        df = pd.DataFrame(data_dict)
        # ширина и высота бокса с текстом
        df['width'] = df['x_ru'] - df['x_lu']
        df['heigth'] = df['y_lb'] - df['y_lu']
        # центр бокса с текстом
        df['center_horizontal'] = (df['x_ru'] + df['x_lu'])/2
        df['center_vertical'] = (df['y_lb'] + df['y_lu'])/2
        
        # (4) отсев мусорных данных
        # поиск верхней и нижней строки с датой
        h_min = max(df['y_lu'])
        h_max = min(df['y_lu'])
        for i in range(len(df)):
            h_curr = self.__get_limit_value(df.iloc[i,:])
            if h_curr != None:
                if h_curr < h_min:
                    h_min = h_curr
                elif h_curr > h_max:
                    h_max = h_curr
        # отсечение по высоте с учетом поправки
        df = df[((df['y_lu'] >= h_min - df['heigth'].min()) & (df['y_lu'] <= h_max + df['heigth'].min()))]
        # проверка наличия данных для обработки
        if len(df) == 0:
            print(f'Warning:4. Line:{sys._getframe().f_lineno}. Отсутствуют данные для обработки!')
            return
        # запись датафрейма
        self.data = df
        
        # (5) парсинг горизонтальных пространственных меток 
        # кластеризация данных на столбцы
        self.__clasterize_data()
        # получение образцов кластеров по порядку, начиная с 0-го
        claster_samples = []
        for j in range(self.n_cols):
            for i in self.data.index:
                if self.data.loc[i, 'cluster_labels'] == j:
                    claster_samples.append(self.data.loc[i, 'center_horizontal'])
                    break
        # получение индекса кластера в соответствие с его расположением
        claster_idx = list(np.argsort(claster_samples))
        # перевод случайных имен кластеров в упорядоченный по горизонтали список
        clasters_dict = {claster_name:idx for idx, claster_name  in enumerate(claster_idx)}
        self.data['claster_sorted'] = self.data['cluster_labels'].apply(lambda x: clasters_dict[x])         
        # проверка количества кластеров и запись числа уникальных полей записи о донации
        if self.n_cols % 3 == 0:
            self.num_fields = 3
        elif self.n_cols % 4 == 0:
            self.num_fields = 4            
        else:
            print(f'Warning:5. Line:{sys._getframe().f_lineno}. Неизвестный формат таблицы!')
            return         
        
        # (6) парсинг вертикальных пространственных меток
        # общий список индексов для всех колонок
        idx_list = []
        # получение индексов  начальной колонки
        idx_init = self.data[self.data['claster_sorted'] == 0].sort_values(by=['center_vertical']).index.to_list()
        # запись индексов начальной колонки
        idx_current = idx_init
        idx_list.append(idx_init)
        # число строк определяется по начальной колонке
        self.n_rows = len(idx_init)
        # предельные значения начальной колонки
        limits_init = []
        for i in range(self.n_rows):
            limits = []
            # верхний и нижний пределы диапазона поиска
            limits.append(self.data.loc[idx_init[i], 'center_vertical'] + 0.5*self.data.loc[idx_init[i], 'heigth'])
            limits.append(self.data.loc[idx_init[i], 'center_vertical'] - 0.5*self.data.loc[idx_init[i], 'heigth'])
            limits_init.append(limits)
        # предельные значения текущей колонки
        limits_current = limits_init
        # проход по столбцам
        for i in range(self.n_cols-1):
            # список индексов следующей колонки
            idx_next_full = self.data[self.data['claster_sorted'] == (i+1)].sort_values(by=['center_vertical']).index.to_list()
            # списки отобранных индексов и предельных значениий следующей колонки
            idx_next_finded, limits_next = [], []
            # поиск соседней ячейки в строке для текущего ряда
            for j in range(self.n_rows):
                # проход по следующему ряду
                for idx in idx_next_full:
                    center_next = self.data.loc[idx, 'center_vertical']
                    # попадание текста в заданные пределы
                    if limits_current[j][0] >= center_next >= limits_current[j][1]:
                        limits = []
                        # запись найденного индекса и диапазона поиска
                        idx_next_finded.append(idx)
                        limits.append(self.data.loc[idx, 'center_vertical'] + 0.5*self.data.loc[idx, 'heigth'])
                        limits.append(self.data.loc[idx, 'center_vertical'] - 0.5*self.data.loc[idx, 'heigth'])
                        limits_next.append(limits)
                        break
                    # соседняя ячейки текущего ряда не найдена
                    elif idx == idx_next_full[-1]:
                        idx_next_finded.append(None)
                        # запись пределов начальной конки для текущей строки
                        limits_next.append(limits_init[j])
            # запись индексов найденных ячеек и смена предельных значений для дальнейшего поиска
            idx_list.append(idx_next_finded)
            limits_current = limits_next          
            
        # (7) сведение распарсенных данных в датасет
        df_consructed = pd.DataFrame()
        for i in range(len(idx_list)):
            # заполнение текущей колонки значений
            cluster_list = []
            for j in idx_list[i]:
                try:
                    cluster_list.append(self.data.loc[j, ['value']].values[0])
                except:
                    cluster_list.append("UNKNOWN")
            df_consructed[i] = cluster_list
            
        # (8) коррекция распознанных значений
        # проверка корректности числа кластеров
        df_general = df_consructed.copy()
        # поправка для случая записи из 4 колонок (с подписью)
        add = 0
        if self.n_cols % 4 == 0:
            add = 1
        # проход по колонкам с учетом их типа
        for i in range(df_general.shape[1]):
            if i % (3 + add) == 0:
                try:
                    # удаление пробелов
                    df_general[i] = df_general[i].apply(lambda x: "".join(x.split()))
                    # коррекция значений
                    df_general[i] = df_general[i].apply(lambda x: self.__check_adequacy(x, col_type='date'))
                except:
                    df_general[i] = "UNKNOWN"
            if i % (3 + add) == 1:
                try:
                    df_general[i] = df_general[i].apply(lambda x: self.__check_adequacy(x, col_type='type'))                    
                except:
                    df_general[i] = "UNKNOWN"
            if i % (3 + add) == 2:
                try:
                    df_general[i] = df_general[i].apply(lambda x: "".join(x.split()))
                    df_general[i] = df_general[i].apply(lambda x: self.__check_adequacy(x, col_type='quantity'))
                except:
                    df_general[i] = "UNKNOWN"
            if i % (3 + add) == 3:
                df_general[i] = "UNKNOWN"
        # сохранение скорректированного результата
        self.df_general = df_general
        
        # (9) формирование финальной таблицы записей из 3 полей
        data_type = ['date', 'type', 'quantity']
        # финальная таблицы записей
        df_recognised = pd.DataFrame({'date':[],'type':[],'quantity':[]})
        # временный словарь для хранения блока записей
        block_dict = {}
        # проход по полям таблицы и заполнение временного словаря
        for i in range(self.n_cols):
            # текущий тип поля
            current_field = (i + self.num_fields) % self.num_fields
             # пропуск поля с подписью
            if self.num_fields == 4 and current_field == 3:
                continue
            # подтверждение типа данных от некоторых ячеек
            if sum([self.__check_cell_type(stroke, data_type[current_field]) for stroke in df_general[i]]) > 0:
                # запись текущего столбца по текущему ключу во временный словарь
                data_type_confirmed = data_type[current_field]
                block_dict[data_type_confirmed] = df_general[i]
                # завершение формирования блока записей
                if current_field == 2:
                    df_added = pd.DataFrame(block_dict)
                    # удаление пустой строки в конце блока
                    if current_field > 0:
                        if sum(list(map(lambda x: x == 'UNKNOWN', df_added.iloc[-1]))) == self.num_fields:
                            df_added = df_added.iloc[:-1]
                    # добавление блока в общий датафрйем
                    df_recognised = pd.concat([df_recognised, df_added], axis=0)
                    # очистка словаря
                    block_dict = {}
                    df_recognised.reset_index(drop=True, inplace=True)
        
        # (10) сохранение результатов распознавания в csv-файл
        if len(df_recognised) > 0:
            # удаление дулирующихся записей
            df_recognised = df_recognised.query("not(date == type == quantity == 'UNKNOWN')").drop_duplicates()
            self.df_recognised = df_recognised                
            df_recognised.to_csv(self.csv_recognised_path)        
    
        
    # функция преобразования первичных результатов распознавания в записи словаря   
    def __fill_data_dict(self):
        data_dict = {'x_lu':[],'y_lu':[],'x_ru':[],'y_ru':[],'x_rb':[],'y_rb':[],'x_lb':[],'y_lb':[],'value':[],'prob':[]}
        for item in self.ocr_result:
            # отсев сомнительных данных (с дробными координатами)
            if type(item[0][0][0]) == np.int32:
                # координаты бокса с текстом (не ячейки!)
                data_dict['x_lu'].append(item[0][0][0])
                data_dict['y_lu'].append(item[0][0][1])
                data_dict['x_ru'].append(item[0][1][0])
                data_dict['y_ru'].append(item[0][1][1])
                data_dict['x_rb'].append(item[0][2][0])
                data_dict['y_rb'].append(item[0][2][1])
                data_dict['x_lb'].append(item[0][3][0])
                data_dict['y_lb'].append(item[0][3][1])
                # значение и верятность распознавания
                data_dict['value'].append(item[1])
                data_dict['prob'].append(item[2])
        return data_dict       
        

    # функция парсинга даты и получения высоты бокса с текстом
    def __get_limit_value(self, stroke):
        # удаление пробелов
        stroke_joined = "".join(stroke['value'].split())
        # поиск даты по маске
        date = None    
        mask = r'(.*)(\d{2})(?:.)(\d{2})(?:.)(\d{2,4})(.*)'
        try:
            parsed = re.match(mask, stroke_joined)      
            # строка в формате даты
            date = parsed.group(2) + '.' + parsed.group(3) + '.' + parsed.group(4)    
            return stroke['y_lu']
        except:
            pass
        return None 
    

    # функция разбивки данных на кластеры по вертикали
    def __clasterize_data(self):
        # выделения признака для кластеризации - середина текста ячейки по горизонтали
        X = self.data['center_horizontal'].values.reshape(-1, 1)
        # проверка наличия данных для обработки
        if X.shape[0] == 0:
            print(f'Warning:4. Line:{sys._getframe().f_lineno}. Отсутствуют данные для обработки!')
            return
        silhouette_max = 0
        self.n_cols = 9
        # подбор числа кластеров
        for n in [3,4,6,8,9]:
            clusterer = KMeans(n_clusters=n, n_init="auto", random_state=42)
            cluster_labels = clusterer.fit_predict(X)
            # оценка качества кластеризации
            silhouette_avg = silhouette_score(X, cluster_labels)    
            # оптимальное число кластеров
            if silhouette_avg > silhouette_max:
                silhouette_max = silhouette_avg
                self.n_cols = n
        # разбивка данных кластеры (столбцы)
        clusterer = KMeans(n_clusters=self.n_cols, n_init="auto", random_state=42)
        cluster_labels = clusterer.fit_predict(X)
        self.data['cluster_labels'] = cluster_labels
    

    # функция коррекции значений и оценки адекватности распознавания
    def __check_adequacy(self, stroke, col_type=0):
        # оценка количества
        if col_type == 'quantity':
            try:
                # проверка нахождения количества сданной крови в пределах от 50 до 700 мл
                if 100 <= int(stroke) <= 700:
                    # дополнительная коррекция нормы 450 мл
                    try:
                        re.match(r'(4(?:6|9)(?:0|9|8))', stroke).group(0)
                        stroke = 450
                    except:
                        pass
                    return stroke
                else:
                    return "UNKNOWN"
            except:
                return "UNKNOWN"
        # оценка даты
        elif col_type == 'date':
            # замена неккорректных разделителей в дате
            try:
                finded = re.match(r'(.*)(\d{2})(?:.)(\d{2})(?:.)(\d{2,4})(.*)', stroke)
                stroke_improved = finded.group(2) + '.' + finded.group(3) + '.' + finded.group(4)
            except:
                return "UNKNOWN"
            # дата приказа о форме справки 
            init_date = datetime.datetime.strptime(r'31.03.2005', '%d.%m.%Y')
            # текущее время
            current_date = datetime.datetime.now()
            date_stroke = ''
            # перебор возможных вариантов записи
            for pattern in [r'%d.%m.%Y', r'%d.%m.%y']:
                try:
                    date_stroke = datetime.datetime.strptime(stroke_improved, pattern)
                    break
                except:
                    pass
            # проверка нахождения дат в пределах от даты приказа до текущей
            if date_stroke != '' and (init_date <= date_stroke <= current_date):
                # вывод унифицированного варианта записи напр. 31.12.2020
                return date_stroke.strftime('%d.%m.%Y')
            else:
                return "UNKNOWN"
        # коррекция записи для типа донорства 
        elif col_type == 'type':
            try:                
                # перевод символов в нижний регистр
                stroke_improved = stroke.lower()
                try:
                    finded = re.match(r'(?:.*)(кр|пл|ц)(?:.?)(?:д)(.*)', stroke_improved) 
                    stroke_improved = finded.group(1) + r'/д' + finded.group(2)              
                except:
                    pass
                # замена цифр подходящими символами
                stroke_improved = re.sub('3|8|9', 'в', stroke_improved)
                stroke_improved = re.sub('5|6', 'б', stroke_improved)
                stroke_improved = re.sub('.ф', '/ф', stroke_improved)
                # восстановление неполной строки для распостраненных значений
                for cutted in ['кр/д', 'к/д', 'пл/д', 'т/ф','п/ф', 'ц/д']:
                    if stroke_improved == cutted:
                        return cutted + ' (бв)'
                if stroke_improved in ['(бв)', '(пл)']:
                    return r'кр/д ' + stroke_improved
                # очистка лишних знаков
                finded = re.match(r'.*((?:кр|к|пл|п|ц)(?:/д)|(?:т|п)(?:/ф))(.*)((?:\(бв\))|(?:\(пл\)))', stroke_improved)
                # проверка частей записи по отдельности
                blood_part = finded.group(1)
                reward_type = finded.group(3)
                if blood_part not in ['кр/д', 'пл/д']:
                    # коррекция значений с помощью расстояния  Левенштайна
                    if lev.distance(blood_part, 'кр/д') < 2:
                        blood_part = 'кр/д'
                    if lev.distance(blood_part, 'пл/д') < 2:
                        blood_part = 'пл/д'
                return finded.group(1) + ' ' + finded.group(3)
            except:
                return "UNKNOWN"
        # вывод без проверки    
        else:
            return stroke
        

    # функция проверки типа столбца
    def __check_cell_type(self, stroke, type_expected):
        # проверка преобразования строки в целое число
        if type_expected == 'quantity':
            try:
                int(stroke)
                return True
            except:
                return False
        # проверка преобразования строки в дату
        elif type_expected == 'date':
            try:
                datetime.datetime.strptime(stroke, '%d.%m.%Y')
                return True
            except:
                return False
        # проверка наличия в строке кириллических букв
        elif type_expected == 'type':
            try:
                re.match(r'^[а-я]{1}', stroke).group()
                return True
            except:
                return False  
        return None