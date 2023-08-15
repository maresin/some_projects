import logging
logging.basicConfig(filename="applog.log",level=logging.INFO, filemode="w", format="%(asctime)s %(levelname)s %(message)s")

print("Modules importing... Wait for input.")

import asyncio
import os
import sched
import time
import datetime
import class_form405


# функция распознавания одиночных изображений
async def single_img_ocr(img_name):
    doc = class_form405.Form405(img_name)
    doc.recognize_image()
    logging.info(f"The process of recognizing the {img_name} is completed at {datetime.datetime.now().isoformat()}.")
    return doc

# функция создания задания для асинхронной обработки
def select_images():
    images = []
    # путь к рабочей директории
    this_dir = os.getcwd()
    # абсолютные пути к файлам в папках рабочей директории и названия файлов
    for root, _, files in os.walk(this_dir):    
        for file in files:
            # только jpg-изображения
            if file.endswith(".jpg"):            
                img_path = os.path.join(root, file)
                # проверка наличия csv-файлов с распознанным текстом
                csv_path = os.path.join(root, file.split('.')[0].strip()) + ".csv"
                if os.path.exists(csv_path) == False:
                    # сохранение нераспознанных файлов с список для обработки
                    images.append((file))    
    logging.info(f"There are {len(images)} files for recognition in total.")
    return images

# функция запуска цикла событий
async def start_tasks(start_type="immediate"):
    img_paths = select_images()
    if len(img_paths) == 0:
        print("There are no new files for recognition.")
        return None
    tasks = []     
    for img in img_paths:
        # задание на распознавание отдельного изображения
        tasks.append(asyncio.create_task(single_img_ocr(img)))
    logging.info(f"The event loop is running at {datetime.datetime.now().isoformat()}, the startup type is {start_type}.")    
    for task in tasks:
         await task
    return None

# функция запуска асинхронной функции для планировщика
def shed_start_tasks(start_type="immediate"):
    asyncio.run(start_tasks(start_type))


# выбор для распознавания одного или нескольких файлов
answer = input("Do you want to recognize single file? [Yes/No]:")
if answer.lower()[0] == "y":
    file = input("Please input file name for recognize:")
    asyncio.run(single_img_ocr(file))
# выбор отложенного или немедленного распознавания батча
elif answer.lower()[0] == "n":
    batch_answer = input("Do you want to schedule batch recognition? [Yes/No]:")
    if batch_answer.lower()[0] == "n":
        logging.info("The immediate batch recognition task is running...")
        asyncio.run(start_tasks())
    # запланировать распознавание батча
    elif batch_answer.lower()[0] == "y":
        start_time = input(f"Please enter the recognition start time in the ISOformat (Example: 2000-12-31T23:59:59):")
        # парсинг введенной даты
        try:
            # проверка формата введенного времени
            start_time = datetime.datetime.fromisoformat(start_time).timestamp()
            # проверка времени запуска распознавания
            if ((start_time - time.time())//3600 <= 24) and (start_time - time.time()) > 0:
                # планирование и запуск
                shed_task = sched.scheduler(timefunc=time.time, delayfunc=time.sleep)
                shed_task.enterabs(start_time, 1, shed_start_tasks, argument=("delayed",))
                logging.info("The delayed batch recognition task is running...") 
                shed_task.run()                          
            else:
                logging.info("An incorrect sleeping time was entered.")
            pass
        except:
            logging.info("An incorrect date format was entered.")
    else:
        logging.info("An incorrect response was entered.")    
else:
    logging.info("An incorrect response was entered.")