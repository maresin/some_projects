## Описание файлов для создания docker-образа
Файлы в директории представляют собой содержимое рабочей папки для создания docker-образа. Помимо указанных файлов в ней должны присутствовать две папки: `model` с данными обученной модели ([архив](https://drive.google.com/file/d/10XZjccbG53hgLTqGCoyjRToD4Cfa-EQV/view?usp=sharing)) и присоединяемая папка resources, где будут находиться изображения со справками и сохраняться распознанные данные в виде csv-файлов.

Команды для сборки образа и запуска контейнера:
~~~
docker build -t ocr-app .
docker run -it --rm --name ocr-app -v c:/temp/form/resources:/usr/src/app/resources ocr-app
~~~
Для работы с файлами в docker монтируется внешняя папка. В примере это `c:/temp/form/resources`. Отсюда будут браться файлы для распознавания и сохраняться csv-файлы.  


__Примечание:__
Иногда, может потребоваться посмотреть как храняться записанные docker-файлы в OS Windows. Docker работает в подсистеме  Linux - WSL. Пдсмотреть и отредактировать файлы образов можно по следующим путям непосредственно в проводнике Windows:
Например, для смонтированного тома с названием `my_volume`
~~~
\\wsl$\docker-desktop-data\data\docker\volumes\my_volume\_data 
~~~
другой вариант записи -  
~~~
\\wsl.localhost\docker-desktop-data\data\docker\volumes\my_volume\_data
~~~
Или пример для данных самого приложения из рабочей папки `\usr\src\app`:  
~~~
\\wsl.localhost\docker-desktop-data\data\docker\overlay2\xpg6m9ctvhbn6imp3krajv6y9\diff\usr\src\app
~~~
