## Описание предыдущих версий
Алгоритм распознавания включал предобработку изображений и был нацелен на автоматической определение таблицы на них с помощью `img2table`. 
Затем отдельные ячейки таблицы распознавались программе `TesseractOCR`. Причем различные типы данных имели свои особенности распознавания.  

Во второй версии скрипта была предпринята попытка отказаться от `img2table`и выделять таблицу самостоятельно,
так как качество выделения таблиц оказалось низким и как следствие низким оказалось и само качество распознавания текста. 
Предполагалось сначала выделить область с таблицей, затем выделить угловые ячейки и по их координатам провести исправление перспективы. К сожалению, подход сработал примерно в половине случаев. 
Для прочих изображений возможности распознавания только ухудшились. В результате от такого подхода пришлось отказаться.
