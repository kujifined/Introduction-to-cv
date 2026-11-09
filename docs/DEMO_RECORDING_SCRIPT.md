# Two-minute demo recording: four speakers

Все четверо говорят по очереди, Руслан управляет notebook весь ролик. Тайминг целевой: один раз прочитайте вслух с переключениями перед финальной записью. Записывайте экран с микрофоном; при удалённом подключении проверьте, что в записи слышно всех.

## До начала записи

1. Откройте `09_live_demo.ipynb` вместе с `live_demo.py` и `demo_assets` из архива Руслана.
2. Поместите `icv_final_results.zip` в `results_download/` и следуйте `notebooks/README_LIVE_DEMO.md`. Автоматический поиск весов работает без личных абсолютных путей.
3. Выполните первую кодовую ячейку `prepare` и дождитесь `Ready: four verified checkpoints, seed 43, CPU. Run scenes 1–3 below.` Она загружает модели и выполняет warm-up. В ролик этот setup не включаем.
4. Очистите прежние outputs только трёх сцен, сохраняя запущенное ядро. Не перезапускайте kernel после подготовки. Проверьте масштаб, чтобы исходное изображение и все четыре строки таблицы читались на записи.
5. Начните запись на разделе `1 · Clean crazing`. Во время ролика Руслан запускает три scene cells по очереди и показывает свежий output каждой.

Ниже считаем только кодовые ячейки: первая `prepare`, вторая `clean_reference`, третья `same_image_blurred`, четвёртая `observed_failure`. Номера execution count в квадратных скобках для навигации не используем.

## 0:00–0:20 | Danil | Introduction

На экране раздел `1 · Clean crazing`; сцену ещё не запускаем.

“We classify cropped steel defects into six classes. Here we compare ResNet eighteen and EfficientNet B zero, each trained with minimal or camera-aware augmentation. We use frozen seed forty-three checkpoints. Each scene runs fresh CPU inference.”

## 0:20–0:50 | Ruslan | Clean input

В начале своей реплики Руслан запускает вторую кодовую ячейку:

```python
demo_results["clean_reference"] = demo.predict_case("clean_reference")
display(HTML(render_case(demo_results["clean_reference"])))
```

Показывает Cr_107 и все четыре строки таблицы.

“I will first run the clean image. The dataset label is crazing. All four models predict it correctly. The table shows the two most likely classes and their probabilities. Every model receives the same input and preprocessing. We load existing weights; no training happens during this demonstration.”

## 0:50–1:25 | Nikita | Blurred input

Когда Никита начинает говорить “Now we apply…”, Руслан запускает третью кодовую ячейку:

```python
demo_results["same_image_blurred"] = demo.predict_case("same_image_blurred")
display(HTML(render_case(demo_results["same_image_blurred"])))
```

В кадре должны быть original/blurred изображения и четыре строки predictions. На словах про ошибки Руслан указывает курсором на minimal-модели, затем на camera-aware.

“Now we apply the fixed severity-three Gaussian blur to the same crop, before resizing. Both minimally trained models make an error: ResNet predicts pitted surface, and EfficientNet predicts inclusion. Both camera-trained models still predict crazing. This illustrates the augmentation package's benefit on this example. Our overall comparison uses the complete test set and all three seeds.”

## 1:25–1:50 | Timur | A real failure

Когда Тимур начинает “Here is…”, Руслан запускает четвёртую кодовую ячейку:

```python
demo_results["observed_failure"] = demo.predict_case("observed_failure")
display(HTML(render_case(demo_results["observed_failure"])))
```

Показывает Sc_69. Курсором выделяет строку EfficientNet-B0 minimal: inclusion около 55.3%, scratches около 39.6%.

“Here is a clean image labelled scratches. Minimal EfficientNet predicts inclusion, while the other three models predict scratches. Its two leading probabilities are about fifty-five and forty percent. The elongated, low-contrast mark may resemble an inclusion. This is a possible explanation, not a proven cause.”

## 1:50–2:00 | Danil | Conclusion

Последний output остаётся на экране, новые ячейки не запускаем.

“Augmentation improves robustness in our experiments, but errors remain. These are illustrative examples; our reported metrics use the full evaluation.”

Остановите запись после заключения. Если закончите немного раньше двух минут, не нужно искусственно заполнять паузу. Если не укладываетесь, сначала сократите паузы между участниками; не ускоряйте речь до неразборчивой.

## Перед отправкой

- Пересмотрите начало и конец, проверьте слышимость всех четырёх голосов и читаемость таблиц.
- Убедитесь, что в кадре видны свежие outputs трёх сцен и что код выполнялся во время записи.
- Сохраните MP4, загрузите в выбранное вами хранилище и проверьте доступ по ссылке без авторизации. Публикация автоматически не выполняется.
- Добавьте ссылку как `Demo video` в technical summary. Не переносите probabilities со слайда для seed 44 в это demo: здесь везде seed 43.
