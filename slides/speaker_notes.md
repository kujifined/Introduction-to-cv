# Presentation script and speaking roles

[Presentation PDF](final_presentation.pdf) | [Demo recording script](../docs/DEMO_RECORDING_SCRIPT.md)

Основа: текст Никиты, сверенный с финальными результатами. Все четверо выступают; Руслан затем управляет notebook. Целевой тайминг слайдов около 4–4,5 минут, до пяти минут остаётся запас на переключения. Двухминутный demo идёт отдельно, после слайдов. Это план времени, а не результат замера: перед записью или защитой прочитайте текст вслух с таймером.

| Время | Кто говорит | Что на экране |
|---|---|---|
| 0:00–0:45 | Данил | Слайд 1: задача и исследовательские вопросы |
| 0:45–1:45 | Руслан | Слайд 1: данные, baseline и общий эксперимент |
| 1:45–3:15 | Никита | Слайд 2: clean/corrupted F1, размер и runtime |
| 3:15–4:15 | Тимур | Слайд 3: ошибка Sc_69, её возможная причина и ограничения |
| 4:15–4:30 | Данил | Слайд 3: вывод и переход к demo |

## Danil: task and question

Our project classifies defects on steel surfaces. Given a cropped image, the model predicts one of six defect classes in the NEU-CLS dataset. The image on the right is an example of crazing.

We asked two questions. First, can a smaller EfficientNet match the quality of our ResNet baseline? Second, does training with camera-like distortions help either model handle degraded images? To answer those questions, we used the same data split and preprocessing for every run.

Передать слово Руслану, оставаясь на слайде 1.

## Ruslan: data, baseline and method

We excluded one exact duplicate, leaving 1,799 active images. The fixed split contains 1,259 training, 180 validation and 360 test crops. Reviewed duplicate groups do not cross splits. Inputs are grayscale images converted to three channels and resized to 224 pixels.

Our baseline is ImageNet-pretrained ResNet-18 with minimal preprocessing. We compared it with EfficientNet-B0 and trained both with minimal or camera-aware augmentation, using seeds forty-two, forty-three and forty-four. Camera-aware training adds effects such as brightness changes, noise and blur.

All layers were fine-tuned using shared data loaders, optimizer settings and evaluation code. Checkpoint selection used clean validation macro-F1. All twelve selected checkpoints were evaluated on the same held-out test set.

Переключить на слайд 2 и передать слово Никите.

## Nikita: results and comparison

Start with the clean scores on the right. All four configurations are close to one in macro-F1. The larger differences appear when we apply synthetic corruptions to the same test crops.

Across twenty-seven corruption conditions, camera-aware training raised ResNet's average macro-F1 from about 0.83 to 0.96. For EfficientNet, it increased from about 0.74 to 0.95. Both architectures improved on this average at every training seed.

We average the twenty-seven condition-level F1 scores within each seed, then report the mean and sample standard deviation across three seeds. The plus-minus values are seed variation, not confidence intervals.

ResNet camera had the higher overall corrupted mean, while EfficientNet camera led on one seed. The ranking was therefore not consistent. Augmentation did not improve every condition or clean probability quality.

EfficientNet has about four million parameters versus eleven million for ResNet. However, its measured batch-one T4 latency was about 8.85 milliseconds versus 3.6 milliseconds. Fewer parameters did not mean faster inference in this implementation.

Параметры и runtime находятся в отчёте и README, а не в таблице на слайде. Не указывать на несуществующую колонку. Опубликованная latency относится к FP32 forward на T4 и исключает загрузку, preprocessing и передачу CPU-to-GPU; это не время CPU demo.

Переключить на слайд 3 и передать слово Тимуру.

## Timur: failure and limitations

This is a real clean-test failure. The true label is scratches, but minimal EfficientNet at seed forty-three predicted inclusion with about fifty-five percent probability. Scratches came second at about forty percent.

The mark is faint and elongated, which may explain the confusion. This is a visual hypothesis, not a proven cause. The report analyzes three different clean-test failure examples; this slide shows the one used in our demo.

These results use 360 source test crops. Corrupted views are paired versions of those images, not new independent samples. Synthetic shifts do not establish reliability on a production line. NEU-CLS also has no normal class, so we cannot estimate false alarms on defect-free steel.

Слайд 3 остаётся на экране. Передать слово Данилу для заключения.

## Danil: conclusion and transition

The main result is improved average corruption robustness from camera-aware training, with remaining trade-offs. We will now demonstrate a clean crop, the same crop under blur, and this scratch error.

Теперь Руслан запускает отдельный двухминутный demo или включает его заранее записанную версию. Сохранённое видео называйте recorded demo. Порядок сцен остаётся прежним: clean Cr_107, blurred Cr_107, clean Sc_69.

## Notes for rehearsal

- Не читайте все строки таблицы вслух. Никита показывает правую clean-колонку, затем левую corrupted-колонку и её прирост.
- Заголовок слайда 2 относится к corrupted F1: clean F1 в среднем немного снизился. Не утверждайте, что augmentation улучшила обе метрики.
- Три seeds используются в итоговых метриках. На слайде 3 и в demo используется один последовательный seed 43; его confidence не следует смешивать с примером Sc_69 при seed 44 из отчёта.
- Объяснение ошибки остаётся предположением. Не называйте confidence доказательством правильности или промышленной надёжности.
- Если потребуется сократить речь, уберите абзац про параметры и latency, оставив его для Q&A. Числа, объяснение SD и ограничения сохраняйте.
