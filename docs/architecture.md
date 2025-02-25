polymer_ai/
├── configs/                          # Конфигурационные файлы
│   ├── forward_model_config.yaml     # Конфигурация Forward VAE
│   ├── inverse_model_config.yaml     # Конфигурация Inverse VAE
│   ├── preprocessing_config.yaml     # Конфигурация предобработки данных
│   └── similarity_config.yaml        # Конфигурация базы схожести материалов
├── data/                             # Хранилище данных
│   ├── material_database/            # Базы данных материалов
│   │   ├── polymer_properties.csv    # Свойства полимеров
│   │   ├── element_compatibility.csv # Совместимость элементов
│   │   ├── reaction_conditions.json  # Условия реакций
│   │   └── material_embeddings.pkl   # Предвычисленные эмбеддинги материалов
│   ├── experimental_data/            # Экспериментальные данные
│   │   ├── experiments.csv           # Записанные эксперименты
│   │   └── validation_results.csv    # Данные для валидации
│   ├── simulation/                   # Генерация синтетических данных
│   │   ├── generate_simulated_data.py # Генерация синтетических данных
│   │   └── simulated_experiments.csv  # Синтетические эксперименты
│   ├── preprocessing/                # Предобработка данных
│   │   ├── normalize.py              # Нормализация данных
│   │   ├── tokenize_equipment.py     # Токенизация оборудования
│   │   ├── aggregate_reactions.py    # Агрегация реакций
│   │   ├── featurize_materials.py    # Извлечение признаков из материалов
│   │   ├── process_time_series.py    # Обработка временных рядов
│   │   └── handle_multi_step_reactions.py # Обработка многоступенчатых реакций
│   └── test_data/                    # Новая директория для тестовых данных
├── feature_engineering/              # Новая директория для функционального инжиниринга
│   ├── create_features.py            # Создание новых признаков
│   └── feature_selection.py          # Отбор важных признаков
├── models/                           # Модели
│   ├── forward_vae/                  # Компоненты Forward VAE
│   │   ├── encoder.py                # Кодировщик Forward VAE
│   │   ├── decoder.py                # Декодировщик Forward VAE
│   │   └── loss.py                   # Пользовательская функция потерь для Forward VAE
│   ├── inverse_vae/                  # Компоненты Inverse VAE
│   │   ├── encoder.py                # Кодировщик Inverse VAE
│   │   ├── decoder.py                # Декодировщик Inverse VAE
│   │   └── loss.py                   # Пользовательская функция потерь для Inverse VAE
│   ├── physics_informed/             # Физически информированные модели
│   │   ├── pinn_encoder.py           # Кодировщик PINN
│   │   └── pinn_decoder.py           # Декодировщик PINN
│   ├── gnn/                          # Графовые нейронные сети
│   │   ├── gnn_encoder.py            # Кодировщик GNN
│   │   └── gnn_decoder.py            # Декодировщик GNN
│   ├── rl/                           # Обучение с подкреплением
│   │   ├── rl_agent.py               # Агент RL
│   │   └── environment.py            # Определение среды для RL
│   ├── similarity_db/                # База схожести материалов
│   │   ├── embeddings.py             # Генерация эмбеддингов материалов
│   │   └── faiss_index.py            # Индекс FAISS для быстрого поиска схожести
│   └── model_utils/                  # Новая директория для утилит моделей
│       ├── save_load_model.py        # Сохранение/загрузка моделей
│       └── inference_utils.py        # Утилиты для инференса
├── knowledge_graph/                  # Знаниевая база
│   ├── kg_builder.py                 # Построение знаниевой базы
│   ├── kg_query.py                   # Запросы к знаниевой базе
│   └── kg_visualization.py           # Визуализация знаниевой базы
├── visualization/                    # Визуализация
│   ├── dashboard.py                  # Веб-панель управления
│   ├── plot_results.py               # Генерация графиков результатов
│   └── explainability.py             # Интерпретация моделей
├── core/                             # Основная логика и утилиты
│   ├── system.py                     # Оркестровка системы
│   ├── utils.py                      # Общие утилиты
│   ├── substitution_logic.py         # Логика замены материалов
│   └── postprocessing.py             # Послепrocessing предсказаний
├── training/                         # Тренировочные пipelines
│   ├── train_forward.py              # Тренировка Forward VAE
│   ├── train_inverse.py              # Тренировка Inverse VAE
│   ├── train_pinn.py                 # Тренировка физически информированных моделей
│   └── callbacks.py                  # Callback'и для тренировки
├── inference/                        # Pipelines для вывода
│   ├── predict_properties.py         # Предсказание свойств
│   ├── suggest_materials.py          # Предложение материалов
│   ├── validate_predictions.py       # Проверка предсказаний
│   └── real_time_monitoring.py       # Реальное мониторинговое интегрирование
├── tests/                            # Тестирование
│   ├── unit_tests/                   # Юнит-тесты
│   │   ├── test_encoder.py           # Тест кодировщиков
│   │   ├── test_decoder.py           # Тест декодировщиков
│   │   ├── test_loss.py              # Тест функций потерь
│   │   └── test_similarity.py        # Тест базы схожести
│   ├── integration_tests/            # Интеграционные тесты
│   │   ├── test_forward_pipeline.py  # Тест пайплайна Forward VAE
│   │   └── test_inverse_pipeline.py  # Тест пайплайна Inverse VAE
│   └── test_more_integration/             # Расширенные интеграционные тесты
├── logs/                             # Логирование
│   ├── training_logs/                # Логи обучения
│   └── inference_logs/               # Логи вывода
├── deployment/                       # Развертывание
│   ├── api/                          # Новая директория для API
│   │   ├── app.py                    # Flask/FastAPI приложение
│   │   └── requirements.txt          # Зависимости для API
│   ├── docker/                       # Dockerfiles и связанные скрипты
│   │   ├── Dockerfile                # Dockerfile для контейнеризации
│   │   └── requirements.txt          # Зависимости Python для Docker
│   └── kubernetes/                   # Конфигурации для Kubernetes
│       ├── deployment.yaml           # Конфигурация развертывания Kubernetes
│       └── service.yaml              # Конфигурация сервиса Kubernetes
├── monitoring/                       # Новая директория для мониторинга
│   ├── model_monitoring.py           # Мониторинг производительности модели
│   └── data_monitoring.py            # Мониторинг качества данных
├── docs/                             # Документация
│   ├── architecture.md               # Обзор архитектуры
│   ├── api_reference.md              # Справочник API
│   ├── user_guide.md                 # Руководство пользователя
│   └── examples/                     # Примеры использования
│       ├── example_forward_vae.ipynb # Пример работы Forward VAE
│       └── example_inverse_vae.ipynb # Пример работы Inverse VAE
├── data_augmentation/                 # Новая директория для генерации данных
│   ├── augment_data.py               # Генерация дополнительных данных
│   └── simulate_reactions.py         # Симуляция реакций
├── ensemble_models/                   # Новая директория для ансамблей моделей
│   ├── ensemble_forward.py           # Ансамбль Forward VAE
│   └── ensemble_inverse.py           # Ансамбль Inverse VAE
├── explanation/                       # Новая директория для интерпретации
│   ├── shap_explainer.py             # SHAP для интерпретации
│   └── lime_explainer.py             # LIME для интерпретации
├── requirements.txt                  # Зависимости Python
└── README.md                         # Главная документация 