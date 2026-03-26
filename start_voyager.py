from voyager import Voyager

# =================================================================
# КОНФИГУРАЦИЯ БОТА VOYAGER
# =================================================================

# 1. Провайдер нейросети (задаем "openai")
LLM_PROVIDER = "openai" 

# 2. Модели (теперь используем только семейство GPT-5.4)
ACTION_MODEL = "gpt-5.4-mini"      # Для написания сложного кода (Action Agent)
SKILL_MODEL = "gpt-5.4-mini"       # Для библиотеки навыков (Skill Manager)
CRITIC_MODEL = "gpt-5.4-nano"      # Для быстрой оценки успеха (Critic Agent)
CURRICULUM_MODEL = "gpt-5.4-nano"  # Для планирования задач (Curriculum Agent)
# ACTION_MODEL = "gpt-5-mini"      # Для написания сложного кода (Action Agent)
# SKILL_MODEL = "gpt-5-mini"       # Для библиотеки навыков (Skill Manager)
# CRITIC_MODEL = "gpt-5-nano"      # Для быстрой оценки успеха (Critic Agent)
# CURRICULUM_MODEL = "gpt-5-nano"  # Для планирования задач (Curriculum Agent)

# 3. Настройки API
# Если используете стандартный OpenAI, укажите только ключ.
# Если используете Puter или LLMAPI, укажите их BASE_URL и TOKEN здесь.
OPENAI_API_KEY = "asd"
OPENAI_BASE_URL = None               # Оставьте None для стандартного OpenAI

# 6. Режим выживания (True - как обычный игрок, False - режим обучения с читами)
SURVIVAL_MODE = False

# 7. Настройки Minecraft
MC_PORT = 55375  # Порт из чата игры Minecraft

# 8. Цель бота (оставьте None для автоматического режима)
FIXED_TASK = None
# =================================================================

# Инициализация Voyager
voyager = Voyager(
    mc_port=MC_PORT,
    llm_provider="openai",
    llm_base_url=OPENAI_BASE_URL,
    openai_api_key=OPENAI_API_KEY,
    openai_api_request_timeout=300,
    
    # Режимы
    skill_library_dir="ckpt",    # Сохраняет старые навыки
    curriculum_agent_mode="manual",
    
    # Распределяем модели по агентам
    action_agent_model_name=ACTION_MODEL,
    skill_manager_model_name=SKILL_MODEL,
    critic_agent_model_name=CRITIC_MODEL,
    curriculum_agent_model_name=CURRICULUM_MODEL,
    
    fixed_task=FIXED_TASK,
    max_iterations=1000,
)

# Запуск процесса обучения (lifelong learning)
voyager.learn()
