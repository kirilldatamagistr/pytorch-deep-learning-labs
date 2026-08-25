"""Минимальный агент с явным ReAct-циклом.

Установка:
    pip install -U langchain langchain-ollama langchain-tavily

Перед запуском скачайте и запустите локальную open-source модель:
    ollama pull llama3.2

Для web search задайте ключ:
    export TAVILY_API_KEY="..."

Пример запуска:
    python simple_react_agent.py "Найди официальную документацию Python 3.13 и проверь версию Python в терминале"

Агент может многократно выбирать инструмент, получать наблюдение (observation)
и на его основе принимать следующее решение. Это и есть практическая схема
ReAct: Reason → Act → Observation. В консоль выводятся действия и наблюдения,
но не внутренние рассуждения модели.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


# Ограниченный набор команд делает учебный пример безопаснее: агент не может
# выполнять произвольные команды, менять файлы или запускать вложенную оболочку.
ALLOWED_COMMANDS = {
    "pwd": ["pwd"],
    "ls": ["ls", "-la"],
    "python_version": ["python", "--version"],
}


@tool
def terminal(command_name: str) -> str:
    """Выполнить одну из безопасных диагностических команд терминала.

    Допустимые значения command_name: pwd, ls, python_version.
    """
    command = ALLOWED_COMMANDS.get(command_name)
    if command is None:
        return f"Ошибка: команда {command_name!r} не разрешена."

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return "Ошибка: время выполнения команды истекло."

    output = (completed.stdout + completed.stderr).strip()
    return output or "Команда завершилась без вывода."


def format_observation(result: Any, limit: int = 6_000) -> str:
    """Преобразовать результат инструмента в компактное наблюдение для модели."""
    if isinstance(result, str):
        text = result
    else:
        text = json.dumps(result, ensure_ascii=False, default=str)
    return text[:limit]


def parse_text_tool_calls(content: Any) -> list[dict[str, Any]]:
    """Распознать JSON-вызовы, если модель вернула их обычным текстом.

    Некоторые локальные модели не поддерживают native tool calling стабильно и
    печатают один или несколько объектов вида
    {"name": "python_version", "parameters": {...}} как финальный текст.
    Такие объекты могут быть разделены точкой с запятой.
    """
    if not isinstance(content, str):
        return []

    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", maxsplit=1)[-1].rsplit("```", maxsplit=1)[0].strip()

    calls: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        # Пропускаем обычный текст, пробелы и разделители между JSON-объектами.
        while position < len(text) and text[position] not in "{[":
            position += 1
        if position == len(text):
            break

        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            position += 1
            continue

        values = value if isinstance(value, list) else [value]
        for call in values:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                continue
            arguments = call.get("args", call.get("parameters", {}))
            if isinstance(arguments, dict):
                calls.append({"name": call["name"], "args": arguments})
    return calls


def run_tool(
    tool_name: str, arguments: dict[str, Any], tools_by_name: dict[str, Any]
) -> tuple[str, str]:
    """Выполнить инструмент и обработать распространённые варианты имени команды."""
    # Локальные модели иногда называют команду именем действия, а не именем
    # инструмента terminal. Поддержим оба варианта без расширения прав доступа.
    if tool_name in ALLOWED_COMMANDS:
        selected_tool = tools_by_name["terminal"]
        display_name = f"terminal(command_name={tool_name!r})"
        arguments = {"command_name": tool_name}
    else:
        selected_tool = tools_by_name.get(tool_name)
        display_name = f"{tool_name}({json.dumps(arguments, ensure_ascii=False)})"

        # Модель может передать буквальную безопасную команду вместо её ключа.
        if tool_name == "terminal" and arguments.get("command_name") == "python --version":
            arguments = {"command_name": "python_version"}

    if selected_tool is None:
        return display_name, f"Инструмент {tool_name!r} недоступен."

    try:
        observation = format_observation(selected_tool.invoke(arguments))

        # Старый end_date нередко отсекает все актуальные страницы документации.
        # Если поиск пуст, повторяем его без временных ограничений и возвращаем
        # именно повторный результат модели.
        is_web_search = selected_tool is tools_by_name.get("tavily_search")
        has_date_filter = any(
            key in arguments for key in ("start_date", "end_date", "time_range")
        )
        if is_web_search and has_date_filter and observation.startswith("No search results found"):
            retry_arguments = {
                key: value
                for key, value in arguments.items()
                if key not in {"start_date", "end_date", "time_range"}
            }
            observation = format_observation(selected_tool.invoke(retry_arguments))
    except Exception as error:  # Демонстрация передачи ошибки модели.
        observation = f"Ошибка инструмента: {type(error).__name__}: {error}"
    return display_name, observation


def finalize_answer(model_name: str, messages: list[Any]) -> str:
    """Получить финальный ответ от модели без доступа к инструментам."""
    final_model = ChatOllama(model=model_name, temperature=0)
    final_messages = [
        *messages,
        HumanMessage(
            content=(
                "На основе всех уже полученных наблюдений дай финальный ответ "
                "на исходный вопрос на русском. Не вызывай инструменты и не "
                "выводи JSON действия."
            )
        ),
    ]
    return str(final_model.invoke(final_messages).content)


def run_react_agent(
    question: str, model_name: str = "llama3.2", max_steps: int = 6
) -> str:
    """Запустить ReAct-цикл и вернуть финальный ответ агента."""
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Не задана переменная окружения TAVILY_API_KEY.")

    # TavilySearch — готовый LangChain-инструмент для поиска в интернете.
    web_search = TavilySearch(max_results=3, topic="general")
    tools = [terminal, web_search]
    tools_by_name = {item.name: item for item in tools}
    # Имена-псевдонимы покрывают частый текстовый формат локальных моделей.
    tools_by_name["tavily_search"] = web_search
    tools_by_name["web_search"] = web_search

    # ChatOllama обращается к локальному серверу Ollama. По умолчанию это
    # http://localhost:11434, поэтому ключ к облачному LLM-провайдеру не нужен.
    model = ChatOllama(model=model_name, temperature=0).bind_tools(tools)
    messages = [
        SystemMessage(
            content=(
                "Ты учебный агент. Используй web search для актуальных сведений, "
                "а terminal — только для локальной диагностики. Если источники из "
                "поиска важны для ответа, укажи ссылки. Не выдумывай наблюдения. "
                "Для документации и общих вопросов не задавай end_date или другие "
                "временные фильтры, если пользователь не попросил ограничить дату. "
                "Когда данных достаточно, дай краткий финальный ответ на русском. "
                "Вызывай инструменты нативно. Если это недоступно, верни только "
                "JSON вида {\"name\": \"terminal\", \"parameters\": {...}}; "
                "не считай такой JSON финальным ответом."
            )
        ),
        HumanMessage(content=question),
    ]

    # Один проход цикла: модель выбирает действие, инструмент возвращает
    # observation, затем это observation добавляется в историю сообщений.
    executed_calls: set[str] = set()
    for step in range(1, max_steps + 1):
        response = model.invoke(messages)
        messages.append(response)
        native_tool_calls = response.tool_calls
        text_tool_calls = parse_text_tool_calls(response.content)

        if not native_tool_calls and not text_tool_calls:
            print(f"Нет инструментов для вызова, агент сформировал финальный ответ.")
            return str(response.content)

        print(f"\n[Шаг {step}: действие]")
        calls = native_tool_calls or text_tool_calls
        for call in calls:
            tool_name = call["name"]
            arguments = call["args"]
            call_signature = json.dumps(
                {"name": tool_name, "args": arguments},
                ensure_ascii=False,
                sort_keys=True,
            )

            # Один и тот же инструмент с теми же аргументами не даст нового
            # observation. Вместо повторения завершаем задачу по уже собранным
            # данным, используя модель без tools.
            if call_signature in executed_calls:
                print("Повторное действие обнаружено; завершаем задачу.")
                print(call_signature)
                return finalize_answer(model_name, messages)
            executed_calls.add(call_signature)

            display_name, observation = run_tool(tool_name, arguments, tools_by_name)
            print(display_name)

            print(f"[Наблюдение]\n{observation}\n")
            if native_tool_calls:
                messages.append(
                    ToolMessage(content=observation, tool_call_id=call["id"])
                )
            else:
                # Для текстового JSON не создаём ToolMessage: у него нет
                # корректного tool_call_id. Observation передаётся как реплика
                # пользователя, чтобы локальная модель продолжила ReAct-цикл.
                messages.append(HumanMessage(content=f"Наблюдение: {observation}"))

    return finalize_answer(model_name, messages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Учебный ReAct-агент LangChain")
    parser.add_argument(
        "question",
        nargs="?",
        default="Проверь версию Python в терминале. Найди актуальную официальную документацию Python, верни ссылку и убедись что она рабочая.",
        help="Вопрос или задача для агента.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Название локальной модели, загруженной в Ollama.",
    )
    args = parser.parse_args()

    answer = run_react_agent(args.question, model_name=args.model)
    print(f"\n[Финальный ответ]\n{answer}")


if __name__ == "__main__":
    main()
