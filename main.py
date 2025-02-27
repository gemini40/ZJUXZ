import re
import os
import json
import time
import copy
import uuid
import gzip
import asyncio
import websockets
import subprocess
import threading
import sqlite3

from flask import Flask, request, redirect, url_for, render_template_string, jsonify, Response, send_file
from openai import OpenAI

# 配置部分
MC_BASE_URL_1 = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MC_API_KEY_1 = "sk-0cf4070025924d03a4b505e892f3bf9f"
MC_MODEL_NAME_1 = "deepseek-r1"

MC_BASE_URL_2 = "https://api.siliconflow.cn/v1"
MC_API_KEY_2 = "sk-qumbdboezpbyzdjqskflttyrkjbjvuqaihqmkrcgniiqusag"
MC_MODEL_NAME_2 = "Pro/deepseek-ai/DeepSeek-R1"

CHECK_BASE_URL = "https://api.siliconflow.cn/v1"
CHECK_API_KEY = "sk-qumbdboezpbyzdjqskflttyrkjbjvuqaihqmkrcgniiqusag"
CHECK_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-128K"

client_mc1 = OpenAI(api_key=MC_API_KEY_1, base_url=MC_BASE_URL_1)
client_mc2 = OpenAI(api_key=MC_API_KEY_2, base_url=MC_BASE_URL_2)
client_check = OpenAI(api_key=CHECK_API_KEY, base_url=CHECK_BASE_URL)

# TTS 配置
appid = "6959536185"
token = "pCff9XSFyC2PZxQvq86zORVQ8jmnzGpA"
cluster = "volcano_tts"
voice_type = "zh_female_linjianvhai_moon_bigtts"
host = "openspeech.bytedance.com"
api_url = f"wss://{host}/api/v1/tts/ws_binary"

default_header = bytearray(b"\x11\x10\x11\x00")

request_json = {
    "app" : {"appid" : appid, "token" : "access_token", "cluster" : cluster},
    "user" : {"uid" : "2103133273"},
    "audio" : {
        "voice_type" : voice_type,
        "encoding" : "mp3",
        "speed_ratio" : 1.0,
        "volume_ratio" : 1.0,
        "pitch_ratio" : 1.0,
    },
    "request" : {
        "reqid" : "",  # 每次动态生成
        "text" : "",  # 待合成文本
        "text_type" : "plain",
        "operation" : "submit",
    },
}

DATABASE = 'database.db'  # 数据库文件路径


# 2. 数据库相关的基础函数
def init_db():
    """初始化数据库和创建表结构"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # 创建表
        cursor.executescript('''
            CREATE TABLE IF NOT EXISTS courses (
                name TEXT PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                course_name TEXT,
                FOREIGN KEY (course_name) REFERENCES courses (name)
            );

            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                chapter_id INTEGER,
                FOREIGN KEY (chapter_id) REFERENCES chapters (id)
            );

            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                type TEXT,
                content TEXT,
                options TEXT,
                answer TEXT,
                explanation TEXT,
                checked INTEGER DEFAULT 0,
                tts_file TEXT,
                section_id INTEGER,
                FOREIGN KEY (section_id) REFERENCES sections (id)
            );
        ''')

        # 验证表是否创建成功
        tables = ['courses', 'chapters', 'sections', 'knowledge']
        for table in tables:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            if cursor.fetchone() is None:
                raise Exception(f"表 {table} 创建失败")

        conn.commit()
        print("数据库表结构创建成功！")
        return True

    except sqlite3.Error as e:
        print(f"SQLite 错误: {e}")
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        print(f"初始化数据库时出错: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# 3. 数据操作函数
def add_course(course_name):
    """添加新课程"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO courses (name) VALUES (?)', (course_name,))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        print(f"课程 '{course_name}' 已存在")
        return False
    finally:
        conn.close()

def add_chapter(chapter_name, course_name):
    """添加新章节"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO chapters (name, course_name) VALUES (?, ?)',
            (chapter_name, course_name)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def add_section(section_name, chapter_id):
    """添加新小节"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO sections (name, chapter_id) VALUES (?, ?)',
            (section_name, chapter_id)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def add_knowledge(knowledge_data, section_id):
    """添加新知识点"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO knowledge 
            (id, type, content, options, answer, explanation, checked, tts_file, section_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            knowledge_data['id'],
            knowledge_data['type'],
            knowledge_data['content'],
            knowledge_data.get('options', ''),
            knowledge_data.get('answer', ''),
            knowledge_data.get('explanation', ''),
            1 if knowledge_data.get('checked', False) else 0,
            knowledge_data.get('tts_file'),
            section_id
        ))
        conn.commit()
        return True
    finally:
        conn.close()



def parse_response(res, file) :
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0F
    header_size = res[0] & 0x0F
    payload = res[header_size * 4 :]

    if message_type == 0xB :  # audio-only server response
        if message_type_specific_flags == 0 :  # no sequence number as ACK
            return False
        else :
            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
            payload_size = int.from_bytes(payload[4 :8], "big", signed=False)
            payload = payload[8 :]
            file.write(payload)
            if sequence_number < 0 :
                return True
            else :
                return False
    elif message_type == 0xF :
        return True
    elif message_type == 0xC :
        return False
    else :
        return True


async def run_tts(text, out_path) :
    submit_req = copy.deepcopy(request_json)
    submit_req["request"]["text"] = text
    submit_req["request"]["reqid"] = str(uuid.uuid4())
    submit_req["request"]["operation"] = "submit"

    payload_bytes = json.dumps(submit_req, ensure_ascii=False).encode("utf-8")
    payload_bytes = gzip.compress(payload_bytes)

    full_client_request = bytearray(default_header)
    full_client_request.extend((len(payload_bytes)).to_bytes(4, "big"))
    full_client_request.extend(payload_bytes)

    with open(out_path, "wb") as fout :
        header = {"Authorization" : f"Bearer; {token}"}
        async with websockets.connect(
                api_url, extra_headers=header, ping_interval=None
        ) as ws :
            await ws.send(full_client_request)
            while True :
                res = await ws.recv()
                done = parse_response(res, fout)
                if done :
                    break
    return True


def call_api_with_retry(client, model_name, messages, max_retries=5) :
    last_exception = None
    for attempt in range(max_retries) :
        try :
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.5,
                timeout=120,
                stream=False,
            )

            if not response or not hasattr(response, "choices") or not response.choices :
                time.sleep(3)
                continue

            content_str = response.choices[0].message.content.strip()
            cleaned_content = re.sub(
                r"^```json\s*|\s*```$",
                "",
                content_str,
                flags=re.IGNORECASE | re.MULTILINE,
            ).strip()

            try :
                parsed_data = json.loads(cleaned_content)
                if isinstance(parsed_data, list) :
                    for item in parsed_data :
                        if not all(
                                k in item
                                for k in ["question", "options", "answer", "explanation"]
                        ) :
                            raise ValueError("选择题数据结构不完整")
                elif isinstance(parsed_data, dict) :
                    if not all(
                            k in parsed_data
                            for k in ["question", "options", "answer", "explanation"]
                    ) :
                        raise ValueError("选择题数据结构不完整")
                else :
                    raise ValueError("无效的数据类型")

                return parsed_data

            except json.JSONDecodeError as e :
                print(
                    f"[call_api_with_retry] 第 {attempt + 1} 次尝试: JSON 解析失败: {e}"
                )
                print("原始内容:", content_str)
                time.sleep(3)
                continue

            except ValueError as e :
                print(
                    f"[call_api_with_retry] 第 {attempt + 1} 次尝试: 数据验证失败: {e}"
                )
                time.sleep(3)
                continue

        except Exception as e :
            print(f"[call_api_with_retry] 第 {attempt + 1} 次尝试失败: {e}")
            last_exception = e
            time.sleep(3)

    print(f"[call_api_with_retry] 所有 {max_retries} 次尝试均失败")
    return None


def call_mc_api_two_stage(messages) :
    result = call_api_with_retry(client_mc1, MC_MODEL_NAME_1, messages, max_retries=5)
    if result is not None :
        return result
    result = call_api_with_retry(client_mc2, MC_MODEL_NAME_2, messages, max_retries=3)
    return result


def call_check_api_with_retry(client, messages, max_retries=10) :
    for attempt in range(max_retries) :
        try :
            response = client.chat.completions.create(
                model=CHECK_MODEL_NAME,
                messages=messages,
                temperature=0.7,
                timeout=120,
                stream=False,
            )
            content_str = response.choices[0].message.content.strip()
            print("纠错API返回内容:", content_str)

            json_match = re.search(
                r"```(?:json)?\s*({.*?})\s*```", content_str, re.DOTALL
            )
            if json_match :
                content_str = json_match.group(1).strip()
            else :
                content_str = content_str.strip("``` ").strip()

            parsed_data = json.loads(content_str)
            if "corrected" in parsed_data :
                return {"original" : parsed_data["corrected"]}
            else :
                return parsed_data

        except Exception as e :
            print(f"[call_check_api_with_retry] 第 {attempt + 1} 次失败, 错误: {e}")
            time.sleep(2 if attempt == 0 else 5)
    print("[call_check_api_with_retry] 多次重试后依然失败")
    return None


app = Flask(__name__)


def load_data() :
    conn = get_db_connection()
    cursor = conn.cursor()
    try :
        # 加载课程
        cursor.execute('SELECT * FROM courses')
        courses = {course['name'] : {'chapters' : []} for course in cursor.fetchall()}

        # 加载章节
        cursor.execute('''
            SELECT ch.*, c.name as course_name 
            FROM chapters ch
            JOIN courses c ON ch.course_name = c.name
        ''')
        chapters = cursor.fetchall()
        for chapter in chapters :
            if chapter['course_name'] in courses :
                courses[chapter['course_name']]['chapters'].append({
                    'name' : chapter['name'],
                    'sections' : []
                })

        # 加载小节
        cursor.execute('''
            SELECT s.*, ch.course_name, ch.name as chapter_name
            FROM sections s
            JOIN chapters ch ON s.chapter_id = ch.id
        ''')
        sections = cursor.fetchall()
        for section in sections :
            if section['course_name'] in courses :
                for chapter in courses[section['course_name']]['chapters'] :
                    if chapter['name'] == section['chapter_name'] :
                        chapter['sections'].append({
                            'name' : section['name'],
                            'knowledge' : []
                        })

        # 加载知识点
        cursor.execute('''
            SELECT k.*, s.name as section_name, ch.name as chapter_name, ch.course_name
            FROM knowledge k
            JOIN sections s ON k.section_id = s.id
            JOIN chapters ch ON s.chapter_id = ch.id
        ''')
        knowledge = cursor.fetchall()
        knowledge_index = {}

        for item in knowledge :
            knowledge_data = dict(item)
            knowledge_index[item['id']] = knowledge_data

            if item['course_name'] in courses :
                for chapter in courses[item['course_name']]['chapters'] :
                    if chapter['name'] == item['chapter_name'] :
                        for section in chapter['sections'] :
                            if section['name'] == item['section_name'] :
                                section['knowledge'].append({
                                    'id' : item['id'],
                                    'type' : item['type'],
                                    'content' : item['content'],
                                    'options' : item['options'],
                                    'answer' : item['answer'],
                                    'explanation' : item['explanation'],
                                    'checked' : bool(item['checked']),
                                    'tts_file' : item['tts_file']
                                })

        return {'courses' : courses, 'index' : knowledge_index}
    finally :
        conn.close()


def save_data(data) :
    conn = get_db_connection()
    cursor = conn.cursor()
    try :
        cursor.execute('BEGIN TRANSACTION')

        # 清空所有表
        cursor.execute('DELETE FROM knowledge')
        cursor.execute('DELETE FROM sections')
        cursor.execute('DELETE FROM chapters')
        cursor.execute('DELETE FROM courses')

        # 保存数据
        for course_name, course_data in data['courses'].items() :
            cursor.execute('INSERT INTO courses (name) VALUES (?)', (course_name,))

            for chapter in course_data['chapters'] :
                cursor.execute(
                    'INSERT INTO chapters (name, course_name) VALUES (?, ?)',
                    (chapter['name'], course_name)
                )
                chapter_id = cursor.lastrowid

                for section in chapter['sections'] :
                    cursor.execute(
                        'INSERT INTO sections (name, chapter_id) VALUES (?, ?)',
                        (section['name'], chapter_id)
                    )
                    section_id = cursor.lastrowid

                    for knowledge in section['knowledge'] :
                        cursor.execute('''
                            INSERT INTO knowledge 
                            (id, type, content, options, answer, explanation, checked, tts_file, section_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            knowledge['id'],
                            knowledge['type'],
                            knowledge['content'],
                            knowledge.get('options', ''),
                            knowledge.get('answer', ''),
                            knowledge.get('explanation', ''),
                            1 if knowledge.get('checked', False) else 0,
                            knowledge.get('tts_file'),
                            section_id
                        ))

        conn.commit()
        print("数据保存成功!")
    except Exception as e :
        conn.rollback()
        print(f"保存数据时出错: {e}")
        raise
    finally :
        conn.close()

# 读取和保存数据示例
data = load_data()
save_data(data)

BASE_HTML_HEAD = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>浙大西溪学长-同等学力申硕知识管理</title>
<link rel="stylesheet" href="{{ url_for('static', filename='css/bootstrap.min.css') }}">
<script src="{{ url_for('static', filename='js/bootstrap.bundle.min.js') }}"></script>
    <style>
        body {
         margin-top: 40px;
        }

        .btn-custom {
            background-color: #007bff;
            border-color: #007bff;
            color: #fff;
        }
        .btn-custom:hover,
        .btn-custom:focus,
        .btn-custom:active {
            background-color: #007bff !important;
            border-color: #007bff !important;
            color: #fff !important;
            opacity: 0.8;
        }
        .orange-text {
            color: #373b3e;
            font-size: smaller;
        }
        .float-btn-container {
            position: fixed;
            right: 20px;
            bottom: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 999;
        }
        .block-container {
            margin-top: 20px;
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 5px;
        }
        .bg-mc { background-color: #fffdf7; }
        .bg-def { background-color: #fff7f7; }
        .bg-qa { background-color: #f3fff9; }
    </style>
    <script>
    function confirmDelete() {
        var answer = prompt("警告：是否决定执行删除？建议改名处理。(输入Y或N)", "N");
        if (!answer) return false;
        if (answer.toUpperCase() !== 'Y') return false;
        return true;
    }
    </script>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-dark bg-dark" style="position: fixed; top: 0; width: 100%; z-index: 9999;">
  <div class="container-fluid">
    <a class="navbar-brand" href="#">浙大西溪学长 - 专业学科题库管理系统</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse"
        data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent"
        aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarSupportedContent">
      <ul class="navbar-nav ms-auto mb-2 mb-lg-0">
        <li class="nav-item"><a class="nav-link" href="/">首页</a></li>
        <li class="nav-item"><a class="nav-link" href="/manage">维护目录</a></li>
        <li class="nav-item"><a class="nav-link" href="/add">添加知识点</a></li>
      </ul>
    </div>
  </div>
</nav>
<div class="container mt-4">

<!-- 在 BASE_HTML_HEAD 部分添加弹窗的 HTML 结构 -->
<div class="modal fade" id="editModal" tabindex="-1" aria-labelledby="editModalLabel" aria-hidden="true">
  <div class="modal-dialog modal-lg">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="editModalLabel">修改知识点</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <form id="editKnowledgeForm" class="row g-3">
          <input type="hidden" name="kid" id="editKid">
          <div class="col-md-4">
            <label for="editCourse" class="form-label">选择课程</label>
            <select class="form-select" name="course" id="editCourse" required>
              {% for course_name, cdata in courses.items() %}
              <option value="{{ course_name }}">{{ course_name }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="col-md-4">
            <label for="editChapter" class="form-label">选择章</label>
            <select class="form-select" name="chapter" id="editChapter" required>
              {% for course_name, cdata in courses.items() %}
                {% for ch in cdata["chapters"] %}
                <option value="{{ ch['name'] }}">{{ course_name }} -> {{ ch['name'] }}</option>
                {% endfor %}
              {% endfor %}
            </select>
          </div>
          <div class="col-md-4">
            <label for="editSection" class="form-label">选择节</label>
            <select class="form-select" name="section" id="editSection" required>
              {% for course_name, cdata in courses.items() %}
                {% for ch in cdata["chapters"] %}
                  {% for sec in ch["sections"] %}
                  <option value="{{ sec['name'] }}">
                    {{ course_name }}->{{ ch['name'] }}->{{ sec['name'] }}
                  </option>
                  {% endfor %}
                {% endfor %}
              {% endfor %}
            </select>
          </div>
          <div class="col-md-12">
            <div class="field-group multiple_choice border p-3 mb-3" style="display:block">
              <label class="form-label">原始内容</label>
              <textarea class="form-control" name="statement" id="editStatement" rows="8"></textarea>
            </div>
            <div class="field-group definition border p-3 mb-3" style="display:none">
              <label class="form-label">术语名称</label>
              <input type="text" class="form-control" name="term" id="editTerm">
              <label class="form-label mt-2">详细解释</label>
              <textarea class="form-control" name="explanation" id="editExplanation" rows="8"></textarea>
            </div>
            <div class="field-group qa border p-3 mb-3" style="display:none">
              <label class="form-label">问题内容</label>
              <input type="text" class="form-control" name="question" id="editQuestion">
              <label class="form-label mt-2">答案</label>
              <textarea class="form-control" name="answer" id="editAnswer" rows="8"></textarea>
            </div>
          </div>
        </form>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
        <button type="button" class="btn btn-primary" onclick="saveEdit()">保存</button>
      </div>
    </div>
  </div>
</div>

"""

BASE_HTML_FOOT = """
</div>
<script src="{{ url_for('static', filename='js/bootstrap.bundle.min.js') }}"></script>
</body>
</html>
"""

INDEX_HTML = (
    BASE_HTML_HEAD
    + """
<div class="row">
    <!-- 左侧导航 -->
    <div class="col-md-2" style="position:fixed;left:0;top:56px;bottom:0;overflow-y:auto;background-color:#f8f9fa;">
        <div class="p-2">
            <h6 class="mt-2">快速导航栏</h6>
            <hr>
            <ul class="nav flex-column small">
                {% for c_name, c_data in courses.items() %}
                    <li class="nav-item mb-1">
                        <strong>{{ c_name }}</strong>
                        <ul class="ms-0">
                            {% for ch in c_data["chapters"] %}
                                <li class="my-0">
                                    <span>{{ ch.name }}</span>
                                    <ul class="ms-0">
                                        {% for sec in ch["sections"] %}
                                            <li>
                                                <a href="/?course={{ c_name }}&chapter={{ ch.name }}&section={{ sec.name }}">
                                                    {{ sec.name }}
                                                </a>
                                            </li>
                                        {% endfor %}
                                    </ul>
                                </li>
                            {% endfor %}
                        </ul>
                    </li>
                {% endfor %}
            </ul>
            <hr>
            <form action="{{ url_for('index') }}" method="get" class="mb-3">
                <div class="mb-2">
                    <label for="courseSelect" class="form-label">课程</label>
                    <select class="form-select" name="course" id="courseSelect">
                        <option value="">所有课程</option>
                        {% for course_name in courses.keys() %}
                            <option value="{{ course_name }}" {% if course_name == selected_course %} selected {% endif %}>
                                {{ course_name }}
                            </option>
                        {% endfor %}
                    </select>
                </div>
                <div class="mb-2">
                    <label for="chapterSelect" class="form-label">章</label>
                    <select class="form-select" name="chapter" id="chapterSelect">
                        <option value="">所有章</option>
                        {% for chapter in chapters %}
                            <option value="{{ chapter }}" {% if chapter == selected_chapter %} selected {% endif %}>
                                {{ chapter }}
                            </option>
                        {% endfor %}
                    </select>
                </div>
                <div class="mb-2">
                    <label for="sectionSelect" class="form-label">节</label>
                    <select class="form-select" name="section" id="sectionSelect">
                        <option value="">所有节</option>
                        {% for section in sections %}
                            <option value="{{ section }}" {% if section == selected_section %} selected {% endif %}>
                                {{ section }}
                            </option>
                        {% endfor %}
                    </select>
                </div>
                <div class="mb-2">
                    <label class="form-label">类型</label><br>
                    <input type="radio" class="form-check-input" name="ktype" value="multiple_choice"
                        {% if selected_type == 'multiple_choice' %}checked{% endif %}> 选择
                    <input type="radio" class="form-check-input" name="ktype" value="definition"
                        {% if selected_type == 'definition' %}checked{% endif %}> 名解
                    <input type="radio" class="form-check-input" name="ktype" value="qa"
                        {% if selected_type == 'qa' %}checked{% endif %}> 问答
                    <input type="radio" class="form-check-input" name="ktype" value="" id="typeAll" onclick="activateAll()"> 全部
                </div>
                <div class="mb-2">
                    <label for="keywordInput" class="form-label">关键词</label>
                    <input type="text" class="form-control" id="keywordInput" name="keyword" placeholder="关键字..." value="{{ keyword|default('') }}">
                </div>
                <button type="submit" class="btn btn-custom w-100">搜索</button>
            </form>
        </div>
    </div>

    <!-- 右侧内容 -->
    <div class="col-md-10 offset-md-2" style="margin-top: 62px;">
        <h1 class="mb-4">浙大西溪学长-同等学力申硕学科综合题库系统</h1>
        <div class="mb-3">
            <button class="btn btn-success me-3" onclick="startBulkCheck()">1. 一键批量检查所有的文本字段</button>
            <button class="btn btn-warning" onclick="startBulkConvert()">2. 一键批量生成所有的选择题</button>
        </div>
        <hr>

        {% if results is not none %}
            {% if not results %}
                <p>没有匹配的知识点。</p>
            {% else %}
                {% if multiple_choices %}
                <div class="block-container bg-mc">
                    <h4>【选择题】</h4>
                    <ul class="list-group">
                    {% for item in multiple_choices %}
                        {% set i = loop.index %}
                        {% set checked = item['content'].get('checked', False) %}
                        {% set tts_file = item['content'].get('tts_file') %}
                        <li class="list-group-item">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <strong>{{ i }}.</strong>
                                    <span id="toggleText-mc-{{ i }}"
                                          style="cursor:pointer;color:#0d6efd;margin-left:8px;"
                                          onclick="toggleContent('mc','{{ i }}')">▷ 展开</span>
                                </div>
                                <div>
                                    {% if item['content']['multiple_choices'] and item['content']['multiple_choices']|length > 0 %}
                                        <!-- 已有 multiple_choices 则禁用按钮 -->
                                        <button type="button" class="btn btn-sm btn-secondary me-2" disabled>转为选择题</button>
                                    {% else %}
                                        <!-- 改为 Ajax 提交 -->
                                        <form action="{{ url_for('convert_to_mc') }}" method="post"
                                              style="display:inline-block; margin-right:4px;"
                                              onsubmit="convertToMcAjax(event, '{{ item['id'] }}','{{ i }}')">
                                            <input type="hidden" name="knowledge_id" value="{{ item['id'] }}">
                                            <button type="submit" class="btn btn-sm btn-warning">转为选择题</button>
                                        </form>
                                    {% endif %}

                                    <form action="{{ url_for('delete_knowledge') }}" method="post"
                                          style="display:inline-block; margin-right:4px;"
                                          onsubmit="return confirmDelete();">
                                        <input type="hidden" name="kid" value="{{ item['id'] }}">
                                        <button type="submit" class="btn btn-danger btn-sm">删除</button>
                                    </form>

                                    <a href="#" class="btn btn-primary btn-sm" onclick="openEditModal('{{ item['id'] }}')" style="margin-right:4px;">修改</a>

                                   {% if not item['content']['checked'] %}
                                                 <button type="button" class="btn btn-success btn-sm"
                                                  id="checkButton-mc-{{ i }}"
                                                 onclick="checkTextAjax('{{ item['id'] }}', 'mc', '{{ i }}')">
                                                                    检查
                                                    </button>
                                                    {% else %}
                                                     <button type="button" class="btn btn-secondary btn-sm" disabled>检查</button>
                                                    {% endif %}

                                    <!-- TTS/收听/下载 操作区 -->
                                    <div id="ttsWrapper-mc-{{ i }}"
                                         style="display:inline-block; float: right; margin-left:8px; margin-right:2px;">
                                        {% if not checked %}
                                            <button type="button" class="btn btn-sm btn-secondary" disabled>转录</button>
                                            <button type="button" class="btn btn-sm btn-secondary" disabled>收听</button>
                                            <button type="button" class="btn btn-sm btn-secondary" disabled>下载</button>
                                        {% else %}
                                            {% if tts_file %}
                                                <!-- 已转录 -->
                                                <button type="button" class="btn btn-sm btn-secondary" disabled>已转录</button>
                                                <button type="button"
                                                        class="btn btn-sm"
                                                        style="background-color:#b8860b;color:white;"
                                                        onclick="listenTTS('{{ item['id'] }}')">
                                                    收听
                                                </button>
                                                <button type="button"
                                                        class="btn btn-sm"
                                                        style="background-color:#ff1493;color:white;"
                                                        onclick="downloadTTS('{{ item['id'] }}')">
                                                    下载
                                                </button>
                                            {% else %}
                                                <button type="button"
                                                        class="btn btn-sm"
                                                        style="background-color:purple;color:white;"
                                                        id="ttsSpeakButton-{{ i }}"
                                                        onclick="ttsSpeak('{{ item['id'] }}','{{ i }}','mc')">
                                                    转录
                                                </button>
                                                <button type="button" class="btn btn-sm btn-secondary" disabled>收听</button>
                                                <button type="button" class="btn btn-sm btn-secondary" disabled>下载</button>
                                            {% endif %}
                                        {% endif %}
                                    </div>
                                </div>
                            </div>
                            <div class="orange-text mt-2">
                                当前所属：▶ {{ item['course'] }} ▶ {{ item['chapter'] }} ▶ {{ item['section'] }}
                            </div>
                            {% set prefix = "<strong>知识点：</strong>" %}
                            {% set full_text = item['content']['original'] %}
                            {% if full_text|length > 125 %}
                                {% set snippet_text = prefix ~ full_text[:186] ~ "..." %}
                            {% else %}
                                {% set snippet_text = prefix ~ full_text %}
                            {% endif %}
                            <div style="margin-top:10px;">
                                <span id="snippet-mc-{{ i }}">{{ snippet_text|safe }}</span>
                            </div>
                            <div id="fullContent-mc-{{ i }}" style="display:none;margin-top:10px;">
                                <div>
                                    <strong>知识点：</strong>
                                    <span id="mc-original-{{ i }}">{{ item['content']['original'] }}</span>
                                </div>
                               {% if item['content']['multiple_choices'] %}
                                   <hr>
                                   <p><strong>已生成的单选题列表：</strong></p>
                                   <ul id="question-list-container">
                                       {% for q_obj in item['content']['multiple_choices'] %}
                                       <li class="mb-3" id="question-{{ q_obj.id }}">
                                           <strong>题目 {{ loop.index }}: </strong> {{ q_obj.question }}
                                           <button class="btn btn-warning btn-sm delete-btn" 
                                                   data-id="{{ q_obj.id }}" 
                                                   onclick="deleteQuestion(this)">删除</button>
                                           <ul>
                                               <li>A. {{ q_obj.options.A }}</li>
                                               <li>B. {{ q_obj.options.B }}</li>
                                               <li>C. {{ q_obj.options.C }}</li>
                                               <li>D. {{ q_obj.options.D }}</li>
                                           </ul>
                                           <div><strong>答案:</strong> {{ q_obj.answer }}</div>
                                           <div><strong>解析:</strong> {{ q_obj.explanation }}</div>
                                       </li>
                                       {% if not loop.last %}
                                       <div style="text-align:left;color:#b1a17057;">
                                           --------------------------------------------
                                       </div>
                                       {% endif %}
                                       {% endfor %}
                                   </ul>
                               {% endif %}
                                {% if item['content'].get('last_error') %}
                                    <div class="text-danger">
                                        <strong>错误:</strong> {{ item['content']['last_error'] }}
                                    </div>
                                {% endif %}
                            </div>
                        </li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}

                {% if definitions %}
                <div class="block-container bg-def">
                    <h4>【名词解释】</h4>
                    <ul class="list-group">
                    {% for item in definitions %}
                        {% set i = loop.index %}
                        {% set checked = item['content'].get('checked', False) %}
                        {% set tts_file = item['content'].get('tts_file') %}
                        <li class="list-group-item">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <strong>{{ i }}.</strong>
                                    <span id="toggleText-def-{{ i }}"
                                          style="cursor:pointer;color:#0d6efd;margin-left:8px;"
                                          onclick="toggleContent('def','{{ i }}')">▷ 展开</span>
                                </div>
                                <div>
                                    <form action="{{ url_for('delete_knowledge') }}" method="post"
                                          style="display:inline-block; margin-right:4px;"
                                          onsubmit="return confirmDelete();">
                                        <input type="hidden" name="kid" value="{{ item['id'] }}">
                                        <button type="submit" class="btn btn-danger btn-sm">删除</button>
                                    </form>
                                    <a href="#" class="btn btn-primary btn-sm" onclick="openEditModal('{{ item['id'] }}')" style="margin-right:4px;">修改</a>

                                   {% if not item['content']['checked'] %}
                                                 <button type="button" class="btn btn-success btn-sm"
                                                  id="checkButton-def-{{ i }}"
                                                 onclick="checkTextAjax('{{ item['id'] }}', 'def', '{{ i }}')">
                                                                    检查
                                                    </button>
                                                    {% else %}
                                                     <button type="button" class="btn btn-secondary btn-sm" disabled>检查</button>
                                                    {% endif %}

                                    <!-- TTS/收听/下载 操作区 -->
                                    <div id="ttsWrapper-def-{{ i }}"
                                         style="display:inline-block; float: right; margin-left:8px; margin-right:2px;">
                                        {% if not checked %}
                                            <button type="button" class="btn btn-sm btn-secondary" disabled>转录</button>
                                            <button type="button" class="btn btn-sm btn-secondary" disabled>收听</button>
                                            <button type="button" class="btn btn-sm btn-secondary" disabled>下载</button>
                                        {% else %}
                                            {% if tts_file %}
                                                <button type="button"
                                                        class="btn btn-sm btn-secondary"
                                                        disabled>已转录
                                                </button>
                                                <button type="button"
                                                        class="btn btn-sm"
                                                        style="background-color:#b8860b;color:white;"
                                                        onclick="listenTTS('{{ item['id'] }}')">
                                                    收听
                                                </button>
                                                <button type="button"
                                                        class="btn btn-sm"
                                                        style="background-color:#ff1493;color:white;"
                                                        onclick="downloadTTS('{{ item['id'] }}')">
                                                    下载
                                                </button>
                                            {% else %}
                                                <button type="button"
                                                        class="btn btn-sm"
                                                        style="background-color:purple;color:white;"
                                                        id="ttsSpeakButton-{{ i }}"
                                                        onclick="ttsSpeak('{{ item['id'] }}','{{ i }}','def')">
                                                    转录
                                                </button>
                                                <button type="button" class="btn btn-sm btn-secondary" disabled>收听</button>
                                                <button type="button" class="btn btn-sm btn-secondary" disabled>下载</button>
                                            {% endif %}
                                        {% endif %}
                                    </div>
                                </div>
                            </div>
                            <div class="orange-text mt-2">
                                当前所属：▶ {{ item['course'] }} ▶ {{ item['chapter'] }} ▶ {{ item['section'] }}
                            </div>
                            {% set prefix_question = "<strong>请解释：</strong>" %}
                            {% set prefix_answer = "<strong>答：</strong>" %}
                            {% set full_text = prefix_question ~ item['term'] ~ " | " ~ prefix_answer ~ item['explanation'] %}
                            {% if full_text|length > 125 %}
                                {% set snippet_text = full_text[:183] ~ "..." %}
                            {% else %}
                                {% set snippet_text = full_text %}
                            {% endif %}
                            <div style="margin-top:10px;">
                                <span id="snippet-def-{{ i }}">{{ snippet_text|safe }}</span>
                            </div>
                            <div id="fullContent-def-{{ i }}" style="display:none;margin-top:10px;">
                                <div id="def-term-{{ i }}">
                                    <strong>请解释：</strong>{{ item['term'] }}
                                </div>
                                <div id="def-exp-{{ i }}">
                                    <strong>答：</strong>{{ item['explanation'] }}
                                </div>
                            </div>
                        </li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}

                {% if qa %}
    <div class="block-container bg-qa">
        <h4>【问答题】</h4>
        <ul class="list-group">
            {% for item in qa %}
                {% set i = loop.index %}
                {% set checked = item['content'].get('checked', False) %}
                {% set tts_file = item['content'].get('tts_file') %}
                <li class="list-group-item">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <strong>{{ i }}.</strong>
                            <span id="toggleText-qa-{{ i }}"
                                  style="cursor:pointer;color:#0d6efd;margin-left:8px;"
                                  onclick="toggleContent('qa','{{ i }}')">▷ 展开</span>
                        </div>
                        <div>
                            <form action="{{ url_for('delete_knowledge') }}" method="post"
                                  style="display:inline-block; margin-right:4px;"
                                  onsubmit="return deleteKnowledgeAjax(event, '{{ item['id'] }}')">
                                <input type="hidden" name="kid" value="{{ item['id'] }}">
                                <button type="submit" class="btn btn-danger btn-sm">删除</button>
                            </form>

                            <a href="#" class="btn btn-primary btn-sm" onclick="openEditModal('{{ item['id'] }}')" style="margin-right:4px;">修改</a>


                            {% if not item['content']['checked'] %}
                            <button type="button" class="btn btn-success btn-sm"
                            id="checkButton-qa-{{ i }}"
                            onclick="checkTextAjax('{{ item['id'] }}', 'qa', '{{ i }}')">
                            检查
                            </button>
                            {% else %}
                            <button type="button" class="btn btn-secondary btn-sm" disabled>检查</button>
                            {% endif %}


                            <!-- TTS/收听/下载 操作区 -->
                            <div id="ttsWrapper-qa-{{ i }}"
                                 style="display:inline-block; float: right; margin-left:8px; margin-right:2px;">
                                {% if not checked %}
                                    <button type="button" class="btn btn-sm btn-secondary" disabled>转录</button>
                                    <button type="button" class="btn btn-sm btn-secondary" disabled>收听</button>
                                    <button type="button" class="btn btn-sm btn-secondary" disabled>下载</button>
                                {% else %}
                                    {% if tts_file %}
                                        <button type="button"
                                                class="btn btn-sm btn-secondary"
                                                disabled>已转录</button>
                                        <button type="button"
                                                class="btn btn-sm"
                                                style="background-color:#b8860b;color:white;"
                                                onclick="listenTTS('{{ item['id'] }}')">
                                            收听
                                        </button>
                                        <button type="button"
                                                class="btn btn-sm"
                                                style="background-color:#ff1493;color:white;"
                                                onclick="downloadTTS('{{ item['id'] }}')">
                                            下载
                                        </button>
                                    {% else %}
                                        <button type="button"
                                                class="btn btn-sm"
                                                style="background-color:purple;color:white;"
                                                id="ttsSpeakButton-{{ i }}"
                                                onclick="ttsSpeak('{{ item['id'] }}','{{ i }}','qa')">
                                            转录
                                        </button>
                                        <button type="button" class="btn btn-sm btn-secondary" disabled>收听</button>
                                        <button type="button" class="btn btn-sm btn-secondary" disabled>下载</button>
                                    {% endif %}
                                {% endif %}
                            </div>
                        </div>
                    </div>
                    <div class="orange-text mt-2">
                        当前所属：▶ {{ item['course'] }} ▶ {{ item['chapter'] }} ▶ {{ item['section'] }}
                    </div>
                    {% set full_text = "<strong>请简述: </strong>" ~ item['question'] ~ " <strong>| 答: </strong>" ~ item['answer'] %}
                    {% if full_text|length > 126 %}
                        {% set snippet_text = full_text[:168] ~ "..." %}
                    {% else %}
                        {% set snippet_text = full_text %}
                    {% endif %}
                    <div style="margin-top:10px;">
                        <span id="snippet-qa-{{ i }}">{{ snippet_text|safe }}</span>
                    </div>
                    <div id="fullContent-qa-{{ i }}" style="display:none;margin-top:10px;">
                        <div id="qa-question-{{ i }}">
                            <strong>请简述:</strong> {{ item['question'] }}
                        </div>
                        <div id="qa-answer-{{ i }}">
                            <strong>答:</strong> {{ item['answer'] }}
                        </div>
                    </div>
                </li>
            {% endfor %}
        </ul>
    </div>
{% endif %}
{% endif %}
{% endif %}

<div class="float-btn-container">
    <button class="btn btn-custom w-100" onclick="scrollToTop()">返回顶部</button>
    <button class="btn btn-custom w-100" onclick="toggleAllExpand()">一键展开/收起</button>
</div>


<!-- SSE进度弹窗 -->
        <div class="modal" tabindex="-1" id="progressModal">
          <div class="modal-dialog" style="margin-top:120px;">
            <div class="modal-content">
              <div class="modal-header">
                <h5 class="modal-title">批量处理进度</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
              </div>
              <div class="modal-body">
                <p id="progressText">正在准备...</p>
              </div>
              <div class="modal-footer">
                <button type="button" class="btn btn-secondary"
                        data-bs-dismiss="modal"
                        onclick="refreshAfterClose()">关闭</button>
              </div>
            </div>
          </div>
        </div>

<!-- 播放音频用的模态框 -->
        <div class="modal" tabindex="-1" id="listenModal">
          <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
              <div class="modal-header">
                <h5 class="modal-title">收听音频</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
              </div>
              <div class="modal-body" id="listenModalBody">
                <!-- 动态插入 audio -->
              </div>
              <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">关闭</button>
              </div>
            </div>
          </div>
        </div>


<script>
/* ========== 第一部分：返回顶部 & 一键展开/收起 ========== */

// 返回顶部
function scrollToTop() {
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// 一键展开/收起
let allExpanded = false;

function toggleAllExpand() {
    const blocks = ["mc", "def", "qa"];

    blocks.forEach(type => {
        const snippetNodes = document.querySelectorAll(`[id^='snippet-${type}-']`);
        const fullNodes = document.querySelectorAll(`[id^='fullContent-${type}-']`);
        const toggleTexts = document.querySelectorAll(`[id^='toggleText-${type}-']`);

        if (!allExpanded) {
            // 展开所有内容
            fullNodes.forEach(el => el.style.display = "block");
            snippetNodes.forEach(el => el.style.display = "none");
            toggleTexts.forEach(tt => tt.textContent = "▷ 收起");
        } else {
            // 收起所有内容
            fullNodes.forEach(el => el.style.display = "none");
            snippetNodes.forEach(el => el.style.display = "inline");
            toggleTexts.forEach(tt => tt.textContent = "▷ 展开");
        }
    });

    // 切换全局状态
    allExpanded = !allExpanded;
}

function toggleContent(type, index) {
    const snippet = document.getElementById(`snippet-${type}-${index}`);
    const fullContent = document.getElementById(`fullContent-${type}-${index}`);
    const toggleText = document.getElementById(`toggleText-${type}-${index}`);

    // 依据当前显示状态进行切换
    if (fullContent.style.display === "none" || fullContent.style.display === "") {
        fullContent.style.display = "block";
        snippet.style.display = "none";
        toggleText.textContent = "▷ 收起";
    } else {
        fullContent.style.display = "none";
        snippet.style.display = "inline";
        toggleText.textContent = "▷ 展开";
    }
}


/* ========== 第二部分：SSE 批量操作(批量检查、批量生成) ========== */

// 打开模态框并开始批量生成
function startBulkConvert() {
    let modal = new bootstrap.Modal(document.getElementById('progressModal'));
    modal.show();
    document.getElementById('progressText').innerText = '正在启动批量生成...';

    let es = new EventSource("/bulk_convert_sse");
    es.onmessage = function(e) {
        let data = JSON.parse(e.data);
        if (data.done) {
            document.getElementById('progressText').innerText =
                `完成！共需处理 ${data.total} 条，已处理完。`;
            es.close();
        } else {
            document.getElementById('progressText').innerText =
                `共需处理 ${data.total} 条，已处理 ${data.progress} 条，还剩 ${data.total - data.progress} 条...`;
        }
    };
    es.onerror = function() {
        document.getElementById('progressText').innerText =
            "出错：请检查网络或服务器！";
        es.close();
    };
}

// 打开模态框并开始批量检查
function startBulkCheck() {
    let modal = new bootstrap.Modal(document.getElementById('progressModal'));
    modal.show();
    document.getElementById('progressText').innerText = '正在启动批量检查...';

    let es = new EventSource("/bulk_check_sse");
    es.onmessage = function(e) {
        let data = JSON.parse(e.data);
        if (data.done) {
            document.getElementById('progressText').innerText =
                `完成！共需检查 ${data.total} 条，已处理完。`;
            es.close();
        } else {
            document.getElementById('progressText').innerText =
                `共需检查 ${data.total} 条，已处理 ${data.progress} 条，还剩 ${data.total - data.progress} 条...`;
        }
    };
    es.onerror = function() {
        document.getElementById('progressText').innerText =
            "出错：请检查网络或服务器！";
        es.close();
    };
}

function refreshAfterClose(){
}

// Ajax 单条检查
async function checkTextAjax(kid, t, i) {
    const buttonId = `checkButton-${t}-${i}`;
    const transcriptButtonId = `ttsSpeakButton-${i}`;

    const checkBtn = document.getElementById(buttonId);
    const ttsBtn = document.getElementById(transcriptButtonId);

    // 设置“检查中” + 小转圈，并禁用按钮以防二次点击
    if (checkBtn) {
        checkBtn.disabled = true;
        checkBtn.innerHTML =
            `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 检查中...`;
    }
    if (ttsBtn) {
        ttsBtn.disabled = true;
    }

    try {
        const resp = await fetch("/check_text_ajax", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ kid })
        });
        const data = await resp.json();

        if (!data.success) {
            alert("检查失败: " + (data.error || "未知错误"));

            // 失败后恢复按钮状态，让用户可以重试
            if (checkBtn) {
                checkBtn.innerHTML = "检查";
                checkBtn.disabled = false;
            }
            if (ttsBtn) {
                ttsBtn.disabled = false;
            }
            return;
        }

        // ========== 如果检查成功 ==========

        // (可选) 更新DOM，比如填充内容
        if (data.type === "definition") {
            // ...更新 definition DOM
        } else if (data.type === "qa") {
            // ...更新 qa DOM
        } else if (data.type === "multiple_choice") {
            // ...更新 multiple_choice DOM
        }

        // 刷新TTS按钮UI
        refreshTTSUI(t, i, kid);

        alert("已执行检查，请核实结果！");

        // ★★★ 关键点：检查成功后，将按钮设置为“已检查”并保持灰色，不可再点击
        if (checkBtn) {
            checkBtn.innerHTML = "已检查";
            checkBtn.disabled = true;  // 一直灰色不可点
        }
        if (ttsBtn) {
            ttsBtn.disabled = false;  // TTS按钮可恢复
        }

    } catch (e) {
        alert("请求异常: " + e);

        // 出错时同样恢复，让用户可以重试
        if (checkBtn) {
            checkBtn.innerHTML = "检查";
            checkBtn.disabled = false;
        }
        if (ttsBtn) {
            ttsBtn.disabled = false;
        }
    }
}


// 更新TTS按钮UI（假设还没有tts_file）
function refreshTTSUI(t, i, kid) {
    const containerId = `ttsWrapper-${t}-${i}`;
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = `
        <button type="button" class="btn btn-sm" style="background-color:purple;color:white;"
            id="ttsSpeakButton-${i}" onclick="ttsSpeak('${kid}','${i}','${t}')">
            转录
        </button>
        <button type="button" class="btn btn-sm btn-secondary" disabled>收听</button>
        <button type="button" class="btn btn-sm btn-secondary" disabled>下载</button>
    `;
}

// 发起TTS转录
async function ttsSpeak(kid, index, t){
    const speakBtnId = `ttsSpeakButton-${index}`;
    const speakBtn = document.getElementById(speakBtnId);
    if(speakBtn) {
        speakBtn.disabled = true;
        speakBtn.innerHTML =
            `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 转录中...`;
    }

    try {
        let resp = await fetch("/tts_speak", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ kid: kid, index: index })
        });
        let data = await resp.json();

        if(data.success){
            alert("转录成功！文件已生成: " + data.file_path);
            // 转录成功后，更新UI => “已转录” + “收听” + “下载”
            const wrapperId = `ttsWrapper-${t}-${index}`;
            const container = document.getElementById(wrapperId);
            if(container){
                container.innerHTML = `
                    <button type="button" class="btn btn-sm btn-secondary" disabled>已转录</button>
                    <button type="button" class="btn btn-sm" style="background-color:#b8860b;color:white;" onclick="listenTTS('${kid}')">收听</button>
                    <button type="button" class="btn btn-sm" style="background-color:#ff1493;color:white;" onclick="downloadTTS('${kid}')">下载</button>
                `;
            }
        } else {
            if(speakBtn){
                speakBtn.disabled = false;
                speakBtn.innerHTML = '转录';
            }
            alert("转录失败：" + (data.error || "未知错误"));
        }
    } catch(e) {
        if(speakBtn){
            speakBtn.disabled = false;
            speakBtn.innerHTML = '转录';
        }
        alert("请求异常:" + e);
    }
}

// 收听音频
async function listenTTS(kid){
    try {
        let resp = await fetch("/tts_listen?kid=" + kid);
        let data = await resp.json();
        if(!data.success){
            alert("找不到文件，请重新转录~");
            return;
        }
        let audioHtml = `
            <audio style="width:100%;" controls autoplay src="${data.file_url}"></audio>
        `;
        document.getElementById("listenModalBody").innerHTML = audioHtml;
        let modal = new bootstrap.Modal(document.getElementById("listenModal"));
        modal.show();
    } catch(e){
        alert("请求错误: " + e);
    }
}

// 下载音频
function downloadTTS(kid){
    // 直接新标签打开即可下载
    window.open('/tts_download?kid='+kid, '_blank');
}

function activateAll(){
    document.querySelector('input[name="ktype"][value=""]').checked = true;
}

// 转为选择题按钮
async function convertToMcAjax(ev, kid, i){
    ev.preventDefault(); // 拦截表单默认提交
    const form = ev.target;
    const submitBtn = form.querySelector("button[type='submit']");

    // 禁用按钮并显示“生成中...”+转圈
    submitBtn.disabled = true;
    submitBtn.innerHTML =
        '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 生成中...';

    try {
        const fd = new FormData(form);
        const resp = await fetch("/convert_to_mc", {
            method: "POST",
            body: fd
        });

        // 检查返回
        console.log("Response status: ", resp.status);
        const data = await resp.json();
        console.log("Response data: ", data);

        if(!data.success){
            alert("转换失败：" + (data.error || "未知错误"));
            // 失败时恢复
            submitBtn.disabled = false;
            submitBtn.textContent = '转为选择题';
            return;
        }

        // 更新DOM: 把 multiple_choices 显示到 fullContent-mc-i
        const mcDiv = document.getElementById("fullContent-mc-" + i);
        if(mcDiv){
            let html = `
                <div>
                    <strong>知识点：</strong>
                    <span id="mc-original-${i}">${data.original}</span>
                </div>
            `;
            if(data.multiple_choices && data.multiple_choices.length > 0){
                html += `<hr><p><strong>已生成的单选题列表：</strong></p>`;
                data.multiple_choices.forEach((q_obj, idx)=>{
                    html += `
                        <div class="mb-3">
                            <strong>题目 ${idx+1}: </strong> ${q_obj.question}
                            <ul id="question-list-container">
                                <li>A. ${q_obj.options.A}</li>
                                <li>B. ${q_obj.options.B}</li>
                                <li>C. ${q_obj.options.C}</li>
                                <li>D. ${q_obj.options.D}</li>
                            </ul>
                            <div><strong>答案:</strong> ${q_obj.answer}</div>
                            <div><strong>解析:</strong> ${q_obj.explanation}</div>
                        </div>
                    `;
                    if(idx < data.multiple_choices.length - 1){
                        html += `
                            <div style="text-align:left;color:#b1a17057;">
                                --------------------------------------------
                            </div>
                        `;
                    }
                });
            }
            if(data.last_error){
                html += `
                    <div class="text-danger">
                        <strong>错误:</strong> ${data.last_error}
                    </div>
                `;
            }
            mcDiv.innerHTML = html;
        }

        alert("已成功生成选择题！");

        submitBtn.disabled = true;
        submitBtn.textContent = '已转换';
    } catch(e){
        console.error("Request failed:", e);
        alert("请求异常：" + e);
    } finally {
        // 如果你希望在 catch 之外，才恢复按钮，这里可以控制
        // 但是如果希望按钮永久禁用，就不要在 finally 中恢复
        // submitBtn.disabled = false;
        // submitBtn.textContent = '转为选择题';
    }
}


/* ========== 第五部分：删除知识点 & 删除选择题 ========== */

// 原始保存
function saveEdit() {
    const form = document.getElementById("editKnowledgeForm");
    const formData = new FormData(form);
    const saveButton = document.querySelector('.btn-primary');

    saveButton.disabled = true;
    saveButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 保存中...';

    fetch("/edit_knowledge_ajax", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            // 成功后恢复按钮状态
            saveButton.disabled = false;
            saveButton.innerHTML = '保存';

            // 关闭弹窗
            const modal = bootstrap.Modal.getInstance(document.getElementById('editModal'));
            modal.hide();
            location.reload(); // 添加页面刷新
            // 提示保存成功
            alert("保存成功！");
        } else {
            // 错误恢复按钮状态
            saveButton.disabled = false;
            saveButton.innerHTML = '保存';
            alert("保存失败: " + (result.error || "未知错误"));
        }
    })
    .catch(err => {
        saveButton.disabled = false;
        saveButton.innerHTML = '保存';
        alert("请求出错: " + err);
    });
}

//修改题目
function openEditModal(kid) {
    fetch(`/get_knowledge/${kid}`)
        .then(response => response.json())
        .then(data => {
            console.log(data);  // 这里调试，确保数据是有效的

            if (data.success) {
                document.getElementById("editKid").value = data.kid;
                document.getElementById("editCourse").value = data.course;
                document.getElementById("editChapter").value = data.chapter;
                document.getElementById("editSection").value = data.section;
                document.getElementById("editStatement").value = data.content.original || ''; 

                // 根据知识点类型显示相应的内容
                if (data.type === 'definition') {
                    document.getElementById("editTerm").value = data.content.term || '';
                    document.getElementById("editExplanation").value = data.content.explanation || '';
                    document.querySelector('.field-group.definition').style.display = 'block';
                    document.querySelector('.field-group.multiple_choice').style.display = 'none';
                    document.querySelector('.field-group.qa').style.display = 'none';
                } else if (data.type === 'qa') {
                    document.getElementById("editQuestion").value = data.content.question || '';
                    document.getElementById("editAnswer").value = data.content.answer || '';
                    document.querySelector('.field-group.qa').style.display = 'block';
                    document.querySelector('.field-group.multiple_choice').style.display = 'none';
                    document.querySelector('.field-group.definition').style.display = 'none';
                } else if (data.type === 'multiple_choice') {
                    document.getElementById("editStatement").value = data.content.original || '';  
                    document.querySelector('.field-group.multiple_choice').style.display = 'block';
                    document.querySelector('.field-group.qa').style.display = 'none';
                    document.querySelector('.field-group.definition').style.display = 'none';
                }

                var modal = new bootstrap.Modal(document.getElementById('editModal'));
                modal.show();
            } else {
                alert("无法加载数据！");
            }
        })
        .catch(err => console.error("请求错误:", err));
}



// 删除知识点 Ajax
async function deleteKnowledgeAjax(ev, kid) {
    ev.preventDefault();
    if(!confirmDelete()) {
        return false;
    }

    try {
        const resp = await fetch("/delete_knowledge_ajax", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ kid: kid })
        });
        const data = await resp.json();

        if(data.success) {
            // 找到对应的 li 元素并移除
            const itemElement = ev.target.closest('li');
            if(itemElement) {
                itemElement.remove();
            }
        } else {
            alert("删除失败: " + (data.error || "未知错误"));
        }
    } catch(e) {
        console.error(e);
        alert("请求出错: " + e);
    }
    return false;
}

// 绑定删除“选择题”按钮事件
document.addEventListener('DOMContentLoaded', function() {
    // 页面加载时，检查是否有要展开的题目ID
    const expandedQuestionId = sessionStorage.getItem('expandedQuestionId');
    if (expandedQuestionId) {
        const questionElement = document.getElementById(expandedQuestionId);
        if (questionElement) {
            const toggleText = questionElement.querySelector('[id^="toggleText-mc-"]');
            if (toggleText && toggleText.textContent === "▷ 展开") {
                toggleText.click(); // 展开题目
            }
        }
        sessionStorage.removeItem('expandedQuestionId'); // 清除记录
    }

    document.body.addEventListener('click', function(e) {
        if (e.target.classList.contains('delete-btn')) {
            const questionId = e.target.dataset.id;
            const questionElement = e.target.closest('.mb-3');  // 获取要删除的问题元素
            const toggleText = questionElement.querySelector('[id^="toggleText-mc-"]'); // 获取展开/收起按钮
            const isExpanded = toggleText && toggleText.textContent === "▷ 收起"; // 判断是否已展开

            if (confirm('确定要删除这个选择题吗？')) {
                fetch('/delete_question', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ id: questionId }),
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // 删除当前题目
                        questionElement.remove();

                        // 如果当前问题是展开状态，记录该问题ID
                        if (isExpanded) {
                            sessionStorage.setItem('expandedQuestionId', questionId);
                        }

                        // 重新编号剩余问题并更新展开/收起按钮ID
                        const remainingQuestions = document.querySelectorAll('.list-group-item');
                        remainingQuestions.forEach((item, index) => {
                            const questionIndex = item.querySelector('strong');
                            if (questionIndex) {
                                questionIndex.textContent = index + 1 + "."; // 更新编号
                            }

                            // 更新展开/收起按钮的 id 和事件
                            const toggleText = item.querySelector('[id^="toggleText-mc-"]');
                            if (toggleText) {
                                toggleText.id = `toggleText-mc-${index + 1}`; // 更新 ID
                                toggleText.setAttribute('onclick', `toggleContent('mc', ${index + 1})`); // 更新事件
                            }

                            // 更新其他元素的 id（如 snippet、fullContent）
                            const snippet = item.querySelector('[id^="snippet-mc-"]');
                            if (snippet) {
                                snippet.id = `snippet-mc-${index + 1}`;
                            }

                            const fullContent = item.querySelector('[id^="fullContent-mc-"]');
                            if (fullContent) {
                                fullContent.id = `fullContent-mc-${index + 1}`;
                            }
                        });

                        // 刷新页面
                        location.reload();
                    } else {
                        alert('删除失败: ' + (data.error || '未知错误'));
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            }
        }
    });
});


</script>


    </div>
</div>
"""
    + BASE_HTML_FOOT
)

MANAGE_HTML = (
    BASE_HTML_HEAD
    + """
<div style="margin-top:100px;">
    <h2>维护学科、章、节</h2>
    <p>在此页面，你可以新增、删除或重命名学科/章/节</p>
    <strong><p>注意：若新建学科与本学科无关联，建议新建json数据库文件</p></strong> 
    <!-- 新增课程表单 -->
    <form method="post" class="row gy-2 gx-3 align-items-center mb-3">
        <div class="col-auto">
            <input type="text" name="new_course_name" placeholder="新增学科名" class="form-control">
        </div>
        <div class="col-auto">
            <button type="submit" name="action" value="add_course" class="btn btn-primary">
                新增学科
            </button>
        </div>
    </form>

    <hr>

    <h3>现有学科/章/节</h3>
    <ul class="list-group">
    {% for course_name, cdata in courses.items() %}
        <li class="list-group-item">
            <div class="mb-1">
                <strong> {{ course_name }}:</strong>

                <form method="post" style="display:inline" onsubmit="return confirmDelete()">
                    <input type="hidden" name="course_name" value="{{ course_name }}">
                    <button type="submit" name="action" value="delete_course" class="btn btn-danger btn-sm">
                        删除课程
                    </button>
                </form>

                <form method="post" style="display:inline">
                    <input type="hidden" name="old_course_name" value="{{ course_name }}">
                    <input type="text" name="new_course_name" placeholder="新课程名" style="width:200px;">
                    <button type="submit" name="action" value="rename_course" class="btn btn-warning btn-sm">
                        重命名
                    </button>
                </form>
            </div>

            <ul class="list-group mt-2">
                {% for ch in cdata["chapters"] %}
                <li class="list-group-item">
                    <div>
                        <strong>{{ ch["name"] }}:</strong>

                        <form method="post" style="display:inline" onsubmit="return confirmDelete()">
                            <input type="hidden" name="course_name" value="{{ course_name }}">
                            <input type="hidden" name="chapter_name" value="{{ ch["name"] }}">
                            <button type="submit" name="action" value="delete_chapter" class="btn btn-danger btn-sm">
                                删除章
                            </button>
                        </form>

                        <form method="post" style="display:inline">
                            <input type="hidden" name="course_name" value="{{ course_name }}">
                            <input type="hidden" name="old_chapter_name" value="{{ ch["name"] }}">
                            <input type="text" name="new_chapter_name" placeholder="新章名" style="width:400px;">
                            <button type="submit" name="action" value="rename_chapter" class="btn btn-warning btn-sm">
                                重命名
                            </button>
                        </form>
                    </div>

                    <ul class="list-group mt-2">
                        {% for sec in ch["sections"] %}
                        <li class="list-group-item">
                            <strong> {{ sec["name"] }}:</strong>

                            <form method="post" style="display:inline" onsubmit="return confirmDelete()">
                                <input type="hidden" name="course_name" value="{{ course_name }}">
                                <input type="hidden" name="chapter_name" value="{{ ch["name"] }}">
                                <input type="hidden" name="section_name" value="{{ sec["name"] }}">
                                <button type="submit" name="action" value="delete_section" class="btn btn-danger btn-sm">
                                    删除节
                                </button>
                            </form>

                            <form method="post" style="display:inline">
                                <input type="hidden" name="course_name" value="{{ course_name }}">
                                <input type="hidden" name="chapter_name" value="{{ ch["name"] }}">
                                <input type="hidden" name="old_section_name" value="{{ sec["name"] }}">
                                <input type="text" name="new_section_name" placeholder="新小节名" style="width:600px;">
                                <button type="submit" name="action" value="rename_section" class="btn btn-warning btn-sm">
                                    重命名
                                </button>
                            </form>
                        </li>
                        {% endfor %}
                        <li class="list-group-item">
                            <form method="post" class="row gx-1 gy-1 align-items-center">
                                <input type="hidden" name="course_name" value="{{ course_name }}">
                                <input type="hidden" name="chapter_name" value="{{ ch["name"] }}">
                                <div class="col-auto">
                                    <input type="text" name="new_section_name" placeholder="新增小节名" class="form-control">
                                </div>
                                <div class="col-auto">
                                    <button type="submit" name="action" value="add_section" class="btn btn-success btn-sm">
                                        新增节
                                    </button>
                                </div>
                            </form>
                        </li>
                    </ul>
                </li>
                {% endfor %}
                <li class="list-group-item">
                    <form method="post" class="row gx-1 gy-1 align-items-center">
                        <input type="hidden" name="course_name" value="{{ course_name }}">
                        <div class="col-auto">
                            <input type="text" name="new_chapter_name" placeholder="新增章名" class="form-control">
                        </div>
                        <div class="col-auto">
                            <button type="submit" name="action" value="add_chapter" class="btn btn-success btn-sm">
                                新增章
                            </button>
                        </div>
                    </form>
                </li>
            </ul>
        </li>
    {% endfor %}
    </ul>
</div>
"""
    + BASE_HTML_FOOT
)

ADD_HTML = (
    BASE_HTML_HEAD
    + """
<div style="margin-top:100px;">
<h1 class="mb-4">📝 添加新知识点</h1>

<div id="saveResult" class="alert alert-info" style="display:none;"></div>

<form id="addKnowledgeForm" class="row g-3">
    <div class="col-md-4">
        <label for="selectCourse" class="form-label">选择课程</label>
        <select class="form-select" name="course" id="selectCourse" required>
            <option value="">请选择</option>
            {% for course_name, cdata in courses.items() %}
            <option value="{{ course_name }}">{{ course_name }}</option>
            {% endfor %}
        </select>
    </div>

    <div class="col-md-4">
        <label for="selectChapter" class="form-label">选择章</label>
        <select class="form-select" name="chapter" id="selectChapter" required>
            <option value="">请选择</option>
            {% for course_name, cdata in courses.items() %}
                {% for ch in cdata["chapters"] %}
                <option value="{{ ch['name'] }}">{{ course_name }} -> {{ ch['name'] }}</option>
                {% endfor %}
            {% endfor %}
        </select>
    </div>

    <div class="col-md-4">
        <label for="selectSection" class="form-label">选择节</label>
        <select class="form-select" name="section" id="selectSection" required>
            <option value="">请选择</option>
            {% for course_name, cdata in courses.items() %}
                {% for ch in cdata["chapters"] %}
                    {% for sec in ch["sections"] %}
                    <option value="{{ sec['name'] }}">
                        {{ course_name }}->{{ ch['name'] }}->{{ sec['name'] }}
                    </option>
                    {% endfor %}
                {% endfor %}
            {% endfor %}
        </select>
    </div>

    <div class="col-md-6">
        <label for="ktypeSelect" class="form-label">知识点类型</label>
        <select class="form-select" name="knowledge_type" id="ktypeSelect" onchange="toggleFields()" required>
            <option value="multiple_choice">选择题</option>
            <option value="definition">名词解释</option>
            <option value="qa">问答题</option>
        </select>
    </div>

    <div class="col-md-12">
        <div class="field-group multiple_choice border p-3 mb-3" style="display:block">
            <label class="form-label">原始内容</label>
            <textarea class="form-control" name="statement" placeholder="此处可输入知识点原文" rows="8"></textarea>
        </div>

        <div class="field-group definition border p-3 mb-3" style="display:none">
            <label class="form-label">术语名称</label>
            <input type="text" class="form-control" name="term" placeholder="必填">

            <label class="form-label mt-2">详细解释</label>
            <textarea class="form-control" name="explanation" rows="8"></textarea>
        </div>

        <div class="field-group qa border p-3 mb-3" style="display:none">
            <label class="form-label">问题内容</label>
            <input type="text" class="form-control" name="question" placeholder="问题">

            <label class="form-label mt-2">答案</label>
            <textarea class="form-control" name="answer" rows="8"></textarea>
        </div>
    </div>

    <div class="col-12">
        <button type="submit" class="btn btn-primary">💾 保存</button>
    </div>
</form>

<script>
const coursesData = {{ courses|tojson|safe }};

document.getElementById("selectCourse").addEventListener("change", function(){
    let course = this.value;
    let chapterSelect = document.getElementById("selectChapter");
    let sectionSelect = document.getElementById("selectSection");
    chapterSelect.innerHTML = '<option value="">请选择</option>';
    sectionSelect.innerHTML = '<option value="">请选择</option>';

    if(course && coursesData[course]){
        let chapters = coursesData[course].chapters;
        chapters.forEach(ch => {
            let opt = document.createElement("option");
            opt.value = ch.name;
            opt.textContent = ch.name;
            chapterSelect.appendChild(opt);
        });
    }
});

document.getElementById("selectChapter").addEventListener("change", function(){
    let course = document.getElementById("selectCourse").value;
    let chapter = this.value;
    let sectionSelect = document.getElementById("selectSection");
    sectionSelect.innerHTML = '<option value="">请选择</option>';

    if(course && coursesData[course]){
        let chapters = coursesData[course].chapters;
        let targetCh = chapters.find(c => c.name === chapter);
        if(targetCh){
            targetCh.sections.forEach(sec => {
                let opt = document.createElement("option");
                opt.value = sec.name;
                opt.textContent = sec.name;
                sectionSelect.appendChild(opt);
            });
        }
    }
});

function toggleFields() {
    const type = document.getElementById('ktypeSelect').value;
    document.querySelectorAll('.field-group').forEach(group => {
        if (group.classList.contains(type)) {
            group.style.display = 'block';
        } else {
            group.style.display = 'none';
        }
    });
}

// ============== 新增：使用AJAX保存 ==============

// 新增保存
const addForm = document.getElementById("addKnowledgeForm");
addForm.addEventListener("submit", async function(e){
    e.preventDefault();

    const formData = new FormData(addForm);
    try{
        let resp = await fetch("/add_knowledge_ajax", {
            method: "POST",
            body: formData
        });
        let result = await resp.json();
        if(result.success){
            let div = document.getElementById("saveResult");
            div.style.display = "block";
            div.innerText = "保存成功！您可以继续添加下一个。";

            let ktype = formData.get("knowledge_type");
            if(ktype === "multiple_choice"){
                addForm.querySelector('textarea[name="statement"]').value = "";
            } else if(ktype === "definition"){
                addForm.querySelector('input[name="term"]').value = "";
                addForm.querySelector('textarea[name="explanation"]').value = "";
            } else if(ktype === "qa"){
                addForm.querySelector('input[name="question"]').value = "";
                addForm.querySelector('textarea[name="answer"]').value = "";
            }
        } else {
            alert("保存失败: " + (result.error || "未知错误"));
        }
    } catch(err){
        console.error(err);
        alert("请求出错: " + err);
    }
});

</script>
</div>
"""
    + BASE_HTML_FOOT
)

EDIT_HTML = (
    BASE_HTML_HEAD
    + """
<div style="margin-top:100px;">
<h1 class="mb-4">✏️ 修改知识点</h1>

{% if not item %}
<div class="alert alert-danger">无效的知识点ID</div>
{% else %}

<form method="post" class="row g-3" style="margin-top:10px;">
    <input type="hidden" name="kid" value="{{ kid }}">

    <div class="col-md-4">
        <label for="selectCourse" class="form-label">选择课程</label>
        <select class="form-select" name="course" id="selectCourse" required>
            {% for course_name, cdata in courses.items() %}
            <option value="{{ course_name }}" 
                {% if course_name == item['course'] %} selected {% endif %}>
                {{ course_name }}
            </option>
            {% endfor %}
        </select>
    </div>

    <div class="col-md-4">
        <label for="selectChapter" class="form-label">选择章</label>
        <select class="form-select" name="chapter" id="selectChapter" required>
            {% for course_name, cdata in courses.items() %}
                {% for ch in cdata["chapters"] %}
                <option value="{{ ch['name'] }}"
                    {% if course_name == item['course'] and ch['name'] == item['chapter'] %} selected {% endif %}>
                    {{ course_name }}->{{ ch['name'] }}
                </option>
                {% endfor %}
            {% endfor %}
        </select>
    </div>

    <div class="col-md-4">
        <label for="selectSection" class="form-label">选择节</label>
        <select class="form-select" name="section" id="selectSection" required>
            {% for course_name, cdata in courses.items() %}
                {% for ch in cdata["chapters"] %}
                    {% for sec in ch["sections"] %}
                    <option value="{{ sec['name'] }}"
                        {% if course_name == item['course'] and ch['name'] == item['chapter'] and sec['name'] == item['section'] %} selected {% endif %}>
                        {{ course_name }}->{{ ch['name'] }}->{{ sec['name'] }}
                    </option>
                    {% endfor %}
                {% endfor %}
            {% endfor %}
        </select>
    </div>

    <!-- 显示类型（只读） -->
    <div class="col-md-4">
        <label class="form-label">当前类型（只读）</label>
        {% set type_label = item['type'] %}
        {% if item['type'] == 'multiple_choice' %}
            {% set type_label = "知识点" %}
        {% elif item['type'] == 'definition' %}
            {% set type_label = "名词解释" %}
        {% elif item['type'] == 'qa' %}
            {% set type_label = "问答题" %}
        {% endif %}

        <input type="hidden" name="knowledge_type" value="{{ item['type'] }}">
        <input type="text" class="form-control" value="{{ type_label }}" readonly>
    </div>

    {% if item['type'] == 'multiple_choice' %}
    <div class="col-md-12 border p-3">
        <label class="form-label">原始内容</label>
        <textarea class="form-control" name="statement" rows="12"
                  style="min-height: calc(8.5em + 1.75rem + 2px);">{{ item['content']['original'] }}</textarea>
    </div>
   {% elif item['type'] == 'definition' %}
    <div class="col-md-12 border p-3">
        <label class="form-label">术语名称</label>
        <input type="text" class="form-control" name="term" value="{{ item['content']['term'] }}">
        <label class="form-label mt-2">详细解释</label>
        <textarea class="form-control" name="explanation" rows="12">{{ item['content']['explanation'] }}</textarea>
    </div>
{% elif item['type'] == 'qa' %}
    <div class="col-md-12 border p-3">
        <label class="form-label">问题内容</label>
        <input type="text" class="form-control" name="question" value="{{ item['content']['question'] }}">
        <label class="form-label mt-2">答案</label>
        <textarea class="form-control" name="answer" rows="12">{{ item['content']['answer'] }}</textarea>
    </div>
{% endif %}
    <div class="col-12">
        <button type="submit" class="btn btn-primary">保存修改</button>
    </div>
</form>

<script>
const coursesData = {{ courses|tojson|safe }};

document.getElementById("selectCourse").addEventListener("change", function(){
    let course = this.value;
    let chapterSelect = document.getElementById("selectChapter");
    let sectionSelect = document.getElementById("selectSection");
    chapterSelect.innerHTML = '';
    sectionSelect.innerHTML = '';

    if(course && coursesData[course]){
        let chapters = coursesData[course].chapters;
        chapters.forEach(ch => {
            let opt = document.createElement("option");
            opt.value = ch.name;
            opt.textContent = course + "->" + ch.name;
            chapterSelect.appendChild(opt);
        });
        if(chapters.length > 0) {
            let firstChName = chapters[0].name;
            let firstSecArr = chapters[0].sections;
            firstSecArr.forEach(sec => {
                let opt = document.createElement("option");
                opt.value = sec.name;
                opt.textContent = course + "->" + firstChName + "->" + sec.name;
                sectionSelect.appendChild(opt);
            });
        }
    }
});

document.getElementById("selectChapter").addEventListener("change", function(){
    let course = document.getElementById("selectCourse").value;
    let chapter = this.value;
    let sectionSelect = document.getElementById("selectSection");
    sectionSelect.innerHTML = '';

    if(course && coursesData[course]){
        let chapters = coursesData[course].chapters;
        let targetCh = chapters.find(c => c.name === chapter);
        if(targetCh){
            targetCh.sections.forEach(sec => {
                let opt = document.createElement("option");
                opt.value = sec.name;
                opt.textContent = course + "->" + chapter + "->" + sec.name;
                sectionSelect.appendChild(opt);
            });
        }
    }
});
</script>

{% endif %}
</div>

"""
    + BASE_HTML_FOOT
)


@app.route("/get_question_list", methods=["GET"])
def get_question_list():
    data = load_data()

    # 获取更新后的所有选择题
    question_list = []
    for kid, meta in data["index"].items():
        if meta["type"] == "multiple_choice":
            question_list.append(
                {
                    "id": kid,
                    "content": meta["content"]["original"],  # 这里只获取题干内容
                }
            )

    return jsonify({"success": True, "questions": question_list})


@app.route("/delete_knowledge_ajax", methods=["POST"])
def delete_knowledge_ajax():
    data = load_data()
    kid = request.form.get("kid")
    if not kid or kid not in data["index"]:
        return jsonify({"success": False, "error": "无效或不存在的ID"})

    del data["index"][kid]
    remove_kid_from_course(data, kid)
    save_data(data)

    return jsonify({"success": True})


@app.route("/get_knowledge/<kid>")
def get_knowledge(kid):
    data = load_data()
    if kid not in data["index"]:
        return jsonify({"success": False, "error": "无效或不存在的知识点ID"})

    item = data["index"][kid]
    return jsonify(
        {
            "success": True,
            "kid": kid,
            "course": item["course"],
            "chapter": item["chapter"],
            "section": item["section"],
            "type": item["type"],
            "content": item["content"],
        }
    )


@app.route("/edit_knowledge_ajax", methods=["POST"])
def edit_knowledge_ajax():
    data = load_data()  # 加载数据
    kid = request.form.get("kid")
    if not kid or kid not in data["index"]:
        return jsonify({"success": False, "error": "无效或不存在的知识点ID"})

    # 获取表单中更新的字段
    item = data["index"][kid]
    item["course"] = request.form.get("course")
    item["chapter"] = request.form.get("chapter")
    item["section"] = request.form.get("section")

    # 根据不同的知识点类型，更新相应的内容
    if item["type"] == "definition":
        item["content"]["term"] = request.form.get("term")
        item["content"]["explanation"] = request.form.get("explanation")
    elif item["type"] == "qa":
        item["content"]["question"] = request.form.get("question")
        item["content"]["answer"] = request.form.get("answer")
    elif item["type"] == "multiple_choice":
        item["content"]["original"] = request.form.get("statement")
        item["content"]["multiple_choices"] = []  # 清空选择题选项
        item["content"]["checked"] = False
        item["content"]["tts_file"] = None

    # 保存更新后的数据
    data["index"][kid] = item
    sync_knowledge_in_course(data, kid)
    save_data(data)
    print(f"保存成功: {item}")  # 打印保存的数据，确认数据是否成功更新
    return jsonify({"success": True})


########################################################
# 数据读写
########################################################
data_lock = threading.Lock()  # 新增线程锁


def load_data() :
    conn = get_db_connection()
    cursor = conn.cursor()
    try :
        # 1. 加载所有课程
        cursor.execute('SELECT * FROM courses')
        courses = {course['name'] : {'chapters' : []} for course in cursor.fetchall()}

        # 2. 加载章节及其关联的课程信息
        cursor.execute('SELECT * FROM chapters')
        chapters = cursor.fetchall()
        for chapter in chapters :
            if chapter['course_name'] in courses :  # 确保课程存在
                courses[chapter['course_name']]['chapters'].append({
                    'name' : chapter['name'],
                    'sections' : []
                })

        # 3. 加载小节信息，使用 JOIN 获取必要的关联信息
        cursor.execute('''
            SELECT s.*, ch.course_name, ch.name as chapter_name
            FROM sections s
            JOIN chapters ch ON s.chapter_id = ch.id
        ''')
        sections = cursor.fetchall()
        for section in sections :
            course_name = section['course_name']
            chapter_name = section['chapter_name']

            if course_name in courses :
                for chapter in courses[course_name]['chapters'] :
                    if chapter['name'] == chapter_name :
                        chapter['sections'].append({
                            'name' : section['name'],
                            'knowledge' : []
                        })

        # 4. 加载知识点信息，使用 JOIN 获取所有必要的关联信息
        cursor.execute('''
            SELECT k.*, s.name as section_name, ch.name as chapter_name, ch.course_name
            FROM knowledge k
            JOIN sections s ON k.section_id = s.id
            JOIN chapters ch ON s.chapter_id = ch.id
        ''')
        knowledge = cursor.fetchall()
        knowledge_index = {}

        for item in knowledge :
            course_name = item['course_name']
            chapter_name = item['chapter_name']
            section_name = item['section_name']

            knowledge_data = {
                'type' : item['type'],
                'id' : item['id'],
                'content' : item['content'],
                'checked' : bool(item['checked']),
                'tts_file' : item['tts_file']
            }

            # 添加到索引
            knowledge_index[item['id']] = knowledge_data

            # 添加到课程结构中
            if course_name in courses :
                for chapter in courses[course_name]['chapters'] :
                    if chapter['name'] == chapter_name :
                        for section in chapter['sections'] :
                            if section['name'] == section_name :
                                section['knowledge'].append(knowledge_data)

        return {'courses' : courses, 'index' : knowledge_index}

    except Exception as e :
        print(f"加载数据时出错: {e}")
        raise
    finally :
        conn.close()


def save_data(data) :
    conn = get_db_connection()
    cursor = conn.cursor()
    try :
        # 开始事务
        cursor.execute('BEGIN TRANSACTION')

        # 清空所有表
        cursor.execute('DELETE FROM knowledge')
        cursor.execute('DELETE FROM sections')
        cursor.execute('DELETE FROM chapters')
        cursor.execute('DELETE FROM courses')

        # 保存课程数据
        for course_name, course_data in data['courses'].items() :
            # 插入课程
            cursor.execute('INSERT INTO courses (name) VALUES (?)', (course_name,))

            # 插入章节
            for chapter in course_data['chapters'] :
                cursor.execute(
                    'INSERT INTO chapters (name, course_name) VALUES (?, ?)',
                    (chapter['name'], course_name)
                )
                chapter_id = cursor.lastrowid

                # 插入小节
                for section in chapter['sections'] :
                    cursor.execute(
                        'INSERT INTO sections (name, chapter_id) VALUES (?, ?)',
                        (section['name'], chapter_id)
                    )
                    section_id = cursor.lastrowid

                    # 插入知识点
                    for knowledge in section['knowledge'] :
                        cursor.execute('''
                            INSERT INTO knowledge 
                            (id, type, content, checked, tts_file, section_id)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            knowledge['id'],
                            knowledge['type'],
                            knowledge['content'],
                            1 if knowledge.get('checked', False) else 0,
                            knowledge.get('tts_file'),
                            section_id
                        ))

        # 提交事务
        conn.commit()
        print("数据保存成功!")

    except Exception as e :
        # 如果出错，回滚事务
        conn.rollback()
        print(f"保存数据时出错: {e}")
        raise

    finally :
        conn.close()

@app.route("/")
def index():
    data = load_data()
    selected_course = request.args.get("course", "")
    selected_chapter = request.args.get("chapter", "")
    selected_section = request.args.get("section", "")
    selected_type = request.args.get("ktype", "")
    keyword = request.args.get("keyword", "").strip()

    all_courses = data["courses"]

    # 计算可用章列表
    if selected_course and selected_course in all_courses:
        chapters_list = [c["name"] for c in all_courses[selected_course]["chapters"]]
    else:
        chapters_list = []
        for c_data in all_courses.values():
            for ch in c_data["chapters"]:
                if ch["name"] not in chapters_list:
                    chapters_list.append(ch["name"])

    # 计算可用节列表
    sections_list = []
    if selected_course and selected_chapter and selected_course in all_courses:
        for ch_obj in all_courses[selected_course]["chapters"]:
            if ch_obj["name"] == selected_chapter:
                sections_list = [s["name"] for s in ch_obj["sections"]]
                break

    # 如果没任何搜索参数就不返回结果
    no_param = (
        not selected_course
        and not selected_chapter
        and not selected_section
        and not selected_type
        and not keyword
    )

    if no_param:
        # 获取第一个课程
        if all_courses:
            first_course = list(all_courses.keys())[0]
            chapters = all_courses[first_course]["chapters"]
            if chapters:
                first_chapter = chapters[0]["name"]
                sections = chapters[0]["sections"]
                if sections:
                    first_section = sections[0]["name"]
                    # 拼接一个带默认参数的URL
                    return redirect(
                        url_for(
                            "index",
                            course=first_course,
                            chapter=first_chapter,
                            section=first_section,
                            ktype="",  # 你想默认展示哪种类型都可以
                        )
                    )
        # 如果根本就没有课程或章节，那就继续返回空白页也行
        return render_template_string(
            INDEX_HTML,
            courses=all_courses,
            selected_course="",
            selected_chapter="",
            selected_section="",
            selected_type="",
            keyword="",
            chapters=[],
            sections=[],
            results=None,
            multiple_choices=[],
            definitions=[],
            qa=[],
        )

    results = []
    for kid, meta in data["index"].items():
        if selected_course and meta["course"] != selected_course:
            continue
        if selected_chapter and meta["chapter"] != selected_chapter:
            continue
        if selected_section and meta["section"] != selected_section:
            continue
        if selected_type and meta["type"] != selected_type:
            continue

        # 关键字过滤
        if keyword:
            if meta["type"] == "definition":
                match_text = meta["content"]["term"] + meta["content"]["explanation"]
            elif meta["type"] == "qa":
                match_text = meta["content"]["question"] + meta["content"]["answer"]
            else:
                match_text = meta["content"]["original"]
            if keyword not in match_text:
                continue

        item = {
            "id": kid,
            "type": meta["type"],
            "course": meta["course"],
            "chapter": meta["chapter"],
            "section": meta["section"],
        }
        if meta["type"] == "definition":
            item["term"] = meta["content"]["term"]
            item["explanation"] = meta["content"]["explanation"]
            item["content"] = meta["content"]
        elif meta["type"] == "qa":
            item["question"] = meta["content"]["question"]
            item["answer"] = meta["content"]["answer"]
            item["content"] = meta["content"]
        else:
            # multiple_choice
            item["content"] = meta["content"]
        results.append(item)

    # 排序函数，按照 (课程顺序, 章顺序, 节顺序) 排
    def get_directory_pos(x):
        c_list = list(all_courses.keys())
        cidx = c_list.index(x["course"]) if x["course"] in c_list else 9999
        chidx, sidx = 9999, 9999
        if x["course"] in all_courses:
            chapters = all_courses[x["course"]]["chapters"]
            for i, ch in enumerate(chapters):
                if ch["name"] == x["chapter"]:
                    chidx = i
                    for j, sec in enumerate(ch["sections"]):
                        if sec["name"] == x["section"]:
                            sidx = j
                            break
                    break
        return (cidx, chidx, sidx)

    results.sort(key=lambda x: get_directory_pos(x))
    mcs = [r for r in results if r["type"] == "multiple_choice"]
    defs = [r for r in results if r["type"] == "definition"]
    qa = [r for r in results if r["type"] == "qa"]

    return render_template_string(
        INDEX_HTML,
        courses=all_courses,
        selected_course=selected_course,
        selected_chapter=selected_chapter,
        selected_section=selected_section,
        selected_type=selected_type,
        keyword=keyword,
        chapters=chapters_list,
        sections=sections_list,
        results=results,
        multiple_choices=mcs,
        definitions=defs,
        qa=qa,
    )


@app.route("/manage", methods=["GET", "POST"])
def manage_directory():
    data = load_data()
    courses = data["courses"]

    if request.method == "POST":
        action = request.form.get("action", "")
        if action == "add_course":
            new_course_name = request.form.get("new_course_name", "").strip()
            if new_course_name and new_course_name not in courses:
                courses[new_course_name] = {"chapters": []}
                save_data(data)
        elif action == "delete_course":
            course_name = request.form.get("course_name", "")
            if course_name in courses:
                to_delete = []
                for kid, meta in data["index"].items():
                    if meta["course"] == course_name:
                        to_delete.append(kid)
                for k in to_delete:
                    del data["index"][k]
                del courses[course_name]
                save_data(data)
        elif action == "rename_course":
            old_course_name = request.form.get("old_course_name", "")
            new_course_name = request.form.get("new_course_name", "").strip()
            if (
                old_course_name in courses
                and new_course_name
                and new_course_name not in courses
            ):
                temp_data = courses.pop(old_course_name)
                courses[new_course_name] = temp_data
                for kid, meta in data["index"].items():
                    if meta["course"] == old_course_name:
                        meta["course"] = new_course_name
                save_data(data)
        elif action == "add_chapter":
            course_name = request.form.get("course_name", "")
            new_chapter_name = request.form.get("new_chapter_name", "").strip()
            if course_name in courses and new_chapter_name:
                chapter_obj = next(
                    (
                        c
                        for c in courses[course_name]["chapters"]
                        if c["name"] == new_chapter_name
                    ),
                    None,
                )
                if not chapter_obj:
                    courses[course_name]["chapters"].append(
                        {"name": new_chapter_name, "sections": []}
                    )
                save_data(data)
        elif action == "delete_chapter":
            course_name = request.form.get("course_name", "")
            chapter_name = request.form.get("chapter_name", "")
            if course_name in courses:
                ch_list = courses[course_name]["chapters"]
                for i, ch in enumerate(ch_list):
                    if ch["name"] == chapter_name:
                        to_delete = []
                        for kid, meta in data["index"].items():
                            if (
                                meta["course"] == course_name
                                and meta["chapter"] == chapter_name
                            ):
                                to_delete.append(kid)
                        for k in to_delete:
                            del data["index"][k]
                        del ch_list[i]
                        break
                save_data(data)
        elif action == "rename_chapter":
            course_name = request.form.get("course_name", "")
            old_chapter_name = request.form.get("old_chapter_name", "")
            new_chapter_name = request.form.get("new_chapter_name", "").strip()
            if course_name in courses and new_chapter_name:
                ch_list = courses[course_name]["chapters"]
                if not any(ch["name"] == new_chapter_name for ch in ch_list):
                    for ch in ch_list:
                        if ch["name"] == old_chapter_name:
                            ch["name"] = new_chapter_name
                            for kid, meta in data["index"].items():
                                if (
                                    meta["course"] == course_name
                                    and meta["chapter"] == old_chapter_name
                                ):
                                    meta["chapter"] = new_chapter_name
                            break
                    save_data(data)
        elif action == "add_section":
            course_name = request.form.get("course_name", "")
            chapter_name = request.form.get("chapter_name", "")
            new_section_name = request.form.get("new_section_name", "").strip()
            if course_name in courses and new_section_name:
                for ch in courses[course_name]["chapters"]:
                    if ch["name"] == chapter_name:
                        if not any(
                            sec["name"] == new_section_name for sec in ch["sections"]
                        ):
                            ch["sections"].append(
                                {"name": new_section_name, "knowledge": []}
                            )
                save_data(data)
        elif action == "delete_section":
            course_name = request.form.get("course_name", "")
            chapter_name = request.form.get("chapter_name", "")
            section_name = request.form.get("section_name", "")
            if course_name in courses:
                for ch in courses[course_name]["chapters"]:
                    if ch["name"] == chapter_name:
                        sec_list = ch["sections"]
                        for i, sec in enumerate(sec_list):
                            if sec["name"] == section_name:
                                to_delete = []
                                for kid, meta in data["index"].items():
                                    if (
                                        meta["course"] == course_name
                                        and meta["chapter"] == chapter_name
                                        and meta["section"] == section_name
                                    ):
                                        to_delete.append(kid)
                                for k in to_delete:
                                    del data["index"][k]
                                del sec_list[i]
                                break
                save_data(data)
        elif action == "rename_section":
            course_name = request.form.get("course_name", "")
            chapter_name = request.form.get("chapter_name", "")
            old_section_name = request.form.get("old_section_name", "")
            new_section_name = request.form.get("new_section_name", "").strip()
            if course_name in courses and new_section_name:
                for ch in courses[course_name]["chapters"]:
                    if ch["name"] == chapter_name:
                        if not any(
                            sec["name"] == new_section_name for sec in ch["sections"]
                        ):
                            for sec in ch["sections"]:
                                if sec["name"] == old_section_name:
                                    sec["name"] = new_section_name
                                    for kid, meta in data["index"].items():
                                        if (
                                            meta["course"] == course_name
                                            and meta["chapter"] == chapter_name
                                            and meta["section"] == old_section_name
                                        ):
                                            meta["section"] = new_section_name
                                    break
                save_data(data)

        return redirect(url_for("manage_directory"))

    return render_template_string(MANAGE_HTML, courses=courses)


@app.route("/add", methods=["GET", "POST"])
def add_knowledge():
    data = load_data()
    all_courses = data["courses"]

    if request.method == "POST":
        course = request.form["course"]
        chapter = request.form["chapter"]
        section = request.form["section"]

        if course not in all_courses:
            all_courses[course] = {"chapters": []}

        chapter_obj = next(
            (c for c in all_courses[course]["chapters"] if c["name"] == chapter), None
        )
        if not chapter_obj:
            chapter_obj = {"name": chapter, "sections": []}
            all_courses[course]["chapters"].append(chapter_obj)

        section_obj = next(
            (s for s in chapter_obj["sections"] if s["name"] == section), None
        )
        if not section_obj:
            section_obj = {"name": section, "knowledge": []}
            chapter_obj["sections"].append(section_obj)

        ktype = request.form["knowledge_type"]
        kid = f"{course}|{chapter}|{section}|{int(time.time() * 1000)}"

        if ktype == "definition":
            term = request.form.get("term", "")
            explanation = request.form.get("explanation", "")
            data["index"][kid] = {
                "course": course,
                "chapter": chapter,
                "section": section,
                "type": ktype,
                "content": {
                    "term": term,
                    "explanation": explanation,
                    "checked": False,
                    "tts_file": None,
                },
            }
            knowledge = {
                "type": ktype,
                "id": kid,
                "term": term,
                "explanation": explanation,
            }
        elif ktype == "qa":
            question = request.form.get("question", "")
            answer = request.form.get("answer", "")
            data["index"][kid] = {
                "course": course,
                "chapter": chapter,
                "section": section,
                "type": ktype,
                "content": {
                    "question": question,
                    "answer": answer,
                    "checked": False,
                    "tts_file": None,
                },
            }
            knowledge = {
                "type": ktype,
                "id": kid,
                "question": question,
                "answer": answer,
            }
        else:
            statement = request.form.get("statement", "")
            data["index"][kid] = {
                "course": course,
                "chapter": chapter,
                "section": section,
                "type": ktype,
                "content": {
                    "original": statement,
                    "multiple_choices": [],
                    "checked": False,
                    "tts_file": None,
                },
            }
            knowledge = {
                "type": ktype,
                "id": kid,
                "content": {"original": statement, "multiple_choices": []},
            }

        section_obj["knowledge"].append(knowledge)
        save_data(data)
        return redirect(url_for("index"))

    return render_template_string(ADD_HTML, courses=all_courses)


@app.route("/add_knowledge_ajax", methods=["POST"])
def add_knowledge_ajax():
    data = load_data()
    all_courses = data["courses"]

    course = request.form.get("course", "")
    chapter = request.form.get("chapter", "")
    section = request.form.get("section", "")
    ktype = request.form.get("knowledge_type", "")

    if not (course and chapter and section and ktype):
        return jsonify({"success": False, "error": "参数不足"})

    if course not in all_courses:
        all_courses[course] = {"chapters": []}

    chapter_obj = next(
        (c for c in all_courses[course]["chapters"] if c["name"] == chapter), None
    )
    if not chapter_obj:
        chapter_obj = {"name": chapter, "sections": []}
        all_courses[course]["chapters"].append(chapter_obj)

    section_obj = next(
        (s for s in chapter_obj["sections"] if s["name"] == section), None
    )
    if not section_obj:
        section_obj = {"name": section, "knowledge": []}
        chapter_obj["sections"].append(section_obj)

    kid = f"{course}|{chapter}|{section}|{int(time.time() * 1000)}"
    if ktype == "definition":
        term = request.form.get("term", "")
        explanation = request.form.get("explanation", "")
        data["index"][kid] = {
            "course": course,
            "chapter": chapter,
            "section": section,
            "type": ktype,
            "content": {
                "term": term,
                "explanation": explanation,
                "checked": False,
                "tts_file": None,
            },
        }
        knowledge = {
            "type": ktype,
            "id": kid,
            "term": term,
            "explanation": explanation,
        }
    elif ktype == "qa":
        question = request.form.get("question", "")
        answer = request.form.get("answer", "")
        data["index"][kid] = {
            "course": course,
            "chapter": chapter,
            "section": section,
            "type": ktype,
            "content": {
                "question": question,
                "answer": answer,
                "checked": False,
                "tts_file": None,
            },
        }
        knowledge = {
            "type": ktype,
            "id": kid,
            "question": question,
            "answer": answer,
        }
    else:
        statement = request.form.get("statement", "")
        data["index"][kid] = {
            "course": course,
            "chapter": chapter,
            "section": section,
            "type": ktype,
            "content": {
                "original": statement,
                "multiple_choices": [],
                "checked": False,
                "tts_file": None,
            },
        }
        knowledge = {
            "type": ktype,
            "id": kid,
            "content": {"original": statement, "multiple_choices": []},
        }

    section_obj["knowledge"].append(knowledge)
    save_data(data)
    return jsonify({"success": True})


@app.route("/delete_question", methods=["POST"])
def delete_question():
    data = request.get_json()
    question_id = data.get("id")

    data_store = load_data()
    deleted = False

    for kid, item in data_store["index"].items():
        if item["type"] == "multiple_choice":
            original_questions = item["content"]["multiple_choices"]
            new_questions = [
                q for q in original_questions if q.get("id") != question_id
            ]
            if len(new_questions) != len(original_questions):
                item["content"]["multiple_choices"] = new_questions
                deleted = True
                if not new_questions:
                    item["content"].pop("last_error", None)
                break

    if deleted:
        save_data(data_store)
        return jsonify(
            {"success": True, "questions": item["content"]["multiple_choices"]}
        )  # 返回更新后的题目列表
    return jsonify({"success": False, "error": "题目不存在"})


def delete_question_by_id(question_id):
    data = load_data()
    for kid, item in data["index"].items():
        if item["type"] == "multiple_choice":
            for q_obj in item["content"]["multiple_choices"]:
                if q_obj.get("id") == question_id:
                    item["content"]["multiple_choices"].remove(q_obj)
                    save_data(data)
                    return True
    return False


@app.route("/edit_knowledge/<kid>", methods=["GET", "POST"])
def edit_knowledge(kid):
    data = load_data()
    if kid not in data["index"]:
        return render_template_string(
            EDIT_HTML, item=None, kid=kid, courses=data["courses"]
        )

    old_item = data["index"][kid]
    if request.method == "POST":
        new_course = request.form.get("course", old_item["course"])
        new_chapter = request.form.get("chapter", old_item["chapter"])
        new_section = request.form.get("section", old_item["section"])
        new_id = f"{new_course}|{new_chapter}|{new_section}|{int(time.time() * 1000)}"
        ktype = old_item["type"]

        # 修改 => 视同为新内容 => checked = False, tts_file = None
        if ktype == "definition":
            new_term = request.form.get("term", "")
            new_explanation = request.form.get("explanation", "")
            new_content = {
                "term": new_term,
                "explanation": new_explanation,
                "checked": False,
                "tts_file": None,
            }
        elif ktype == "qa":
            new_question = request.form.get("question", "")
            new_answer = request.form.get("answer", "")
            new_content = {
                "question": new_question,
                "answer": new_answer,
                "checked": False,
                "tts_file": None,
            }
        elif ktype == "multiple_choice":
            new_statement = request.form.get("statement", "")
            new_content = {
                "original": new_statement,
                "multiple_choices": [],
                "checked": False,
                "tts_file": None,
            }
        else:
            new_content = old_item["content"]
            new_content["checked"] = False
            new_content["tts_file"] = None

        data["index"][new_id] = {
            "course": new_course,
            "chapter": new_chapter,
            "section": new_section,
            "type": ktype,
            "content": new_content,
        }

        if new_course not in data["courses"]:
            data["courses"][new_course] = {"chapters": []}

        # 挂到目录树
        ch_obj = next(
            (
                c
                for c in data["courses"][new_course]["chapters"]
                if c["name"] == new_chapter
            ),
            None,
        )
        if not ch_obj:
            ch_obj = {"name": new_chapter, "sections": []}
            data["courses"][new_course]["chapters"].append(ch_obj)

        sec_obj = next(
            (s for s in ch_obj["sections"] if s["name"] == new_section), None
        )
        if not sec_obj:
            sec_obj = {"name": new_section, "knowledge": []}
            ch_obj["sections"].append(sec_obj)

        sec_obj["knowledge"].append(convert_item_to_stub(data["index"][new_id]))

        # 删旧kid
        if kid in data["index"]:
            del data["index"][kid]
            remove_kid_from_course(data, kid)

        save_data(data)
        return redirect(url_for("index"))

    return render_template_string(
        EDIT_HTML, item=old_item, kid=kid, courses=data["courses"]
    )


def remove_kid_from_course(data, kid):
    for c_name, c_val in data["courses"].items():
        for ch_obj in c_val["chapters"]:
            for sec_obj in ch_obj["sections"]:
                new_k_list = []
                for k_obj in sec_obj.get("knowledge", []):
                    if k_obj.get("id") != kid:
                        new_k_list.append(k_obj)
                sec_obj["knowledge"] = new_k_list


def sync_knowledge_in_course(data, kid):
    """
    将 data["index"][kid] 的最新内容，写回到 data["courses"] 的目录树中。
    """
    if kid not in data["index"]:
        return

    item = data["index"][kid]
    # 先删除旧的kid
    remove_kid_from_course(data, kid)

    # 准备在 courses 里找到对应位置
    c_name = item["course"]
    ch_name = item["chapter"]
    s_name = item["section"]

    if c_name not in data["courses"]:
        data["courses"][c_name] = {"chapters": []}

    chapters = data["courses"][c_name]["chapters"]
    ch_obj = next((c for c in chapters if c["name"] == ch_name), None)
    if not ch_obj:
        ch_obj = {"name": ch_name, "sections": []}
        chapters.append(ch_obj)

    sec_obj = next((s for s in ch_obj["sections"] if s["name"] == s_name), None)
    if not sec_obj:
        sec_obj = {"name": s_name, "knowledge": []}
        ch_obj["sections"].append(sec_obj)

    sec_obj["knowledge"].append(convert_item_to_stub(item))


def convert_item_to_stub(item):
    ktype = item["type"]
    kid = item.get("id")
    c = item["content"]
    if ktype == "definition":
        return {
            "type": "definition",
            "id": kid,
            "term": c["term"],
            "explanation": c["explanation"],
        }
    elif ktype == "qa":
        return {
            "type": "qa",
            "id": kid,
            "question": c["question"],
            "answer": c["answer"],
        }
    elif ktype == "multiple_choice":
        return {"type": "multiple_choice", "id": kid, "content": c}
    else:
        return {"type": ktype, "id": kid, "content": c}


@app.route("/delete_knowledge", methods=["POST"])
def delete_knowledge():
    data = load_data()
    kid = request.form.get("kid")
    if not kid or kid not in data["index"]:
        return redirect(url_for("index"))
    del data["index"][kid]
    remove_kid_from_course(data, kid)
    save_data(data)
    return redirect(url_for("index"))


def generate_unique_id():
    return str(uuid.uuid4())


# ------------------ 下面是：改为AJAX的“转为选择题”路由 ------------------
@app.route("/convert_to_mc", methods=["POST"])
def convert_to_mc():
    data = load_data()
    kid = request.form.get("knowledge_id")
    if not kid or kid not in data["index"]:
        # AJAX时返回JSON
        return jsonify({"success": False, "error": "无效或不存在的ID"})

    item = data["index"][kid]
    if item["type"] != "multiple_choice":
        return jsonify({"success": False, "error": "非 multiple_choice 类型"})

    original_text = item["content"]["original"].strip()
    if not original_text:
        return jsonify({"success": False, "error": "原文为空，无法转换"})

    prompt_text = f"""请理解以下考研知识点，转换为1-6个的**独立**单选题。每个选择题的标题需信息完整，**禁止过分解读知识点、禁止无中生有**：\n\n {original_text} """
    messages = [
        {
            "role": "system",
            "content": """你是一个试题生成专家，请严格遵守以下规则：

                                0. **必须直接输出严格符合规范的JSON数组，禁止任何额外文字（包括 ```json、注释、说明等）**

                                1. **题干要求**：
                                - 仅使用提供的文字内容，禁止添加外部文字
                                - 题干应完整呈现核心内容
                                - 禁止随意举例
                                - 标点符号正确，''单引号请转换为“”双引号。

                                2. **选项要求**：
                                - 生成4个选项（A/B/C/D）
                                - 正确选项需随机
                                - 保持选项文字简洁

                                3. **输出格式**：
                                - 严格生成JSON数组，每个对象包含：
                                  - `question`: 题干（字符串）
                                  - `options`: 选项字典，包含A、B、C、D四个选项
                                  - `answer`: 正确选项字母（A/B/C/D）
                                  - `explanation`: 简短解析，需要输出完整的答案解析。
                                - 禁止任何额外文字，直接输出JSON数组。

                                4. **示例格式**：
                                [
                                    {
                                        "question": "《说文解字》中“教”字的释义强调了什么教育特征？",
                                        "options": {
                                            "A": "知识系统性",
                                            "B": "上行下效的示范性",
                                            "C": "因材施教的针对性",
                                            "D": "考核标准的统一性"
                                        },
                                        "answer": "B",
                                        "explanation": "《说文解字》中“教”字的释义明确体现上行下效的示范特征。"
                                    }
                                ]
                                """,
        },
        {"role": "user", "content": prompt_text},
    ]
    parsed_data = call_mc_api_two_stage(messages)
    if parsed_data is None:
        # 两轮API都失败
        item["content"]["last_error"] = "调用API多次失败(2*10)"
        item["content"]["multiple_choices"] = []
        data["index"][kid] = item
        save_data(data)
        return jsonify({"success": False, "error": "两API均失败，未能生成"})

    if isinstance(parsed_data, list):
        for q_obj in parsed_data:
            q_obj["id"] = generate_unique_id()
        item["content"]["multiple_choices"] = parsed_data
    else:
        parsed_data["id"] = generate_unique_id()
        item["content"]["multiple_choices"] = [parsed_data]
    if "last_error" in item["content"]:
        del item["content"]["last_error"]

    # 转换成功后，tts_file 清空
    item["content"]["tts_file"] = None
    data["index"][kid] = item
    save_data(data)

    return jsonify(
        {
            "success": True,
            "original": original_text,
            "multiple_choices": item["content"]["multiple_choices"],
            "last_error": "",
        }
    )


@app.route("/bulk_convert_sse")
def bulk_convert_sse():
    def generate():
        data = load_data()
        to_process = []
        # 仅批量生成"还没生成"的 multiple_choice
        for kid, item in data["index"].items():
            if item["type"] != "multiple_choice":
                continue
            mc_list = item["content"].get("multiple_choices", [])
            last_err = item["content"].get("last_error")
            if mc_list and not last_err:
                continue
            to_process.append(kid)

        total = len(to_process)
        processed = 0
        yield f"data: {json.dumps({'progress' : processed, 'total' : total})}\n\n"

        for kid in to_process:
            item = data["index"][kid]
            original_text = item["content"]["original"].strip()
            if original_text:
                prompt_text = f"""请围绕以下知识点，输出1-6个**独立的**单选题。每个选择题的标题需信息完整，**请合理控制出题数量**：\n\n {original_text} """
                messages = [
                    {
                        "role": "system",
                        "content": """你是一个试题生成专家，请严格遵守以下规则：

                                                0. **必须直接输出严格符合规范的JSON数组，禁止任何额外文字（包括 ```json、注释、说明等）**

                                                1. **题干要求**：
                                                - 仅使用提供的文字内容，禁止添加外部文字
                                                - 题干应完整呈现核心内容
                                                - 禁止随意举例
                                                - 标点符号正确，''单引号请转换为“”双引号。

                                                2. **选项要求**：
                                                - 生成4个选项（A/B/C/D）
                                                - 正确选项需随机
                                                - 保持选项文字简洁

                                                3. **输出格式**：
                                                - 严格生成JSON数组，每个对象包含：
                                                  - `question`: 题干（字符串）
                                                  - `options`: 选项字典，包含A、B、C、D四个选项
                                                  - `answer`: 正确选项字母（A/B/C/D）
                                                  - `explanation`: 直接给出正确的完整解析，禁止添加“根据题干明确说明”、“根据原文”、“原文指出”等提示词。
                                                - 禁止任何额外文字，直接输出JSON数组。

                                                4. **示例格式**：
                                                [
                                                    {
                                                        "question": "《说文解字》中“教”字的释义强调了什么教育特征？",
                                                        "options": {
                                                            "A": "知识系统性",
                                                            "B": "上行下效的示范性",
                                                            "C": "因材施教的针对性",
                                                            "D": "考核标准的统一性"
                                                        },
                                                        "answer": "B",
                                                        "explanation": "《说文解字》中“教”字的释义明确体现上行下效的示范特征。"
                                                    }
                                                ]
                                                """,
                    },
                    {"role": "user", "content": prompt_text},
                ]

                parsed_data = call_mc_api_two_stage(messages)
                if parsed_data is not None:
                    if isinstance(parsed_data, list):
                        item["content"]["multiple_choices"] = parsed_data
                    else:
                        item["content"]["multiple_choices"] = [parsed_data]
                    item["content"].pop("last_error", None)
                else:
                    item["content"]["multiple_choices"] = []
                    item["content"]["last_error"] = "批量转换失败(2*10)"

                data["index"][kid] = item
                save_data(data)

            processed += 1
            yield f"data: {json.dumps({'progress' : processed, 'total' : total})}\n\n"

        yield f"data: {json.dumps({'progress' : total, 'total' : total, 'done' : True})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/bulk_check_sse")
def bulk_check_sse():
    def generate():
        data = load_data()
        # 仅检查未 checked 的
        to_process = [
            kid
            for kid, itm in data["index"].items()
            if not itm["content"].get("checked")
        ]
        total = len(to_process)
        processed = 0
        yield f"data: {json.dumps({'progress' : 0, 'total' : total})}\n\n"

        for kid in to_process:
            item = data["index"][kid]
            check_and_correct_item(item)
            item["content"]["tts_file"] = None
            data["index"][kid] = item
            sync_knowledge_in_course(data, kid)
            save_data(data)

            processed += 1
            yield f"data: {json.dumps({'progress' : processed, 'total' : total})}\n\n"

        yield f"data: {json.dumps({'progress' : total, 'total' : total, 'done' : True})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


########################################################
# 纠错逻辑
########################################################
@app.route("/check_text_ajax", methods=["POST"])
def check_text_ajax():
    data = load_data()
    kid = request.form.get("kid", "")
    if not kid or kid not in data["index"]:
        return jsonify({"success": False, "error": "无效或不存在的知识点ID"})

    item = data["index"][kid]
    # 调用你已有的纠错核心函数
    check_and_correct_item(item)

    # 检查完成后，顺便把 tts_file 置空，因为它和已检查内容对应
    item["content"]["tts_file"] = None
    # 设置 checked 为 True，表示已经检查过
    item["content"]["checked"] = True
    data["index"][kid] = item

    # 同步到目录树
    sync_knowledge_in_course(data, kid)
    save_data(data)

    # 前端要根据不同类型来局部更新页面
    t = item["type"]
    c = item["content"]

    if t == "definition":
        return jsonify(
            {
                "success": True,
                "type": "definition",
                "term": c["term"],
                "explanation": c["explanation"],
            }
        )
    elif t == "qa":
        return jsonify(
            {
                "success": True,
                "type": "qa",
                "question": c["question"],
                "answer": c["answer"],
            }
        )
    elif t == "multiple_choice":
        return jsonify(
            {"success": True, "type": "multiple_choice", "original": c["original"]}
        )
    else:
        return jsonify({"success": True, "type": t})


def check_and_correct_item(item):
    content = item["content"]
    # 若已检查，就不再重复检查
    if content.get("checked"):
        return

    t = item["type"]
    if t == "definition":
        to_check = {"term": content["term"], "explanation": content["explanation"]}
    elif t == "qa":
        to_check = {"question": content["question"], "answer": content["answer"]}
    elif t == "multiple_choice":
        to_check = {"original": content["original"]}
    else:
        return

    original_json_str = json.dumps(to_check, ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": "你是一个文本纠错专家，请仅返回纠正后的JSON，不要做其它解释。",
        },
        {
            "role": "user",
            "content": f"请充分理解以下JSON文本，文本中可能含有错别字、错误的标点、多余的空格等，请修正后输出修正后的JSON， "
            f"在此过程中："
            f"数字序号(如有)，请转换为： 1. 2. 3. ......这样的阿拉伯数字序号;"
            f"数字序号1. 2. 3. 之下的序号(如有)，请转换为 ① ② ③..."
            f"语言可以适当的重构和组织，使文字转为更适合TTS播报朗读的文本."
            f"\n\n {original_json_str}",
        },
    ]
    corrected_data = call_check_api_with_retry(client_check, messages, 10)
    if corrected_data is None:
        return

    # 关键修改：根据类型更新对应的字段
    if t == "definition":
        content["term"] = corrected_data.get("term", content["term"])
        content["explanation"] = corrected_data.get(
            "explanation", content["explanation"]
        )
    elif t == "qa":
        content["question"] = corrected_data.get("question", content["question"])
        content["answer"] = corrected_data.get("answer", content["answer"])
    elif t == "multiple_choice":
        content["original"] = corrected_data.get("original", content["original"])

    content["checked"] = True


########################################################
# TTS部分
########################################################
@app.route("/tts_speak", methods=["POST"])
def tts_speak():
    data = load_data()
    kid = request.form.get("kid", "")
    index = request.form.get("index", "")
    if not kid or kid not in data["index"]:
        return jsonify({"success": False, "error": "无效kid"})

    item = data["index"][kid]
    ctype = item["type"]
    content = item["content"]

    if content.get("tts_file"):
        return jsonify(
            {"success": False, "error": "该知识点已转录过，请先修改/检查后再重试"}
        )

    if ctype == "multiple_choice":
        text_for_tts = "考点速记：" + content["original"]
        suffix = "考点"
    elif ctype == "definition":
        term = content["term"]
        explanation = content["explanation"]
        text_for_tts = f"请解释 {term}：\n答：{explanation}"
        suffix = "名词解释"
    elif ctype == "qa":
        question = content["question"]
        answer = content["answer"]
        text_for_tts = f"请简述 {question}\n答：{answer}"
        suffix = "问答题"
    else:
        return jsonify({"success": False, "error": "暂不支持此类型"})

    course = item["course"]
    chapter = item["chapter"]
    section = item["section"]
    base_dir = os.path.join("MP3", course, chapter, section)
    os.makedirs(base_dir, exist_ok=True)

    ts = int(time.time())
    if not index:
        index = "1"
    filename = f"{section}-{suffix}{index}_{ts}.mp3"
    out_path = os.path.join(base_dir, filename)

    try:
        asyncio.run(run_tts(text_for_tts, out_path))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    rel_path = os.path.relpath(out_path)
    content["tts_file"] = rel_path
    data["index"][kid] = item
    save_data(data)

    return jsonify({"success": True, "file_path": rel_path})


@app.route("/tts_listen", methods=["GET"])
def tts_listen():
    kid = request.args.get("kid", "")
    data = load_data()
    if not kid or kid not in data["index"]:
        return jsonify({"success": False, "error": "kid无效"})

    content = data["index"][kid]["content"]
    tts_file = content.get("tts_file")
    if not tts_file:
        return jsonify({"success": False, "error": "没有tts_file"})

    abs_path = os.path.abspath(tts_file)
    if not os.path.exists(abs_path):
        return jsonify({"success": False, "error": "文件已丢失"})

    file_url = f"/mp3_file?file={tts_file}"
    return jsonify({"success": True, "file_url": file_url})


@app.route("/tts_download", methods=["GET"])
def tts_download():
    kid = request.args.get("kid", "")
    data = load_data()
    if not kid or kid not in data["index"]:
        return jsonify({"success": False, "error": "kid无效"})

    content = data["index"][kid]["content"]
    tts_file = content.get("tts_file")
    if not tts_file:
        return jsonify({"success": False, "error": "没有tts_file"})

    abs_path = os.path.abspath(tts_file)
    if not os.path.exists(abs_path):
        return jsonify({"success": False, "error": "文件已丢失"})

    # 生成一个临时的 32k mp3，命名可随意
    tmp_filename = f"tmp_{uuid.uuid4().hex}.mp3"
    tmp_path = os.path.join(os.path.dirname(abs_path), tmp_filename)

    # 调用ffmpeg转码为32k
    # 也可以用'ffmpeg-python'之类的库，这里演示用命令行
    try:
        cmd = [
            "ffmpeg",
            "-y",  # -y表示若文件已存在则直接覆盖
            "-i",
            abs_path,  # 输入
            "-b:a",
            "32k",  # 音频码率设为32k
            tmp_path,  # 输出
        ]
        subprocess.run(cmd, check=True)
    except Exception as e:
        return jsonify({"success": False, "error": f"ffmpeg转码失败: {str(e)}"})

    # 准备让浏览器下载时，文件名和原tts_file保持一致(用户需求: “文件名不变”)
    original_name = os.path.basename(tts_file)  # 取原本的文件名

    # 用send_file以附件形式下载
    response = send_file(
        tmp_path,
        as_attachment=True,
        download_name=original_name,  # 保持原文件名
        mimetype="audio/mpeg",
    )

    # 可以在请求完成后把临时文件删除
    # 这里用回调函数的方式，等response完成后再删
    @response.call_on_close
    def remove_temp():
        try:
            os.remove(tmp_path)
        except:
            pass

    return response


@app.route("/mp3_file")
def mp3_file():
    f = request.args.get("file", "")
    abs_path = os.path.abspath(f)
    if not os.path.exists(abs_path):
        return "File not found", 404
    return send_file(abs_path, mimetype="audio/mpeg")

if __name__ == "__main__":
    # 初始化数据库
    try :
        # 1. 检查数据库文件是否存在
        db_exists = os.path.exists(DATABASE)
        if not db_exists :
            print(f"数据库文件不存在，创建新数据库: {DATABASE}")
        else :
            print(f"使用现有数据库: {DATABASE}")

        # 2. 创建数据库连接并初始化表
        print("开始初始化数据库...")
        init_db()
        print("数据库初始化完成")

        # 3. 验证表是否创建成功
        conn = get_db_connection()
        cursor = conn.cursor()

        # 检查表是否存在
        cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='courses'
            """)
        table_exists = cursor.fetchone() is not None

        if not table_exists :
            print("错误：courses 表未能成功创建")
            raise Exception("数据库表创建失败")

        # 4. 检查是否需要添加示例数据
        cursor.execute("SELECT COUNT(*) FROM courses")
        count = cursor.fetchone()[0]
        conn.close()

        if count == 0 :
            print("数据库为空，添加示例数据...")
            # 添加示例数据
            add_course("示例课程")
            chapter_id = add_chapter("第一章", "示例课程")
            section_id = add_section("第一节", chapter_id)

            test_knowledge = {
                'id' : 'test001',
                'type' : 'multiple_choice',
                'content' : '这是一道测试题目',
                'options' : 'A.选项1|B.选项2|C.选项3|D.选项4',
                'answer' : 'A',
                'explanation' : '这是答案解释',
                'checked' : 0,
                'tts_file' : None
            }
            add_knowledge(test_knowledge, section_id)
            print("示例数据添加完成")

        # 5. 尝试加载数据
        print("开始加载数据...")
        data = load_data()
        print("数据加载成功!")

        # 6. 启动 Flask 应用
        print("启动 Web 服务器...")
        app.run(host='0.0.0.0', port=5000, debug=True)

    except sqlite3.Error as e :
        print(f"数据库错误: {e}")
        raise
    except Exception as e :
        print(f"程序启动时出错: {e}")
        raise
