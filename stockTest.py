import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv
import re
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# .env 파일 로드
load_dotenv()

# db 연결
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PASSWORD_SAFE = quote_plus(DB_PASSWORD)
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT", 3306)

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD_SAFE}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    echo=True
)

with engine.connect() as conn:
    result = conn.execute(text("SELECT NOW();"))
    print("DB 연결 성공 ✅ 현재 시간:", result.scalar())
    

KRX_FILE = "krx_temp.xls"

# 날짜별 섹션 뉴스(일반 모드)
NEWS_BY_DATE_URL = "https://finance.naver.com/news/mainnews.naver?&date={date}&page={page}"
CLOVA_API_KEY = os.getenv("CLOVA_API_KEY")
CLOVA_URL = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003"

st.set_page_config(page_title="KRX 뉴스-AI 자동화", layout="centered")
st.title("KRX 상장종목 뉴스 + AI 분석 자동화")

# -------------------- 세션/캐시 --------------------
@st.cache_resource
def get_session():
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    })

    # 커넥션 풀/재시도
    retry = Retry(
        total=3, backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = get_session()

@st.cache_data(ttl=3600)
def load_krx_excel(path: str) -> pd.DataFrame:
    try:
        try:
            df = pd.read_excel(path, dtype=str, engine="xlrd")
        except Exception:
            df = pd.read_excel(path, dtype=str)
        df["종목코드"] = df["종목코드"].apply(lambda x: str(x).zfill(6))
        return df
    except Exception as e:
        raise e

@st.cache_data(ttl=1800)
def crawl_naver_daily_price_cached(stock_code, max_days=60) -> pd.DataFrame:
    base_url = "https://finance.naver.com/item/sise_day.naver"
    all_rows, page = [], 1
    while True:
        params = {"code": stock_code, "page": page}
        res = SESSION.get(base_url, params=params, timeout=10)
        soup = BeautifulSoup(res.content, "lxml")
        table = soup.find("table", class_="type2")
        if not table:
            break
        rows = table.find_all("tr")
        data_rows = [row for row in rows if len(row.find_all("td")) == 7]
        if not data_rows:
            break
        for row in data_rows:
            cols = row.find_all("td")
            date = cols[0].get_text(strip=True).replace(".", "-")
            close = cols[1].get_text(strip=True).replace(",", "")
            open_price = cols[3].get_text(strip=True).replace(",", "")
            high = cols[4].get_text(strip=True).replace(",", "")
            low = cols[5].get_text(strip=True).replace(",", "")
            volume = cols[6].get_text(strip=True).replace(",", "")
            all_rows.append({
                "날짜": date,
                "종가": float(close) if close else None,
                "시가": float(open_price) if open_price else None,
                "고가": float(high) if high else None,
                "저가": float(low) if low else None,
                "거래량": int(volume) if volume else None
            })
            if len(all_rows) >= max_days:
                break
        if len(all_rows) >= max_days:
            break
        page += 1
    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["종가"])
    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.sort_values("날짜").reset_index(drop=True)
    return df

# 환경변수에서 API 키 읽기
api_key_env = os.getenv("OPENAI_API_KEY")
if "api_key" not in st.session_state:
    st.session_state.api_key = api_key_env

def submit_api_key():
    st.session_state.api_key = st.session_state.api_key_input

if st.session_state.api_key is None:
    st.text_input("API 키가 있는 ENV 파일을 받아 주세요.", type="password", key="api_key_input", on_change=submit_api_key)
else:
    st.success("✅ ENV 파일 인식 완료")
    

# 모드 토글 상태
if "specific_mode" not in st.session_state:
    st.session_state.specific_mode = False

# 파일 체크
file_exists = os.path.isfile(KRX_FILE)
if file_exists and (st.session_state.api_key is not None):
    st.success(f"✅ '{KRX_FILE}' 파일이 확인되었습니다.")
else:
    if not file_exists:
        st.warning(
            "❌ '{KRX_FILE}' 파일이 폴더에 없습니다.\n"
            "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13\n"
            "위 주소에서 파일 받은 후\n"
            "파일 -> 내보내기 -> 파일 형식 변경 -> Excel 97 - 2003 통합 문서로 내보내기\n"
            "'krx_temp.xls'로 이름을 변경하여 이 파이썬 파일과 같은 폴더에 위치시켜 주세요."
        )
    if st.session_state.api_key is None:
        st.warning("❌ OpenAI API 키를 입력해주세요.")

# KRX 종목 로딩
if file_exists:
    try:
        df_stocklist = load_krx_excel(KRX_FILE)
        stock_names = sorted(set(df_stocklist["회사명"].dropna()))
    except Exception as e:
        st.error(f"엑셀 파일 읽기 오류: {e}")
        stock_names = []
else:
    stock_names = []

def get_code_from_name(stock_name):
    try:
        matched_rows = df_stocklist[df_stocklist["회사명"] == stock_name]
        if not matched_rows.empty:
            code = matched_rows.iloc[0]["종목코드"]
            return str(code).zfill(6)
        return None
    except Exception:
        return None

# ---------- 날짜 파싱 (일반 모드 필요 시) ----------
def parse_news_date(dt_text: str) -> datetime.date:
    raw = (dt_text or "").strip().replace(".", "-")
    token = raw.split()[0]
    parts = token.split("-")
    if len(parts) == 2:
        year = datetime.today().year
        token = f"{year}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
    return datetime.strptime(token, "%Y-%m-%d").date()

# ---------- 회사뉴스 날짜 파싱 ----------
def parse_company_news_date(s: str) -> datetime.date:
    s = (s or "").strip()
    s = s.replace("-", ".").replace("/", ".")
    token = s.split()[0]
    return datetime.strptime(token, "%Y.%m.%d").date()

# ---------- 텍스트 전처리(공백 제거 + 소문자) ----------
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace(" ", "").lower()

# ---------- 공통 유틸 ----------
def find_stock_in_text(text, stock_names_set):
    if not isinstance(text, str):
        return None
    for name in stock_names_set:
        if name in text:
            return name
    return None

def classify_news(title, summary):
    news_text = f"{title} {summary}"
    prompt = f"""
아래 뉴스를 보고 JSON 형식으로만 출력해라.

형식:
{{
  "stock_nm": "뉴스에서 직접적으로 지목하거나 영향을 받을 주식명 (예: 삼성전자, 현대차, LG화학). 
               특정 종목이 아니라 전체 시장에 영향을 준다면 '전체'",
  "category": "관련 산업 카테고리 (예: 반도체, 자동차, 음식료, 금융, IT, 헬스케어 등)",
  "tag": "호재 or 악재 or 중립",
  "description": "한 문장 설명",
  "power": 0~100 사이 정수 (뉴스가 해당 종목/시장에 미치는 영향력 강도, 숫자가 클수록 영향 큼)
}}

뉴스: {news_text}
"""
    headers = {
        "Authorization": f"Bearer {CLOVA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "topP": 0.8,
        "maxTokens": 512
    }

    try:
        res = requests.post(CLOVA_URL, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        answer = res.json()["result"]["message"]["content"].strip()
        return json.loads(answer)
    except Exception as e:
        return {
            "stock_nm": "분석불가",
            "category": "기타",
            "tag": "중립",
            "description": f"CLOVA 오류: {e}",
            "power": 0
        }


def safe_float(val):
    try:
        return float(str(val).replace(",", ""))
    except Exception:
        return None

def crawl_naver_daily_price(stock_code, max_days=60):
    return crawl_naver_daily_price_cached(stock_code, max_days=max_days)

def get_existing_news(engine, title, stock_nm):
    sql = text("SELECT id, stock_nm, tag, category, description, date FROM NEWS_DATA WHERE title = :title AND stock_nm = :stock_nm")
    with engine.connect() as conn:
        row = conn.execute(sql, {"title": title, "stock_nm": stock_nm}).fetchone()
        if row:
            return dict(row._mapping)
    return None


def insert_news_to_db(engine, title, stock_nm, tag, category, description, power, date_str):
    sql = text("""
        INSERT INTO NEWS_DATA (title, stock_nm, tag, category, description, power, date)
        VALUES (:title, :stock_nm, :tag, :category, :description, :power, :date)
    """)
    with engine.begin() as conn:
        conn.execute(sql, {
            "title": title,
            "stock_nm": stock_nm,
            "tag": tag,
            "category": category,
            "description": description,
            "power": power,
            "date": date_str
        })


def plot_bollinger_20day(df_price, stock_name, stock_code, key=None, news_dict=None):
    if len(df_price) < 20:
        st.warning(f"{stock_name} 시세 데이터가 부족해 볼린저 밴드를 그릴 수 없습니다.")
        return
    window = 20
    df_price[f"MA{window}"] = df_price["종가"].rolling(window=window).mean()
    df_price[f"STD{window}"] = df_price["종가"].rolling(window=window).std()
    df_price[f"Upper{window}"] = df_price[f"MA{window}"] + 2 * df_price[f"STD{window}"]
    df_price[f"Lower{window}"] = df_price[f"MA{window}"] - 2 * df_price[f"STD{window}"]
    dates = df_price["날짜"]; max_date = dates.max(); min_date = dates.min()
    st.write(f"### {stock_name} 20일 볼린저 밴드")
    st.markdown(f"[주가 상세 보기](https://finance.naver.com/item/main.nhn?code={stock_code})", unsafe_allow_html=True)
    fig = go.Figure(
        data=[
            go.Scatter(x=dates, y=df_price["종가"], mode="lines", name="종가"),
            go.Scatter(x=dates, y=df_price[f"MA{window}"], mode="lines", name=f"MA{window}"),
            go.Scatter(x=dates, y=df_price[f"Upper{window}"], mode="lines", name="상단 밴드"),
            go.Scatter(x=dates, y=df_price[f"Lower{window}"], mode="lines", name="하단 밴드", fill="tonexty", fillcolor="rgba(200,200,200,0.2)"),
        ],
        layout=go.Layout(
            xaxis_title="날짜", yaxis_title="가격",
            xaxis=dict(range=[min_date, max_date + pd.Timedelta(days=3)], fixedrange=True),
            yaxis=dict(autorange=True, fixedrange=True),
            dragmode="pan"
        )
    )
    if news_dict:
        for nd, news_list in news_dict.items():
            try:
                nd_dt = pd.to_datetime(nd)
                price_row = df_price[df_price["날짜"] == nd_dt]
                if not price_row.empty:
                    price = price_row.iloc[0]["종가"]
                    news_text = "<br>".join([n["title"] for n in news_list])
                    fig.add_trace(go.Scatter(
                        x=[nd_dt], y=[price], mode="markers+text",
                        marker=dict(color="orange", size=7, symbol="circle"),
                        text=["📰"], textposition="top center",
                        name=f"뉴스({nd})", hoverinfo="text", hovertext=news_text, showlegend=False
                    ))
            except Exception:
                pass
    st.plotly_chart(fig, use_container_width=True, key=key)

# ------------------- 모드 전환 버튼 (단일) -------------------
toggle_label = "↩ 일반 모드로 전환" if st.session_state.specific_mode else "📰 특정 기사만 보기 (BETA)"
if st.button(toggle_label):
    st.session_state.specific_mode = not st.session_state.specific_mode

# ------------------- 속도 개선 도우미 -------------------
@st.cache_data(ttl=600)
def get_last_page_for_date(date_param: str) -> int:
    url = NEWS_BY_DATE_URL.format(date=date_param, page=1)
    res = SESSION.get(url, timeout=10)
    soup = BeautifulSoup(res.content, "lxml")

    last_link = soup.select_one("td.pgRR a")
    if last_link and last_link.get("href"):
        m = re.search(r"page=(\d+)", last_link["href"])
        if m:
            return int(m.group(1))

    pages = []
    for a in soup.select("table.Nnavi a"):
        txt = (a.get_text(strip=True) or "").strip()
        if txt.isdigit():
            pages.append(int(txt))
    return max(pages) if pages else 1

def parse_mainnews_page(html_bytes, fallback_date_str: str, stock_key_norm: str):
    """
    mainnews 한 페이지에서 해당 종목 키워드가 제목/요약에 있으면 결과 리스트로 반환
    """
    soup = BeautifulSoup(html_bytes, "lxml")
    items = soup.select("ul.newsList > li")
    results = []
    if not items:
        return results

    for li in items:
        a = li.select_one("dd.articleSubject a")
        sm = li.select_one("dd.articleSummary")
        if not a or not sm:
            continue

        title = a.get_text(strip=True)
        href = a.get("href", "")
        link = href if href.startswith("http") else "https://finance.naver.com" + href
        summary = sm.get_text(" ", strip=True)

        dt_tag = sm.select_one("span.wdate")
        if dt_tag:
            news_date_only = dt_tag.get_text(strip=True).split(" ")[0].replace(".", "-")
        else:
            news_date_only = fallback_date_str

        title_key = normalize_text(title)
        summary_key = normalize_text(summary)
        if (stock_key_norm in title_key) or (stock_key_norm in summary_key):
            results.append({
                "종목명": None,
                "뉴스": title,
                "링크": link,
                "요약": summary,
                "뉴스날짜": news_date_only,
            })
    return results


# ------------------- 모드별 UI/로직 -------------------
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# ------------------- 챗봇 ----------------------------
# --- 챗봇 버튼 위치 (CSS) ---

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 버튼 표시
if st.button("챗봇에게 물어보기", key="chat_btn"):
    st.session_state.chat_open = not st.session_state.chat_open

# --- 챗봇 창 ---
if st.session_state.chat_open:
    st.sidebar.title("RAG 챗봇")

    chat_container = st.sidebar.container()
    input_container = st.sidebar.container()

    # 1) 대화 메시지 표시
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 2) 입력창을 항상 아래 고정
    with input_container:
        question = st.chat_input("뉴스/주식 질문하기")

    if question:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)

        st.session_state.chat_history.append({"role": "user", "content": question})

        # 로딩 중 메시지 먼저 출력
        with chat_container:
            with st.chat_message("assistant"):
                loading_placeholder = st.empty()
                loading_placeholder.markdown("로딩 중.....")

        # --- CLOVA 질문 해석 ---
        parse_prompt = f"""
        사용자의 질문을 바탕으로 아래 JSON 형식으로만 답해라 무조건.
        한국어 그대로 주식명을 사용해야 하고, 만약 이름이 이상하다 싶으면 예측해 예를들어
        "삼전" 이라고 하면 "삼성전자" 겠구나 이런식으로, 전체적으로 물어보는거 같으면 "전체"로 검색 하면 다 나오니까 필요하면 이걸로 해.
        {{
          "stock": "조회할 주식명 (예: 삼성전자, 현대차, 전체)",
          "days": 최근 N일 (정수, 없으면 기본 30),
          "limit": 최대 결과 개수 (없으면 20)
        }}
        질문: "{question}"
        """
        headers = {"Authorization": f"Bearer {CLOVA_API_KEY}", "Content-Type": "application/json"}
        payload = {"messages": [{"role": "user", "content": parse_prompt}]}
        res = requests.post(CLOVA_URL, headers=headers, json=payload, timeout=30)
        parsed = json.loads(res.json()["result"]["message"]["content"])

        stock = parsed.get("stock", "전체")
        days = int(parsed.get("days", 30))
        limit = int(parsed.get("limit", 20))

        # --- DB 검색 ---
        with engine.connect() as conn:
            sql = text("""
                SELECT title, stock_nm, tag, category, description, power, date
                FROM NEWS_DATA
                WHERE (:stock = '전체' OR stock_nm = :stock)
                  AND date >= :start_date
                ORDER BY power DESC, date DESC
                LIMIT :limit
            """)
            rows = conn.execute(sql, {
                "stock": stock,
                "start_date": (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d"),
                "limit": limit
            }).fetchall()

        if rows:
            df = pd.DataFrame(rows, columns=["title","stock_nm","tag","category","description","power","date"])
            context = "\n".join([
                f"{r['date']} [{r['stock_nm']}] {r['title']} "
                f"({r['tag']}, {r['category']}, power={r['power']}): {r['description']}"
                for r in df.to_dict(orient="records")
            ])
        else:
            context = "관련 뉴스 없음."

        answer_prompt = f"""
        사용자가 "{question}" 라고 물었습니다.
        아래는 최근 {days}일 동안 {stock} 관련 뉴스 데이터입니다.
        반드시 이 데이터를 근거로 종합적으로 분석해서 답변하세요.
        데이터에 없는 내용은 추측하지 말고 '관련 뉴스 없음'이라고 말하세요.
        power 값은 해당 뉴스의 영향력을 예측한 값이니까 만약 질문이 예측해줘 혹은 추천해줘 이런내용이면 이 power 값을 참조하고
        가장 영향을 준 기사를 보여주는 것도 좋아
        데이터:
        {context}
        """
        payload = {
            "messages": [{"role": "user", "content": answer_prompt}],
            "temperature": 0.4,
            "topP": 0.8,
            "maxTokens": 512
        }
        res = requests.post(CLOVA_URL, headers=headers, json=payload, timeout=30)
        answer = res.json()["result"]["message"]["content"]

        # --- 로딩 메시지 교체 ---
        loading_placeholder.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})



# ===== 특정 기사만 보기 모드 (날짜 섹션 전체 순회 + 종목명 필터) =====
if st.session_state.specific_mode:
    st.subheader("📰 특정 기사만 보기")

    days_to_fetch = st.number_input("가져올 뉴스 일자 (최근 N일)", min_value=1, max_value=100, value=30, step=1)
    selected_stocks = st.multiselect(
        "종목 선택 (복수 선택 가능, 검색 가능)",
        options=stock_names,
        default=stock_names[:0] if stock_names else []
    )

    can_run_specific = file_exists and (len(selected_stocks) > 0)
    run_specific = st.button("🔎 뉴스 검색 (특정 기사만 보기) 실행", disabled=not can_run_specific)

    def crawl_mainnews_by_dates_for_stocks(stock_names, days: int = 30, max_pages_per_day: int = 200):
        results = []
        today = datetime.today().date()

        stock_norm_to_orig = {}
        stock_norm_set = set()
        for nm in stock_names or []:
            sn = normalize_text(nm)
            if sn:
                stock_norm_set.add(sn)
                stock_norm_to_orig.setdefault(sn, nm)

        if not stock_norm_set:
            return results

        pattern = re.compile("|".join(map(re.escape, stock_norm_set)))

        for i in range(days):
            d = today - timedelta(days=i)
            date_param = d.strftime("%Y%m%d")

            page = 1
            max_page = None

            while True:
                url = NEWS_BY_DATE_URL.format(date=date_param, page=page)
                try:
                    res = SESSION.get(url, timeout=10)
                    soup = BeautifulSoup(res.content, "lxml")
                except Exception as e:
                    st.warning(f"[{date_param}] 주요뉴스 요청 실패(page {page}): {e}")
                    break

                if page == 1:
                    nav_nums = [
                        int(a.get_text(strip=True))
                        for a in soup.select("table.Nnavi a")
                        if a.get_text(strip=True).isdigit()
                    ]
                    max_page = max(nav_nums) if nav_nums else 1

                items = soup.select("ul.newsList > li")
                if not items:
                    soup.decompose()
                    break

                for li in items:
                    a = li.select_one("dd.articleSubject a")
                    sm = li.select_one("dd.articleSummary")
                    if not (a and sm):
                        continue

                    title = a.get_text(strip=True)
                    href = a.get("href", "")
                    link = href if href.startswith("http") else "https://finance.naver.com" + href
                    summary = sm.get_text(" ", strip=True)

                    dt_tag = sm.select_one("span.wdate")
                    if dt_tag:
                        news_date_only = dt_tag.get_text(strip=True).split(" ")[0].replace(".", "-")
                    else:
                        news_date_only = d.strftime("%Y-%m-%d")

                    tkey = normalize_text(title)
                    skey = normalize_text(summary)
                    m = pattern.search(tkey) or pattern.search(skey)
                    if m:
                        hit_stock = stock_norm_to_orig.get(m.group(0))
                        if hit_stock:
                            results.append({
                                "종목명": hit_stock,
                                "뉴스": title,
                                "링크": link,
                                "요약": summary,
                                "뉴스날짜": news_date_only,
                            })

                soup.decompose()
                page += 1

                if max_page is not None and page > max_page:
                    break
                if page > max_pages_per_day:
                    break

        # 중복 제거 + 정렬
        results = list({(r["종목명"], r["뉴스"], r["뉴스날짜"]): r for r in results}.values())
        results.sort(key=lambda x: (x["뉴스날짜"], x["종목명"], x["뉴스"]))
        return results

    if run_specific:
        st.info(f"선택 종목: {', '.join(selected_stocks)} / 최근 {int(days_to_fetch)}일 기사 수집 중… (기사의 개수가 많을 수록 좀 걸려용)")
        st.info(f"시작 버튼을 연속해서 누르지 마세요 지금 실행중입니다.")
        news_list = crawl_mainnews_by_dates_for_stocks(selected_stocks, days=int(days_to_fetch))

        if len(news_list) == 0:
            st.warning("해당 기간에 선택한 종목명이 포함된 기사가 없습니다.")
        else:
            df_sel = pd.DataFrame(news_list).drop_duplicates(["종목명", "뉴스"])
            st.write("### 📄 수집 기사")
            st.dataframe(df_sel[["뉴스날짜", "종목명", "뉴스", "링크", "요약"]], use_container_width=True)

            # 종목별 차트
            st.write(f"## 🗓 선택 종목 최근 {int(days_to_fetch)}일 뉴스 차트")
            for stock in df_sel["종목명"].dropna().unique():
                code = get_code_from_name(stock)
                if not code:
                    st.warning(f"{stock} 종목코드 없음, 차트 생략")
                    continue
                try:
                    df_price = crawl_naver_daily_price(code, max_days=60)
                except Exception as e:
                    st.warning(f"{stock} 시세 크롤링 실패: {e}")
                    continue

                # 해당 종목 뉴스만 매핑
                date_news_map = {}
                for _, nrow in df_sel[df_sel["종목명"] == stock].iterrows():
                    dt = nrow["뉴스날짜"]
                    date_news_map.setdefault(dt, [])
                    if not any(n["title"] == nrow["뉴스"] for n in date_news_map[dt]):
                        date_news_map[dt].append({"title": nrow["뉴스"], "link": nrow["링크"]})

                plot_bollinger_20day(
                    df_price, stock, code,
                    key=f"bollinger_specific_{code}_{days_to_fetch}",
                    news_dict=date_news_map
                )

# ===== 일반 모드 =====
else:
    news_count = st.number_input("가져올 뉴스 기사 개수 입력 (표시용)", min_value=1, max_value=200, value=20, step=1)
    today_only = st.checkbox("금일 기사만", value=True)

    start_btn_disabled = not (file_exists and st.session_state.api_key is not None)
    start = st.button("🚀 시작 (기본 모드)", disabled=start_btn_disabled or st.session_state.is_running)

    def color_news(tag):
        if tag == "호재":
            return "color: red; font-weight: bold;"
        elif tag == "악재":
            return "color: blue; font-weight: bold;"
        elif tag == "중립":
            return "color: gray;"
        else:
            return ""

    # -------- 일반 모드: 메인뉴스 전체 페이지 순회 크롤러 --------
    def crawl_mainnews_all_pages(stock_names, today_only=True, max_pages=200):
        results = []
        stock_set = set(stock_names)
        today_str = datetime.today().strftime("%Y-%m-%d")

        page = 1
        while True:
            news_url = f"https://finance.naver.com/news/mainnews.naver?&page={page}"
            try:
                res = SESSION.get(news_url, timeout=10)
                soup = BeautifulSoup(res.content, "lxml")
            except Exception as e:
                st.error(f"네이버 뉴스 크롤링 오류 (페이지 {page}): {e}")
                return results

            items = soup.select("ul.newsList > li")
            if not items:
                return results

            hit_non_today = False

            for li in items:
                a = li.select_one("dd.articleSubject a")
                sm = li.select_one("dd.articleSummary")
                dt_tag = li.select_one("dd.articleSummary span.wdate")
                if not (a and dt_tag):
                    continue

                # 날짜
                news_date_only = dt_tag.get_text(strip=True).split(" ")[0].replace(".", "-")

                if today_only and news_date_only != today_str:
                    hit_non_today = True
                    continue

                # 제목/요약
                title = a.get_text(strip=True)
                link = a.get("href", "")
                link = link if link.startswith("http") else "https://finance.naver.com" + link
                summary = sm.get_text(" ", strip=True) if sm else ""

                # 제목과 요약 둘 다 검사
                stock = find_stock_in_text(title, stock_set) or find_stock_in_text(summary, stock_set)
                if stock:
                    results.append({
                        "종목명": stock,
                        "뉴스": title,
                        "링크": link,
                        "요약": summary,
                        "뉴스날짜": news_date_only
                    })

            if today_only and hit_non_today:
                return results

            page += 1
            if page > max_pages:
                return results

    if start:
        st.session_state.is_running = True
        st.info(f"기사 수집 중… (기사의 개수가 많을 수록 좀 걸려용)")
        st.info(f"시작 버튼을 연속해서 누르지 마세요 지금 실행중입니다.")
        news_results = crawl_mainnews_all_pages(
            stock_names=stock_names,
            today_only=today_only,
            max_pages=200,
        )

        if len(news_results) == 0:
            st.warning("뉴스에서 상장종목명이 포함된 기사가 없습니다.")
            st.session_state.is_running = False
            st.stop()
            
        news_results_display = news_results[:news_count]
        news_df = pd.DataFrame(news_results_display).drop_duplicates(["종목명", "뉴스"])

        openai_api_key = st.session_state.api_key

        news_df["뉴스판별"] = ""
        news_df["AI설명"] = ""
        today_str = datetime.today().strftime("%Y-%m-%d")

        for idx, row in news_df.iterrows():
            existing = get_existing_news(engine, row["뉴스"], row["종목명"])
            if existing:
                news_df.at[idx, "뉴스판별"] = existing["tag"]
                news_df.at[idx, "AI설명"] = existing["description"]
                news_df.at[idx, "카테고리"] = existing["category"]
                st.info(f"[DB] {row['종목명']}: {row['뉴스']} => {existing['tag']}")
            else:
                ai_result = classify_news(row["뉴스"], row["요약"])
                news_df.at[idx, "뉴스판별"] = ai_result["tag"]
                news_df.at[idx, "AI설명"] = ai_result["description"]
                news_df.at[idx, "카테고리"] = ai_result["category"]

                insert_news_to_db(engine,row["뉴스"],row["종목명"],ai_result["tag"],ai_result["category"],ai_result["description"],ai_result["power"],row["뉴스날짜"])
                st.info(f"[AI분석]{row['종목명']}: {row['뉴스']} => {ai_result['tag']}")
                time.sleep(0.05)

        # 재무
        finance_results = []
        for stock in set(news_df["종목명"]):
            code = get_code_from_name(stock)
            if code is None:
                st.warning(f"{stock} 종목코드 미발견, 재무 데이터 생략")
                continue
            url = f"https://finance.naver.com/item/main.nhn?code={code}"
            try:
                res = SESSION.get(url, timeout=10)
                soup = BeautifulSoup(res.content, "lxml")
            except Exception as e:
                st.warning(f"{stock} 재무 데이터 요청 실패: {e}")
                continue

            price = "N/A"
            try:
                price_tag = soup.select_one("p.no_today span.blind")
                if price_tag:
                    price = price_tag.get_text(strip=True).replace(",", "")
            except Exception:
                pass

            per = pbr = roe = eps = "N/A"
            table = soup.select_one("table.tb_type1_ifrs")
            if table:
                try:
                    rows = table.select("tr")
                    for r in rows:
                        th = r.find("th")
                        if not th:
                            continue
                        th_text = th.get_text(strip=True)
                        tds = r.find_all("td")
                        val = next((td.get_text(strip=True) for td in tds if td.get_text(strip=True)), "N/A")
                        if "PER" in th_text:
                            per = val
                        elif "PBR" in th_text:
                            pbr = val
                        elif "ROE" in th_text:
                            roe = val
                        elif "EPS" in th_text:
                            eps = val
                except Exception:
                    pass

            finance_results.append({"종목명": stock, "PER": per, "PBR": pbr, "ROE": roe, "EPS": eps, "현재가": price})

        df_finance = pd.DataFrame(finance_results)
        if not df_finance.empty:
            df_finance["PER_f"] = df_finance["PER"].apply(safe_float)
            df_finance["ROE_f"] = df_finance["ROE"].apply(safe_float)
            df_finance["EPS_f"] = df_finance["EPS"].apply(safe_float)
            df_finance["현재가_f"] = df_finance["현재가"].apply(safe_float)
            df_finance["예상주가"] = df_finance["EPS_f"] * 10
            df_finance["예상주가_표시"] = df_finance["예상주가"].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
            df_finance["상승여력"] = df_finance["예상주가"] - df_finance["현재가_f"]
            
            def format_sign(x):
                if pd.isnull(x): return "N/A"
                if x > 0: return f"+{x:,.0f}"
                if x < 0: return f"{x:,.0f}"
                return "0"

            df_finance["상승여력_표시"] = df_finance["상승여력"].apply(format_sign)
            df_finance["ROE"] = df_finance["ROE"].apply(lambda x: f"{x}%" if x != "N/A" and not str(x).endswith("%") else x)
            df_finance["현재가_표시"] = df_finance["현재가_f"].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")

        news_df["종목명"] = news_df["종목명"].astype(str)
        df_finance["종목명"] = df_finance["종목명"].astype(str)
        result_table = pd.merge(news_df, df_finance, on="종목명", how="left")
        result_table["오늘기사"] = result_table["뉴스판별"]

        show_cols = ["종목명", "뉴스", "요약", "뉴스날짜", "뉴스판별", "AI설명", "카테고리",
             "예상주가_표시", "현재가_표시", "상승여력_표시", "오늘기사", "링크"]

        for col in ["예상주가_표시", "현재가_표시", "상승여력_표시"]:
            if col not in result_table.columns:
                result_table[col] = "N/A"

        final_df = result_table[show_cols].rename(columns={
            "예상주가_표시": "예상주가", "현재가_표시": "현재가", "상승여력_표시": "예상주가-현재가"
        })

        def highlight_news(row):
            return [color_news(row["뉴스판별"])] * len(row)

        st.write("### 📰 오늘 종목 뉴스 전체")
        st.dataframe(final_df.style.apply(highlight_news, axis=1), use_container_width=True)

        # 호재 상위 5
        expecting_stocks = final_df[(final_df["뉴스판별"] == "호재") & (final_df["예상주가-현재가"] != "N/A")]
        def to_float(x):
            try: return float(str(x).replace(",", "").replace("+", ""))
            except Exception: return -9999999
        expecting_stocks_unique = expecting_stocks.sort_values("예상주가-현재가", ascending=False).drop_duplicates(subset=["종목명"], keep="first")
        top5_expect = expecting_stocks_unique.head(5)
        if not top5_expect.empty:
            st.write("### 🚀 가장 기대되는 종목 TOP 5 (호재 + 상승여력 높은 순)")
            st.dataframe(top5_expect[["종목명","뉴스","요약","뉴스판별","AI설명","예상주가","현재가","예상주가-현재가","링크"]])
        else:
            st.info("호재에 해당하는 종목이 없거나 상승여력 데이터가 부족합니다.")

        # 호재 차트
        st.write("## 📊 호재 종목 20일 볼린저 밴드 차트")
        for idx, row in expecting_stocks_unique.iterrows():
            stock_name = row["종목명"]; code = get_code_from_name(stock_name)
            if not code:
                st.warning(f"{stock_name} 종목코드 없음, 차트 생략"); continue
            try:
                df_price = crawl_naver_daily_price(code, max_days=60)
            except Exception as e:
                st.warning(f"{stock_name} 시세 크롤링 실패: {e}"); continue
            date_news_map = {}
            for _, nrow in news_df[news_df["종목명"] == stock_name].iterrows():
                dt = nrow["뉴스날짜"]; date_news_map.setdefault(dt, []).append({"title": nrow["뉴스"], "link": nrow["링크"]})
            plot_bollinger_20day(df_price, stock_name, code, key=f"bollinger_good_{code}_{idx}", news_dict=date_news_map)

        # 악재 차트
        st.write("## 📊 악재 종목 20일 볼린저 밴드 차트")
        bad_stocks = final_df[(final_df["뉴스판별"] == "악재")]
        bad_stocks_unique = bad_stocks.drop_duplicates(subset=["종목명"], keep="first")
        for idx, row in bad_stocks_unique.iterrows():
            stock_name = row["종목명"]; code = get_code_from_name(stock_name)
            if not code:
                st.warning(f"{stock_name} 종목코드 없음, 차트 생략"); continue
            try:
                df_price = crawl_naver_daily_price(code, max_days=60)
            except Exception as e:
                st.warning(f"{stock_name} 시세 크롤링 실패: {e}"); continue
            date_news_map = {}
            for _, nrow in news_df[news_df["종목명"] == stock_name].iterrows():
                dt = nrow["뉴스날짜"]; date_news_map.setdefault(dt, []).append({"title": nrow["뉴스"], "link": nrow["링크"]})
            plot_bollinger_20day(df_price, stock_name, code, key=f"bollinger_bad_{code}_{idx}", news_dict=date_news_map)

        st.session_state.is_running = False
