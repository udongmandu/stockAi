import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import json
from datetime import datetime, timedelta
from openai import OpenAI
import plotly.graph_objects as go
from dotenv import load_dotenv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# .env 파일 로드
load_dotenv()

KRX_FILE = "krx_temp.xls"
CACHE_FILE = "cache_news_ai.json"

# 날짜별 섹션 뉴스(일반 모드)
NEWS_BY_DATE_URL = "https://finance.naver.com/news/mainnews.naver?&date={date}&page={page}"

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
    token = raw.split()[0]  # 'YYYY-MM-DD' or 'MM-DD'
    parts = token.split("-")
    if len(parts) == 2:  # 'MM-DD' -> attach current year
        year = datetime.today().year
        token = f"{year}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
    return datetime.strptime(token, "%Y-%m-%d").date()

# ---------- 회사뉴스 날짜 파싱 ----------
def parse_company_news_date(s: str) -> datetime.date:
    # 예: "2025.08.10 13:24"
    s = (s or "").strip()
    s = s.replace("-", ".").replace("/", ".")
    token = s.split()[0]  # "YYYY.MM.DD"
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
    prompt = f"""아래 뉴스가 해당 기업에 호재(상승 가능성), 악재(하락 가능성), 중립 중 어떤 영향을 미칠지 한글로 단답(호재/악재/중립)과 이유(1문장)를 알려줘.
뉴스: {news_text}"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
        if answer.startswith("호재"):
            tag = "호재"
        elif answer.startswith("악재"):
            tag = "악재"
        else:
            tag = "중립"
        return tag, answer
    except Exception as e:
        return "분석불가", f"API 오류: {e}"

def safe_float(val):
    try:
        return float(str(val).replace(",", ""))
    except Exception:
        return None

def crawl_naver_daily_price(stock_code, max_days=60):
    return crawl_naver_daily_price_cached(stock_code, max_days=max_days)

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
toggle_label = "↩ 일반 모드로 전환" if st.session_state.specific_mode else "📰 특정 기사만 보기 (미완)"
if st.button(toggle_label):
    st.session_state.specific_mode = not st.session_state.specific_mode

# ------------------- 속도 개선 도우미 -------------------
@st.cache_data(ttl=600)
def get_last_page_for_date(date_param: str) -> int:
    """
    해당 날짜(date=YYYYMMDD)의 mainnews 마지막 페이지 번호를 추정.
    - '맨뒤' 링크가 있으면 그 page 파라미터
    - 없으면 페이징 숫자 링크 중 최대값
    """
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
                "종목명": None,  # 나중에 채움
                "뉴스": title,
                "링크": link,
                "요약": summary,
                "뉴스날짜": news_date_only,
            })
    return results

# ------------------- 모드별 UI/로직 -------------------
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# ===== 특정 기사만 보기 모드 (날짜 섹션 전체 순회 + 종목명 필터) =====
if st.session_state.specific_mode:
    st.subheader("📰 특정 기사만 보기")

    days_to_fetch = st.number_input("가져올 뉴스 일자 (최근 N일)", min_value=1, max_value=100, value=30, step=1)
    selected_stock = st.selectbox("종목 선택 (검색 가능)", options=stock_names, index=0 if stock_names else None)

    can_run_specific = file_exists and (selected_stock is not None)
    run_specific = st.button("🔎 뉴스 검색 (특정 기사만 보기) 실행", disabled=not can_run_specific)

    def crawl_mainnews_by_dates_for_stock(stock_name: str, days: int = 30, max_pages_per_day: int = 200):
        """
        주요뉴스(mainnews)에서 date=YYYYMMDD의 1..last_page를 병렬 크롤링.
        각 li에서 제목/요약 둘 다에 stock_name(정규화)이 등장하면 수집.
        """
        results = []
        today = datetime.today().date()
        stock_key_norm = normalize_text(stock_name)
        MAX_WORKERS = 12  # 네트워크/머신 환경 따라 8~16 사이에서 조절 추천

        for i in range(days):
            d = today - timedelta(days=i)
            date_param = d.strftime("%Y%m%d")
            fallback_date_str = d.strftime("%Y-%m-%d")

            try:
                last_page = get_last_page_for_date(date_param)
                last_page = min(last_page, max_pages_per_day)
            except Exception as e:
                st.warning(f"[{date_param}] 마지막 페이지 파악 실패: {e} (1페이지만 시도)")
                last_page = 1

            page_indices = list(range(1, last_page + 1))

            # 병렬 크롤링
            page_results = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                future_to_page = {
                    ex.submit(
                        lambda p: parse_mainnews_page(
                            SESSION.get(NEWS_BY_DATE_URL.format(date=date_param, page=p), timeout=10).content,
                            fallback_date_str, stock_key_norm
                        ), page
                    ): page for page in page_indices
                }
                for fut in as_completed(future_to_page):
                    try:
                        part = fut.result() or []
                        page_results.extend(part)
                    except Exception:
                        # 개별 페이지 실패는 무시
                        pass

            # 종목명 채우고 합치기
            for r in page_results:
                r["종목명"] = stock_name
            results.extend(page_results)

        # 중복 제거 + 정렬
        results = list({(r["뉴스"], r["뉴스날짜"]): r for r in results}.values())
        results.sort(key=lambda x: (x["뉴스날짜"], x["뉴스"]))
        return results

    if run_specific:
        st.info(f"선택 종목: {selected_stock} / 최근 {int(days_to_fetch)}일 기사 수집 중… (가져올 뉴스의 갯수가 많을 수록 느립니다요)")
        news_list = crawl_mainnews_by_dates_for_stock(selected_stock, days=int(days_to_fetch))
        if len(news_list) == 0:
            st.warning("해당 기간에 해당 종목명이 포함된 기사가 없습니다.")
        else:
            df_sel = pd.DataFrame(news_list).drop_duplicates(["종목명", "뉴스"])
            st.write("### 📄 수집 기사")
            st.dataframe(df_sel[["뉴스날짜", "뉴스", "링크", "요약"]], use_container_width=True)

            # 뉴스-차트 매핑
            date_news_map = {}
            for _, nrow in df_sel.iterrows():
                dt = nrow["뉴스날짜"]
                date_news_map.setdefault(dt, [])
                if not any(n["title"] == nrow["뉴스"] for n in date_news_map[dt]):
                    date_news_map[dt].append({"title": nrow["뉴스"], "link": nrow["링크"]})

            # 차트용 가격 (종목코드 필요)
            code = get_code_from_name(selected_stock)
            if not code:
                st.warning(f"{selected_stock} 종목코드 없음, 차트 생략")
            else:
                try:
                    df_price = crawl_naver_daily_price(code, max_days=60)
                except Exception as e:
                    st.warning(f"{selected_stock} 시세 크롤링 실패: {e}")
                    df_price = None
                if df_price is not None:
                    st.write(f"## 🗓 선택 종목 최근 {int(days_to_fetch)}일 뉴스 차트")
                    plot_bollinger_20day(df_price, selected_stock, code,
                                         key=f"bollinger_specific_{code}_{days_to_fetch}",
                                         news_dict=date_news_map)

# ===== 일반 모드 =====
else:
    news_count = st.number_input("가져올 뉴스 기사 개수 입력 (표시용)", min_value=1, max_value=200, value=20, step=1)
    today_only = st.checkbox("금일 기사만", value=True)
    graph_show_in_cache = st.checkbox("지금 까지 모든 데이터 보기", value=False)

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
        """
        네이버 금융 메인뉴스를 page=1부터 기사 없을 때까지 전부 순회.
        제목/요약에 종목명이 포함되면 수집.
        today_only=True면 '오늘 날짜'가 아닌 기사부터는 이후 페이지 순회 조기 종료.
        """
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
                return results  # 실패 시 조기 반환

            items = soup.select("ul.newsList > li")
            if not items:
                return results  # 더 이상 페이지 없음 -> 즉시 반환

            hit_non_today = False  # 오늘만 보기일 때, 비-오늘 기사 만나면 이후 페이지 중단

            for li in items:
                a = li.select_one("dd.articleSubject a")
                sm = li.select_one("dd.articleSummary")
                dt_tag = li.select_one("dd.articleSummary span.wdate")
                if not (a and dt_tag):
                    continue

                # 날짜
                news_date_only = dt_tag.get_text(strip=True).split(" ")[0].replace(".", "-")

                # 오늘만 보기면, 비-오늘 기사부터는 종료 플래그
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
                return results  # 오늘 기사 끝났으니 즉시 반환

            page += 1
            if page > max_pages:
                return results  # 안전장치

    if start:
        st.session_state.is_running = True

        # ▼▼▼ 일반 모드 수집
        news_results = crawl_mainnews_all_pages(
            stock_names=stock_names,
            today_only=today_only,
            max_pages=200,
        )

        if len(news_results) == 0:
            st.warning("뉴스에서 상장종목명이 포함된 기사가 없습니다.")
            st.session_state.is_running = False
            st.stop()

        # 화면 표시용으로 상위 news_count개만
        news_results_display = news_results[:news_count]
        news_df = pd.DataFrame(news_results_display).drop_duplicates(["종목명", "뉴스"])

        # 캐시
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache_dict = json.load(f)
        else:
            cache_dict = {}

        openai_api_key = st.session_state.api_key
        client = OpenAI(api_key=openai_api_key)

        news_df["뉴스판별"] = ""
        news_df["AI설명"] = ""
        today_str = datetime.today().strftime("%Y-%m-%d")

        for idx, row in news_df.iterrows():
            cached = cache_dict.get(row["뉴스"])
            if cached:
                if today_only and cached.get("뉴스날짜") != today_str:
                    tag, explanation = classify_news(row["뉴스"], row["요약"])
                    news_df.at[idx, "뉴스판별"] = tag
                    news_df.at[idx, "AI설명"] = explanation
                    st.info(f"[AI분석]{row['종목명']}: {row['뉴스']} => {tag}")
                    time.sleep(0.05)
                    cache_dict[row["뉴스"]] = {"뉴스판별": tag, "AI설명": explanation, "뉴스날짜": row["뉴스날짜"]}
                else:
                    news_df.at[idx, "뉴스판별"] = cached["뉴스판별"]
                    news_df.at[idx, "AI설명"] = cached["AI설명"]
                    st.info(f"[캐시] {row['종목명']}: {row['뉴스']} => {cached['뉴스판별']}")
            else:
                tag, explanation = classify_news(row["뉴스"], row["요약"])
                news_df.at[idx, "뉴스판별"] = tag
                news_df.at[idx, "AI설명"] = explanation
                st.info(f"[AI분석]{row['종목명']}: {row['뉴스']} => {tag}")
                time.sleep(0.05)
                cache_dict[row["뉴스"]] = {"뉴스판별": tag, "AI설명": explanation, "뉴스날짜": row["뉴스날짜"]}

        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_dict, f, ensure_ascii=False, indent=2)

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

        show_cols = ["종목명", "뉴스", "요약", "뉴스날짜", "뉴스판별", "AI설명",
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
        expecting_stocks_unique = expecting_stocks.sort_values("예상주가-현재가", ascending=False)\
                                                 .drop_duplicates(subset=["종목명"], keep="first")
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
            if os.path.exists(CACHE_FILE):
                for news_text, val in cache_dict.items():
                    dt = val.get("뉴스날짜")
                    if dt and (stock_name in news_text):
                        date_news_map.setdefault(dt, [])
                        if not any(n["title"] == news_text for n in date_news_map[dt]):
                            date_news_map[dt].append({"title": news_text, "link": None})
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
            if os.path.exists(CACHE_FILE):
                for news_text, val in cache_dict.items():
                    dt = val.get("뉴스날짜")
                    if dt and (stock_name in news_text):
                        date_news_map.setdefault(dt, [])
                        if not any(n["title"] == news_text for n in date_news_map[dt]):
                            date_news_map[dt].append({"title": news_text, "link": None})
            plot_bollinger_20day(df_price, stock_name, code, key=f"bollinger_bad_{code}_{idx}", news_dict=date_news_map)

        # 캐시 전체 보기
        if graph_show_in_cache:
            st.write("## 지금까지의 데이터")
            cached_rows = []
            for news_title, val in cache_dict.items():
                stock = find_stock_in_text(news_title, set(stock_names))
                if stock:
                    cached_rows.append({
                        "종목명": stock, "뉴스": news_title, "링크": None, "요약": None,
                        "뉴스날짜": val.get("뉴스날짜"), "뉴스판별": val.get("뉴스판별", "분석불가"), "AI설명": val.get("AI설명", "")
                    })
            if len(cached_rows) == 0:
                st.info("캐시에 종목명이 포함된 저장 데이터가 없습니다.")
            else:
                cached_df = pd.DataFrame(cached_rows).drop_duplicates(["종목명","뉴스"])
                def _color(tag):
                    if tag == "호재": return "color: red; font-weight: bold;"
                    if tag == "악재": return "color: blue; font-weight: bold;"
                    if tag == "중립": return "color: gray;"
                    return ""
                st.write("### 🗂 캐시된 전체 뉴스")
                st.dataframe(
                    cached_df[["종목명","뉴스","뉴스날짜","뉴스판별","AI설명"]].style.apply(
                        lambda row: [_color(row["뉴스판별"])]*5, axis=1
                    ),
                    use_container_width=True
                )
                for idx, stock_name in enumerate(cached_df["종목명"].dropna().unique().tolist()):
                    code = get_code_from_name(stock_name)
                    if not code:
                        st.warning(f"{stock_name} 종목코드 없음, 차트 생략"); continue
                    try:
                        df_price = crawl_naver_daily_price(code, max_days=60)
                    except Exception as e:
                        st.warning(f"{stock_name} 시세 크롤링 실패: {e}"); continue
                    date_news_map = {}
                    subset = cached_df[cached_df["종목명"] == stock_name]
                    for _, nrow in subset.iterrows():
                        dt = nrow["뉴스날짜"]
                        if dt:
                            date_news_map.setdefault(dt, [])
                            if not any(n["title"] == nrow["뉴스"] for n in date_news_map[dt]):
                                date_news_map[dt].append({"title": nrow["뉴스"], "link": None})
                    plot_bollinger_20day(df_price, stock_name, code, key=f"bollinger_cache_{code}_{idx}", news_dict=date_news_map)

        st.session_state.is_running = False
