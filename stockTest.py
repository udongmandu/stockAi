import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import json
from datetime import datetime
from openai import OpenAI
import plotly.graph_objects as go
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

KRX_FILE = "krx_temp.xls"
CACHE_FILE = "cache_news_ai.json"

st.set_page_config(page_title="KRX 뉴스-AI 자동화", layout="centered")
st.title("KRX 상장종목 뉴스 + AI 분석 자동화")

# 환경변수에서 API 키 읽기
api_key_env = os.getenv("OPENAI_API_KEY")

if 'api_key' not in st.session_state:
    st.session_state.api_key = api_key_env

def submit_api_key():
    st.session_state.api_key = st.session_state.api_key_input

if st.session_state.api_key is None:
    st.text_input("API 키가 있는 ENV 파일을 받아 주세요.", type="password", key="api_key_input", on_change=submit_api_key)
else:
    st.success("✅ ENV 파일 인식 완료")

news_count = st.number_input("가져올 뉴스 기사 개수 입력", min_value=1, max_value=50, value=15, step=1)
today_only = st.checkbox("금일 기사만", value=False)

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

start_btn_disabled = not (file_exists and st.session_state.api_key is not None)
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
start = st.button("🚀 시작", disabled=start_btn_disabled or st.session_state.is_running)

def find_stock_in_text(text, stock_names):
    if not isinstance(text, str):
        return None
    for name in stock_names:
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

def get_code_from_name(stock_name):
    try:
        matched_rows = df_stocklist[df_stocklist['회사명'] == stock_name]
        if not matched_rows.empty:
            code = matched_rows.iloc[0]['종목코드']
            return str(code).zfill(6)
        return None
    except:
        return None

def safe_float(val):
    try:
        return float(str(val).replace(',', ''))
    except:
        return None

def crawl_naver_daily_price(stock_code, max_days=60):
    base_url = "https://finance.naver.com/item/sise_day.naver"
    all_rows = []
    page = 1

    while True:
        params = {'code': stock_code, 'page': page}
        res = requests.get(base_url, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'html.parser')

        table = soup.find('table', class_='type2')
        if not table:
            break

        rows = table.find_all('tr')
        data_rows = [row for row in rows if len(row.find_all('td')) == 7]
        if not data_rows:
            break

        for row in data_rows:
            cols = row.find_all('td')
            date = cols[0].get_text(strip=True).replace('.', '-')
            close = cols[1].get_text(strip=True).replace(',', '')
            open_price = cols[3].get_text(strip=True).replace(',', '')
            high = cols[4].get_text(strip=True).replace(',', '')
            low = cols[5].get_text(strip=True).replace(',', '')
            volume = cols[6].get_text(strip=True).replace(',', '')
            all_rows.append({
                '날짜': date,
                '종가': float(close) if close else None,
                '시가': float(open_price) if open_price else None,
                '고가': float(high) if high else None,
                '저가': float(low) if low else None,
                '거래량': int(volume) if volume else None
            })

            if len(all_rows) >= max_days:
                break

        if len(all_rows) >= max_days:
            break

        page += 1

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=['종가'])
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values('날짜')
    df.reset_index(drop=True, inplace=True)
    return df

def plot_bollinger_20day(df_price, stock_name, stock_code, key=None, news_dates=None):
    if len(df_price) < 20:
        st.warning(f"{stock_name} 시세 데이터가 부족해 볼린저 밴드를 그릴 수 없습니다.")
        return

    window = 20
    df_price[f'MA{window}'] = df_price['종가'].rolling(window=window).mean()
    df_price[f'STD{window}'] = df_price['종가'].rolling(window=window).std()
    df_price[f'Upper{window}'] = df_price[f'MA{window}'] + 2 * df_price[f'STD{window}']
    df_price[f'Lower{window}'] = df_price[f'MA{window}'] - 2 * df_price[f'STD{window}']

    dates = df_price['날짜']

    st.write(f"### {stock_name} 20일 볼린저 밴드")
    st.markdown(f"[주가 상세 보기](https://finance.naver.com/item/main.nhn?code={stock_code})")

    fig = go.Figure(
        data=[
            go.Scatter(x=dates, y=df_price['종가'], mode='lines', name='종가'),
            go.Scatter(x=dates, y=df_price[f'MA{window}'], mode='lines', name=f'MA{window}'),
            go.Scatter(x=dates, y=df_price[f'Upper{window}'], mode='lines', name='상단 밴드'),
            go.Scatter(x=dates, y=df_price[f'Lower{window}'], mode='lines', name='하단 밴드', fill='tonexty', fillcolor='rgba(200,200,200,0.2)'),
        ],
        layout=go.Layout(
            xaxis_title="날짜",
            yaxis_title="가격",
            xaxis=dict(range=[dates.min(), dates.max()]),
            yaxis=dict(autorange=True),
        )
    )

    if news_dates:
        for nd in news_dates:
            try:
                nd_dt = pd.to_datetime(nd)
                price_row = df_price[df_price['날짜'] == nd_dt]
                if not price_row.empty:
                    price = price_row.iloc[0]['종가']
                    fig.add_trace(go.Scatter(
                        x=[nd_dt],
                        y=[price],
                        mode='markers+text',
                        marker=dict(color='orange', size=12, symbol='star'),
                        text=["📰 뉴스"],
                        textposition="top center",
                        name=f"뉴스({nd})",
                        showlegend=False
                    ))
            except Exception:
                pass

    st.plotly_chart(fig, use_container_width=True, key=key)


if start:
    st.session_state.is_running = True
    try:
        try:
            df_stocklist = pd.read_excel(KRX_FILE, dtype=str, engine='xlrd')
        except Exception:
            df_stocklist = pd.read_excel(KRX_FILE, dtype=str)
        stock_names = set(df_stocklist['회사명'].dropna())
        st.info(f"상장종목 수: {len(stock_names)}개")
    except Exception as e:
        st.error(f"엑셀 파일 읽기 오류: {e}")
        st.session_state.is_running = False
        st.stop()

    news_results = []
    cnt = 0
    page = 1
    count_limit = news_count
    today_str = datetime.today().strftime("%Y-%m-%d")

    while cnt < count_limit:
        news_url = f"https://finance.naver.com/news/mainnews.naver?&page={page}"
        try:
            res = requests.get(news_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
        except Exception as e:
            st.error(f"네이버 뉴스 크롤링 오류 (페이지 {page}): {e}")
            break

        articles = soup.select('ul.newsList > li')
        if not articles:
            break

        for li in articles:
            if cnt >= count_limit:
                break
            article_subject = li.select_one('dd.articleSubject a')
            article_summary = li.select_one('dd.articleSummary')
            article_date_tag = li.select_one('dd.articleSummary span.wdate')

            if article_subject and article_date_tag:
                news_date = article_date_tag.get_text(strip=True)
                news_date_only = news_date.split(' ')[0]

                if today_only and news_date_only != today_str:
                    break

                title = article_subject.get_text(strip=True)
                link = article_subject['href']
                if not link.startswith('http'):
                    link = "https://finance.naver.com" + link
                summary = article_summary.get_text(" ", strip=True) if article_summary else ""
                stock = find_stock_in_text(title, stock_names) or find_stock_in_text(summary, stock_names)
                if stock:
                    news_results.append({
                        '종목명': stock,
                        '뉴스': title,
                        '링크': link,
                        '요약': summary,
                        '뉴스날짜': news_date_only
                    })
                    cnt += 1

        else:
            page += 1
            continue
        break

    if len(news_results) == 0:
        st.warning("뉴스에서 상장종목명이 포함된 기사가 없습니다.")
        st.session_state.is_running = False
        st.stop()

    news_df = pd.DataFrame(news_results).drop_duplicates(['종목명', '뉴스'])

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_dict = json.load(f)
    else:
        cache_dict = {}

    openai_api_key = st.session_state.api_key
    client = OpenAI(api_key=openai_api_key)

    news_df['뉴스판별'] = ""
    news_df['AI설명'] = ""

    for idx, row in news_df.iterrows():
        cached = cache_dict.get(row['뉴스'])
        if cached:
            if today_only and cached.get('뉴스날짜') != today_str:
                tag, explanation = classify_news(row['뉴스'], row['요약'])
                news_df.at[idx, '뉴스판별'] = tag
                news_df.at[idx, 'AI설명'] = explanation
                st.info(f"[AI분석]{row['종목명']}: {row['뉴스']} => {tag}")
                time.sleep(0.5)
                cache_dict[row['뉴스']] = {
                    '뉴스판별': tag,
                    'AI설명': explanation,
                    '뉴스날짜': row['뉴스날짜']
                }
            else:
                news_df.at[idx, '뉴스판별'] = cached['뉴스판별']
                news_df.at[idx, 'AI설명'] = cached['AI설명']
                st.info(f"[캐시] {row['종목명']}: {row['뉴스']} => {cached['뉴스판별']}")
        else:
            tag, explanation = classify_news(row['뉴스'], row['요약'])
            news_df.at[idx, '뉴스판별'] = tag
            news_df.at[idx, 'AI설명'] = explanation
            st.info(f"[AI분석]{row['종목명']}: {row['뉴스']} => {tag}")
            time.sleep(0.5)
            cache_dict[row['뉴스']] = {
                '뉴스판별': tag,
                'AI설명': explanation,
                '뉴스날짜': row['뉴스날짜']
            }

    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_dict, f, ensure_ascii=False, indent=2)

    finance_results = []
    unique_stocks = set(news_df['종목명'])
    for stock in unique_stocks:
        code = get_code_from_name(stock)
        if code is None:
            st.warning(f"{stock} 종목코드 미발견, 재무 데이터 생략")
            continue
        url = f"https://finance.naver.com/item/main.nhn?code={code}"
        try:
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(res.text, 'html.parser')
        except Exception as e:
            st.warning(f"{stock} 재무 데이터 요청 실패: {e}")
            continue

        price = 'N/A'
        try:
            price_tag = soup.select_one('p.no_today span.blind')
            if price_tag:
                price = price_tag.get_text(strip=True).replace(',', '')
        except:
            pass

        per = pbr = roe = eps = 'N/A'
        table = soup.select_one('table.tb_type1_ifrs')
        if table:
            try:
                rows = table.select('tr')
                for row in rows:
                    th = row.find('th')
                    if not th:
                        continue
                    th_text = th.get_text(strip=True)
                    tds = row.find_all('td')
                    val = next((td.get_text(strip=True) for td in tds if td.get_text(strip=True)), 'N/A')
                    if 'PER' in th_text:
                        per = val
                    elif 'PBR' in th_text:
                        pbr = val
                    elif 'ROE' in th_text:
                        roe = val
                    elif 'EPS' in th_text:
                        eps = val
            except:
                pass

        finance_results.append({
            '종목명': stock,
            'PER': per,
            'PBR': pbr,
            'ROE': roe,
            'EPS': eps,
            '현재가': price
        })

    df_finance = pd.DataFrame(finance_results)

    if not df_finance.empty:
        df_finance['PER_f'] = df_finance['PER'].apply(safe_float)
        df_finance['ROE_f'] = df_finance['ROE'].apply(safe_float)
        df_finance['EPS_f'] = df_finance['EPS'].apply(safe_float)
        df_finance['현재가_f'] = df_finance['현재가'].apply(safe_float)
        df_finance['예상주가'] = df_finance['EPS_f'] * 10
        df_finance['예상주가_표시'] = df_finance['예상주가'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else 'N/A')
        df_finance['상승여력'] = df_finance['예상주가'] - df_finance['현재가_f']

        def format_sign(x):
            if pd.isnull(x):
                return 'N/A'
            if x > 0:
                return f"+{x:,.0f}"
            elif x < 0:
                return f"{x:,.0f}"
            else:
                return "0"

        df_finance['상승여력_표시'] = df_finance['상승여력'].apply(format_sign)
        df_finance['ROE'] = df_finance['ROE'].apply(lambda x: f"{x}%" if x != 'N/A' and not str(x).endswith('%') else x)
        df_finance['현재가_표시'] = df_finance['현재가_f'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else 'N/A')

    news_df['종목명'] = news_df['종목명'].astype(str)
    df_finance['종목명'] = df_finance['종목명'].astype(str)
    result_table = pd.merge(news_df, df_finance, on='종목명', how='left')
    result_table['오늘기사'] = result_table['뉴스판별']

    show_cols = ['종목명', '뉴스', '요약', '뉴스날짜', '뉴스판별', 'AI설명',
                 '예상주가_표시', '현재가_표시', '상승여력_표시', '오늘기사', '링크']

    for col in ['예상주가_표시', '현재가_표시', '상승여력_표시']:
        if col not in result_table.columns:
            result_table[col] = 'N/A'

    final_df = result_table[show_cols].rename(columns={
        '예상주가_표시': '예상주가',
        '현재가_표시': '현재가',
        '상승여력_표시': '예상주가-현재가'
    })

    def color_news(tag):
        if tag == '호재':
            return 'color: red; font-weight: bold;'
        elif tag == '악재':
            return 'color: blue; font-weight: bold;'
        elif tag == '중립':
            return 'color: gray;'
        else:
            return ''

    def highlight_news(row):
        color = color_news(row['뉴스판별'])
        return [color] * len(row)

    st.write("### 📰 오늘 종목 뉴스 전체")
    st.dataframe(final_df.style.apply(highlight_news, axis=1), use_container_width=True)

    expecting_stocks = final_df[(final_df['뉴스판별'] == '호재') & (final_df['예상주가-현재가'] != 'N/A')]

    def to_float(x):
        try:
            return float(str(x).replace(',', '').replace('+',''))
        except:
            return -9999999

    expecting_stocks_unique = expecting_stocks.sort_values('예상주가-현재가', ascending=False) \
                                             .drop_duplicates(subset=['종목명'], keep='first')

    top5_expect = expecting_stocks_unique.head(5)

    if not top5_expect.empty:
        st.write("### 🚀 가장 기대되는 종목 TOP 5 (호재 + 상승여력 높은 순)")
        st.dataframe(top5_expect[[
            '종목명', '뉴스', '요약', '뉴스판별', 'AI설명', '예상주가', '현재가', '예상주가-현재가', '링크'
        ]])
    else:
        st.info("호재에 해당하는 종목이 없거나 상승여력 데이터가 부족합니다.")

    st.write("## 📊 호재 종목 20일 볼린저 밴드 차트")
    for idx, row in expecting_stocks_unique.iterrows():
        stock_name = row['종목명']
        code = get_code_from_name(stock_name)
        if not code:
            st.warning(f"{stock_name} 종목코드 없음, 차트 생략")
            continue
        try:
            df_price = crawl_naver_daily_price(code, max_days=60)
        except Exception as e:
            st.warning(f"{stock_name} 시세 크롤링 실패: {e}")
            continue

        news_dates_for_stock = news_df[
            (news_df['종목명'] == stock_name) & (news_df['뉴스판별'] == '호재')
        ]['뉴스날짜'].unique().tolist()

        plot_bollinger_20day(df_price, stock_name, code, key=f"bollinger_{code}_{idx}", news_dates=news_dates_for_stock)

    def to_html_download(df):
        html = df.to_html(index=False)
        b64 = html.encode('utf-8')
        return b64

    b64_html = to_html_download(final_df)
    st.download_button(
        label="📥 HTML 다운로드",
        data=b64_html,
        file_name="stock_news_ai_result.html",
        mime="text/html"
    )

    st.session_state.is_running = False
