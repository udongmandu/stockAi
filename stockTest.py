import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
from openai import OpenAI

st.set_page_config(page_title="KRX 뉴스-AI 자동화", layout="centered")

KRX_FILE = "krx_temp.xls"

st.title("KRX 상장종목 뉴스 + AI 분석 자동화")

# API 키와 뉴스 개수 입력란 (입력 완료 시 숨김 처리)
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'news_count' not in st.session_state:
    st.session_state.news_count = 15

def submit_api_key():
    st.session_state.api_key = st.session_state.api_key_input

def submit_news_count():
    st.session_state.news_count = st.session_state.news_count_input

if st.session_state.api_key is None:
    st.text_input("OpenAI API 키를 입력하세요", type="password", key="api_key_input", on_change=submit_api_key)
else:
    st.success("✅ API 키 입력 완료")

if st.session_state.news_count is None:
    st.number_input("가져올 뉴스 기사 개수 입력", min_value=1, max_value=50, value=10, step=1, key="news_count_input", on_change=submit_news_count)
else:
    st.info(f"가져올 뉴스 기사 개수: {st.session_state.news_count}")

# KRX 파일 존재 여부 체크
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
start = st.button("🚀 시작", disabled=start_btn_disabled)

if start:
    with st.spinner("AI 분석 및 데이터 처리중..."):
        # 엑셀 파일 읽기
        try:
            try:
                df_stocklist = pd.read_excel(KRX_FILE, dtype=str, engine='xlrd')
            except Exception:
                df_stocklist = pd.read_excel(KRX_FILE, dtype=str)
            stock_names = set(df_stocklist['회사명'].dropna())
            st.info(f"상장종목 수: {len(stock_names)}개")
        except Exception as e:
            st.error(f"엑셀 파일 읽기 오류: {e}")
            st.stop()

        # 네이버 주요 뉴스 크롤링
        news_url = "https://finance.naver.com/news/mainnews.naver"
        try:
            res = requests.get(news_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
        except Exception as e:
            st.error(f"네이버 뉴스 크롤링 오류: {e}")
            st.stop()

        def find_stock_in_text(text, stock_names):
            if not isinstance(text, str):
                return None
            for name in stock_names:
                if name in text:
                    return name
            return None

        news_results = []
        count_limit = st.session_state.news_count or 10
        cnt = 0
        for li in soup.select('ul.newsList > li'):
            if cnt >= count_limit:
                break
            article_subject = li.select_one('dd.articleSubject a')
            article_summary = li.select_one('dd.articleSummary')
            if article_subject:
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
                        '요약': summary
                    })
                    cnt += 1

        if len(news_results) == 0:
            st.warning("뉴스에서 상장종목명이 포함된 기사가 없습니다.")
            st.stop()

        news_df = pd.DataFrame(news_results).drop_duplicates(['종목명', '뉴스'])

        st.write("### 📰 오늘 종목 뉴스 미리보기", news_df[['종목명', '뉴스', '요약']].head(5))

        # OpenAI API 클라이언트 초기화
        openai_api_key = st.session_state.api_key
        client = OpenAI(api_key=openai_api_key)

        # AI 분석 함수
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

        news_df['뉴스판별'] = ""
        news_df['AI설명'] = ""

        for idx, row in news_df.iterrows():
            tag, explanation = classify_news(row['뉴스'], row['요약'])
            news_df.at[idx, '뉴스판별'] = tag
            news_df.at[idx, 'AI설명'] = explanation
            st.info(f"[AI분석]{row['종목명']}: {row['뉴스']} => {tag}")

            time.sleep(0.5)  # API 과부하 방지

        # 네이버 재무 데이터 조회 함수
        def get_code_from_name(stock_name):
            try:
                # krx_temp.xls의 '종목코드' 컬럼과 매핑 시도 (회사명 기준)
                matched_rows = df_stocklist[df_stocklist['회사명'] == stock_name]
                if not matched_rows.empty:
                    code = matched_rows.iloc[0]['종목코드']
                    return str(code).zfill(6)
                return None
            except:
                return None

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

        # 숫자 변환 함수
        def safe_float(val):
            try:
                return float(str(val).replace(',', ''))
            except:
                return None

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

        # 결과 병합
        news_df['종목명'] = news_df['종목명'].astype(str)
        df_finance['종목명'] = df_finance['종목명'].astype(str)
        result_table = pd.merge(news_df, df_finance, on='종목명', how='left')
        result_table['오늘기사'] = result_table['뉴스판별']

        # 컬럼 준비
        show_cols = ['종목명', '뉴스', '요약', '뉴스판별', 'AI설명',
                     '예상주가_표시', '현재가_표시', '상승여력_표시', '오늘기사', '링크']

        # 컬럼이 존재하지 않는 경우 N/A 채움
        for col in ['예상주가_표시', '현재가_표시', '상승여력_표시']:
            if col not in result_table.columns:
                result_table[col] = 'N/A'

        final_df = result_table[show_cols].rename(columns={
            '예상주가_표시': '예상주가',
            '현재가_표시': '현재가',
            '상승여력_표시': '예상주가-현재가'
        })

        st.write("## 🚩 결과표 (오늘의 종목별 뉴스 AI 분석 + 재무데이터)")
        st.dataframe(final_df)

        # HTML 다운로드 버튼
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
