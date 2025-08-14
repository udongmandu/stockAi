# stockAi
KRX 상장종목 뉴스 + AI 분석 자동화 프로그램

필요 환경 : python3 이상의 버젼, 몇몇 파이썬 라이브러리, openAi key 가 들어간 ENV 파일

사용 방법 : 

  % 파이썬이 다운로드 되어있지 않다면 https://www.python.org/downloads/ 해당 주소에서 다운로드 %

  
  % ENV 파일은 나에게 받도록 혹은 본인이 OPEN Ai 키를 가지고 있다면, ' OPENAI_API_KEY= 본인의 키 ' 내용으로 .env 파일을 만들어 넣도록 하자 %


  ## 해당 파일 내부의 '윈도우 전용 실행기.bat' 실행 (mac 이라면 start_Mac.sh 실행)



  ## 가져올 기사 갯수 설정 후 (최대 100개)
  <img width="1253" height="831" alt="image" src="https://github.com/user-attachments/assets/8291e9bf-b2b7-45d9-8c72-e61791658acb" />


  ## [일반모드] 가져온 기사들을 한국거래소에서 제공되는 상장된 데이터와 매핑되는 지 파악 후, 해당 기사를 AI에게 주어 해당 기사가 호재 / 악재 / 중립 인지를 판단 하여 화면에 띄워줌
  + 한번 AI 가 판단한 기사는 캐시로 저장되어 새로운 기사가 아닌 이상 AI는 한번만 사용되며('캐시' 라고 써있는게 캐싱된 기사, 'AI 판단' 이라고 써있는게 방금 AI가 판단한 것) 빠르게 화면에 출력되도록 설정. (이유: AI 쓸 때마다 돈 나감)
  + 나오는 내용은 AI 가 호재 / 악재 / 중립 을 판단 하고, 판단 된 주식의 PER, PBR, ROE, EPS 를 통해 정상 주가의 값을 계산 후, 현재 주가로부터 앞으로 얼마 성장 가능성을 표시
  + 위 내용을 정리한 데이터 표를 표시
  + 호재 / 악재 에 해당하는 주식들의 주가 및 볼린저 밴드를 표시하였음.
  <img width="738" height="793" alt="image" src="https://github.com/user-attachments/assets/cba11724-003e-4b8d-9ad4-2ec23ac2800a" />
  <img width="762" height="746" alt="image" src="https://github.com/user-attachments/assets/86106d82-5886-4fd9-81d5-8a5339439c28" />
  <img width="838" height="767" alt="image" src="https://github.com/user-attachments/assets/ea70b01b-5651-41d7-821c-013d1752ae1d" />


  ## [특정기사 모드] 입력된 N일 동안의 지정된 주식에 대한 기사들을 모두 가져와 볼린저 밴드와 함께 화면에 표시
  + 해당 기능은 추후에 AI 인식을 추가할 계획이며 아직 추가 되지 않아 BETA로 표시해 두었음
  <img width="809" height="734" alt="image" src="https://github.com/user-attachments/assets/46fa3e54-6ce1-4549-a2a7-a0132f717474" />
  <img width="807" height="504" alt="image" src="https://github.com/user-attachments/assets/3790767c-4960-42e4-a451-84c5ffdd7193" />
  <img width="847" height="822" alt="image" src="https://github.com/user-attachments/assets/63a0176c-2fa7-4dba-a194-27d3289a1963" />

