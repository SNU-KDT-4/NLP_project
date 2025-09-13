import os
import re
import pandas as pd
from bs4 import BeautifulSoup

class AuditReportParser:
    def parse_html(self, file_path):
        """HTML 파일을 읽고 BeautifulSoup 객체 반환 (인코딩 자동처리)"""
        for encoding in ('utf-8', 'cp949', 'euc-kr'):
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return BeautifulSoup(f.read(), 'html.parser')
            except Exception:
                continue
        raise ValueError('Unknown encoding or file error')
    
    def extract_sections(self, soup):
        """주요 섹션(제목 블록 등) 텍스트 추출"""
        sections = {}
        # h1~h3 혹은 <b>, <strong> 등으로 주요 구간 구분
        for header_tag in soup.find_all(['h1','h2','h3','b','strong']):
            section_title = header_tag.get_text(strip=True)
            # 인접한 div, p 등 본문 텍스트 추출
            next_node = header_tag.find_next_sibling(['div','p'])
            body_text = next_node.get_text(strip=True) if next_node else ''
            sections[section_title] = body_text
        return sections
    
    def extract_tables(self, soup):
        """모든 테이블을 pandas.DataFrame 리스트로 추출"""
        tables = []
        for tbl in soup.find_all('table'):
            rows = []
            for tr in tbl.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td','th'])]
                if cells: rows.append(cells)
            if rows:
                # 첫행(0) 헤더, 나머지 데이터
                df = pd.DataFrame(rows)
                tables.append(df)
        return tables
    
    def normalize_text(self, text):
        """금융 텍스트 정규화 (숫자, 단위, 특수문자 등 처리)"""
        text = re.sub(r'[,\u200b]', '', text)       # 콤마/이상한 공백 제거
        text = re.sub(r'\\n|\\r|\\t', ' ', text)    # 줄바꿈/탭 제거
        text = re.sub(r'[ ]+', ' ', text)           # 다중공백 정리
        text = re.sub(r'[()\\[\\]\\-]', '', text)   # 괄호 등 제거
        # 한글 "단위: 백만원"/"단위: 원" 등 판별, 단위 통일로 변환 가능
        text = text.replace('백만원','*1_000_000').replace('천원','*1_000')
        return text.strip()




# 위에 제공한 AuditReportParser 클래스를 먼저 정의 후에 실행하세요!
parser = AuditReportParser()

# ❶ HTML 파일 파싱
soup = parser.parse_html("감사보고서_2014.htm")

# ❷ 섹션별 텍스트 추출
sections = parser.extract_sections(soup)
print("==== 주요 섹션 ====")
for title, content in sections.items():
    print(f"\n[{title}]\n{parser.normalize_text(content[:500])} ...")  # 길이제한/정규화 후 샘플 출력

# ❸ 표 데이터 추출
tables = parser.extract_tables(soup)
print("\n==== 추출된 표(샘플) ====")
for idx, df in enumerate(tables[:50]):  # 처음 두 개 표만 샘플로
    print(f"\n[표 {idx+1}]\n", df.head())

# ❹ 특정 표 CSV로 저장도 가능
if tables:
    tables[0].to_csv("first_table_sample.csv", index=False)







tables = []
for tbl in soup.find_all('table'):
    rows = tbl.find_all('tr')
    if len(rows) >= 2 and any(tr.find_all('th') for tr in rows[:2]):
        arr = [[td.get_text(strip=True) for td in tr.find_all(['td','th'])] for tr in rows]
        if any(arr): tables.append(arr)






print(f"추출된 표 개수: {len(tables)}")
for idx, table in enumerate(tables[:5]):  # 처음 5개 표만 샘플 출력
    df = pd.DataFrame(table)
    print(f"\n[표 {idx+1}]\n", df.head())
