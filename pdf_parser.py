### 경영진 의견

# -*- coding: utf-8 -*-
import re, json, unicodedata as ud
from pathlib import Path
import pdfplumber

# ===== 경로 =====
DIR_PDFS = Path("사업보고서_preprocessed/경영진의견")
OUT_DIR  = Path("jsonl")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 파일명 매칭 =====
RE_NAME = re.compile(r"경영진\s*의견[ _-]*(\d{4})\.pdf$", re.IGNORECASE)

# ===== 구간 추출용 =====
RE_START  = re.compile(r"^\s*2\.\s*개요\s*$|^\s*2\.\s*개요\b", re.MULTILINE)
RE_FOOT   = re.compile(r"전자공시시스템\s+dart\.fss\.or\.kr\s+Page\s+\d+")
RE_SECTION = re.compile(r"\((?:[^)]*부문[^)]*|Harman|SDC)\)")
RE_FOOTER  = RE_FOOT  # 사업부문 처리 시에도 동일하게 footer 제거에 사용

# ===== 텍스트 추출 =====
def build_non_table_text(page) -> str:
    """표 제외 텍스트 추출 (개요 추출에서 사용)"""
    tables = page.find_tables(table_settings={
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
        "snap_tolerance": 3,
    }) or []
    bboxes = [t.bbox for t in tables]

    def in_table(b):
        x0, top, x1, bottom = b
        for (tx0, ttop, tx1, tbottom) in bboxes:
            if (x0 >= tx0 and x1 <= tx1 and top >= ttop and bottom <= tbottom):
                return True
        return False

    def keep_obj(obj):
        if obj.get("object_type") == "char":
            bbox = (obj["x0"], obj["top"], obj["x1"], obj["bottom"])
            return not in_table(bbox)
        return True

    filtered = page.filter(keep_obj)
    return filtered.extract_text(x_tolerance=1, y_tolerance=3) or ""

def extract_text_all(pdf_path: Path) -> str:
    """페이지 전체 텍스트 단순 합치기 (사업부문 추출에서 사용)"""
    parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
    return "\n".join(parts)

# ===== 개요 슬라이스 =====
def slice_overview(full_text: str) -> str | None:
    m_start = RE_START.search(full_text)
    if not m_start:
        return None
    m_foot = RE_FOOT.search(full_text, m_start.end())
    end = m_foot.start() if m_foot else len(full_text)
    return full_text[m_start.end():end].strip()

# ===== 사업부문 슬라이스 =====
def slice_sections(text: str) -> list[tuple[str,str]]:
    out = []
    matches = list(RE_SECTION.finditer(text))
    if not matches:
        return out
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        title = m.group().strip("()")
        out.append((title, chunk))
    return out

# ===== 텍스트 정리 =====
def normalize_text(raw: str) -> str:
    s = raw
    s = s.replace("\xa0", " ").replace("\u200b", "")
    s = RE_FOOTER.sub(" ", s)                # footer 제거
    s = re.sub(r"^\s*-\s*", "", s, flags=re.MULTILINE)  # 맨 앞 bullet 제거
    s = re.sub(r"([가-힣])\s*[\r\n]+\s*([가-힣])", r"\1\2", s)  # 한글 단어 찢김 복원
    s = re.sub(r"[\r\n]+", " ", s)           # 줄바꿈 → 공백
    s = re.sub(r" {2,}", " ", s).strip()     # 공백 정리
    s = s.replace("당사", "삼성전자")         # '당사' 치환
    return s

# ===== 메인 =====
def main():
    pdfs = list(DIR_PDFS.glob("*.pdf"))
    for pdf_path in sorted(pdfs):
        name_nfc = ud.normalize("NFC", pdf_path.name)
        m = RE_NAME.search(name_nfc)
        if not m:
            continue
        year = int(m.group(1))

        # 1) 개요 추출
        with pdfplumber.open(str(pdf_path)) as pdf:
            parts = [build_non_table_text(p).strip()
                     for p in pdf.pages if build_non_table_text(p).strip()]
        full_text_overview = "\n\n".join(parts)
        chunk = slice_overview(full_text_overview)
        overview_records = []
        if chunk:
            norm = normalize_text(chunk)
            overview_records.append({
                "id": f"감사보고서-{year}-경영진의견",
                "document": "[개요] " + norm,
                "metadata": {
                    "year": year,
                    "source": name_nfc,
                    "section": "경영진의견-개요"
                }
            })

        # 2) 사업부문 추출
        full_text_sections = extract_text_all(pdf_path)
        sections = slice_sections(full_text_sections)
        section_records = []
        for title, chunk in sections:
            norm = normalize_text(chunk)
            section_records.append({
                "id": f"감사보고서-{year}-경영진의견",
                "document": f"[사업부문] {norm}",
                "metadata": {
                    "year": year,
                    "source": name_nfc,
                    "section": f"경영진의견-사업부문-{title}"
                }
            })

        # 3) JSONL 저장 (overview + 사업부문을 같은 연도 파일에)
        out_path = OUT_DIR / f"경영진의견_{year}.overview.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in overview_records + section_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[OK] {year} → {out_path} ({len(overview_records)}개 개요, {len(section_records)}개 사업부문)")

if __name__ == "__main__":
    main()


### 배당정보

import pdfplumber
import pandas as pd
import os

ROOT_DIR = Path(__file__).resolve().parent


def normalize_fiscal_year(header_list):
    """
    회계 기수를 실제 연도로 변환합니다.
    44기 -> 2012년, 45기 -> 2013년, ...
    """
    normalized_header = []
    for header in header_list:
        if isinstance(header, str) and '제' in header and '기' in header:
            try:
                # '제46 기' -> '46' 추출
                fiscal_year_str = header.replace('제', '').replace('기', '').replace(' ', '').strip()
                fiscal_year_int = int(fiscal_year_str)
                # 연도 계산: 연도 = 2012 + (기수 - 44)
                actual_year = 2012 + (fiscal_year_int - 44)
                normalized_header.append(f"{actual_year}년")
            except (ValueError, IndexError):
                normalized_header.append(header)
        else:
            normalized_header.append(header)
    return normalized_header

def clean_table_data(data):
    """
    테이블 데이터에서 쉼표(,)를 제거하고 하이픈(-)을 None으로 변환합니다.
    """
    cleaned_data = []
    for row in data:
        cleaned_row = []
        for cell in row:
            if isinstance(cell, str):
                # 쉼표(,) 제거
                cleaned_cell = cell.replace(',', '')
                # 하이픈(-)을 None으로 변환
                if cleaned_cell.strip() == '-':
                    cleaned_cell = None
                cleaned_row.append(cleaned_cell)
            else:
                cleaned_row.append(cell)
        cleaned_data.append(cleaned_row)
    return cleaned_data

def parse_and_save_all_tables(input_dir, output_dir):
    """
    지정된 디렉토리의 모든 PDF 파일에서 첫 번째 표를 추출하여
    정규화 후 마크다운 파일로 저장합니다.
    """
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 입력 디렉토리 내의 모든 PDF 파일 목록 가져오기
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"오류: '{input_dir}' 디렉토리에 PDF 파일이 없습니다.")
        return

    for filename in pdf_files:
        file_path = os.path.join(input_dir, filename)
        
        try:
            # 파일 이름에서 연도 추출 (예: '배당_2020.pdf' -> 2020)
            year = int(filename.split('_')[1].split('.')[0])
        except (ValueError, IndexError):
            print(f"경고: '{filename}' 파일 이름에서 연도를 추출할 수 없습니다. '배당_YYYY.pdf' 형식을 사용해 주세요.")
            continue
        
        print(f"'{filename}' 파일 처리 중...")

        try:
            with pdfplumber.open(file_path) as pdf:
                # 첫 페이지의 첫 번째 표만 추출
                first_page = pdf.pages[0]
                tables = first_page.extract_tables()
                
                if not tables:
                    print(f"경고: '{filename}'의 첫 페이지에서 표를 찾을 수 없습니다. 건너뜝니다.")
                    continue

                raw_table_data = tables[0]
                
                # 헤더가 2줄인 경우를 고려하여 유효성 검사
                if not raw_table_data or len(raw_table_data) < 2:
                    print(f"경고: '{filename}'의 첫 번째 표 데이터가 유효하지 않습니다. 건너뜁니다.")
                    continue

                # 두 줄로 된 헤더를 하나로 합치는 로직 추가
                header_row1 = raw_table_data[0]
                header_row2 = raw_table_data[1]
                
                combined_header = []
                for i, cell in enumerate(header_row2):
                    if isinstance(cell, str) and '기' in cell:
                        # '제47기'와 같은 셀은 그대로 사용
                        combined_header.append(cell)
                    else:
                        # 그 외의 셀은 첫 번째 헤더 행의 값을 사용
                        combined_header.append(header_row1[i])
                
                # 데이터는 세 번째 행부터 시작
                data = raw_table_data[2:]
                
                # 데이터 정제 함수 호출
                cleaned_data = clean_table_data(data)

                # DataFrame 생성 및 정규화된 헤더 적용
                normalized_header = normalize_fiscal_year(combined_header)
                df = pd.DataFrame(cleaned_data, columns=normalized_header)

                # 마크다운 내용 생성
                md_content = f"## 배당 정보 - {year}년\n"
                md_content += df.fillna('').to_markdown(index=False)
                md_content += "\n\n"

                # 마크다운 파일로 저장
                md_filename = f"배당_{year}.md"
                md_filepath = os.path.join(output_dir, md_filename)
                
                with open(md_filepath, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                print(f"마크다운 파일 저장 완료: {md_filepath}")
        
        except Exception as e:
            print(f"'{filename}' 파일을 처리하는 중 오류가 발생했습니다: {e}")

# 실행
input_directory_path = ROOT_DIR / "사업보고서_preprocessed" / "배당"
output_directory_path = ROOT_DIR / "parsed_data"

parse_and_save_all_tables(input_directory_path, output_directory_path)




