#!/usr/bin/env python
"""Excel Q&A 데이터를 지식베이스 형식으로 변환"""
import pandas as pd

# Excel 파일 읽기
df = pd.read_excel('ChatBot_Q&A(통합).xlsx')
df.columns = ['번호', '질문', '답변']

# 헤더 제거 (첫 2개 행)
df = df.iloc[2:].reset_index(drop=True)

# Python 코드 생성
python_code = """from typing import List, Dict, Any

def get_excel_knowledge_base() -> List[Dict[str, Any]]:
    \"\"\"Excel 파일에서 읽은 Q&A 지식 베이스\"\"\"

    knowledge_data = [
"""

for idx, row in df.iterrows():
    if pd.notna(row['질문']) and pd.notna(row['답변']):
        # 답변 내용 정리
        answer = str(row['답변']).strip()
        question = str(row['질문']).strip() if pd.notna(row['질문']) else ''
        
        # 특수 문자 이스케이프
        answer = answer.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
        question = question.replace('\\', '\\\\').replace('"', '\\"')
        
        num = int(row['번호']) if pd.notna(row['번호']) and str(row['번호']).isdigit() else idx+1
        
        python_code += f"""        {{
            "id": "excel_{num}",
            "content": \"\"\"{answer}\"\"\",
            "metadata": {{
                "category": "노인학대",
                "type": "Q&A",
                "priority": "high",
                "question": "{question}",
            }},
        }},
"""

python_code += """    ]
    return knowledge_data
"""

# 파일로 저장
with open('excel_kb_generated.py', 'w', encoding='utf-8') as f:
    f.write(python_code)

print(f"✅ Excel 데이터 변환 완료: {len(df)}개 항목")
print("📝 excel_kb_generated.py 파일 생성됨")

# 샘플 출력
sample_df = df[['질문', '답변']].head(5)
print("\n=== 샘플 데이터 ===")
for idx, row in sample_df.iterrows():
    if pd.notna(row['질문']):
        q = str(row['질문'])[:50] + "..." if len(str(row['질문'])) > 50 else str(row['질문'])
        print(f"\nQ: {q}")

