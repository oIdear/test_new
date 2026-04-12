import os
import json
import csv
from PyPDF2 import PdfReader

# 解析CSV文件
def parse_csv(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = ' '.join([f'{k}: {v}' for k, v in row.items() if v])
                data.append({
                    'type': 'csv',
                    'file': os.path.basename(file_path),
                    'content': text
                })
    except Exception as e:
        print(f'Error parsing {file_path}: {e}')
    return data

# 解析PDF文件
def parse_pdf(file_path):
    data = []
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
            
            # 简单分割文本
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            for i, chunk in enumerate(chunks):
                data.append({
                    'type': 'pdf',
                    'file': os.path.basename(file_path),
                    'content': chunk
                })
    except Exception as e:
        print(f'Error parsing {file_path}: {e}')
    return data

# 主函数
def main():
    data_dir = '/home/zzx-king/zzx/routing-anomaly-detection-master/bgp-embedding-data'
    all_data = []
    
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if file.endswith('.csv'):
            all_data.extend(parse_csv(file_path))
        elif file.endswith('.pdf'):
            all_data.extend(parse_pdf(file_path))
    
    print(f'Parsed {len(all_data)} items')
    
    # 保存结果
    output_file = '/home/zzx-king/zzx/routing-anomaly-detection-master/rag_system/parsed_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f'Data saved to {output_file}')

if __name__ == '__main__':
    main()
