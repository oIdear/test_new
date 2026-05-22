import os
import json
import csv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

# 使用LangChain的智能文本分割器
def semantic_text_splitter(text, chunk_size=1000, chunk_overlap=150):
    """
    使用LangChain的RecursiveCharacterTextSplitter进行智能语义分割
    
    优势：
    - 按优先级尝试多种分隔符（段落→句子→单词→字符）
    - 保持文本的语义完整性
    - 支持重叠机制保持上下文连贯性
    - 生产级验证，稳定可靠
    
    Args:
        text: 待分割的文本
        chunk_size: 每个chunk的目标字符数（默认1000）
        chunk_overlap: chunk之间的重叠字符数（默认150）
    
    Returns:
        list: 分割后的文本块列表
    """
    if not text:
        return []
    
    # 创建递归字符分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",      # 段落分隔（最高优先级）
            "\n",        # 换行
            "。", "！", "？",  # 中文句子结束符
            ". ", "! ", "? ",  # 英文句子结束符
            " ",         # 空格/单词边界
            ""           # 字符级别（最后手段）
        ]
    )
    
    chunks = text_splitter.split_text(text)
    # 过滤空字符串
    return [chunk for chunk in chunks if chunk.strip()]

# 解析TXT文件（RFC纯文本格式）
def parse_txt(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        chunks = semantic_text_splitter(text, chunk_size=1000, chunk_overlap=150)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                data.append({
                    'type': 'txt',
                    'file': os.path.basename(file_path),
                    'chunk_index': i,
                    'content': chunk.strip()
                })
    except Exception as e:
        print(f'Error parsing {file_path}: {e}')
    return data

# 解析PDF文件（使用LangChain语义分割）
def parse_pdf(file_path):
    data = []
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
            
            # 使用LangChain智能语义分割
            chunks = semantic_text_splitter(text, chunk_size=1000, chunk_overlap=150)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # 只保存非空chunk
                    data.append({
                        'type': 'pdf',
                        'file': os.path.basename(file_path),
                        'chunk_index': i,
                        'content': chunk.strip()
                    })
    except Exception as e:
        print(f'Error parsing {file_path}: {e}')
    return data

# 主函数
def main():
    # 使用相对路径，基于项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'bgp-embedding-data')
    
    if not os.path.exists(data_dir):
        print(f'❌ 数据目录不存在: {data_dir}')
        return
    
    print(f'📂 开始解析数据目录: {data_dir}')
    print(f'📄 文件列表: {os.listdir(data_dir)}')
    
    all_data = []
    
    for file in os.listdir(data_dir):
        # 跳过Windows元数据文件
        if 'Zone.Identifier' in file:
            continue
            
        file_path = os.path.join(data_dir, file)
        
        if file.endswith('.csv'):
            print(f'📊 解析CSV文件: {file}')
            all_data.extend(parse_csv(file_path))
        elif file.endswith('.pdf'):
            print(f'📕 解析PDF文件: {file}')
            all_data.extend(parse_pdf(file_path))
        elif file.endswith('.txt'):
            print(f'📄 解析TXT文件: {file}')
            all_data.extend(parse_txt(file_path))
    
    print(f'\n✅ 解析完成！共生成 {len(all_data)} 个文本块')
    
    # 保存结果到rag_system目录
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, 'parsed_data.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f'💾 数据已保存到: {output_file}')
    print(f'📏 文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB')
    
    # 打印统计信息
    type_stats = {}
    for item in all_data:
        file_type = item['type']
        type_stats[file_type] = type_stats.get(file_type, 0) + 1
    
    print(f'\n📈 统计信息:')
    for file_type, count in type_stats.items():
        print(f'   - {file_type}: {count} 个文本块')

if __name__ == '__main__':
    main()
