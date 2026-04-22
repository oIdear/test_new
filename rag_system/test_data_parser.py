#!/usr/bin/env python3
"""
data_parser.py 功能测试脚本
用于验证PDF和CSV解析功能是否正常工作
"""

import sys
import os
from pathlib import Path

# 添加rag_system目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from data_parser import parse_pdf, parse_csv, semantic_text_splitter

def test_langchain_import():
    """测试LangChain是否正确安装"""
    print("=" * 60)
    print("测试1: LangChain导入检查")
    print("=" * 60)
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✅ LangChain导入成功")
        return True
    except ImportError as e:
        print(f"❌ LangChain导入失败: {e}")
        print("💡 请运行: pip install langchain langchain-core")
        return False

def test_semantic_splitter():
    """测试语义分割器功能"""
    print("\n" + "=" * 60)
    print("测试2: 语义分割器功能测试")
    print("=" * 60)
    
    # 创建测试文本（包含中英文混合）
    test_text = """
    BGP路由泄露是一种严重的网络安全威胁。
    
    Route leak occurs when an AS improperly propagates BGP routes.
    This can cause traffic to be misdirected through unintended paths.
    
    根据RFC7908的定义，路由泄露可以分为以下几种类型：
    1. 意外泄露（Accidental Leak）
    2. 恶意泄露（Malicious Leak）
    
    Valley-free规则要求BGP路径必须遵循特定的传播模式。
    Customer-to-provider (c2p), peer-to-peer (p2p), and provider-to-customer (p2c) relationships must be respected.
    
    违反valley-free规则的路径通常被认为是异常的。
    Such anomalies may indicate route hijacking or misconfiguration.
    """
    
    try:
        chunks = semantic_text_splitter(test_text, chunk_size=200, chunk_overlap=50)
        print(f"✅ 分割成功！生成了 {len(chunks)} 个文本块\n")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"--- 文本块 {i} ({len(chunk)} 字符) ---")
            print(chunk[:150] + "..." if len(chunk) > 150 else chunk)
            print()
        
        return len(chunks) > 0
    except Exception as e:
        print(f"❌ 分割失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_parsing():
    """测试CSV文件解析"""
    print("=" * 60)
    print("测试3: CSV文件解析测试")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent / 'bgp-embedding-data'
    csv_files = list(base_dir.glob('*.csv'))
    
    if not csv_files:
        print("⚠️  未找到CSV文件，跳过测试")
        return True
    
    # 测试第一个CSV文件
    test_file = csv_files[0]
    print(f"📊 测试文件: {test_file.name}")
    
    try:
        data = parse_csv(str(test_file))
        print(f"✅ CSV解析成功！生成了 {len(data)} 条记录")
        
        if data:
            print(f"📝 示例数据:")
            print(f"   - 类型: {data[0]['type']}")
            print(f"   - 文件名: {data[0]['file']}")
            print(f"   - 内容长度: {len(data[0]['content'])} 字符")
            print(f"   - 内容预览: {data[0]['content'][:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ CSV解析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdf_parsing():
    """测试PDF文件解析"""
    print("\n" + "=" * 60)
    print("测试4: PDF文件解析测试")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent / 'bgp-embedding-data'
    pdf_files = [f for f in base_dir.glob('*.pdf') if 'Zone.Identifier' not in str(f)]
    
    if not pdf_files:
        print("⚠️  未找到PDF文件，跳过测试")
        return True
    
    # 测试最小的PDF文件（rfc7908.txt.pdf）
    test_file = min(pdf_files, key=lambda x: x.stat().st_size)
    print(f"📕 测试文件: {test_file.name} ({test_file.stat().st_size / 1024:.2f} KB)")
    
    try:
        data = parse_pdf(str(test_file))
        print(f"✅ PDF解析成功！生成了 {len(data)} 个文本块")
        
        if data:
            print(f"📝 示例数据:")
            print(f"   - 类型: {data[0]['type']}")
            print(f"   - 文件名: {data[0]['file']}")
            print(f"   - 块索引: {data[0]['chunk_index']}")
            print(f"   - 内容长度: {len(data[0]['content'])} 字符")
            print(f"   - 内容预览: {data[0]['content'][:150]}...")
        
        return True
    except Exception as e:
        print(f"❌ PDF解析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "🧪" * 30)
    print("开始测试 data_parser.py 功能")
    print("🧪" * 30 + "\n")
    
    results = []
    
    # 测试1: LangChain导入
    results.append(("LangChain导入", test_langchain_import()))
    
    # 测试2: 语义分割器
    results.append(("语义分割器", test_semantic_splitter()))
    
    # 测试3: CSV解析
    results.append(("CSV解析", test_csv_parsing()))
    
    # 测试4: PDF解析
    results.append(("PDF解析", test_pdf_parsing()))
    
    # 打印测试结果汇总
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20s} : {status}")
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！data_parser.py 可以正常使用")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查错误信息")
        return 1

if __name__ == '__main__':
    exit(main())
