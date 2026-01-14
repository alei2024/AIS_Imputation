"""
环境配置验证脚本
运行此脚本检查CSDI项目的环境是否正确配置
"""
import sys

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 7:
        print("✓ Python版本符合要求 (>= 3.7)")
        return True
    else:
        print("✗ Python版本过低，需要Python 3.7或更高版本")
        return False

def check_dependencies():
    """检查依赖包"""
    dependencies = {
        'torch': 'PyTorch',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML',
        'matplotlib': 'Matplotlib',
        'wget': 'wget',
        'requests': 'requests',
    }
    
    results = {}
    for module, name in dependencies.items():
        try:
            if module == 'yaml':
                import yaml
                results[name] = True
                print(f"✓ {name}: 已安装")
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', '未知版本')
                if module == 'torch':
                    cuda_available = mod.cuda.is_available()
                    print(f"✓ {name}: {version} (CUDA可用: {cuda_available})")
                else:
                    print(f"✓ {name}: {version}")
                results[name] = True
        except ImportError:
            print(f"✗ {name}: 未安装")
            results[name] = False
    
    # 检查linear_attention_transformer
    try:
        from linear_attention_transformer import LinearAttentionTransformer
        print("✓ linear_attention_transformer: 已安装")
        results['linear_attention_transformer'] = True
    except ImportError:
        print("✗ linear_attention_transformer: 未安装")
        results['linear_attention_transformer'] = False
    
    return results

def main():
    print("=" * 50)
    print("CSDI 项目环境配置检查")
    print("=" * 50)
    print()
    
    # 检查Python版本
    python_ok = check_python_version()
    print()
    
    # 检查依赖
    print("检查依赖包:")
    print("-" * 50)
    results = check_dependencies()
    print()
    
    # 总结
    print("=" * 50)
    all_ok = python_ok and all(results.values())
    if all_ok:
        print("✓ 环境配置完成！所有依赖已正确安装。")
    else:
        print("✗ 环境配置不完整，请安装缺失的依赖包。")
        print("\n安装命令:")
        print("  pip install -r requirements.txt")
        print("  pip install linear-attention-transformer")
    print("=" * 50)

if __name__ == "__main__":
    main()

