#!/usr/bin/env python3
"""
Amazon 2023数据集完整训练流程
从JSONL格式到RQ-VAE训练的一键执行脚本
"""

import os
import json
import torch
import shutil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ========== 配置区 ==========
DATA_BASE = "/data1/datasets/amazon2023"  # 2023数据集路径
TARGET_DIR = "/data1/xinyuefeng/recommender-system/RQ-VAE"
DATASET_NAME = "Toys_and_Games"  # 使用玩具数据集（TIGER标准基准）
# ===========================

def copy_data_files():
    """复制数据集文件到训练目录"""
    print("步骤1: 准备数据集文件...")
    
    # 复制元数据文件
    meta_source = os.path.join(DATA_BASE, "meta", f"meta_{DATASET_NAME}.jsonl")
    meta_target = os.path.join(TARGET_DIR, f"meta_{DATASET_NAME}.json")
    
    if not os.path.exists(meta_target):
        if os.path.exists(meta_source):
            shutil.copy2(meta_source, meta_target)
            print(f"  ✓ 已复制元数据: {meta_target}")
        else:
            print(f"  ❌ 元数据文件不存在: {meta_source}")
            return False
    else:
        print(f"  ✓ 已存在: {meta_target}")
    
    # 查找并复制评论文件
    reviews_sources = [
        os.path.join(DATA_BASE, "reviews", f"reviews_{DATASET_NAME}_5.jsonl"),  # 5-star版本
        os.path.join(DATA_BASE, f"reviews_{DATASET_NAME}.jsonl"),  # 完整版本
        os.path.join(DATA_BASE, f"reviews_{DATASET_NAME}.json")  # json格式
    ]
    
    reviews_target = os.path.join(TARGET_DIR, f"reviews_{DATASET_NAME}.json")
    
    if not os.path.exists(reviews_target):
        found = False
        for source in reviews_sources:
            if os.path.exists(source):
                shutil.copy2(source, reviews_target)
                print(f"  ✓ 已复制评论数据: {reviews_target} (来源: {source})")
                found = True
                break
        
        if not found:
            print(f"  ⚠ 未找到评论文件，尝试手动查找...")
            # 列出可能的位置
            print("  可用的评论文件:")
            for root, dirs, files in os.walk(DATA_BASE):
                for file in files:
                    if DATASET_NAME.lower() in file.lower() and "review" in file.lower():
                        print(f"    {os.path.join(root, file)}")
            return False
    else:
        print(f"  ✓ 已存在: {reviews_target}")
    
    return True

def generate_bge_embeddings():
    """生成BGE embeddings"""
    print("\n步骤2: 生成BGE embeddings...")
    
    meta_file = os.path.join(TARGET_DIR, f"meta_{DATASET_NAME}.json")
    output_file = os.path.join(TARGET_DIR, f"bge_embeddings.pt")
    
    if os.path.exists(output_file):
        print(f"  ✓ 已存在: {output_file}")
        return True
    
    # 加载BGE模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  * 加载BGE模型到 {device}...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    
    # 读取元数据（JSONL格式）
    print(f"  * 读取元数据: {meta_file}")
    items = []
    with open(meta_file, 'r') as f:
        for line in f:
            try:
                items.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"  ⚠ JSON解析错误: {e}")
                continue
    
    print(f"  * 总物品数: {len(items)}")
    
    # 构建文本映射
    item_text_map = {}
    for item in tqdm(items, desc="处理物品文本"):
        # Amazon 2023元数据结构
        asin = item.get('asin', item.get('parent_asin', ''))
        title = item.get('title', '')
        description = item.get('description', '')
        features = item.get('features', [])
        
        # 构建描述文本
        text_parts = []
        if title:
            text_parts.append(title)
        
        if isinstance(description, str) and description.strip():
            text_parts.append(description[:500])  # 截断长描述
        elif isinstance(description, list):
            text_parts.extend([d[:200] for d in description if isinstance(d, str)][:3])
        
        if isinstance(features, list):
            features_text = " ".join([f[:100] for f in features if isinstance(f, str)][:5])
            if features_text:
                text_parts.append(f"Features: {features_text}")
        
        if text_parts:
            item_text_map[asin] = " [SEP] ".join(text_parts)
        else:
            continue
    
    print(f"  * 有效物品数量: {len(item_text_map)}")
    
    if len(item_text_map) == 0:
        print(f"  ❌ 没有有效的物品文本数据")
        return False
    
    # 生成embeddings
    texts = list(item_text_map.values())
    asins = list(item_text_map.keys())
    
    embeddings = []
    batch_size = 128
    
    print("  * 开始生成embeddings...")
    for i in tqdm(range(0, len(texts), batch_size), desc="生成embeddings"):
        batch = texts[i:i+batch_size]
        batch_embeds = model.encode(batch, 
                                    convert_to_tensor=True,
                                    normalize_embeddings=True,
                                    device=device)
        embeddings.append(batch_embeds.cpu())
    
    embeddings_tensor = torch.cat(embeddings, dim=0)
    
    # 保存
    output_dict = {
        "item_ids": asins,
        "embeddings": embeddings_tensor
    }
    torch.save(output_dict, output_file)
    print(f"  ✓ 已保存: {output_file} (维度: {embeddings_tensor.shape})")
    
    return True

def create_dataset_structure():
    """创建目录结构"""
    print("\n步骤3: 创建目录结构...")
    
    dataset_lower = DATASET_NAME.lower().replace("_and_", "").replace("_", "")
    dirs_to_create = [
        os.path.join(TARGET_DIR, "data", dataset_lower),
        os.path.join(TARGET_DIR, "rqvae_data", dataset_lower),
        os.path.join(TARGET_DIR, "logs"),
        os.path.join(TARGET_DIR, "saves")
    ]
    
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓ 创建: {d}")
    
    return dataset_lower

def prepare_data_files(dataset_lower):
    """准备训练数据"""
    print("\n步骤4: 准备序列数据...")
    
    reviews_file = os.path.join(TARGET_DIR, f"reviews_{DATASET_NAME}.json")
    data_dir = os.path.join(TARGET_DIR, "data", dataset_lower)
    
    all_seqs_file = os.path.join(data_dir, "all_item_seqs.json")
    mapping_file = os.path.join(data_dir, "id_mapping.json")
    
    if os.path.exists(all_seqs_file) and os.path.exists(mapping_file):
        print(f"  ✓ 已存在序列文件")
        return True
    
    # 读取评论数据
    print(f"  * 读取评论数据...")
    reviews = []
    with open(reviews_file, 'r') as f:
        for line in f:
            try:
                reviews.append(json.loads(line.strip()))
            except:
                continue
    
    print(f"  * 总评论数: {len(reviews)}")
    
    # 按用户分组
    user_seqs = {}
    for review in tqdm(reviews, desc="处理评论"):
        user = review.get('reviewerID', review.get('reviewer_id', ''))
        item = review.get('asin', review.get('parent_asin', ''))
        time = review.get('unixReviewTime', review.get('timestamp', 0))
        
        if user and item and time:
            if user not in user_seqs:
                user_seqs[user] = []
            user_seqs[user].append((time, item))
    
    # 按时间排序
    for user in user_seqs:
        user_seqs[user].sort(key=lambda x: x[0])
        user_seqs[user] = [item for _, item in user_seqs[user]]
    
    # 过滤短序列
    user_seqs = {u: s for u, s in user_seqs.items() if len(s) >= 4}
    
    # 生成ID映射
    print(f"  * 生成ID映射...")
    all_items = set()
    for seq in user_seqs.values():
        all_items.update(seq)
    
    item2id = {item: i+1 for i, item in enumerate(all_items)}
    user2id = {user: i+1 for i, user in enumerate(user_seqs.keys())}
    
    # 转换序列
    all_item_seqs = {}
    for user, seq in user_seqs.items():
        all_item_seqs[str(user2id[user])] = [str(item2id[item]) for item in seq]
    
    # 保存
    with open(all_seqs_file, 'w') as f:
        json.dump(all_item_seqs, f, indent=2)
    
    id_mapping = {
        "user2id": user2id,
        "item2id": item2id
    }
    with open(mapping_file, 'w') as f:
        json.dump(id_mapping, f, indent=2)
    
    print(f"  ✓ 用户数: {len(user2id)}, 物品数: {len(item2id)}")
    print(f"  ✓ 已保存: {all_seqs_file}")
    print(f"  ✓ 已保存: {mapping_file}")
    
    return True

def copy_embeddings_to_data(dataset_lower):
    """复制embeddings到data目录"""
    print("\n步骤5: 准备embeddings文件...")
    
    bge_file = os.path.join(TARGET_DIR, f"bge_embeddings.pt")
    target_bge = os.path.join(TARGET_DIR, "data", dataset_lower, "bge_embeddings.pt")
    
    if not os.path.exists(target_bge):
        if os.path.exists(bge_file):
            shutil.copy2(bge_file, target_bge)
            print(f"  ✓ 复制embeddings到: {target_bge}")
        else:
            print(f"  ❌ embeddings文件不存在: {bge_file}")
            return False
    
    return True

def update_training_config(dataset_lower):
    """更新训练配置"""
    print("\n步骤6: 更新训练配置...")
    
    train_script = os.path.join(TARGET_DIR, "train_rqvae.py")
    if not os.path.exists(train_script):
        print(f"  ⚠ 训练脚本不存在: {train_script}")
        print(f"  * 请手动设置训练脚本中的 dataset_name = '{dataset_lower}'")
    else:
        try:
            with open(train_script, 'r') as f:
                content = f.read()
            
            # 查找并替换数据集名称
            lines = content.split('\n')
            updated = False
            for i, line in enumerate(lines):
                if 'dataset_name' in line and '=' in line:
                    lines[i] = f"dataset_name = '{dataset_lower}'"
                    updated = True
                    break
            
            if updated:
                with open(train_script, 'w') as f:
                    f.write('\n'.join(lines))
                print(f"  ✓ 已更新训练脚本中的数据集名称: {dataset_lower}")
            else:
                print(f"  ⚠ 未找到dataset_name配置，请在脚本中添加: dataset_name = '{dataset_lower}'")
        except Exception as e:
            print(f"  ⚠ 更新训练脚本失败: {e}")
    
    return True

def main():
    """主执行函数"""
    print("=" * 60)
    print("Amazon 2023数据集完整训练流程")
    print(f"数据集: {DATASET_NAME}")
    print(f"工作目录: {TARGET_DIR}")
    print("=" * 60)
    
    # 确保目标目录存在
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    try:
        # 执行步骤
        if not copy_data_files():
            return
        
        if not generate_bge_embeddings():
            return
        
        dataset_lower = create_dataset_structure()
        
        if not prepare_data_files(dataset_lower):
            return
        
        if not copy_embeddings_to_data(dataset_lower):
            return
        
        update_training_config(dataset_lower)
        
        print("\n" + "=" * 60)
        print("✅ 数据预处理完成!")
        print("\n下一步操作:")
        print(f"1. 启动RQ-VAE训练:")
        print(f"   cd {TARGET_DIR}")
        print(f"   python train_rqvae.py")
        print(f"")
        print(f"2. 训练完成后生成语义ID:")
        print(f"   python prepare_tiger_ids.py")
        print(f"")
        print(f"3. 训练TIGER模型:")
        print(f"   accelerate launch train_tiger_mul.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()