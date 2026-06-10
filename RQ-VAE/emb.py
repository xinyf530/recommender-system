#!/usr/bin/env python3
"""
Amazon 2023数据集 - 全类别批量处理
批量处理所有6个类别的数据
"""

import os
import json
import torch
import shutil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ========== 配置 ==========
DATA_BASE = "/data1/datasets/amazon2023"
TARGET_DIR = "/data1/xinyuefeng/recommender-system/RQ-VAE"

# 所有6个类别
DATASET_CATEGORIES = [
    "Books",
    "Cell_Phones_and_Accessories", 
    "Electronics",
    "Movies_and_TV",
    "Toys_and_Games", 
    "Video_Games"
]
# ==========================

def setup_gpu():
    """GPU设置"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device

def copy_all_dataset_files():
    """复制所有类别的数据文件"""
    print("步骤1: 复制所有类别的数据文件...")
    
    for category in DATASET_CATEGORIES:
        print(f"\n  {category}:")
        
        # 复制元数据
        meta_source = os.path.join(DATA_BASE, "meta", f"meta_{category}.jsonl")
        meta_target = os.path.join(TARGET_DIR, f"meta_{category}.json")
        
        if not os.path.exists(meta_target) and os.path.exists(meta_source):
            shutil.copy2(meta_source, meta_target)
            print(f"    ✓ 元数据")
        
        # 复制评论数据
        reviews_source = os.path.join(DATA_BASE, "review", f"{category}.jsonl")
        reviews_target = os.path.join(TARGET_DIR, f"reviews_{category}.json")
        
        if not os.path.exists(reviews_target) and os.path.exists(reviews_source):
            shutil.copy2(reviews_source, reviews_target)
            print(f"    ✓ 评论数据")
    
    print(f"\n✅ 完成: 已复制 {len(DATASET_CATEGORIES)} 个类别的数据")
    return True

def generate_bge_embeddings_all(device):
    """为所有类别生成BGE embeddings"""
    print("\n步骤2: 为所有类别生成BGE embeddings...")
    
    # 加载模型
    print(f"  * 加载BGE模型到 {device}...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=str(device))
    
    for category in DATASET_CATEGORIES:
        print(f"\n  {category}:")
        
        meta_file = os.path.join(TARGET_DIR, f"meta_{category}.json")
        output_file = os.path.join(TARGET_DIR, f"bge_embeddings_{category}.pt")
        
        if os.path.exists(output_file):
            print(f"    ✓ 已存在embeddings")
            continue
        
        if not os.path.exists(meta_file):
            print(f"    ⚠ 元数据文件不存在")
            continue
        
        # 读取元数据
        items = []
        with open(meta_file, 'r') as f:
            for line in f:
                try:
                    items.append(json.loads(line.strip()))
                except:
                    continue
        
        # 构建文本映射
        item_text_map = {}
        for item in tqdm(items, desc=f"    处理文本", leave=False):
            asin = item.get('asin', '')
            title = item.get('title', '')
            description = item.get('description', '')
            
            if title and description:
                text = f"{title} [SEP] {description}"
            elif title:
                text = title
            else:
                continue
                
            item_text_map[asin] = text
        
        if len(item_text_map) == 0:
            print(f"    ⚠ 无有效物品")
            continue
        
        # 生成embeddings
        texts = list(item_text_map.values())
        asins = list(item_text_map.keys())
        
        all_embeddings = []
        batch_size = 256
        
        for i in tqdm(range(0, len(texts), batch_size), 
                     desc=f"    生成embeddings", 
                     leave=False,
                     total=(len(texts)+batch_size-1)//batch_size):
            batch = texts[i:i+batch_size]
            batch_embeds = model.encode(batch, 
                                       convert_to_tensor=True,
                                       normalize_embeddings=True,
                                       device=str(device),
                                       show_progress_bar=False)
            all_embeddings.append(batch_embeds.cpu())
        
        if all_embeddings:
            embeddings_tensor = torch.cat(all_embeddings, dim=0)
            torch.save({"item_ids": asins, "embeddings": embeddings_tensor}, output_file)
            print(f"    ✓ 已保存: {embeddings_tensor.shape}")
    
    return True

def create_dataset_structure_all():
    """为所有类别创建目录结构"""
    print("\n步骤3: 为所有类别创建目录结构...")
    
    for category in DATASET_CATEGORIES:
        dataset_lower = category.lower().replace("_and_", "").replace("_", "")
        
        # 创建目录
        dirs_to_create = [
            os.path.join(TARGET_DIR, "data", dataset_lower),
            os.path.join(TARGET_DIR, "rqvae_data", dataset_lower)
        ]
        
        for d in dirs_to_create:
            os.makedirs(d, exist_ok=True)
    
    print(f"✅ 完成: 已为 {len(DATASET_CATEGORIES)} 个类别创建目录")
    return True

def prepare_sequence_data_all():
    """为所有类别准备序列数据"""
    print("\n步骤4: 为所有类别准备序列数据...")
    
    for category in DATASET_CATEGORIES:
        print(f"\n  {category}:")
        
        dataset_lower = category.lower().replace("_and_", "").replace("_", "")
        reviews_file = os.path.join(TARGET_DIR, f"reviews_{category}.json")
        data_dir = os.path.join(TARGET_DIR, "data", dataset_lower)
        
        all_seqs_file = os.path.join(data_dir, "all_item_seqs.json")
        mapping_file = os.path.join(data_dir, "id_mapping.json")
        
        if os.path.exists(all_seqs_file) and os.path.exists(mapping_file):
            print(f"    ✓ 序列数据已存在")
            continue
        
        if not os.path.exists(reviews_file):
            print(f"    ⚠ 评论文件不存在")
            continue
        
        # 读取评论数据
        reviews = []
        with open(reviews_file, 'r') as f:
            for line in f:
                try:
                    reviews.append(json.loads(line.strip()))
                except:
                    continue
        
        # 构建用户序列
        user_seqs = {}
        for review in tqdm(reviews, desc=f"    处理评论", leave=False):
            user = review.get('reviewerID', '')
            item = review.get('asin', '')
            time = review.get('unixReviewTime', 0)
            
            if user and item and time:
                if user not in user_seqs:
                    user_seqs[user] = []
                user_seqs[user].append((time, item))
        
        # 过滤短序列
        user_seqs = {u: s for u, s in user_seqs.items() if len(s) >= 3}
        
        if len(user_seqs) == 0:
            print(f" 无有效用户序列")
            continue
        
        # 排序
        for user in user_seqs:
            user_seqs[user].sort(key=lambda x: x[0])
            user_seqs[user] = [item for _, item in user_seqs[user]]
        
        # 生成ID映射
        all_items = set()
        for seq in user_seqs.values():
            all_items.update(seq)
        
        item2id = {item: i+1 for i, item in enumerate(all_items)}
        user2id = {user: i+1 for i, user in enumerate(user_seqs.keys())}
        
        # 保存序列
        all_item_seqs = {}
        for user, seq in user_seqs.items():
            all_item_seqs[str(user2id[user])] = [str(item2id[item]) for item in seq]
        
        with open(all_seqs_file, 'w') as f:
            json.dump(all_item_seqs, f, indent=2)
        
        with open(mapping_file, 'w') as f:
            json.dump({"user2id": user2id, "item2id": item2id}, f, indent=2)
        
        # 复制embeddings
        bge_source = os.path.join(TARGET_DIR, f"bge_embeddings_{category}.pt")
        bge_target = os.path.join(data_dir, "bge_embeddings.pt")
        if os.path.exists(bge_source) and not os.path.exists(bge_target):
            shutil.copy2(bge_source, bge_target)
        
        print(f"    ✓ 用户: {len(user2id)}, 物品: {len(item2id)}")
    
    return True

def create_training_scripts():
    """为所有类别创建训练脚本"""
    print("\n步骤5: 创建训练脚本...")
    
    # 为每个类别创建训练脚本
    for category in DATASET_CATEGORIES:
        dataset_lower = category.lower().replace("_and_", "").replace("_", "")
        
        train_script = os.path.join(TARGET_DIR, f"train_rqvae_{dataset_lower}.py")
        
        if not os.path.exists(train_script):
            with open(train_script, 'w') as f:
                f.write(f"""#!/usr/bin/env python3
# RQ-VAE训练脚本 - {category}
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练 {category} 在设备: {{device}}")

# 数据集配置
dataset_name = "{dataset_lower}"

# 这里需要你的train_rqvae.py主训练逻辑
# 确保修改数据集路径指向 {dataset_lower}
""")
            os.chmod(train_script, 0o755)
    
    # 创建批处理脚本
    batch_script = os.path.join(TARGET_DIR, "train_all_datasets.sh")
    with open(batch_script, 'w') as f:
        f.write("""#!/bin/bash
# 批处理所有数据集的训练脚本
cd /data1/xinyuefeng/recommender-system/RQ-VAE

# 训练所有类别
for script in train_rqvae_*.py; do
    if [ -f "$script" ]; then
        echo "========================================"
        echo "开始训练: $script"
        echo "========================================"
        python "$script"
        echo ""
    fi
done

echo "所有训练完成！"
""")
    os.chmod(batch_script, 0o755)
    
    print(f" 完成: 已创建训练脚本")
    print(f"    - 单个脚本: train_rqvae_*.py")
    print(f"    - 批处理脚本: train_all_datasets.sh")
    return True

def main():
    """主函数"""
    print("="*60)
    print("Amazon 2023 - 全类别批量处理")
    print(f"处理 {len(DATASET_CATEGORIES)} 个类别")
    print("="*60)
    
    device = setup_gpu()
    
    try:
        copy_all_dataset_files()
        generate_bge_embeddings_all(device)
        create_dataset_structure_all()
        prepare_sequence_data_all()
        create_training_scripts()
        
        print("\n" + "="*60)
        print(" 全类别数据预处理完成！")
        print(f"\n类别列表: {', '.join(DATASET_CATEGORIES)}")
        print("\n后续步骤:")
        print(f"1. 检查数据:")
        print(f"   ls {TARGET_DIR}/data/")
        print(f"2. 训练单个数据集 (如Toys_and_Games):")
        print(f"   python train_rqvae_toysandgames.py")
        print(f"3. 训练所有数据集:")
        print(f"   bash train_all_datasets.sh")
        print("="*60)
        
    except Exception as e:
        print(f"\n 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()