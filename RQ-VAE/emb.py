#!/usr/bin/env python3
"""
Amazon 2023数据集完整训练流程 - GPU版本
"""

import os
import json
import torch
import shutil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ========== GPU配置 ==========
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
# ============================

# 路径配置
DATA_BASE = "/data1/datasets/amazon2023"
TARGET_DIR = "/data1/xinyuefeng/recommender-system/RQ-VAE"
DATASET_NAME = "Toys_and_Games"

def setup_gpu():
    """GPU设置"""
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("⚠  GPU不可用，将使用CPU")
        return False

def copy_data_files():
    """复制数据集文件"""
    print("\n步骤1: 准备数据集文件...")
    
    # 复制元数据
    meta_source = os.path.join(DATA_BASE, "meta", f"meta_{DATASET_NAME}.jsonl")
    meta_target = os.path.join(TARGET_DIR, f"meta_{DATASET_NAME}.json")
    
    if not os.path.exists(meta_target):
        shutil.copy2(meta_source, meta_target)
        print(f"  ✓ 已复制元数据")
    
    # 复制评论数据
    reviews_source = os.path.join(DATA_BASE, "review", f"{DATASET_NAME}.jsonl")
    reviews_target = os.path.join(TARGET_DIR, f"reviews_{DATASET_NAME}.json")
    
    if not os.path.exists(reviews_target):
        shutil.copy2(reviews_source, reviews_target)
        print(f"  ✓ 已复制评论数据")
    
    return True

def generate_bge_embeddings_gpu():
    """使用GPU生成BGE embeddings"""
    print("\n步骤2: 使用GPU生成BGE embeddings...")
    
    meta_file = os.path.join(TARGET_DIR, f"meta_{DATASET_NAME}.json")
    output_file = os.path.join(TARGET_DIR, f"bge_embeddings.pt")
    
    if os.path.exists(output_file):
        print(f"  ✓ 已存在: {output_file}")
        return True
    
    # 加载模型到GPU
    print(f"  * 加载BGE模型到 {device}...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=str(device))
    
    # 读取元数据
    items = []
    with open(meta_file, 'r') as f:
        for line in tqdm(f, desc="读取元数据"):
            try:
                items.append(json.loads(line.strip()))
            except:
                continue
    
    # 构建文本
    item_text_map = {}
    for item in tqdm(items, desc="处理文本"):
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
    
    print(f"  * 有效物品: {len(item_text_map)}")
    
    # GPU批量编码
    texts = list(item_text_map.values())
    asins = list(item_text_map.keys())
    
    all_embeddings = []
    batch_size = 512  # GPU可加大批量大小
    
    for i in tqdm(range(0, len(texts), batch_size), desc="GPU编码"):
        batch = texts[i:i+batch_size]
        batch_embeds = model.encode(batch, 
                                   convert_to_tensor=True,
                                   normalize_embeddings=True,
                                   device=str(device),
                                   show_progress_bar=False)
        all_embeddings.append(batch_embeds.cpu())
    
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    
    # 保存
    torch.save({"item_ids": asins, "embeddings": embeddings_tensor}, output_file)
    print(f"  ✓ 已保存: {output_file} (维度: {embeddings_tensor.shape})")
    
    return True

def create_dirs_and_data():
    """创建目录和处理数据"""
    print("\n步骤3: 准备训练数据...")
    
    dataset_lower = DATASET_NAME.lower().replace("_and_", "").replace("_", "")
    
    # 创建目录
    for d in [f"data/{dataset_lower}", f"rqvae_data/{dataset_lower}", "logs", "saves"]:
        os.makedirs(os.path.join(TARGET_DIR, d), exist_ok=True)
    
    # 处理评论数据
    reviews_file = os.path.join(TARGET_DIR, f"reviews_{DATASET_NAME}.json")
    data_dir = os.path.join(TARGET_DIR, "data", dataset_lower)
    
    if os.path.exists(os.path.join(data_dir, "all_item_seqs.json")):
        print(f"  ✓ 数据已存在")
        return dataset_lower
    
    reviews = []
    with open(reviews_file, 'r') as f:
        for line in f:
            try:
                reviews.append(json.loads(line.strip()))
            except:
                continue
    
    # 构建用户序列
    user_seqs = {}
    for review in tqdm(reviews, desc="处理评论"):
        user = review.get('reviewerID', '')
        item = review.get('asin', '')
        time = review.get('unixReviewTime', 0)
        
        if user and item:
            if user not in user_seqs:
                user_seqs[user] = []
            user_seqs[user].append((time, item))
    
    # 排序
    for user in user_seqs:
        user_seqs[user].sort(key=lambda x: x[0])
        user_seqs[user] = [item for _, item in user_seqs[user]]
    
    # 生成映射
    all_items = set()
    for seq in user_seqs.values():
        all_items.update(seq)
    
    item2id = {item: i+1 for i, item in enumerate(all_items)}
    user2id = {user: i+1 for i, user in enumerate(user_seqs.keys())}
    
    # 保存
    all_item_seqs = {}
    for user, seq in user_seqs.items():
        all_item_seqs[str(user2id[user])] = [str(item2id[item]) for item in seq]
    
    with open(os.path.join(data_dir, "all_item_seqs.json"), 'w') as f:
        json.dump(all_item_seqs, f, indent=2)
    
    with open(os.path.join(data_dir, "id_mapping.json"), 'w') as f:
        json.dump({"user2id": user2id, "item2id": item2id}, f, indent=2)
    
    # 复制embeddings
    shutil.copy2(os.path.join(TARGET_DIR, "bge_embeddings.pt"),
                os.path.join(data_dir, "bge_embeddings.pt"))
    
    print(f"  ✓ 数据准备完成: {len(user2id)} users, {len(item2id)} items")
    return dataset_lower

def update_training_scripts(dataset_lower):
    """更新训练脚本启用GPU"""
    print("\n步骤4: 更新训练脚本...")
    
    # 1. 更新RQ-VAE训练脚本
    rqvae_script = os.path.join(TARGET_DIR, "train_rqvae.py")
    if os.path.exists(rqvae_script):
        with open(rqvae_script, 'r') as f:
            content = f.read()
        
        # 添加GPU支持
        if "device = torch.device" not in content:
            content = content.replace("import torch", "import torch\n\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        
        # 更新数据集名称
        content = content.replace('dataset_name = "ele"', f'dataset_name = "{dataset_lower}"')
        
        with open(rqvae_script, 'w') as f:
            f.write(content)
        print(f"  ✓ 已更新RQ-VAE脚本")
    
    # 2. 创建GPU训练启动脚本
    train_gpu_script = os.path.join(TARGET_DIR, "train_gpu.sh")
    with open(train_gpu_script, 'w') as f:
        f.write(f"""#!/bin/bash
# GPU训练脚本
cd {TARGET_DIR}

echo "1. 训练RQ-VAE..."
python train_rqvae.py

echo "2. 生成语义ID..."
python prepare_tiger_ids.py

echo "3. 训练TIGER模型..."
accelerate launch --multi_gpu train_tiger_mul.py
""")
    os.chmod(train_gpu_script, 0o755)
    print(f"  ✓ 已创建GPU训练脚本: {train_gpu_script}")
    
    return True

def main():
    """主函数"""
    print("="*60)
    print("Amazon 2023数据集GPU训练流程")
    print("="*60)
    
    setup_gpu()
    
    try:
        copy_data_files()
        generate_bge_embeddings_gpu()
        dataset_lower = create_dirs_and_data()
        update_training_scripts(dataset_lower)
        
        print("\n" + "="*60)
        print("✅ 数据预处理完成！")
        print(f"\n使用GPU训练:")
        print(f"1. 启动RQ-VAE训练:")
        print(f"   cd {TARGET_DIR}")
        print(f"   python train_rqvae.py")
        print(f"\n2. 或运行完整流程:")
        print(f"   bash train_gpu.sh")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()