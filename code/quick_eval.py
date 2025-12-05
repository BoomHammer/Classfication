#!/usr/bin/env python3
"""
quick_eval.py: 分层分类模型验证脚本

使用方式：
1. 确保已安装所需的 Python 包。
2. 在终端中运行以下命令：
   ```
   python code/quick_eval.py --run_dir ../experiments/outputs/XXXXXXXX_XXXX_EXP_2023_001
   ```。
"""
print("💡 脚本正在启动...")

import torch
import json
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, accuracy_score

# 导入本地模块
sys.path.insert(0, str(Path(__file__).parent))
from config_manager import ConfigManager
from label_encoder import LabelEncoder
from point_timeseries_dataset import PointTimeSeriesDataset, collate_fn
from model_architecture import DualStreamSpatio_TemporalFusionNetwork

def load_model_weights(model, path, device):
    """安全加载模型权重"""
    try:
        # print(f"   ⏳ 加载权重: {path.name} ...")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return True
    except Exception as e:
        print(f"   ❌ 权重加载失败: {e}")
        return False

def predict_subset(model, dataset, indices, device, batch_size):
    """辅助函数：对指定索引的子集进行预测，返回局部预测结果"""
    # [修复] 使用 len() 判断，兼容 List 和 NumPy Array
    if len(indices) == 0:
        return []
    
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    local_preds = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            dyn = batch['dynamic'].to(device)
            sta = batch['static'].to(device)
            outputs = model(dyn, sta)
            preds = torch.argmax(outputs['probabilities'], dim=1)
            local_preds.extend(preds.cpu().numpy())
            
    return local_preds

def main():
    parser = argparse.ArgumentParser(description='分层模型快速评估')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--run_dir', type=str, help='指定实验输出目录')
    parser.add_argument('--split', type=str, default='val', help='评估数据集: val 或 test')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    print("="*60)
    print("🚀 启动全链路评估脚本")
    print("="*60)
    
    # 1. 初始化配置与路径
    config_path = Path(__file__).parent / args.config
    config = ConfigManager(str(config_path))
    
    if args.run_dir:
        output_dir = Path(args.run_dir)
        if not output_dir.exists():
            print(f"❌ 目录不存在: {output_dir}")
            sys.exit(1)
        print(f"📂 实验目录: {output_dir}")
    else:
        print("⚠️ 未指定 --run_dir，使用默认目录")
        output_dir = config.get_experiment_output_dir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  配置: Split={args.split}, Device={device}")
    
    # 2. 确定通道数
    param_file = output_dir / 'detected_parameters.json'
    if param_file.exists():
        with open(param_file, 'r') as f:
            params = json.load(f)
        dyn_ch = params['dynamic_channels']
        sta_ch = params['static_channels']
    else:
        print("⚠️ 自动推断通道数...")
        temp_ds = PointTimeSeriesDataset(config, None, split='val', verbose=False)
        dyn_ch = temp_ds.num_channels
        sta_ch = temp_ds.num_static_channels
    print(f"📊 通道: Dynamic={dyn_ch}, Static={sta_ch}")

    # 3. 加载映射
    major_map_file = output_dir / 'major_labels_map.json'
    detailed_map_file = output_dir / 'detailed_labels_map.json'
    
    if not major_map_file.exists():
        print(f"❌ 缺少映射文件，请检查目录")
        sys.exit(1)
        
    with open(major_map_file, 'r', encoding='utf-8') as f:
        major_map = json.load(f)
    with open(detailed_map_file, 'r', encoding='utf-8') as f:
        detailed_map = json.load(f)
    
    inverse_detailed_map = {v: k for k, v in detailed_map.items()}
    encoder = LabelEncoder(config=config, output_dir=output_dir)
    
    # 4. 加载数据集
    print(f"\n📦 加载 {args.split} 数据集...")
    dataset = PointTimeSeriesDataset(config, encoder, split=args.split, verbose=True)
    if len(dataset) == 0:
        print("❌ 数据集为空")
        sys.exit(1)
        
    # 获取用于索引的 DataFrame
    eval_df = dataset.points_df.iloc[dataset.indices].reset_index(drop=True)
    num_samples = len(dataset)
    
    # 初始化结果数组
    true_major_array = np.array(eval_df['major_label'])
    true_detail_array = np.array(eval_df['detail_label'])
    
    pred_major_array = np.full(num_samples, -1)
    
    # [关键] 两个小类预测数组
    # 1. Upper Bound: 假设大类已知，送入正确的小类模型 (反映小类模型本身能力)
    pred_detail_upper = np.full(num_samples, -1) 
    # 2. Pipeline: 依据预测的大类，送入对应的小类模型 (反映真实系统能力)
    pred_detail_pipeline = np.full(num_samples, -1)

    # =========================================================================
    # 阶段 A: 评估大类模型
    # =========================================================================
    print("\n" + "-"*50)
    print("🏗️  Step 1: 大类预测 (Major Prediction)")
    print("-"*50)
    
    major_model_path = output_dir / 'major_model' / 'best_model.pth'
    if major_model_path.exists():
        major_model = DualStreamSpatio_TemporalFusionNetwork(
            in_channels_dynamic=dyn_ch, in_channels_static=sta_ch, num_classes=len(major_map)
        ).to(device)
        
        if load_model_weights(major_model, major_model_path, device):
            # 对所有数据进行大类预测
            all_indices = list(range(num_samples))
            preds = predict_subset(major_model, dataset, all_indices, device, args.batch_size)
            pred_major_array = np.array(preds)
            
            # 输出报告
            print("\n📋 大类分类报告:")
            major_names = [k for k, v in sorted(major_map.items(), key=lambda x: x[1])]
            print(classification_report(true_major_array, pred_major_array, target_names=major_names, digits=4, zero_division=0))
    else:
        print(f"❌ 大类模型缺失: {major_model_path}")

    # =========================================================================
    # 阶段 B: 评估小类模型 (双路径)
    # =========================================================================
    print("\n" + "-"*50)
    print("🏗️  Step 2: 小类预测 (Detail Prediction)")
    print("-"*50)

    # 遍历每一个大类 ID
    for major_name, major_id in major_map.items():
        sub_model_dir = output_dir / f"detail_model_{major_id}_{major_name}"
        model_path = sub_model_dir / "best_model.pth"
        mapping_path = sub_model_dir / "class_mapping.json"
        
        # 如果该大类没有训练好的小类模型
        if not model_path.exists():
            continue 

        # 加载局部映射
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            local_to_global = {int(k): int(v) for k, v in mapping_data['local_to_global_map'].items()}
        except:
            continue
            
        num_sub_classes = len(local_to_global)
        
        # 加载模型
        sub_model = DualStreamSpatio_TemporalFusionNetwork(
            in_channels_dynamic=dyn_ch, in_channels_static=sta_ch, num_classes=num_sub_classes
        ).to(device)
        
        if not load_model_weights(sub_model, model_path, device):
            continue
        
        # --- 路径 1: Upper Bound (基于真实标签) ---
        true_indices = np.where(true_major_array == major_id)[0]
        if len(true_indices) > 0:
            local_preds = predict_subset(sub_model, dataset, true_indices, device, args.batch_size)
            global_preds = [local_to_global[p] for p in local_preds]
            pred_detail_upper[true_indices] = global_preds
            
        # --- 路径 2: Pipeline (基于大类预测) ---
        # 找出大类模型预测为当前 major_id 的所有样本 (可能包含误判进来的)
        pred_indices = np.where(pred_major_array == major_id)[0]
        if len(pred_indices) > 0:
            local_preds = predict_subset(sub_model, dataset, pred_indices, device, args.batch_size)
            global_preds = [local_to_global[p] for p in local_preds]
            pred_detail_pipeline[pred_indices] = global_preds
            
        print(f"👉 模型 [{major_name}]: 处理真实样本 {len(true_indices)} 个, 处理预测样本 {len(pred_indices)} 个")

    # =========================================================================
    # 阶段 C: 生成报告
    # =========================================================================
    print("\n" + "="*60)
    print("📊 最终评估报告")
    print("="*60)
    
    # 1. Upper Bound 报告
    valid_mask_upper = pred_detail_upper != -1
    if np.sum(valid_mask_upper) > 0:
        y_true = true_detail_array[valid_mask_upper]
        y_pred = pred_detail_upper[valid_mask_upper]
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        names = [inverse_detailed_map.get(i, str(i)) for i in unique_labels]
        
        print("\n✅ 小类分类报告 (Upper Bound - 假设大类正确):")
        print("   (仅包含已训练小类模型的类别)")
        print(classification_report(y_true, y_pred, target_names=names, digits=4, zero_division=0))
    
    # 2. Pipeline 报告
    valid_mask_pipe = pred_detail_pipeline != -1
    
    if np.sum(valid_mask_pipe) > 0:
        y_true = true_detail_array[valid_mask_pipe]
        y_pred = pred_detail_pipeline[valid_mask_pipe]
        
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        names = [inverse_detailed_map.get(i, str(i)) for i in unique_labels]
        
        print("\n🚀 总体各小类分类报告 (Pipeline - 真实流水线):")
        print("   (包含大类错误导致的传递误差)")
        print(classification_report(y_true, y_pred, target_names=names, digits=4, zero_division=0))
        
        acc = accuracy_score(y_true, y_pred)
        print(f"🏆 总体小类准确率 (Pipeline Accuracy): {acc:.2%}")
    else:
        print("\n❌ 无法生成流水线报告 (可能是大类模型未预测出任何有效类别)")

    # 3. 保存详细结果
    id_col = config.get('data_specs.csv_columns.id', 'Index')
    if id_col not in eval_df.columns:
        id_col = 'sample_id_generated'
        eval_df[id_col] = eval_df.index

    results_df = pd.DataFrame({
        'sample_id': eval_df[id_col],
        'true_major': [list(major_map.keys())[list(major_map.values()).index(i)] for i in true_major_array],
        'pred_major': [list(major_map.keys())[list(major_map.values()).index(i)] if i!=-1 else 'N/A' for i in pred_major_array],
        'true_detail': [inverse_detailed_map.get(i, str(i)) for i in true_detail_array],
        'pred_detail_upper': [inverse_detailed_map.get(i, str(i)) if i!=-1 else 'N/A' for i in pred_detail_upper],
        'pred_detail_pipeline': [inverse_detailed_map.get(i, str(i)) if i!=-1 else 'N/A' for i in pred_detail_pipeline]
    })
    
    csv_name = f"eval_full_report_{args.split}.csv"
    save_path = output_dir / csv_name
    results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 详细预测结果已保存: {save_path}")

if __name__ == "__main__":
    main()