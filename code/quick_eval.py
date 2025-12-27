#!/usr/bin/env python3
"""
quick_eval.py: 分层分类模型验证脚本 (适配 Segmentation 架构 + 完善指标输出)

使用方式：
   python code/quick_eval.py --run_dir experiments/outputs/XXXXXXXX_XXXX_EXP_2023_001
"""

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
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# 导入本地模块
sys.path.insert(0, str(Path(__file__).parent))
from config_manager import ConfigManager
from label_encoder import LabelEncoder
from point_timeseries_dataset import PointTimeSeriesDataset, collate_fn
from model_architecture import DualStreamSpatio_TemporalFusionNetwork

# [重要] 必须与 main.py 中的设置保持一致
MAX_TEMPORAL_STEPS = 64 

def load_model_weights(model, path, device):
    """安全加载模型权重"""
    try:
        # 使用 weights_only=True 消除警告
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return True
    except RuntimeError as e:
        print(f"   ❌ 权重加载失败 (尺寸不匹配?): {e}")
        return False
    except Exception as e:
        print(f"   ❌ 权重加载失败: {e}")
        return False

def get_center_predictions(outputs_dict):
    """
    从分割模型的输出 (B, C, H, W) 中提取中心像素的预测
    """
    # (B, C, H, W)
    probs = torch.softmax(outputs_dict['logits'], dim=1)
    B, C, H, W = probs.shape
    
    # 取中心像素
    center_h, center_w = H // 2, W // 2
    center_probs = probs[:, :, center_h, center_w] # (B, C)
    
    return center_probs

def predict_subset_ensemble(models_list, dataset, indices, device, batch_size, method='voting'):
    """
    集合预测：使用多个模型进行投票或概率平均
    适配：提取中心像素进行评估
    """
    if len(indices) == 0:
        return []
    
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    ensemble_preds = []
    
    for batch in dataloader:
        dyn = batch['dynamic'].to(device)
        sta = batch['static'].to(device)
        batch_size_actual = dyn.size(0)
        
        if method == 'voting':
            # 多数投票
            all_preds = []
            for model in models_list:
                model.eval()
                with torch.no_grad():
                    outputs = model(dyn, sta)
                    # 提取中心像素概率 -> 预测类别
                    center_probs = get_center_predictions(outputs)
                    preds = torch.argmax(center_probs, dim=1)
                    all_preds.append(preds.cpu().numpy())
            
            all_preds = np.array(all_preds)  # (num_models, batch_size)
            
            ensemble_batch = []
            for i in range(batch_size_actual):
                votes = all_preds[:, i]
                vote_result = np.bincount(votes.astype(int))
                pred_class = np.argmax(vote_result)
                ensemble_batch.append(pred_class)
            ensemble_preds.extend(ensemble_batch)
            
        elif method == 'averaging':
            # 概率平均
            all_probs = []
            for model in models_list:
                model.eval()
                with torch.no_grad():
                    outputs = model(dyn, sta)
                    # 提取中心像素概率
                    center_probs = get_center_predictions(outputs)
                    all_probs.append(center_probs.cpu().numpy())
            
            all_probs = np.array(all_probs)  # (num_models, batch_size, num_classes)
            avg_probs = np.mean(all_probs, axis=0)  # (batch_size, num_classes)
            preds = np.argmax(avg_probs, axis=1)
            ensemble_preds.extend(preds)
    
    return ensemble_preds

def main():
    parser = argparse.ArgumentParser(description='分层模型快速评估')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--run_dir', type=str, help='指定实验输出目录')
    parser.add_argument('--split', type=str, default='val', help='评估数据集: val 或 test')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    print("="*60)
    print("🚀 启动全链路评估脚本 (适配 Segmentation 架构)")
    print("="*60)
    
    # 1. 初始化配置与路径
    config_path = Path(__file__).parent / args.config
    config = ConfigManager(str(config_path), create_experiment_dir=False)
    
    if args.run_dir:
        output_dir = Path(args.run_dir)
        if not output_dir.exists():
            print(f"❌ 目录不存在: {output_dir}")
            sys.exit(1)
        print(f"📂 实验目录: {output_dir}")
    else:
        output_dir = config.get_experiment_output_dir()
        print(f"⚠️ 未指定 --run_dir，将在基础目录寻找资源: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 确定参数
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
    
    # 获取 patch_size
    patch_size = config.get('data_specs.spatial.patch_size', 64)
    print(f"📊 参数: Dynamic={dyn_ch}, Static={sta_ch}, Patch={patch_size}, T_Steps={MAX_TEMPORAL_STEPS}")

    # 3. 加载映射
    major_map_file = output_dir / 'major_labels_map.json'
    detailed_map_file = output_dir / 'detailed_labels_map.json'
    
    if not major_map_file.exists():
        print(f"❌ 缺少映射文件，请检查目录: {output_dir}")
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
        
    eval_df = dataset.points_df.iloc[dataset.indices].reset_index(drop=True)
    num_samples = len(dataset)
    
    true_major_array = np.array(eval_df['major_label'])
    true_detail_array = np.array(eval_df['detail_label'])
    
    pred_major_array = np.full(num_samples, -1)
    pred_detail_upper = np.full(num_samples, -1) 
    pred_detail_pipeline = np.full(num_samples, -1)

    report_lines = []
    def append_report(s):
        text = str(s)
        print(text)
        report_lines.append(text)

    # =========================================================================
    # 阶段 A: 评估大类模型
    # =========================================================================
    print("\n" + "-"*50)
    print("🏗️  Step 1: 大类预测 (Major Prediction)")
    print("-"*50)
    
    major_model_dir = output_dir / 'major_model'
    fold_models = []
    
    classifier_hidden_dims = config.get('model.classifier.hidden_dims', [128, 64, 32])
    
    for fold_idx in range(1, 6):
        fold_path = major_model_dir / f'fold_{fold_idx}' / 'best_model.pth'
        if fold_path.exists():
            # [修正] 传入新架构所需的完整参数
            major_model = DualStreamSpatio_TemporalFusionNetwork(
                in_channels_dynamic=dyn_ch, 
                in_channels_static=sta_ch, 
                num_classes=len(major_map),
                patch_size=patch_size,
                temporal_steps=MAX_TEMPORAL_STEPS, # 关键修复
                classifier_hidden_dims=classifier_hidden_dims
            ).to(device)
            
            if load_model_weights(major_model, fold_path, device):
                fold_models.append(major_model)
                print(f"   ✅ 加载大类模型 fold_{fold_idx}")
    
    if len(fold_models) > 0:
        all_indices = list(range(num_samples))
        preds = predict_subset_ensemble(fold_models, dataset, all_indices, device, args.batch_size, method='voting')
        pred_major_array = np.array(preds)
        
        append_report(f"\n📊 大类集合预测 (Models: {len(fold_models)}):")
        major_names = [k for k, v in sorted(major_map.items(), key=lambda x: x[1])]
        major_report = classification_report(true_major_array, pred_major_array, target_names=major_names, digits=4, zero_division=0)
        append_report(major_report)
        
        # [新增] 大类总体指标
        m_oa = accuracy_score(true_major_array, pred_major_array)
        m_prec = precision_score(true_major_array, pred_major_array, average='macro', zero_division=0)
        m_rec = recall_score(true_major_array, pred_major_array, average='macro', zero_division=0)
        m_f1 = f1_score(true_major_array, pred_major_array, average='macro', zero_division=0)
        
        append_report("-" * 40)
        append_report(f"🔢 大类总体指标 (Major Overall Metrics):")
        append_report(f"   • OA (Accuracy) : {m_oa:.4f}")
        append_report(f"   • Macro Precision: {m_prec:.4f}")
        append_report(f"   • Macro Recall   : {m_rec:.4f}")
        append_report(f"   • Macro F1       : {m_f1:.4f}")
        append_report("-" * 40)

    else:
        print(f"❌ 未找到大类模型权重")

    # =========================================================================
    # 阶段 B: 评估小类模型
    # =========================================================================
    print("\n" + "-"*50)
    print("🏗️  Step 2: 小类预测 (Detail Prediction)")
    print("-"*50)

    for major_name, major_id in major_map.items():
        sub_model_dir = output_dir / f"detail_model_{major_id}_{major_name}"
        mapping_path = sub_model_dir / "class_mapping.json"
        
        if not (sub_model_dir / 'fold_1' / 'best_model.pth').exists():
            continue 

        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            local_to_global = {int(k): int(v) for k, v in mapping_data['local_to_global_map'].items()}
        except:
            continue
            
        num_sub_classes = len(local_to_global)
        fold_models = []
        
        for fold_idx in range(1, 6):
            fold_path = sub_model_dir / f'fold_{fold_idx}' / 'best_model.pth'
            if fold_path.exists():
                # [修正] 传入新架构所需的完整参数
                sub_model = DualStreamSpatio_TemporalFusionNetwork(
                    in_channels_dynamic=dyn_ch, 
                    in_channels_static=sta_ch, 
                    num_classes=num_sub_classes,
                    patch_size=patch_size,
                    temporal_steps=MAX_TEMPORAL_STEPS, # 关键修复
                    classifier_hidden_dims=classifier_hidden_dims
                ).to(device)
                
                if load_model_weights(sub_model, fold_path, device):
                    fold_models.append(sub_model)
        
        if len(fold_models) == 0:
            continue
        
        # Upper Bound
        true_indices = np.where(true_major_array == major_id)[0]
        if len(true_indices) > 0:
            local_preds = predict_subset_ensemble(fold_models, dataset, true_indices, device, args.batch_size, method='voting')
            global_preds = [local_to_global[p] for p in local_preds]
            pred_detail_upper[true_indices] = global_preds
            
        # Pipeline
        pred_indices = np.where(pred_major_array == major_id)[0]
        if len(pred_indices) > 0:
            local_preds = predict_subset_ensemble(fold_models, dataset, pred_indices, device, args.batch_size, method='voting')
            global_preds = [local_to_global[p] for p in local_preds]
            pred_detail_pipeline[pred_indices] = global_preds
            
        print(f"👉 [{major_name}] Models: {len(fold_models)} | Samples: True {len(true_indices)}, Pred {len(pred_indices)}")

    # =========================================================================
    # 阶段 C: 生成报告
    # =========================================================================
    print("\n" + "="*60)
    print("📊 最终评估报告")
    print("="*60)
    
    # Pipeline Report
    valid_mask_pipe = pred_detail_pipeline != -1
    if np.sum(valid_mask_pipe) > 0:
        y_true = true_detail_array[valid_mask_pipe]
        y_pred = pred_detail_pipeline[valid_mask_pipe]
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        names = [inverse_detailed_map.get(i, str(i)) for i in unique_labels]
        
        append_report("\n🚀 总体各小类分类报告 (Pipeline):")
        pipe_report = classification_report(y_true, y_pred, target_names=names, digits=4, zero_division=0)
        append_report(pipe_report)
        
        # [新增] 小类总体指标 (完善版)
        oa = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        append_report("-" * 40)
        append_report(f"🔢 小类总体指标 (Detail Overall Metrics):")
        append_report(f"   • OA (Accuracy) : {oa:.4f}")
        append_report(f"   • Macro Precision: {prec:.4f}")
        append_report(f"   • Macro Recall   : {rec:.4f}")
        append_report(f"   • Macro F1       : {f1:.4f}")
        append_report("-" * 40)
    else:
        append_report("\n❌ 无法生成流水线报告")

    # Save
    id_col = config.get('data_specs.csv_columns.id', 'Index')
    if id_col not in eval_df.columns:
        id_col = 'sample_id_generated'
        eval_df[id_col] = eval_df.index

    results_df = pd.DataFrame({
        'sample_id': eval_df[id_col],
        'true_major': [list(major_map.keys())[list(major_map.values()).index(i)] for i in true_major_array],
        'pred_major': [list(major_map.keys())[list(major_map.values()).index(i)] if i!=-1 else 'N/A' for i in pred_major_array],
        'true_detail': [inverse_detailed_map.get(i, str(i)) for i in true_detail_array],
        'pred_detail_pipeline': [inverse_detailed_map.get(i, str(i)) if i!=-1 else 'N/A' for i in pred_detail_pipeline]
    })
    
    csv_name = f"eval_full_report_{args.split}.csv"
    save_path = output_dir / csv_name
    results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 结果已保存: {save_path}")

    try:
        report_path = output_dir / f"eval_report_{args.split}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    except Exception as e:
        print(f"❌ 保存报告失败: {e}")

if __name__ == "__main__":
    main()