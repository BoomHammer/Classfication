#!/usr/bin/env python3
"""
quick_eval.py: 分层分类模型验证脚本

使用方式：
1. 确保已安装所需的 Python 包。
2. 在终端中运行以下命令：
   ```
   cd code
   python quick_eval.py --run_dir ../experiments/outputs/XXXXXXXX_XXXX_EXP_2023_001
   ```

修复说明：
强制使用本地的 config.yaml 而不是实验目录下的 config_used.yaml，
以防止相对路径解析错误 (FileNotFoundError)。
"""
print("💡 脚本正在启动...")

import sys
import json
import logging
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# 导入本地模块
sys.path.insert(0, str(Path(__file__).parent))
from config_manager import ConfigManager
from label_encoder import LabelEncoder
from raster_crawler import RasterCrawler
from point_timeseries_dataset import PointTimeSeriesDataset, collate_fn
from model_architecture import DualStreamSpatio_TemporalFusionNetwork

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_major_model(run_dir, num_classes, input_channels, device):
    """加载大类模型"""
    model_path = run_dir / "major_model" / "best_model.pth"
    if not model_path.exists():
        # 尝试加载 last_model.pth 作为备选
        model_path = run_dir / "major_model" / "last_model.pth"
        if not model_path.exists():
             raise FileNotFoundError(f"❌ 大类模型文件未找到: {model_path}")
    
    print(f"📦 加载大类模型: {model_path}")
    model = DualStreamSpatio_TemporalFusionNetwork(
        in_channels_dynamic=input_channels['dynamic'],
        in_channels_static=input_channels['static'],
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_detail_models(run_dir, hierarchical_map, input_channels, device):
    """
    加载所有小类模型
    """
    models = {}
    mappings = {}
    single_class_map = {}
    
    print("📦 加载小类模型...")
    
    for major_name, info in hierarchical_map.items():
        major_id = info['major_id']
        detail_classes = info['detail_classes']
        
        # 情况1：只有一个小类，没有训练模型，直接记录ID
        if len(detail_classes) <= 1:
            global_id = list(detail_classes.values())[0]
            single_class_map[major_id] = global_id
            continue
            
        # 情况2：有多个小类，加载对应的模型
        model_folder = run_dir / f"detail_model_{major_id}_{major_name}"
        model_path = model_folder / "best_model.pth"
        if not model_path.exists():
             model_path = model_folder / "last_model.pth"

        mapping_path = model_folder / "class_mapping.json"
        
        if not model_path.exists() or not mapping_path.exists():
            print(f"  ⚠️  警告: 未找到大类 {major_name} 的模型文件，跳过。")
            continue
            
        # 加载映射配置
        with open(mapping_path, 'r', encoding='utf-8') as f:
            map_data = json.load(f)
        # 转换 key 为 int
        local_to_global = {int(k): int(v) for k, v in map_data['local_to_global_map'].items()}
        mappings[major_id] = local_to_global
        
        # 加载模型
        sub_model = DualStreamSpatio_TemporalFusionNetwork(
            in_channels_dynamic=input_channels['dynamic'],
            in_channels_static=input_channels['static'],
            num_classes=len(detail_classes)
        )
        sub_model.load_state_dict(torch.load(model_path, map_location=device))
        sub_model.to(device)
        sub_model.eval()
        models[major_id] = sub_model
        
    return models, mappings, single_class_map

def predict_batch(dynamic, static, major_model, detail_models, detail_mappings, single_class_map, device):
    """
    对一个 Batch 进行级联预测
    """
    batch_size = dynamic.size(0)
    
    # 1. 预测大类
    with torch.no_grad():
        major_outputs = major_model(dynamic, static)
        major_preds = torch.argmax(major_outputs['logits'], dim=1) 
    
    detail_preds_global = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # 2. 预测小类 (路由逻辑)
    unique_major_ids = torch.unique(major_preds)
    
    for mid in unique_major_ids:
        mid_item = mid.item()
        indices = (major_preds == mid)
        
        sub_dynamic = dynamic[indices]
        sub_static = static[indices]
        
        if mid_item in detail_models:
            # A. 调用小类模型
            model = detail_models[mid_item]
            mapping = detail_mappings[mid_item]
            
            with torch.no_grad():
                sub_out = model(sub_dynamic, sub_static)
                sub_preds_local = torch.argmax(sub_out['logits'], dim=1)
            
            # 映射回全局ID
            sub_preds_local_np = sub_preds_local.cpu().numpy()
            sub_preds_global_np = [mapping[loc_id] for loc_id in sub_preds_local_np]
            
            detail_preds_global[indices] = torch.tensor(sub_preds_global_np, device=device)
            
        elif mid_item in single_class_map:
            # B. 只有一个小类
            target_global_id = single_class_map[mid_item]
            detail_preds_global[indices] = target_global_id
            
        else:
            # C. 异常情况
            detail_preds_global[indices] = -1 
            
    return major_preds, detail_preds_global

def main():
    parser = argparse.ArgumentParser(description="分层模型验证脚本")
    parser.add_argument('--run_dir', type=str, required=True, help="实验输出目录路径")
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'train'], help="数据集划分")
    parser.add_argument('--batch_size', type=int, default=32, help="批次大小")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"❌ 目录不存在: {run_dir}")
        return

    # =========================================================
    # 关键修复: 始终加载本地的 config.yaml
    # =========================================================
    # 假设 evaluate.py 和 config.yaml 在同一个目录 (code/)
    local_config_path = Path(__file__).parent / 'config.yaml'
    
    if not local_config_path.exists():
        print(f"❌ 找不到本地配置文件: {local_config_path}")
        print("请确保脚本运行在 code 目录下，且 config.yaml 存在。")
        return
        
    print(f"📋 加载配置文件: {local_config_path}")
    # 使用本地路径初始化，这样相对路径 (../data) 才会解析正确
    config = ConfigManager(str(local_config_path))
    
    # 2. 准备数据集
    print("🔄 初始化数据加载器...")
    encoder = LabelEncoder(config=config)
    
    dynamic_crawler = RasterCrawler(
        config=config, 
        raster_dir=config.get_resolved_path('dynamic_images_dir'), 
        filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'),
        file_extensions=['.tif']
    )
    static_crawler = RasterCrawler(
        config=config, 
        raster_dir=config.get_resolved_path('static_images_dir'), 
        filename_pattern=config.get('data_specs.raster_crawler.filename_pattern'),
        file_extensions=['.tif']
    )
    
    # 自动检测通道数
    try:
        dyn_ch = dynamic_crawler.detect_num_channels()['most_common']
        sta_ch = static_crawler.detect_num_channels()['most_common']
    except Exception as e:
        print(f"⚠️ 无法自动检测通道数，尝试读取 detected_parameters.json")
        # 尝试从运行目录读取
        param_file = run_dir / 'detected_parameters.json'
        if param_file.exists():
            with open(param_file, 'r') as f:
                params = json.load(f)
                dyn_ch = params.get('dynamic_channels', 4)
                sta_ch = params.get('static_channels', 1)
        else:
            print("❌ 无法确定输入通道数，请检查数据路径。")
            return

    input_channels = {'dynamic': dyn_ch, 'static': sta_ch}
    
    dataset = PointTimeSeriesDataset(
        config=config, 
        encoder=encoder, 
        dynamic_crawler=dynamic_crawler, 
        static_crawler=static_crawler, 
        split=args.split, 
        cache_metadata=True, 
        verbose=False
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0) # Windows下设为0更安全
    
    print(f"📊 验证集样本数: {len(dataset)}")
    
    # 3. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    major_map = encoder.get_major_labels_map()
    hierarchical_map = encoder.get_hierarchical_map()
    
    try:
        major_model = load_major_model(run_dir, len(major_map), input_channels, device)
        detail_models, detail_mappings, single_class_map = load_detail_models(
            run_dir, hierarchical_map, input_channels, device
        )
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 执行推理
    print("\n🚀 开始分层推理...")
    all_results = []
    
    pbar = tqdm(dataloader, desc="Eval")
    for batch in pbar:
        dynamic = batch['dynamic'].to(device)
        static = batch['static'].to(device)
        major_true = batch['major_label'].to(device)
        detail_true = batch['detail_label'].to(device)
        # 获取ID, 兼容不同 dataset 实现
        ids = batch.get('id', torch.zeros(len(major_true))).cpu().numpy()
        
        major_preds, detail_preds = predict_batch(
            dynamic, static, 
            major_model, detail_models, detail_mappings, single_class_map, 
            device
        )
        
        for i in range(len(ids)):
            all_results.append({
                'id': ids[i],
                'major_true': major_true[i].item(),
                'major_pred': major_preds[i].item(),
                'detail_true': detail_true[i].item(),
                'detail_pred': detail_preds[i].item()
            })
            
    # 5. 生成报告
    if not all_results:
        print("❌ 未生成任何预测结果，请检查数据加载器。")
        return

    df_res = pd.DataFrame(all_results)
    
    inv_major_map = {v: k for k, v in major_map.items()}
    detailed_map = encoder.get_detailed_labels_map()
    inv_detail_map = {v: k for k, v in detailed_map.items()}
    
    df_res['major_true_name'] = df_res['major_true'].map(inv_major_map)
    df_res['major_pred_name'] = df_res['major_pred'].map(inv_major_map)
    df_res['detail_true_name'] = df_res['detail_true'].map(inv_detail_map)
    df_res['detail_pred_name'] = df_res['detail_pred'].map(inv_detail_map)
    
    df_res['major_correct'] = df_res['major_true'] == df_res['major_pred']
    df_res['detail_correct'] = df_res['detail_true'] == df_res['detail_pred']
    
    print("\n" + "="*60)
    print("📊 验证结果报告")
    print("="*60)
    
    # 指标计算
    major_acc = accuracy_score(df_res['major_true'], df_res['major_pred'])
    print(f"\n✅ 大类总体准确率 (Major Accuracy): {major_acc:.2%}")
    # 避免 warning: 指定 labels
    unique_major = sorted(list(df_res['major_true'].unique()))
    print("\n大类分类报告:")
    print(classification_report(
        df_res['major_true'], 
        df_res['major_pred'], 
        labels=unique_major,
        target_names=[inv_major_map.get(i, str(i)) for i in unique_major], 
        digits=4,
        zero_division=0
    ))
    
    detail_acc = accuracy_score(df_res['detail_true'], df_res['detail_pred'])
    print(f"\n✅ 小类总体准确率 (Detail Accuracy): {detail_acc:.2%}")
    
    conditional_df = df_res[df_res['major_correct']]
    if len(conditional_df) > 0:
        cond_acc = accuracy_score(conditional_df['detail_true'], conditional_df['detail_pred'])
        print(f"👉 大类正确条件下的小类准确率: {cond_acc:.2%}")
    
    # 保存结果
    output_csv = run_dir / f"evaluation_predictions_{args.split}.csv"
    cols = ['id', 'major_true_name', 'major_pred_name', 'major_correct', 
            'detail_true_name', 'detail_pred_name', 'detail_correct',
            'major_true', 'major_pred', 'detail_true', 'detail_pred']
    df_res[cols].to_csv(output_csv, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()