"""
智能文件名时间解析模块

功能：
1. 自动检测可变长度的数据类型前缀（2-4字母）
2. 解析不同格式的时间信息（日期、月份、年份）
3. 智能区分歧义情况（如2020可能是年份或月份）
4. 支持多种数据类型及特殊后缀（如SR系列波段）

示例：
- GPP230101.tif → 2023年1月1日 (类型: GPP)
- NDVI220306.tif → 2022年3月6日 (类型: NDVI)
- SR230210B1.tif → 2023年2月10日 (类型: SR_B1)
- PR2002.tif → 2020年2月 (类型: PR)
"""

import re
from datetime import datetime
from typing import Optional, Tuple, Dict

import logging


logger = logging.getLogger(__name__)


def extract_time_from_filename(filename: str) -> Tuple[Optional[datetime], Optional[int], Optional[int], str]:
    """
    从文件名智能提取时间信息
    
    Args:
        filename: 文件名（例如 'GPP230101.tif' 或 'SR230210B1.tif'）
    
    Returns:
        Tuple[date, year, month, data_type]: (完整日期对象, 年份, 月份, 数据类型)
        - date: 完整的日期对象（如果能解析的话）
        - year: 仅年份（从任何数字序列中提取）
        - month: 仅月份（从任何数字序列中提取）
        - data_type: 识别的数据类型标识（如 'GPP', 'SR_B1'）
    
    示例：
        >>> extract_time_from_filename('GPP230101.tif')
        (datetime(2023,1,1), 2023, 1, 'GPP')
        
        >>> extract_time_from_filename('SR230210B1.tif')
        (datetime(2023,2,10), 2023, 2, 'SR_B1')
    """
    
    # 移除文件扩展名
    name_without_ext = filename.rsplit('.', 1)[0]
    
    # 第1步：提取前缀、数字部分和后缀
    # 匹配：2-4个字母(前缀) + 数字序列 + 剩余部分(后缀)
    # 这里的 (.*) 会捕获如 'B1' 这样的后缀
    match = re.match(r'^([A-Za-z]{2,4})(\d+)(.*)$', name_without_ext)
    
    if not match:
        logger.debug(f"❌ 无法解析文件名格式: {filename}")
        return None, None, None, ''
    
    prefix = match.group(1)  # 数据类型前缀（如 'GPP', 'SR'）
    number_str = match.group(2)  # 数字序列（如 '230101', '230210'）
    suffix = match.group(3)      # 后缀部分（如 'B1', ''）
    
    # 特殊处理：SR系列数据，将后缀（波段号）合并到数据类型中
    # 例如：SR...B1 -> 数据类型记为 SR_B1
    if prefix.upper() == 'SR' and suffix:
        prefix = f"{prefix}_{suffix}"
    
    logger.debug(f"✓ 解析 {filename}: type={prefix}, numbers={number_str}")
    
    # 第2步：根据数字长度判断时间格式
    date_obj = None
    year = None
    month = None
    
    num_len = len(number_str)
    
    if num_len == 6:
        # 格式: YYMMDD （日期数据）
        try:
            yy = int(number_str[:2])
            mm = int(number_str[2:4])
            dd = int(number_str[4:6])
            
            # 将两位数年份转换为四位数
            # 假设00-50是2000-2050, 51-99是1951-1999
            if yy <= 50:
                yyyy = 2000 + yy
            else:
                yyyy = 1900 + yy
            
            # 验证月份和日期的有效性
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                date_obj = datetime(yyyy, mm, dd)
                year = yyyy
                month = mm
                logger.debug(f"  ✓ 6位数字格式: {yyyy}-{mm:02d}-{dd:02d}")
            else:
                logger.debug(f"  ⚠️  月份或日期超出范围: MM={mm}, DD={dd}")
        except (ValueError, TypeError) as e:
            logger.debug(f"  ⚠️  无法解析6位数字为日期: {e}")
    
    elif num_len == 4:
        # 格式: YYMM 或 YYYY（需要智能判断）
        try:
            first_two = int(number_str[:2])
            last_two = int(number_str[2:4])
            
            # 判断逻辑：
            # - 如果 last_two > 12，不可能是月份，则 number_str 是 YYYY（年份）
            # - 如果 last_two <= 12，则判断为 YYMM（年月）
            
            if last_two > 12:
                # YYYY 格式（年份数据）
                yyyy = int(number_str)
                if 1900 <= yyyy <= 2100:
                    year = yyyy
                    logger.debug(f"  ✓ 4位数字格式（年份）: {yyyy}")
                else:
                    logger.debug(f"  ⚠️  年份超出合理范围: {yyyy}")
            else:
                # YYMM 格式（年月数据）
                # 两位数年份转换
                if first_two <= 50:
                    yyyy = 2000 + first_two
                else:
                    yyyy = 1900 + first_two
                
                mm = last_two
                if 1 <= mm <= 12:
                    year = yyyy
                    month = mm
                    date_obj = datetime(yyyy, mm, 1)  # 使用月初作为代表日期
                    logger.debug(f"  ✓ 4位数字格式（年月）: {yyyy}-{mm:02d}")
                else:
                    logger.debug(f"  ⚠️  月份超出范围: {mm}")
        except (ValueError, TypeError) as e:
            logger.debug(f"  ⚠️  无法解析4位数字: {e}")
    
    else:
        # 其他长度的数字
        logger.debug(f"  ⚠️  数字长度 {num_len} 不支持（只支持4或6位）")
    
    return date_obj, year, month, prefix


def test_time_parser():
    """测试时间解析函数"""
    test_cases = [
        ('GPP230101.tif', (2023, 1, 1), 'GPP'),
        ('NDVI220306.tif', (2022, 3, 6), 'NDVI'),
        ('SR230210B1.tif', (2023, 2, 10), 'SR_B1'),  # 新增测试用例：SR系列 Band 1
        ('SR230210B7.tif', (2023, 2, 10), 'SR_B7'),  # 新增测试用例：SR系列 Band 7
        ('PR2002.tif', (2020, 2, None), 'PR'),
        ('PR2020.tif', (2020, None, None), 'PR'),
    ]
    
    print("\n" + "=" * 80)
    print("时间解析测试")
    print("=" * 80)
    
    for filename, (exp_year, exp_month, exp_day), exp_prefix in test_cases:
        date_obj, year, month, prefix = extract_time_from_filename(filename)
        
        status = "✅"
        # 验证日期
        if date_obj:
            if date_obj.year != exp_year or date_obj.month != exp_month or date_obj.day != exp_day:
                status = "❌ (日期错误)"
        elif year != exp_year or month != exp_month:
            status = "❌ (年月错误)"
            
        # 验证前缀(数据类型)
        if prefix != exp_prefix:
            status = f"❌ (类型错误: 期望 {exp_prefix}, 实际 {prefix})"
        
        print(f"\n{status} {filename}")
        print(f"   解析类型: {prefix}")
        print(f"   解析日期: {date_obj if date_obj else f'{year}-{month}'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    test_time_parser()
    print("\n" + "=" * 80)
    print("✅ 时间解析模块测试完成")
    print("=" * 80 + "\n")