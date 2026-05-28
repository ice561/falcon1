"""
每日自动化流程 — 一键运行完整的盘后工作流

自动判断:
  - 是否为交易日（简单处理：跳过周末）
  - 是否为调仓日（每20个交易日）

流程:
  1. 更新K线数据
  2. 更新因子数据
  3. 生成选股信号
  4. 若为调仓日 → 执行调仓
  5. 更新账户净值
  6. 生成报告 + 风险检查

用法:
  python auto_daily.py                    # 每日自动
  python auto_daily.py --force-rebalance  # 强制调仓
  python auto_daily.py --date 2026-05-26  # 指定日期
"""
import os
import sys
import io
import json
import argparse
import subprocess
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PYTHON = r"C:/Users/86199/AppData/Local/Programs/Python/Python312/python.exe"
PROJECT_DIR = r"c:\Users\86199\Desktop\新建文件夹"
REBALANCE_FREQ = 20  # 每20个交易日调仓

# 调仓日计数器文件
COUNTER_FILE = os.path.join(PROJECT_DIR, ".rebalance_counter")


def run_cmd(script: str, args: str = "", timeout: int = 300) -> bool:
    """执行一个python脚本，返回是否成功"""
    cmd = f'cd "{PROJECT_DIR}" && "{PYTHON}" {script} {args}'
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        print(result.stdout)
        if result.stderr:
            # 忽略 matplotlib 字体警告之类的
            for line in result.stderr.strip().split("\n"):
                if "WARNING" not in line and "INFO" not in line:
                    print(f"  [stderr] {line}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  [超时] {script} 超过 {timeout}s")
        return False
    except Exception as e:
        print(f"  [异常] {script}: {e}")
        return False


def is_trading_day(date_str: str) -> bool:
    """简单判断交易日：不是周末"""
    dt = pd.Timestamp(date_str)
    if dt.weekday() >= 5:
        return False
    # 简单节假日（可扩展）
    holidays = {
        "2026-01-01", "2026-01-02",  # 元旦
        "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",  # 春节
        "2026-04-06",  # 清明
        "2026-05-01", "2026-05-04",  # 劳动节
        "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07",  # 国庆
    }
    return date_str not in holidays


def get_rebalance_counter() -> int:
    """获取调仓日计数"""
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE) as f:
            return int(f.read().strip())
    return 0


def set_rebalance_counter(n: int):
    with open(COUNTER_FILE, "w") as f:
        f.write(str(n))


def main(date_str: str = "", force_rebalance: bool = False):
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    print("=" * 55)
    print(f"  猎鹰一号 — 每日自动化 ({date_str})")
    print("=" * 55)

    # ---- 0. 交易日检查 ----
    if not is_trading_day(date_str):
        print(f"\n  {date_str} 非交易日（周末/节假日），跳过")
        return

    # ---- 1. 生成信号 ----
    print(f"\n[1/4] 生成选股信号...")
    print("-" * 40)
    ok = run_cmd("daily_signals.py", f'--date {date_str} --no-timing')
    if not ok:
        print("  [警告] 信号生成部分失败")

    # ---- 2. 判断是否调仓 ----
    counter = get_rebalance_counter()
    is_rebalance = force_rebalance or (counter % REBALANCE_FREQ == 0 and counter > 0)

    if counter == 0:
        print(f"\n  首次运行，初始化调仓（counter=0）")
        is_rebalance = True

    if is_rebalance:
        print(f"\n[2/4] 调仓执行 (counter={counter})...")
        print("-" * 40)
        # 找最新信号文件
        json_files = sorted([
            f for f in os.listdir(PROJECT_DIR)
            if f.startswith("signals_") and f.endswith(".json")
        ])
        if json_files:
            signal_file = json_files[-1]
            ok = run_cmd("paper_trade.py", f'--signal {signal_file}', timeout=120)
            if ok:
                print(f"  调仓完成: {signal_file}")
            else:
                print("  [警告] 调仓部分失败")
        else:
            print("  [错误] 无信号文件，跳过调仓")
    else:
        print(f"\n[2/4] 非调仓日 (counter={counter})，跳过")

    # 更新计数器
    set_rebalance_counter(counter + 1)

    # ---- 3. 更新净值 ----
    print(f"\n[3/4] 更新账户净值...")
    print("-" * 40)
    run_cmd("paper_trade.py", "--update", timeout=60)

    # ---- 4. 生成报告 ----
    print(f"\n[4/4] 生成报告...")
    print("-" * 40)
    run_cmd("dashboard.py", timeout=30)
    run_cmd("dashboard.py", "--html --output dashboard.html", timeout=30)

    print(f"\n{'=' * 55}")
    print(f"  完成! 下次调仓日: counter={counter + 1} (每{REBALANCE_FREQ}日)")
    print(f"{'=' * 55}")


def setup_scheduled_task():
    """输出 Windows 计划任务配置说明"""
    print("""
=== Windows 计划任务配置 ===

1. 打开 任务计划程序 (taskschd.msc)

2. 创建任务:
   - 名称: 猎鹰一号-每日盘后
   - 触发器: 每天 15:30 (收盘后30分钟)
   - 操作: 启动程序
     程序: C:/Users/86199/AppData/Local/Programs/Python/Python312/python.exe
     参数: c:/Users/86199/Desktop/新建文件夹/auto_daily.py
     起始于: c:/Users/86199/Desktop/新建文件夹
   - 条件: 取消 "仅当计算机使用交流电源时启动"

3. 或者用命令行创建 (管理员):
   schtasks /create /tn "猎鹰一号-每日盘后" /tr "C:/Users/86199/AppData/Local/Programs/Python/Python312/python.exe c:/Users/86199/Desktop/新建文件夹/auto_daily.py" /sc daily /st 15:30 /f

=== 手动运行 ===
   python auto_daily.py                    每日自动
   python auto_daily.py --force-rebalance  强制今天调仓
""")


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description="猎鹰一号 — 每日自动化")
    parser.add_argument("--date", type=str, default="", help="目标日期 YYYY-MM-DD")
    parser.add_argument("--force-rebalance", action="store_true", help="强制调仓")
    parser.add_argument("--setup", action="store_true", help="显示计划任务配置")
    args = parser.parse_args()

    if args.setup:
        setup_scheduled_task()
    else:
        main(date_str=args.date, force_rebalance=args.force_rebalance)
