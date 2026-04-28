import os

# ==========================================
# 核心网络修复：强制在代码级别注入国内高速镜像节点
# 必须放在 import huggingface_hub 之前！
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import shutil
import subprocess
from huggingface_hub import snapshot_download

# ==========================================
# 1. 目录结构配置
# ==========================================
BASE_DIR = os.getcwd()  
DATA_DIR = os.path.join(BASE_DIR, "data")

DIRS = {
    "scalevln_episode": os.path.join(DATA_DIR, "datasets", "scalevln"),
    "trajectory": os.path.join(DATA_DIR, "trajectory_data"),
    "llava_video": os.path.join(DATA_DIR, "co-training_data", "LLaVA-Video-178K"),
    "scannet": os.path.join(DATA_DIR, "co-training_data", "ScanNet"),
    "mmc4": os.path.join(DATA_DIR, "co-training_data", "MMC4-core")
}

def create_directories():
    print("[*] 正在初始化目录树...")
    for name, path in DIRS.items():
        os.makedirs(path, exist_ok=True)
        print(f"    - 创建/确认目录: {path}")

# ==========================================
# 2. 下载核心逻辑
# ==========================================
def download_huggingface_datasets():
    
    # 2.1 下载 ScaleVLN & Trajectory Data
    print("\n[1/3] 开始下载轨迹数据与 ScaleVLN Episodes...")
    snapshot_download(
        repo_id="cywan/StreamVLN-Trajectory-Data",
        repo_type="dataset",
        local_dir=DIRS["trajectory"],
        max_workers=8
        # 已移除过期的 local_dir_use_symlinks 参数
    )
    
    source_json = os.path.join(DIRS["trajectory"], "ScaleVLN", "scalevln_subset_150k.json.gz")
    target_json = os.path.join(DIRS["scalevln_episode"], "scalevln_subset_150k.json.gz")
    
    if os.path.exists(source_json):
        shutil.move(source_json, target_json)
        print(f"    -> [文件分离完成] 已移动 ScaleVLN Episode 数据。")
    elif os.path.exists(target_json):
        print("    -> [跳过] ScaleVLN Episode 数据已存在目标位置。")
    else:
        print("    -> [警告] 未找到 json.gz，请检查网络是否完整下载。")

    # 2.2 下载 LLaVA-Video-178K
    print("\n[2/3] 开始下载协同训练数据: LLaVA-Video-178K...")
    snapshot_download(
        repo_id="lmms-lab/LLaVA-Video-178K",
        repo_type="dataset",
        local_dir=DIRS["llava_video"],
        max_workers=8
    )

    # 2.3 下载 ScanQA & SQA3D
    print("\n[3/3] 开始下载协同训练数据: ScanNet 标注...")
    snapshot_download(
        repo_id="chchnii/StreamVLN-ScanQA-SQA3D-Data",
        repo_type="dataset",
        local_dir=DIRS["scannet"],
        max_workers=4
    )

# ==========================================
# 3. GitHub 仓库克隆逻辑
# ==========================================
def clone_github_repositories():
    print("\n[4] 开始获取 MMC4-core 脚本仓库...")
    
    if not os.listdir(DIRS["mmc4"]):
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/allenai/mmc4.git", DIRS["mmc4"]],
                check=True
            )
            print(f"    -> MMC4-core 仓库已克隆。")
        except subprocess.CalledProcessError as e:
            print(f"    -> [错误] Git Clone MMC4 失败: {e}")
    else:
        print(f"    -> [跳过] MMC4-core 目录非空。")

# ==========================================
# 主函数入口
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("StreamVLN 数据集自动化下载部署脚本启动 (镜像加速版)")
    print("="*60)
    
    create_directories()
    download_huggingface_datasets()
    clone_github_repositories()
    
    print("\n" + "="*60)
    print("所有服务器端自动化下载任务执行完毕。")
    print("="*60)