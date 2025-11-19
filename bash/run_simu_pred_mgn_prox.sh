#!/bin/bash
#SBATCH -A p32685        ## 你的账户
#SBATCH -p normal        ## CPU 分区（R不需要GPU）
#SBATCH --nodes=1        ## 请求一个节点
#SBATCH --ntasks-per-node=64  ## 每个节点运行64个并行任务（可根据节点调整：52/64/128）
#SBATCH -t 48:00:00      ## 最大运行时间 48 小时
#SBATCH --mem=80G    
#SBATCH --output=run_simu_pred_mgn_prox_%j.out  ## 输出日志
#SBATCH --error=run_simu_pred_mgn_prox_%j.err   ## 错误日志

# 说明：
# Quest节点CPU数量：
# - quest10: 52 CPUs (可用52个任务)
# - quest11/quest12: 64 CPUs (可用64个任务)
# - quest13: 128 CPUs (可用128个任务)
# 根据实际分配的节点调整 --ntasks-per-node 的值

# 1) 加载R 4.4模块
module load R/4.4

# 2) 切换到项目目录
cd /home/hhz6461/cfsensitivity_paper

# 3) 运行仿真实验（marginal + prox + prox_est）
# 使用 xargs -P 并行执行多个 R 脚本，每个脚本使用 1 个 CPU 核心
# 固定 p=20, n=2000，测试 w_dim=18/20/22 三种情况

# 创建参数列表并并行执行（使用 SLURM 分配的 CPU 核心数）
# 运行缺失的配置：w_dim=22, alpha_id=1, gamma_id=4, seed=45/46/47
# 只针对 w_dim=22 运行 3 个 seed，共 3 个任务
for seed in 45 46 47; do
  echo "20 2000 22 1 4 $seed"
done | xargs -n 6 -P $SLURM_NTASKS_PER_NODE sh -c '
  cd simulations && \
  Rscript pred_mgn_prox.R "$@" && \
  Rscript pred_mgn_prox_est.R "$@"' _