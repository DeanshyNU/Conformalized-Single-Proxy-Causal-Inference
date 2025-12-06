#!/bin/bash
#SBATCH -A p32685        ## 你的账户
#SBATCH -p normal        ## CPU 分区（R不需要GPU）
#SBATCH --nodes=1        ## 请求一个节点
#SBATCH --ntasks-per-node=52  ## 每个节点运行52个并行任务（可根据节点调整：52/64/128）
#SBATCH -t 48:00:00      ## 最大运行时间 48 小时
#SBATCH --mem=200G    
#SBATCH --output=run_simu_pred_mgn_prox_discrete_est_%j.out  ## 输出日志（离散版本估计边界）
#SBATCH --error=run_simu_pred_mgn_prox_discrete_est_%j.err   ## 错误日志（离散版本估计边界）

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

# 3) 运行仿真实验（离散版本的估计边界）
# 使用 xargs -P 并行执行多个 R 脚本，每个脚本使用 1 个 CPU 核心
# 固定 p=20, n=2000，测试 w_dim=18/20/22 三种情况
# 使用离散版本数据生成（discrete_ind=1）

# 创建参数列表并并行执行（使用 SLURM 分配的 CPU 核心数）
# 固定 p=20, n=2000，遍历 w_dim ∈ {18,20,22}、alpha_id=1..9、gamma_id=1..5、seed=1..100
# 第7个参数为1表示使用离散版本

# # 第一步：执行所有 true 版本（pred_mgn_prox.R）
# echo "Starting true bounds version (pred_mgn_prox.R)..."
# for w_dim in 18 20 22; do
#   for alpha_id in {1..9}; do
#     for gamma_id in {1..5}; do
#       for seed in {1..100}; do
#         echo "20 2000 $w_dim $alpha_id $gamma_id $seed"
#       done
#     done
#   done
# done | xargs -n 6 -P $SLURM_NTASKS_PER_NODE sh -c '
#   cd simulations && \
#   Rscript pred_mgn_prox.R "$@"' _

# # 等待所有 true 版本完成
# echo "True bounds version completed. Starting estimated bounds version (pred_mgn_prox_est.R)..."

# 第二步：执行所有 est 版本（pred_mgn_prox_est.R）- 离散版本
echo "Starting discrete estimated bounds version (pred_mgn_prox_est.R with discrete_ind=1)..."
for w_dim in 18 20 22; do
  for alpha_id in {1..9}; do
    for gamma_id in {1..5}; do
      for seed in {1..100}; do
        echo "20 2000 $w_dim $alpha_id $gamma_id $seed 1"
      done
    done
  done
done | xargs -n 7 -P $SLURM_NTASKS_PER_NODE sh -c '
  cd simulations && \
  Rscript pred_mgn_prox_est.R "$@"' _

echo "All discrete simulations completed."