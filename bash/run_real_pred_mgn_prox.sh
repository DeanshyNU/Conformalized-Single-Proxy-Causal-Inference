#!/bin/bash
#SBATCH -A p32685        ## 你的账户
#SBATCH -p normal        ## CPU 分区（R不需要GPU）
#SBATCH --nodes=1        ## 请求一个节点
#SBATCH --ntasks-per-node=64  ## 每个节点运行64个并行任务（可根据节点调整：52/64/128）
#SBATCH -t 48:00:00      ## 最大运行时间 48 小时
#SBATCH --mem=200G    
#SBATCH --output=run_real_pred_mgn_prox_%j.out  ## 输出日志
#SBATCH --error=run_real_pred_mgn_prox_%j.err   ## 错误日志

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

# 3) 运行真实数据反事实预测（边际有效方法 + proximal方法）
# 使用xargs -P并行执行多个R脚本，每个R脚本使用1个CPU核心

# 创建参数列表并并行执行（使用SLURM分配的CPU核心数）
# 支持多个 w_dim 值（与 simulation 保持一致）
for w_dim in 18 20 22; do
    for alpha_id in {1..9}; do
        for gamma_id in {1..5}; do
            for seed in {1..100}; do
                echo "$alpha_id $gamma_id $seed $w_dim"
            done
        done
    done
done | xargs -n 4 -P $SLURM_NTASKS_PER_NODE sh -c 'cd realdata && Rscript syn_pred_mgn_prox.R "$@"' _