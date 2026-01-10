# 测试离散和连续数据生成函数
# 验证 U 的生成和 T 的生成逻辑，并对比两个版本

# 加载函数文件（假设脚本在 utils 目录下运行）
# 如果从其他目录运行，请修改路径
# 注意：两个文件都定义了 data.gen.ate，需要分别加载测试

cat("========================================\n")
cat("  离散 vs 连续数据生成对比测试\n")
cat("========================================\n\n")

# 设置参数
n = 2000
p = 5
Gamma = 2.0
set.seed(2025)
beta = runif(p, -1, 1)
alpha0 = 0
u_dim = 20  # 必须是偶数（仅用于离散版本）

# ============================================
# 测试离散版本
# ============================================
cat("【离散版本测试】\n")
cat("----------------------------------------\n")

# 加载离散版本函数
source("util_ate_discrete.R")

set.seed(2024)
data_discrete = data.gen.ate(n, p, Gamma, beta, alpha0, obs=FALSE, 
                             u_dim=u_dim)

# 验证 1: U 的范围应该是 [-(u_dim-1)/2, (u_dim-1)/2]
cat("\n=== 验证 1: U 的取值范围 ===\n")
cat("U 的最小值:", min(data_discrete$U), "（应接近", -(u_dim-1)/2, "）\n")
cat("U 的最大值:", max(data_discrete$U), "（应接近", (u_dim-1)/2, "）\n")
cat("U 的均值:", round(mean(data_discrete$U), 4), "（应接近 0）\n")
cat("U 的标准差:", round(sd(data_discrete$U), 4), "\n")
cat("U 的唯一值数量:", length(unique(data_discrete$U)), "（应为", u_dim, "）\n")

# 验证 2: 检查 E[P(A|U,X)|X] = P(A|X)
cat("\n=== 验证 2: E[P(A|U,X)|X] = P(A|X) ===\n")
# 创建 bins 来分组相似的 prop.x 值
n_bins = 10
prop.x.bins = cut(data_discrete$ex, breaks=n_bins, labels=FALSE)

expected_mean = numeric(n_bins)
actual_mean = numeric(n_bins)
max_diff = 0

for (b in 1:n_bins) {
  idx = which(prop.x.bins == b)
  if (length(idx) > 0) {
    expected_mean[b] = mean(data_discrete$ex[idx])
    actual_mean[b] = mean(data_discrete$exu[idx])
    diff = abs(actual_mean[b] - expected_mean[b])
    if (diff > max_diff) max_diff = diff
  }
}

cat("Bin | E[P(A|U,X)|X] | P(A|X) | 差异\n")
cat("----|----------------|-------|------\n")
for (b in 1:n_bins) {
  if (expected_mean[b] > 0) {
    diff = abs(actual_mean[b] - expected_mean[b])
    cat(sprintf("%3d |     %.4f    | %.4f | %.4f\n", 
                b, actual_mean[b], expected_mean[b], diff))
  }
}
cat("\n最大差异:", round(max_diff, 6), "（应接近 0）\n")

# 整体相关性
cor_val = cor(data_discrete$ex, data_discrete$exu)
cat("整体相关性 cor(P(A|X), mean(P(A|U,X)|X)):", round(cor_val, 6), "\n")
cat("（应接近 1.0）\n")

# 验证 3: 检查 prop.xu 的取值是否只包含 a(x) 和 b(x)
cat("\n=== 验证 3: prop.xu 的取值分布 ===\n")
a.x = data_discrete$ex / (data_discrete$ex + Gamma * (1 - data_discrete$ex))
b.x = data_discrete$ex / (data_discrete$ex + (1 - data_discrete$ex) / Gamma)

# 检查每个 prop.xu[i] 是否等于 a.x[i] 或 b.x[i]
tolerance = 1e-10
n_matches_a = sum(abs(data_discrete$exu - a.x) < tolerance)
n_matches_b = sum(abs(data_discrete$exu - b.x) < tolerance)
n_total = length(data_discrete$exu)

cat("匹配 a(x) 的数量:", n_matches_a, "（", 
    round(100*n_matches_a/n_total, 2), "%）\n")
cat("匹配 b(x) 的数量:", n_matches_b, "（", 
    round(100*n_matches_b/n_total, 2), "%）\n")
cat("总数:", n_total, "\n")
cat("匹配率:", round(100*(n_matches_a + n_matches_b)/n_total, 2), "%（应为 100%）\n")

# 验证 4: 检查 U 的分布是否依赖于 X
cat("\n=== 验证 4: U 与 X 的依赖关系 ===\n")
cor_u_x = cor(data_discrete$U, data_discrete$X[,1])
cat("U 与 X[,1] 的相关性:", round(cor_u_x, 4), "\n")
cat("（应不为 0，表明存在混淆）\n")

# 验证 5: 检查 T 的生成
cat("\n=== 验证 5: T 的生成 ===\n")
cat("T=1 的比例:", round(mean(data_discrete$T), 4), "\n")
cat("P(A|X) 的均值:", round(mean(data_discrete$ex), 4), "\n")
cat("P(A|U,X) 的均值:", round(mean(data_discrete$exu), 4), "\n")
cat("（P(A|U,X) 的均值应接近 P(A|X) 的均值）\n")

# ============================================
# 测试连续版本
# ============================================
cat("\n\n【连续版本测试】\n")
cat("----------------------------------------\n")

# 重新加载连续版本函数（会覆盖离散版本）
source("util_ate.R")

set.seed(2024)
data_continuous = data.gen.ate(n, p, Gamma, beta, alpha0, obs=FALSE)

# 验证 1: U 的分布
cat("\n=== 验证 1: U 的分布 ===\n")
cat("U 的最小值:", round(min(data_continuous$U), 4), "\n")
cat("U 的最大值:", round(max(data_continuous$U), 4), "\n")
cat("U 的均值:", round(mean(data_continuous$U), 4), "（应接近 0）\n")
cat("U 的标准差:", round(sd(data_continuous$U), 4), "\n")

# 验证 2: 检查 E[P(A|U,X)|X] = P(A|X)
cat("\n=== 验证 2: E[P(A|U,X)|X] = P(A|X) ===\n")
prop.x.bins_cont = cut(data_continuous$ex, breaks=n_bins, labels=FALSE)

expected_mean_cont = numeric(n_bins)
actual_mean_cont = numeric(n_bins)
max_diff_cont = 0

for (b in 1:n_bins) {
  idx = which(prop.x.bins_cont == b)
  if (length(idx) > 0) {
    expected_mean_cont[b] = mean(data_continuous$ex[idx])
    actual_mean_cont[b] = mean(data_continuous$exu[idx])
    diff = abs(actual_mean_cont[b] - expected_mean_cont[b])
    if (diff > max_diff_cont) max_diff_cont = diff
  }
}

cat("Bin | E[P(A|U,X)|X] | P(A|X) | 差异\n")
cat("----|----------------|-------|------\n")
for (b in 1:n_bins) {
  if (expected_mean_cont[b] > 0) {
    diff = abs(actual_mean_cont[b] - expected_mean_cont[b])
    cat(sprintf("%3d |     %.4f    | %.4f | %.4f\n", 
                b, actual_mean_cont[b], expected_mean_cont[b], diff))
  }
}
cat("\n最大差异:", round(max_diff_cont, 6), "（应接近 0）\n")

# 整体相关性
cor_val_cont = cor(data_continuous$ex, data_continuous$exu)
cat("整体相关性 cor(P(A|X), mean(P(A|U,X)|X)):", round(cor_val_cont, 6), "\n")
cat("（应接近 1.0）\n")

# 验证 3: 检查 prop.xu 的取值
cat("\n=== 验证 3: prop.xu 的取值分布 ===\n")
a.x_cont = data_continuous$ex / (data_continuous$ex + Gamma * (1 - data_continuous$ex))
b.x_cont = data_continuous$ex / (data_continuous$ex + (1 - data_continuous$ex) / Gamma)

n_matches_a_cont = sum(abs(data_continuous$exu - a.x_cont) < tolerance)
n_matches_b_cont = sum(abs(data_continuous$exu - b.x_cont) < tolerance)
n_total_cont = length(data_continuous$exu)

cat("匹配 a(x) 的数量:", n_matches_a_cont, "（", 
    round(100*n_matches_a_cont/n_total_cont, 2), "%）\n")
cat("匹配 b(x) 的数量:", n_matches_b_cont, "（", 
    round(100*n_matches_b_cont/n_total_cont, 2), "%）\n")
cat("总数:", n_total_cont, "\n")
cat("匹配率:", round(100*(n_matches_a_cont + n_matches_b_cont)/n_total_cont, 2), "%（应为 100%）\n")

# 验证 4: 检查 U 与 X 的依赖关系
cat("\n=== 验证 4: U 与 X 的依赖关系 ===\n")
cor_u_x_cont = cor(data_continuous$U, data_continuous$X[,1])
cat("U 与 X[,1] 的相关性:", round(cor_u_x_cont, 4), "\n")
cat("（应不为 0，表明存在混淆）\n")

# 验证 5: 检查 T 的生成
cat("\n=== 验证 5: T 的生成 ===\n")
cat("T=1 的比例:", round(mean(data_continuous$T), 4), "\n")
cat("P(A|X) 的均值:", round(mean(data_continuous$ex), 4), "\n")
cat("P(A|U,X) 的均值:", round(mean(data_continuous$exu), 4), "\n")
cat("（P(A|U,X) 的均值应接近 P(A|X) 的均值）\n")

# ============================================
# 对比总结
# ============================================
cat("\n\n【对比总结】\n")
cat("========================================\n")
cat("指标                    | 离散版本  | 连续版本\n")
cat("------------------------|-----------|----------\n")
cat(sprintf("E[P(A|U,X)|X] vs P(A|X) | %.6f  | %.6f\n", 
            max_diff, max_diff_cont))
cat(sprintf("相关性                | %.6f  | %.6f\n", 
            cor_val, cor_val_cont))
cat(sprintf("T=1 比例              | %.4f  | %.4f\n", 
            mean(data_discrete$T), mean(data_continuous$T)))
cat(sprintf("P(A|X) 均值           | %.4f  | %.4f\n", 
            mean(data_discrete$ex), mean(data_continuous$ex)))
cat(sprintf("P(A|U,X) 均值         | %.4f  | %.4f\n", 
            mean(data_discrete$exu), mean(data_continuous$exu)))

cat("\n=== 测试完成 ===\n")
