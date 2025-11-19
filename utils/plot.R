#!/usr/bin/env Rscript
########################################
## 绘制覆盖率校准图表和平均长度图表
## 比较 proximal, confounding-aware 和 confounding-unaware 方法在不同gamma值下的表现
########################################

# 加载必要的库
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(tidyr))

########################################
## 设置参数
########################################
# 结果文件目录
results_dir <- "/projects/p32685/cfsensitivity_results/simulation_prox_est/wdim_20"

# 创建输出目录
output_dir <- "/home/hhz6461/cfsensitivity_paper/plots"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("=" , rep("=", 60), "\n", sep="")
cat("覆盖率校准和平均长度图表生成工具\n")
cat("=" , rep("=", 60), "\n", sep="")
cat("结果目录:", results_dir, "\n")
cat("输出目录:", output_dir, "\n\n")

########################################
## 读取所有结果文件
########################################
# 查找所有结果文件（支持两种文件名格式）
files1 <- list.files(results_dir, 
                     pattern = "pred_marginal_prox.*\\.csv", 
                     full.names = TRUE)
files2 <- list.files(results_dir, 
                     pattern = "wdim_.*pred_marginal_prox.*\\.csv", 
                     full.names = TRUE)
files <- unique(c(files1, files2))

if (length(files) == 0) {
  stop(paste("错误：在", results_dir, "中未找到结果文件"))
}

cat("找到", length(files), "个结果文件\n")

# 读取所有CSV文件
all_data <- lapply(files, function(f) {
  tryCatch({
    df <- read.csv(f)
    return(df)
  }, error = function(e) {
    cat("警告：无法读取文件", f, ":", e$message, "\n")
    return(NULL)
  })
})

# 移除NULL值
all_data <- all_data[!sapply(all_data, is.null)]

if (length(all_data) == 0) {
  stop("错误：没有成功读取任何文件")
}

# 合并所有数据
combined_data <- do.call(rbind, all_data)
cat("总共读取", nrow(combined_data), "条记录\n\n")

# 从数据中推断参数
n_value <- ifelse("n" %in% names(combined_data), 
                  unique(combined_data$n)[1], 
                  NA)
w_dim <- ifelse("w_dim" %in% names(combined_data), 
                unique(combined_data$w_dim)[1], 
                NA)

cat("推断的参数:\n")
cat("  - 样本大小 n =", n_value, "\n")
cat("  - W维度 w_dim =", w_dim, "\n\n")

########################################
## 计算汇总统计
########################################
# 按alpha和gamma分组，计算平均覆盖率和平均长度
summary_data <- combined_data %>%
  group_by(alpha, gamma) %>%
  summarise(
    # Coverage
    c.cov_mean = mean(c.cov, na.rm = TRUE),      # confounding-aware平均覆盖率
    nc.cov_mean = mean(nc.cov, na.rm = TRUE),    # confounding-unaware平均覆盖率
    prox.cov_mean = mean(prox.cov, na.rm = TRUE), # proximal平均覆盖率
    # Length
    c.len_mean = mean(c.len, na.rm = TRUE),      # confounding-aware平均长度
    nc.len_mean = mean(nc.len, na.rm = TRUE),    # confounding-unaware平均长度
    prox.len_mean = mean(prox.len, na.rm = TRUE), # proximal平均长度
    # 标准差
    c.cov_sd = sd(c.cov, na.rm = TRUE),
    nc.cov_sd = sd(nc.cov, na.rm = TRUE),
    prox.cov_sd = sd(prox.cov, na.rm = TRUE),
    c.len_sd = sd(c.len, na.rm = TRUE),
    nc.len_sd = sd(nc.len, na.rm = TRUE),
    prox.len_sd = sd(prox.len, na.rm = TRUE),
    n_seeds = n(),                                # 使用的seed数量
    .groups = 'drop'
  ) %>%
  mutate(
    target_coverage = 1 - alpha   # 目标覆盖率 = 1 - alpha
  )

cat("数据汇总:\n")
cat("  - 总配置数:", nrow(summary_data), "\n")
cat("  - Alpha范围:", min(summary_data$alpha), "-", max(summary_data$alpha), "\n")
cat("  - Gamma值:", paste(sort(unique(summary_data$gamma)), collapse=", "), "\n")
cat("  - 每个配置的平均seed数:", round(mean(summary_data$n_seeds), 1), "\n\n")

########################################
## 图1: Empirical Coverage vs Target Coverage
########################################
# 准备绘图数据（长格式）
plot_data_cov <- summary_data %>%
  select(alpha, gamma, target_coverage, c.cov_mean, nc.cov_mean, prox.cov_mean) %>%
  pivot_longer(
    cols = c(c.cov_mean, nc.cov_mean, prox.cov_mean),
    names_to = "method_type",
    values_to = "actual_coverage"
  ) %>%
  mutate(
    method = case_when(
      method_type == "c.cov_mean" ~ paste0("gamma_", gamma, "_cov"),
      method_type == "nc.cov_mean" ~ "no_confounding_cov",
      method_type == "prox.cov_mean" ~ "proximal_cov",
      TRUE ~ "unknown"
    ),
    gamma_label = paste0("gamma = ", gamma),
    method_label = case_when(
      method_type == "c.cov_mean" ~ "confounding_aware",
      method_type == "nc.cov_mean" ~ "no_confounding",
      method_type == "prox.cov_mean" ~ "proximal",
      TRUE ~ "unknown"
    ),
    line_group = interaction(method_label, gamma, drop = TRUE),
    method_style = case_when(
      method_label == "no_confounding" ~ "no_confounding",
      method_label == "proximal" ~ "proximal",
      TRUE ~ "confounding_aware"
    )
  ) %>%
  arrange(alpha, gamma, method)

# 获取所有唯一的gamma值和方法
unique_gammas <- sort(unique(summary_data$gamma))

gamma_colors <- c(
  "gamma = 1.5" = "#E69F00",
  "gamma = 2" = "#D55E00",
  "gamma = 2.5" = "#56B4E9",
  "gamma = 3" = "#009E73",
  "gamma = 5" = "#8B4513"
)

available_gamma_labels <- unique(plot_data_cov$gamma_label)
gamma_colors_actual <- gamma_colors[available_gamma_labels]

method_shapes <- c(
  "confounding_aware" = 16,  # circle
  "no_confounding" = 17,     # triangle
  "proximal" = 15            # square
)

# 创建图1: Empirical Coverage vs Target Coverage
p1 <- ggplot(plot_data_cov, aes(x = target_coverage, y = actual_coverage, 
                                 color = gamma_label, shape = method_label,
                                 group = line_group, linetype = method_style)) +
  # 理想对角线（实际覆盖率 = 目标覆盖率）
  geom_abline(intercept = 0, slope = 1, color = "black", 
              linewidth = 0.5, linetype = "dashed", alpha = 0.7) +
  # 数据线
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  # 坐标轴设置
  scale_x_continuous(
    limits = c(0, 1), 
    breaks = seq(0, 1, 0.1),
    name = "Target Coverage"
  ) +
  scale_y_continuous(
    limits = c(0, 1), 
    breaks = seq(0, 1, 0.1),
    name = "Empirical Coverage"
  ) +
  # 颜色和形状映射
  scale_color_manual(
    values = gamma_colors_actual,
    name = "Gamma",
    drop = FALSE
  ) +
  scale_shape_manual(
    values = method_shapes,
    name = "Method",
    drop = FALSE
  ) +
  scale_linetype_manual(
    values = c("confounding_aware" = "solid",
               "no_confounding" = "dashed",
               "proximal" = "dotdash"),
    name = "Method",
    drop = FALSE
  ) +
  # 主题设置
  theme_minimal() +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    axis.title = element_text(size = 12),
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 10)
  ) +
  labs(
    title = paste0("Figure 3: Empirical Coverage |W| = |U| = ", w_dim),
    subtitle = paste0("Empirical Coverage vs Target Coverage, with n = ", n_value, ", card_w=", w_dim)
  )

# 保存图1
output_filename1 <- paste0("coverage_plot_n", n_value, "_wdim", w_dim, ".png")
output_path1 <- file.path(output_dir, output_filename1)
ggsave(output_path1, p1, width = 10, height = 7, dpi = 300)
cat("图1 (Empirical Coverage) 已保存到:", output_path1, "\n")

########################################
## 图2: Average Length vs Target Coverage
########################################
# 准备绘图数据（长格式）
plot_data_len <- summary_data %>%
  select(alpha, gamma, target_coverage, c.len_mean, nc.len_mean, prox.len_mean) %>%
  pivot_longer(
    cols = c(c.len_mean, nc.len_mean, prox.len_mean),
    names_to = "method_type",
    values_to = "average_length"
  ) %>%
  mutate(
    method = case_when(
      method_type == "c.len_mean" ~ paste0("gamma_", gamma, "_len"),
      method_type == "nc.len_mean" ~ "no_confounding_len",
      method_type == "prox.len_mean" ~ "proximal_len",
      TRUE ~ "unknown"
    ),
    gamma_label = paste0("gamma = ", gamma),
    method_label = case_when(
      method_type == "c.len_mean" ~ "confounding_aware",
      method_type == "nc.len_mean" ~ "no_confounding",
      method_type == "prox.len_mean" ~ "proximal",
      TRUE ~ "unknown"
    ),
    line_group = interaction(method_label, gamma, drop = TRUE),
    method_style = case_when(
      method_label == "no_confounding" ~ "no_confounding",
      method_label == "proximal" ~ "proximal",
      TRUE ~ "confounding_aware"
    )
  ) %>%
  arrange(alpha, gamma, method)

# 创建颜色映射（只包含实际存在的方法）
available_gamma_labels_len <- unique(plot_data_len$gamma_label)
gamma_colors_actual_len <- gamma_colors[available_gamma_labels_len]

# 创建图2: Average Length vs Target Coverage
p2 <- ggplot(plot_data_len, aes(x = target_coverage, y = average_length, 
                                 color = gamma_label, shape = method_label,
                                 group = line_group, linetype = method_style)) +
  # 数据线
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  # 坐标轴设置
  scale_x_continuous(
    limits = c(0, 1), 
    breaks = seq(0, 1, 0.1),
    name = "Target Coverage"
  ) +
  scale_y_continuous(
    name = "Average Length"
  ) +
  # 颜色和形状映射
  scale_color_manual(
    values = gamma_colors_actual_len,
    name = "Gamma",
    drop = FALSE
  ) +
  scale_shape_manual(
    values = method_shapes,
    name = "Method",
    drop = FALSE
  ) +
  scale_linetype_manual(
    values = c("confounding_aware" = "solid",
               "no_confounding" = "dashed",
               "proximal" = "dotdash"),
    name = "Method",
    drop = FALSE
  ) +
  # 主题设置
  theme_minimal() +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    axis.title = element_text(size = 12),
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 10)
  ) +
  labs(
    title = paste0("Figure 2: Average Length |W| = |U| = ", w_dim),
    subtitle = paste0("Average Length vs Target Coverage, with n = ", n_value, ", card_w=", w_dim)
  )

# 保存图2
output_filename2 <- paste0("length_plot_n", n_value, "_wdim", w_dim, ".png")
output_path2 <- file.path(output_dir, output_filename2)
ggsave(output_path2, p2, width = 10, height = 7, dpi = 300)
cat("图2 (Average Length) 已保存到:", output_path2, "\n\n")

########################################
## 显示统计信息
########################################
cat("覆盖率统计:\n")
cat("  Proximal平均覆盖率:", 
    round(mean(summary_data$prox.cov_mean, na.rm = TRUE), 3), 
    "±", round(sd(summary_data$prox.cov_mean, na.rm = TRUE), 3), "\n")
cat("  No-confounding平均覆盖率:", 
    round(mean(summary_data$nc.cov_mean, na.rm = TRUE), 3), 
    "±", round(sd(summary_data$nc.cov_mean, na.rm = TRUE), 3), "\n")
cat("  Confounding-aware平均覆盖率:", 
    round(mean(summary_data$c.cov_mean, na.rm = TRUE), 3), 
    "±", round(sd(summary_data$c.cov_mean, na.rm = TRUE), 3), "\n\n")

cat("平均长度统计:\n")
cat("  Proximal平均长度:", 
    round(mean(summary_data$prox.len_mean, na.rm = TRUE), 3), 
    "±", round(sd(summary_data$prox.len_mean, na.rm = TRUE), 3), "\n")
cat("  No-confounding平均长度:", 
    round(mean(summary_data$nc.len_mean, na.rm = TRUE), 3), 
    "±", round(sd(summary_data$nc.len_mean, na.rm = TRUE), 3), "\n")
cat("  Confounding-aware平均长度:", 
    round(mean(summary_data$c.len_mean, na.rm = TRUE), 3), 
    "±", round(sd(summary_data$c.len_mean, na.rm = TRUE), 3), "\n\n")

cat("完成！\n")
