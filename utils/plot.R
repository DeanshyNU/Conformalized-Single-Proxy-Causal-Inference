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
# 多个结果文件目录
results_dirs <- c(
  "/projects/p32685/cfsensitivity_results/simulation_prox_true/wdim_18",
  "/projects/p32685/cfsensitivity_results/simulation_prox_true/wdim_20",
  "/projects/p32685/cfsensitivity_results/simulation_prox_true/wdim_22",
  "/projects/p32685/cfsensitivity_results/simulation_prox_est/wdim_18"
)

# 创建输出目录
output_dir <- "/home/hhz6461/cfsensitivity_paper/plots"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("=" , rep("=", 60), "\n", sep="")
cat("覆盖率校准和平均长度图表生成工具\n")
cat("=" , rep("=", 60), "\n", sep="")
cat("结果目录数量:", length(results_dirs), "\n")
for (i in seq_along(results_dirs)) {
  cat("  目录", i, ":", results_dirs[i], "\n")
}
cat("输出目录:", output_dir, "\n\n")

########################################
## 读取所有结果文件
########################################
all_data_list <- list()

for (results_dir in results_dirs) {
  if (!dir.exists(results_dir)) {
    cat("警告：目录不存在，跳过:", results_dir, "\n")
    next
  }
  
  # 从目录路径中提取配置信息
  if (grepl("prox_true", results_dir)) {
    config_type <- "prox_true"
  } else if (grepl("prox_est", results_dir)) {
    config_type <- "prox_est"
  } else {
    config_type <- "unknown"
  }
  
  w_dim_from_path <- gsub(".*wdim_([0-9]+).*", "\\1", results_dir)
  
  cat("正在读取目录:", results_dir, "\n")
  cat("  配置类型:", config_type, ", w_dim:", w_dim_from_path, "\n")
  
  # 查找所有结果文件（支持两种文件名格式）
  files1 <- list.files(results_dir, 
                       pattern = "pred_marginal_prox.*\\.csv", 
                       full.names = TRUE)
  files2 <- list.files(results_dir, 
                       pattern = "wdim_.*pred_marginal_prox.*\\.csv", 
                       full.names = TRUE)
  files <- unique(c(files1, files2))
  
  if (length(files) == 0) {
    cat("  警告：在", results_dir, "中未找到结果文件\n")
    next
  }
  
  cat("  找到", length(files), "个结果文件\n")
  
  # 读取所有CSV文件
  dir_data <- lapply(files, function(f) {
    tryCatch({
      df <- read.csv(f)
      # 添加配置标识
      df$config_type <- config_type
      df$w_dim_from_path <- as.numeric(w_dim_from_path)
      return(df)
    }, error = function(e) {
      cat("  警告：无法读取文件", f, ":", e$message, "\n")
      return(NULL)
    })
  })
  
  # 移除NULL值
  dir_data <- dir_data[!sapply(dir_data, is.null)]
  
  if (length(dir_data) > 0) {
    all_data_list[[length(all_data_list) + 1]] <- do.call(rbind, dir_data)
    cat("  成功读取", nrow(all_data_list[[length(all_data_list)]]), "条记录\n")
  }
}

if (length(all_data_list) == 0) {
  stop("错误：没有成功读取任何文件")
}

# 合并所有数据
combined_data <- do.call(rbind, all_data_list)
cat("\n总共读取", nrow(combined_data), "条记录\n")

# 从数据中推断参数
n_value <- ifelse("n" %in% names(combined_data), 
                  unique(combined_data$n)[1], 
                  NA)

cat("\n推断的参数:\n")
cat("  - 样本大小 n =", n_value, "\n")
cat("  - 配置类型:", paste(unique(combined_data$config_type), collapse=", "), "\n")
cat("  - W维度:", paste(sort(unique(combined_data$w_dim_from_path)), collapse=", "), "\n\n")

########################################
## 计算汇总统计（按配置分组）
########################################
# 按配置类型、w_dim、alpha和gamma分组，计算平均覆盖率和平均长度
summary_data <- combined_data %>%
  group_by(config_type, w_dim_from_path, alpha, gamma) %>%
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
    target_coverage = 1 - alpha,   # 目标覆盖率 = 1 - alpha
    config_label = paste0(config_type, "_wdim", w_dim_from_path)
  )

cat("数据汇总:\n")
cat("  - 总配置数:", nrow(summary_data), "\n")
cat("  - 配置类型:", paste(unique(summary_data$config_type), collapse=", "), "\n")
cat("  - W维度:", paste(sort(unique(summary_data$w_dim_from_path)), collapse=", "), "\n")
cat("  - Alpha范围:", min(summary_data$alpha), "-", max(summary_data$alpha), "\n")
cat("  - Gamma值:", paste(sort(unique(summary_data$gamma)), collapse=", "), "\n")
cat("  - 每个配置的平均seed数:", round(mean(summary_data$n_seeds), 1), "\n\n")

########################################
## 为每个配置生成图像
########################################
# 获取所有唯一的配置
unique_configs <- summary_data %>%
  ungroup() %>%
  select(config_type, w_dim_from_path) %>%
  distinct()

cat("将为", nrow(unique_configs), "个配置生成图像\n\n")

# 定义颜色和形状
gamma_colors <- c(
  "gamma = 1.5" = "#E69F00",
  "gamma = 2" = "#D55E00",
  "gamma = 2.5" = "#56B4E9",
  "gamma = 3" = "#009E73",
  "gamma = 5" = "#8B4513"
)

method_shapes <- c(
  "confounding_aware" = 16,  # circle
  "no_confounding" = 17,     # triangle
  "proximal" = 15            # square
)

# 为每个配置生成图像
for (i in 1:nrow(unique_configs)) {
  config_type_i <- unique_configs$config_type[i]
  w_dim_i <- unique_configs$w_dim_from_path[i]
  
  cat("=" , rep("=", 60), "\n", sep="")
  cat("处理配置", i, "/", nrow(unique_configs), ":", config_type_i, ", w_dim =", w_dim_i, "\n")
  cat("=" , rep("=", 60), "\n", sep="")
  
  # 筛选当前配置的数据
  config_data <- summary_data %>%
    filter(config_type == config_type_i, w_dim_from_path == w_dim_i)
  
  if (nrow(config_data) == 0) {
    cat("警告：该配置没有数据，跳过\n\n")
    next
  }
  
  ########################################
  ## 图1: Empirical Coverage vs Target Coverage
  ########################################
  # 准备绘图数据（不需要gather，直接使用）
  plot_data_cov <- config_data %>%
    ungroup() %>%
    mutate(
      gamma_label = paste0("gamma = ", gamma)
    )
  
  available_gamma_labels <- unique(plot_data_cov$gamma_label)
  gamma_colors_actual <- gamma_colors[available_gamma_labels]
  
  # 创建图1: Empirical Coverage vs Target Coverage
  p1 <- ggplot(plot_data_cov, aes(x = target_coverage)) +
    # 理想对角线（实际覆盖率 = 目标覆盖率）
    geom_abline(intercept = 0, slope = 1, color = "black", 
                linewidth = 0.5, linetype = "dashed", alpha = 0.7) +
    # Confounding-aware 方法（实线）
    geom_line(aes(y = c.cov_mean, color = gamma_label, group = interaction(gamma, "confounding_aware")), 
              linewidth = 0.8, linetype = "solid") +
    geom_point(aes(y = c.cov_mean, color = gamma_label, shape = "confounding_aware"), 
               size = 2.5) +
    # Confounding-unaware 方法（虚线）
    geom_line(aes(y = nc.cov_mean, color = gamma_label, group = interaction(gamma, "no_confounding")), 
              linewidth = 0.8, linetype = "dashed") +
    geom_point(aes(y = nc.cov_mean, color = gamma_label, shape = "no_confounding"), 
               size = 2.5) +
    # Proximal 方法（点划线）
    geom_line(aes(y = prox.cov_mean, color = gamma_label, group = interaction(gamma, "proximal")), 
              linewidth = 0.8, linetype = "dotdash") +
    geom_point(aes(y = prox.cov_mean, color = gamma_label, shape = "proximal"), 
               size = 2.5) +
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
      title = paste0("Empirical Coverage (", config_type_i, ", |W| = ", w_dim_i, ")"),
      subtitle = paste0("Empirical Coverage vs Target Coverage, n = ", n_value, ", card_w = ", w_dim_i)
    )
  
  # 保存图1
  output_filename1 <- paste0("coverage_plot_", config_type_i, "_wdim", w_dim_i, "_n", n_value, ".png")
  output_path1 <- file.path(output_dir, output_filename1)
  ggsave(output_path1, plot = p1, width = 10, height = 7, dpi = 300)
  cat("图1 (Empirical Coverage) 已保存到:", output_path1, "\n")

  ########################################
  ## 图2: Average Length vs Target Coverage
  ########################################
  # 准备绘图数据（不需要gather，直接使用）
  plot_data_len <- config_data %>%
    ungroup() %>%
    mutate(
      gamma_label = paste0("gamma = ", gamma)
    )
  
  # 创建颜色映射（只包含实际存在的方法）
  available_gamma_labels_len <- unique(plot_data_len$gamma_label)
  gamma_colors_actual_len <- gamma_colors[available_gamma_labels_len]
  
  # 创建图2: Average Length vs Target Coverage
  p2 <- ggplot(plot_data_len, aes(x = target_coverage)) +
    # Confounding-aware 方法（实线）
    geom_line(aes(y = c.len_mean, color = gamma_label, group = interaction(gamma, "confounding_aware")), 
              linewidth = 0.8, linetype = "solid") +
    geom_point(aes(y = c.len_mean, color = gamma_label, shape = "confounding_aware"), 
               size = 2.5) +
    # Confounding-unaware 方法（虚线）
    geom_line(aes(y = nc.len_mean, color = gamma_label, group = interaction(gamma, "no_confounding")), 
              linewidth = 0.8, linetype = "dashed") +
    geom_point(aes(y = nc.len_mean, color = gamma_label, shape = "no_confounding"), 
               size = 2.5) +
    # Proximal 方法（点划线）
    geom_line(aes(y = prox.len_mean, color = gamma_label, group = interaction(gamma, "proximal")), 
              linewidth = 0.8, linetype = "dotdash") +
    geom_point(aes(y = prox.len_mean, color = gamma_label, shape = "proximal"), 
               size = 2.5) +
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
      title = paste0("Average Length (", config_type_i, ", |W| = ", w_dim_i, ")"),
      subtitle = paste0("Average Length vs Target Coverage, n = ", n_value, ", card_w = ", w_dim_i)
    )
  
  # 保存图2
  output_filename2 <- paste0("length_plot_", config_type_i, "_wdim", w_dim_i, "_n", n_value, ".png")
  output_path2 <- file.path(output_dir, output_filename2)
  ggsave(output_path2, plot = p2, width = 10, height = 7, dpi = 300)
  cat("图2 (Average Length) 已保存到:", output_path2, "\n")
  
  ########################################
  ## 显示统计信息
  ########################################
  cat("\n覆盖率统计:\n")
  cat("  Proximal平均覆盖率:", 
      round(mean(config_data$prox.cov_mean, na.rm = TRUE), 3), 
      "±", round(sd(config_data$prox.cov_mean, na.rm = TRUE), 3), "\n")
  cat("  No-confounding平均覆盖率:", 
      round(mean(config_data$nc.cov_mean, na.rm = TRUE), 3), 
      "±", round(sd(config_data$nc.cov_mean, na.rm = TRUE), 3), "\n")
  cat("  Confounding-aware平均覆盖率:", 
      round(mean(config_data$c.cov_mean, na.rm = TRUE), 3), 
      "±", round(sd(config_data$c.cov_mean, na.rm = TRUE), 3), "\n")
  
  cat("\n平均长度统计:\n")
  cat("  Proximal平均长度:", 
      round(mean(config_data$prox.len_mean, na.rm = TRUE), 3), 
      "±", round(sd(config_data$prox.len_mean, na.rm = TRUE), 3), "\n")
  cat("  No-confounding平均长度:", 
      round(mean(config_data$nc.len_mean, na.rm = TRUE), 3), 
      "±", round(sd(config_data$nc.len_mean, na.rm = TRUE), 3), "\n")
  cat("  Confounding-aware平均长度:", 
      round(mean(config_data$c.len_mean, na.rm = TRUE), 3), 
      "±", round(sd(config_data$c.len_mean, na.rm = TRUE), 3), "\n\n")
}

cat("\n所有配置的图像生成完成！\n")
