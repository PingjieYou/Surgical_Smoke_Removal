import re

# 读取日志文件
def read_log_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

# 提取指标数据
def extract_metrics(log_lines):
    metrics = []
    for line in log_lines:
        # 使用正则提取 Epoch、MSE、SSIM 和 PSNR
        match = re.search(r"Epoch (\d+): .*Val \(MSE: ([\d\.]+)\) \(SSIM: ([\d\.]+)\) \(PSNR: ([\d\.]+)\)", line)
        if match:
            metrics.append({
                "epoch": int(match.group(1)),
                "MSE": float(match.group(2)),
                "SSIM": float(match.group(3)),
                "PSNR": float(match.group(4))
            })
    return metrics

# 按照SSIM排序
def sort_by_ssim(metrics):
    return sorted(metrics, key=lambda x: x['SSIM'], reverse=True)

# 保存结果到文件
def save_to_file(sorted_metrics, output_file):
    with open(output_file, 'w') as f:
        f.write("Epoch\tMSE\tSSIM\tPSNR\n")
        for item in sorted_metrics:
            f.write(f"{item['epoch']}\t{item['MSE']:.6f}\t{item['SSIM']:.6f}\t{item['PSNR']:.6f}\n")

# 主函数
def main(log_file, output_file):
    log_lines = read_log_file(log_file)
    metrics = extract_metrics(log_lines)
    if not metrics:
        print("未能在日志中提取到任何指标，请检查日志内容或正则表达式。")
        return
    sorted_metrics = sort_by_ssim(metrics)
    save_to_file(sorted_metrics, output_file)
    print(f"结果已按照 SSIM 排序并保存到 {output_file}")


# 使用示例
log_file_path = "/media/disk6t/ypj/code/research/surgical desmoking/atomosphere cyclegan/checkpoints/CycleGANResnet9/log.txt"  # 替换为你的日志文件路径
output_file_path = "sorted_metrics.txt"
main(log_file_path, output_file_path)
