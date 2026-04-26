"""
统一可视化脚本 - 生成所有实验图表

使用方法:
    python scripts/generate_figures.py                              # 生成所有图表
    python scripts/generate_figures.py --plots training performance # 只生成训练曲线和性能对比图
    python scripts/generate_figures.py --results-dir path/to/results # 指定结果目录
    python scripts/generate_figures.py --figures-dir path/to/docs    # 指定输出目录

可用的图表类型:
    training    - 训练曲线
    confusion   - 混淆矩阵
    performance - 性能对比图
    roc         - ROC曲线
    tsne        - t-SNE特征分布图
"""
import os
import sys
import argparse
import json

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils import config


def generate_all_figures(results_dir=None, figures_dir=None, plots_to_run=None):
    """
    生成实验图表

    参数:
        results_dir: 评估结果目录（None则从配置读取）
        figures_dir: 图表保存目录（None则从配置读取）
        plots_to_run: 要绘制的图表列表，None表示全部。
                      可选值: 'training', 'confusion', 'performance', 'roc', 'tsne'
    """
    # 获取参数
    if results_dir is None:
        results_dir = config.get('evaluation', 'results_dir', default='checkpoints/evaluation_results')
    if figures_dir is None:
        figures_dir = config.get('visualization', 'figures_dir', default='docs/figures')

    # 转换为绝对路径
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(PROJECT_ROOT, results_dir)
    if not os.path.isabs(figures_dir):
        figures_dir = os.path.join(PROJECT_ROOT, figures_dir)

    print("=" * 60)
    print("开始生成实验图表...")
    print(f"结果目录: {results_dir}")
    print(f"图表目录: {figures_dir}")
    if plots_to_run:
        print(f"绘制图表: {', '.join(plots_to_run)}")
    else:
        print(f"绘制图表: 全部")
    print("=" * 60)

    # 确保目录存在
    os.makedirs(figures_dir, exist_ok=True)

    # 1. 绘制训练曲线
    if plots_to_run is None or 'training' in plots_to_run:
        print("\n[1/5] 绘制训练曲线...")
        try:
            from scripts.visualize.visualize_training import plot_training_curves, plot_loss_comparison
            log_dir = config.get_path('logs_dir')

            # 检查是否通过命令行参数指定了日志文件映射
            global training_log_map
            if training_log_map:
                # 使用用户指定的映射
                for key, loss_file_path in training_log_map.items():
                    save_path = os.path.join(figures_dir, key)
                    plot_training_curves(loss_log_file=loss_file_path, save_path=save_path)
                plot_loss_comparison(list(training_log_map.values()), list(training_log_map.keys()), save_path=figures_dir)
                print(f"✓ 训练曲线对比绘制完成 (共{len(training_log_map)}个模型)")
            else:
                # 降级：绘制单个最新的训练曲线
                loss_files = [f for f in os.listdir(log_dir) if f.endswith('_loss.csv')]
                if loss_files:
                    latest_loss = sorted(loss_files)[-1]
                    loss_file = os.path.join(log_dir, latest_loss)
                    plot_training_curves(loss_log_file=loss_file, save_path=figures_dir)
                    print(f"✓ 训练曲线绘制完成 (使用loss日志: {latest_loss})")
                else:
                    print("✗ 未找到loss日志文件")
        except Exception as e:
            print(f"✗ 训练曲线绘制失败: {e}")

    # 2. 绘制混淆矩阵
    if plots_to_run is None or 'confusion' in plots_to_run:
        print("\n[2/5] 绘制混淆矩阵...")
        try:
            from scripts.visualize.visualize_confusion import plot_confusion_matrix
            results = load_evaluation_results(results_dir)
            if results:
                for model_name, data in results.items():
                    plot_confusion_matrix(data['labels'], data['preds'], save_path=figures_dir)
                    old_file = os.path.join(figures_dir, 'confusion_matrix.png')
                    new_file = os.path.join(figures_dir, f'confusion_matrix_{model_name}.png')
                    if os.path.exists(old_file):
                        os.rename(old_file, new_file)
                print("✓ 混淆矩阵绘制完成")
        except Exception as e:
            print(f"✗ 混淆矩阵绘制失败: {e}")

    # 3. 绘制性能对比图
    if plots_to_run is None or 'performance' in plots_to_run:
        print("\n[3/5] 绘制性能对比图...")
        try:
            from scripts.visualize.visualize_performance import plot_performance_comparison, plot_performance_grouped, plot_improvement_chart
            results = load_evaluation_results(results_dir)
            if results and len(results) >= 2:
                metrics_dict = {name: data['metrics'] for name, data in results.items()}
                plot_performance_comparison(metrics_dict, save_path=figures_dir)
                plot_performance_grouped(metrics_dict, save_path=figures_dir)

                baseline_name = list(metrics_dict.keys())[0]
                ours_name = list(metrics_dict.keys())[-1]
                plot_improvement_chart(metrics_dict[baseline_name], metrics_dict[ours_name], save_path=figures_dir)
                print("✓ 性能对比图绘制完成")
            else:
                print("  需要至少2个模型的评估结果才能绘制对比图")
        except Exception as e:
            print(f"✗ 性能对比图绘制失败: {e}")

    # 4. 绘制ROC曲线
    if plots_to_run is None or 'roc' in plots_to_run:
        print("\n[4/5] 绘制ROC曲线...")
        try:
            from scripts.visualize.visualize_roc import plot_roc_curve, plot_multiple_roc
            results = load_evaluation_results(results_dir)
            if results:
                roc_results = {name: (data['labels'], data['probs']) for name, data in results.items()}
                if len(roc_results) >= 2:
                    plot_multiple_roc(roc_results, save_path=figures_dir)
                else:
                    model_name = list(roc_results.keys())[0]
                    plot_roc_curve(roc_results[model_name][0], roc_results[model_name][1],
                                 model_name, save_path=figures_dir)
                print("✓ ROC曲线绘制完成")
        except Exception as e:
            print(f"✗ ROC曲线绘制失败: {e}")

    # 5. 绘制t-SNE图
    if plots_to_run is None or 'tsne' in plots_to_run:
        print("\n[5/5] 绘制t-SNE特征分布图...")
        print("  注意: t-SNE需要加载模型和数据，可能需要较长时间...")
        try:
            import torch
            from torch.utils.data import DataLoader
            from transformers import BertTokenizer
            from scripts.visualize.visualize_tsne import extract_features, plot_tsne, plot_multiple_tsne

            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            data_processed_dir = config.get_path('data_processed_dir')
            test_path = os.path.join(data_processed_dir, 'test.csv')
            from scripts.dataset import HateSpeechDataset
            test_dataset = HateSpeechDataset(test_path, tokenizer)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            checkpoints_dir = config.get_path('checkpoints_dir')
            model_names = ['model_supcon', 'model_supcon_no_proj', 'model_baseline']

            tsne_results = {}
            for model_name in model_names:
                model_path = os.path.join(checkpoints_dir, f'{model_name}.pth')
                if os.path.exists(model_path):
                    print(f"  正在处理 {model_name}...")
                    from models.model import ContrastiveHateSpeechModel
                    model_config = config.get('model', 'model_name', default='bert-base-chinese')
                    # 根据模型名称判断是否使用投影层
                    use_projection = True
                    if model_name and 'no_proj' in model_name:
                        use_projection = False
                    model = ContrastiveHateSpeechModel(model_config, use_projection=use_projection).to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()

                    features, labels = extract_features(model, test_loader, device)
                    display_name = model_name.replace('model_', '').replace('_', ' ').title()
                    tsne_results[display_name] = (features, labels)

            if tsne_results:
                if len(tsne_results) >= 2:
                    plot_multiple_tsne(tsne_results, save_path=figures_dir)
                else:
                    model_name = list(tsne_results.keys())[0]
                    plot_tsne(tsne_results[model_name][0], tsne_results[model_name][1],
                             title=f't-SNE Visualization - {model_name}', save_path=figures_dir)
                print("✓ t-SNE图绘制完成")
        except Exception as e:
            print(f"✗ t-SNE图绘制失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("图表生成完成!")
    print(f"所有图表已保存至: {figures_dir}")
    print("=" * 60)


def load_evaluation_results(results_dir):
    """加载评估结果"""
    results = {}
    if not os.path.exists(results_dir):
        print(f"  评估结果目录不存在: {results_dir}")
        return results

    json_files = [f for f in os.listdir(results_dir) if f.endswith('_results.json')]

    for json_file in json_files:
        model_name = json_file.replace('_results.json', '')
        file_path = os.path.join(results_dir, json_file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                results[model_name] = json.load(f)
            print(f"  加载: {json_file}")
        except Exception as e:
            print(f"  加载失败 {json_file}: {e}")

    return results


training_log_map = None  # 全局变量，用于存储命令行指定的日志映射


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成所有实验图表')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='评估结果目录')
    parser.add_argument('--figures-dir', type=str, default=None,
                       help='图表保存目录')
    parser.add_argument('--training-logs', type=str, default=None, nargs='+',
                       help='训练日志映射，格式: "模型名:日志路径" 多个空格分隔，如: "Baseline:/path/to/baseline_loss.csv SupCon:/path/to/supcon_loss.csv"')
    parser.add_argument('--plots', type=str, default=None, nargs='+',
                       choices=['training', 'confusion', 'performance', 'roc', 'tsne'],
                       help='指定要绘制的图表类型，可选: training, confusion, performance, roc, tsne（可多选，如: --plots training performance）')
    args = parser.parse_args()

    # 解析训练日志映射
    if args.training_logs:
        training_log_map = {}
        for log_mapping in args.training_logs:
            parts = log_mapping.split(':')
            if len(parts) == 2:
                model_name, log_path = parts
                training_log_map[model_name.strip()] = log_path.strip()
            else:
                print(f"警告: 无效的日志映射格式: {log_mapping}，应为 '模型名:日志路径'")

    generate_all_figures(args.results_dir, args.figures_dir, args.plots)