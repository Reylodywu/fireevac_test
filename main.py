from training.config import get_config
from training.utils import set_seed
from training.trainer import MATrainer
from training.evaluator import MATester


def main():
    # 1. 获取配置
    args = get_config()

    # 2. 设置随机种子
    set_seed(args.seed)

    # 3. 分发任务
    if args.mode == 'train':
        trainer = MATrainer(args)
        trainer.train()

    elif args.mode == 'test':
        tester = MATester(args)
        tester.test()

    elif args.mode == 'both':
        print("--- Phase 1: Training ---")
        trainer = MATrainer(args)
        trainer.train()

        print("\n--- Phase 2: Testing ---")
        # 确保测试加载的是刚刚训练好的
        args.load_path = 'checkpoints/latest_checkpoint.pth'
        tester = MATester(args)
        tester.test()


if __name__ == "__main__":
    main()