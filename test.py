import numpy as np
from simple_driving.envs.simple_driving_env import SimpleDrivingEnv  # 请根据你的实际路径修改


def test_get_metrics():
    print("🔍 正在初始化环境...")
    try:
        # 创建环境实例（无渲染）
        env = SimpleDrivingEnv(renders=False)

        print("🔁 调用 env.reset()...")
        env.reset()

        print("🧪 检查是否存在 get_metrics 方法...")
        if hasattr(env, "get_metrics") and callable(env.get_metrics):
            print("✅ get_metrics() 方法存在，正在调用...")

            metrics = env.get_metrics()

            print("🧪 检查返回格式...")
            if isinstance(metrics, dict) and "distance_to_goal" in metrics:
                print("✅ get_metrics() 调用成功！")
                print(f"📏 距离目标: {metrics['distance_to_goal']:.2f} 米")
                print(f"🚗 小车位置: {np.round(metrics['car_position'], 2)}")
                print(f"🎯 目标位置: {np.round(metrics['goal_position'], 2)}")
            else:
                print("❌ get_metrics() 返回格式不正确，应包含 'distance_to_goal', 'car_position', 'goal_position'")
        else:
            print("❌ get_metrics() 方法不存在，或不可调用。")

        env.close()
        print("✅ 测试完成！")

    except Exception as e:
        print("❌ 测试过程中发生异常：")
        print(e)

if __name__ == "__main__":
    test_get_metrics()
    print("当前使用的 SimpleDrivingEnv 来自文件：", SimpleDrivingEnv.__module__)
    import inspect
    print("完整路径：", inspect.getfile(SimpleDrivingEnv))

