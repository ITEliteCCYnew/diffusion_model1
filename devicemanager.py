import torch


class DeviceManager:
    def __init__(self):
        """
        初始化设备管理器，自动检测 MPS 可用性
        """
        self.mps_available = self._check_mps_support()
        self.device = self._select_device()

    def _check_mps_support(self):
        """
        检查 MPS 支持状态
        """
        if not torch.backends.mps.is_available():
            return False

        # 验证 MPS 是否真正可用
        try:
            # 创建测试张量
            test_tensor = torch.randn(100, 100, device=torch.device("mps"))
            # 执行简单计算
            _ = test_tensor * test_tensor
            return True
        except Exception:
            return False

    def _select_device(self):
        """
        根据可用性选择设备
        """
        if self.mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    def get_device(self):
        """
        返回最佳可用设备
        """
        print("使用设备:", self.device)
        return self.device

    def print_device_info(self):
        """
        打印设备信息
        """
        print(f"⚡ MPS 可用: {self.mps_available}")
        print(f"📱 当前设备: {self.device}")

        # 如果是 CPU 且 MPS 应该可用时给出警告
        if not self.mps_available and torch.backends.mps.is_built():
            print("⚠️ 警告: MPS 已构建但不可用，请检查系统兼容性")
        elif not self.mps_available:
            print("ℹ️ 提示: 使用 CPU 模式，性能可能较低")


# 使用示例
if __name__ == "__main__":
    # 创建设备管理器
    device_manager = DeviceManager()

    # 获取设备信息
    # device_manager.print_device_info()

    # 获取设备
    device = device_manager.get_device()