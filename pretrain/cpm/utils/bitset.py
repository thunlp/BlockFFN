import os
import random

import bmtrain as bmt
import numpy as np


class BitSet:
    def __init__(self, size=1024**2):
        self.size = size
        self.bitset = np.zeros(self.size, dtype=bool)

    def _ensure_capacity(self, num):
        """确保bitset有足够的容量来存储指定的数字"""
        if num >= self.size:
            # 扩展bitset大小
            new_size = max(num + 1, self.size * 2)
            new_bitset = np.zeros(new_size, dtype=bool)
            new_bitset[: self.size] = self.bitset
            self.bitset = new_bitset
            self.size = new_size
            bmt.print_rank("enlarge size to {}".format(self.size))

    def add(self, num):
        """向bitset中添加一个数字"""
        self._ensure_capacity(num)
        self.bitset[num] = True

    def remove(self, num):
        """从bitset中移除一个数字"""
        if num < self.size:
            self.bitset[num] = False

    def contains(self, num):
        """检查bitset是否包含某个数字"""
        return num < self.size and self.bitset[num]

    def __contains__(self, num):
        return self.contains(num)

    def update(self, iterable_or_bitset):
        """使用可迭代对象或另一个BitSet中的元素更新当前bitset"""
        if isinstance(iterable_or_bitset, BitSet):
            # 如果参数是BitSet，则使用numpy的向量化操作更新
            self._ensure_capacity(iterable_or_bitset.size)
            self.bitset[: iterable_or_bitset.size] |= iterable_or_bitset.bitset
        else:
            # 如果参数是可迭代对象，则遍历并添加每个元素
            for num in iterable_or_bitset:
                self.add(num)

    def __sub__(self, other):
        """实现减法运算符，使用numpy向量化操作来高效地创建一个新的bitset"""
        # 创建一个新的bitset实例
        result = BitSet(max(self.size, other.size))
        # 使用numpy的向量化逻辑运算
        result.bitset[: self.size] = self.bitset & ~other.bitset[: self.size]
        return result

    def __isub__(self, other):
        """实现就地减法运算符，利用numpy向量化操作进行高效的元素移除"""
        # 首先确保other的大小不超过当前bitset的大小
        min_size = min(self.size, other.size)
        # 使用numpy的向量化逻辑运算进行元素移除
        self.bitset[:min_size] &= ~other.bitset[:min_size]
        return self

    def __str__(self):
        """返回bitset的字符串表示，列出所有为真的位的索引"""
        # 找出所有为真的位的索引
        true_indices = np.where(self.bitset)[0]
        # 将这些索引转换为字符串并用逗号分隔
        indices_str = ", ".join(map(str, true_indices))
        return f"BitSet({indices_str})"

    def __len__(self):
        """返回bitset中为真的元素个数"""
        return self.bitset.sum()

    def capacity(self):
        return self.size

    def density(self):
        return len(self) / self.size

    def memory_usage(self):
        """返回bitset所占用的内存大小，以KB、MB或GB为单位"""
        bytes_usage = self.bitset.nbytes
        if bytes_usage < 1024:
            return f"{bytes_usage} B"
        elif bytes_usage < 1024**2:
            return f"{bytes_usage / 1024:.2f} KB"
        elif bytes_usage < 1024**3:
            return f"{bytes_usage / 1024**2:.2f} MB"
        else:
            return f"{bytes_usage / 1024**3:.2f} GB"

    def to_list(self):
        """返回一个包含所有为真位索引的列表"""
        return list(np.where(self.bitset)[0])

    def save(self, filename):
        """将bitset保存到文件"""

        def random_hash():
            """返回一个随机哈希值"""
            return random.randint(0, 2**64 - 1)

        filename_with_suffix = filename + ".{}.npy".format(random_hash())
        dirname = os.path.dirname(filename_with_suffix)
        os.makedirs(dirname, exist_ok=True)
        np.save(filename_with_suffix, self.bitset)
        return os.path.basename(filename_with_suffix)  # 返回最后的名字，不带前缀，支持tranfer项目

    @classmethod
    def load(cls, filename_with_suffix):
        """从文件加载bitset并创建一个新的BitSet实例"""
        bitset_array = np.load(filename_with_suffix)
        bitset = cls(bitset_array.size)
        bitset.bitset = bitset_array
        return bitset


def bitset_diff(normal_set, bitset):
    """返回存在于normal_set中但不在bitset中的元素集合"""
    ret = {elem for elem in normal_set if not bitset.contains(elem)}
    return ret


if __name__ == "__main__":
    # 示例使用
    bitset1 = BitSet(1024)
    bitset1.update([100, 200, 300, 1023])

    bitset2 = BitSet(1024)
    bitset2.update([100, 400, 1023])

    result_bitset = bitset1 - bitset2
    print(100 in result_bitset)  # 应该输出False
    print(200 in result_bitset)  # 应该输出True
    print(300 in result_bitset)  # 应该输出True
    print(1023 in result_bitset)  # 应该输出False

    bitset1 -= bitset2
    print(result_bitset)  # BitSet(200, 300)
    print(bitset1)  # BitSet(200, 300)
    print(bitset2)  # BitSet(100, 400, 1023)

    bitsetlarge = BitSet(1024**3)
    print(len(bitsetlarge), bitsetlarge.capacity(), bitsetlarge.density(), bitset1.density())
    print("BitSet memory usage:", bitsetlarge.memory_usage())

    print(bitset_diff({100, 200}, bitset2))

    bitset1.update(bitset2)
    bitsetlarge.add(52260134)
    bitset2.update(bitsetlarge)
    print(bitset1)  # BitSet(100, 200, 300, 400, 1023)
    print(bitset2)  # BitSet(100, 400, 1023, 52260134)
