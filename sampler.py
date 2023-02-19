#https://github.com/issamemari/pytorch-multilabel-balanced-sampler/blob/master/sampler.py
import random
import numpy as np

from torch.utils.data.sampler import Sampler


class MultilabelBalancedRandomSampler(Sampler):
    """
MultilabelBalancedRandomSampler: 길이가 n_samples인 다중 레이블 데이터 세트와
    클래스 n_classes 수, 클래스당 동일한 확률을 가진 데이터의 샘플
    소수 클래스를 효과적으로 오버샘플링하고 다수 클래스를 언더샘플링합니다.
    같은 시간. 이 샘플러를 사용해도
    출력 샘플의 클래스는 데이터 세트가 다중 레이블이고
    샘플링은 단일 클래스를 기반으로 합니다. 그러나 이것은 모든 클래스가
    batch_size가 무한대에 가까워지면 적어도 batch_size / n_classes 샘플을 갖게 됩니다.
    """

    def __init__(self, labels, indices=None, class_choice="least_sampled"):
        """
        Parameters:
        -----------
            label: 모양의 멀티 핫 인코딩 numpy 배열(n_samples, n_classes)
            indices: 샘플링할 인덱스 목록을 나타내는 임의 길이의 1차원 numpy 배열
            class_choice: 모든 클래스에 대해 클래스를 선택하는 방법을 나타내는 문자열
            sample:
                "least_sampled": 지금까지 샘플링된 레이블 수가 가장 적은 클래스
                "random": 클래스가 무작위로 균일하게 선택됩니다.
                "cycle": 샘플러가 클래스를 순차적으로 순환합니다.
        """
        self.labels = labels
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))

        self.num_classes = self.labels.shape[1]

        # List of lists of example indices per class
        self.class_indices = []
        for class_ in range(self.num_classes):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

        self.counts = [0] * self.num_classes

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = np.random.choice(class_indices)
        if self.class_choice == "least_sampled":
            for class_, indicator in enumerate(self.labels[chosen_index]):
                if indicator == 1:
                    self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class + 1) % self.labels.shape[1]
        elif self.class_choice == "least_sampled":
            min_count = self.counts[0]
            min_classes = [0]
            for class_ in range(1, self.num_classes):
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = np.random.choice(min_classes)
        return class_

    def __len__(self):
        return len(self.indices)
