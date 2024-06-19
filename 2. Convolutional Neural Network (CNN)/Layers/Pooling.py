import numpy as np
from Layers import Base


class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.max_indexes = []

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        out_height = int(np.ceil((input_tensor.shape[2]-self.pooling_shape[0] + 1) / (self.stride_shape[0])))
        out_width = int(np.ceil((input_tensor.shape[3]-self.pooling_shape[1] + 1) / (self.stride_shape[1])))

        pool_out = np.zeros((input_tensor.shape[0], input_tensor.shape[1], out_height, out_width))
        for image in range(input_tensor.shape[0]):
            for channel in range(input_tensor.shape[1]):
                for y in range(out_height):
                    for x in range(out_width):
                        block = input_tensor[image][channel][
                                y * self.stride_shape[0]:y * self.stride_shape[0] + self.pooling_shape[0],
                                x * self.stride_shape[1]: x * self.stride_shape[1] + self.pooling_shape[1]]
                        highest, ind = self.max_and_index(block)
                        zero = np.zeros(self.input_tensor.shape)
                        zero_temp = np.zeros(self.pooling_shape)
                        zero_temp[ind] = 1
                        zero[image][channel][
                                y * self.stride_shape[0]:y * self.stride_shape[0] + self.pooling_shape[0],
                                x * self.stride_shape[1]: x * self.stride_shape[1] + self.pooling_shape[1]] = zero_temp

                        ind_temp = np.unravel_index(zero.argmax(), zero.shape)
                        temp_dict = {"from": ind_temp,
                                     "to": (image, channel, y, x)}
                        self.max_indexes.append(temp_dict)
                        pool_out[image][channel][y][x] = highest

        return pool_out

    def max_and_index(self, block):
        ind = np.unravel_index(block.argmax(), block.shape)
        return np.max(block), ind

    def backward(self, error_tensor):
        out_back = np.zeros(self.input_tensor.shape)

        for i_dict in self.max_indexes:
            fm, to = i_dict.values()
            out_back[fm] += error_tensor[to]

        return out_back

