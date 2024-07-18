# pylint: skip-file 

import random

def cekirdek(sayi: int) -> None:
    random.seed(sayi)

def rastgele_dogal(boyut, aralik=(0,100), dagilim='uniform'):
    if dagilim != 'uniform':
        raise ValueError("Invalid dagilim parameter. Must be 'uniform'.")
    if boyut==():
        return gergen(random.randint(aralik[0], aralik[1]))
    def dogal_helper(current_depth=0):
        if current_depth == len(boyut) - 1:
            return [random.randint(aralik[0], aralik[1]) for _ in range(boyut[current_depth])]
        else:
            return [dogal_helper(current_depth + 1) for _ in range(boyut[current_depth])]

    data = dogal_helper()
    return gergen(data)

def rastgele_gercek(boyut, aralik=(0.0, 1.0), dagilim='uniform'):
    if dagilim not in ['uniform']:
        raise ValueError("Invalid dagilim parameter. Must be 'uniform'.")
    if boyut==():
        gergen(random.uniform(aralik[0], aralik[1]))
    def gercek_helper(current_depth=0):
        if current_depth == len(boyut) - 1:
            return [random.uniform(aralik[0], aralik[1]) for _ in range(boyut[current_depth])]
        else:
            return [gercek_helper(current_depth + 1) for _ in range(boyut[current_depth])]

    data = gercek_helper()
    return gergen(data)


class Operation:
    def __call__(self, *operands):
        self.operands = operands
        self.outputs = None
        return self.ileri(*operands)

    def ileri(self, *operands):
        """
        Defines the forward pass of the operation.
        Must be implemented by subclasses to perform the actual operation.

        Parameters:
            *operands: Variable length operand list.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError

import math
from typing import Union

class gergen:

    __veri = None #A nested list of numbers representing the data
    D = None # Transpose of data
    __boyut = None #Dimensions of the derivative (Shape)


    def __init__(self, veri=None):
        self.D = None  # Transpose placeholder
        self.__veri = veri

        if isinstance(veri, gergen):
            self.__boyut  = veri.__boyut
        elif isinstance(veri, (int, float)):
            self.__boyut = ()
        elif isinstance(veri, list):
            dimensions = []
            while isinstance(veri, list):
                dimensions.append(len(veri))
                veri = veri[0] if veri else []
            self.__boyut = tuple(dimensions)
        else:
            raise TypeError("Unsupported data type for gergen initialization")


    def __getitem__(self, index):
    #Indexing for gergen objects
        if isinstance(index, int) or isinstance(index, slice):
            result = self.__veri[index]
        elif isinstance(index, tuple):
            result = self.__veri
            for idx in index:
                result = result[idx]
        else:
            raise TypeError("Index must be an int or slice")
        
        if isinstance(result, (list, int, float)):
            return gergen(result) if isinstance(result, list) else result
        else:
            raise ValueError("Invalid index operation")

    def __str__(self):
        if self.__veri is None or not self.__veri:
            return "Empty gergen"
        elif isinstance(self.__veri, (int, float)):
            return f"0 boyutlu skaler gergen:\n{self.__veri}"
        elif len(self.__boyut)==1:
            return f"1x{self.__boyut[0]} boyutlu gergen:\n[{self.__veri}]"
        else:
            dim_str = 'x'.join(map(str, self.__boyut)) + ' boyutlu gergen:'

            if all(isinstance(item, (float,int)) for item in self.__veri):
                data_str = '[[' + ', '.join(map(str, self.__veri)) + ']]'
            else:
                print_rows = ['[' + ', '.join(map(str, row)) + ']' for row in self.__veri]
                data_str = '[' + '\n'.join(print_rows) + ']'
                
            return f"{dim_str}\n{data_str}"


    def __mul__(self, other: Union['gergen', int, float]) -> 'gergen':
        if isinstance(self, (int, float)):
            return other*self.__veri
        if isinstance(other, gergen):
            if self.__boyut != other.__boyut:
                raise ValueError("Dimensions mismatch for element-wise multiplication.")
            def multiply_gergens(g1, g2):
                if isinstance(g1, list) and isinstance(g2, list):
                    return [multiply_gergens(sub1, sub2) for sub1, sub2 in zip(g1, g2)]
                else:
                    return g1 * g2
            result_data = multiply_gergens(self.__veri, other.__veri)
        elif isinstance(other, (int, float)):
            def scalar_multiply(g, scalar):
                if isinstance(g, list):
                    return [scalar_multiply(sub, scalar) for sub in g]
                else:
                    return g * scalar
            result_data = scalar_multiply(self.__veri, other)
        else:
            raise TypeError("Operand must be gergen, int, or float.")

        return gergen(result_data)
    
    def __rmul__(self, other: Union[int, float]) -> 'gergen':
        return self.__mul__(other)

    def __truediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        if isinstance(other, gergen):
            if self.__boyut != other.__boyut:
                raise ValueError("Dimensions mismatch for element-wise division.")
            def divide_gergens(g1, g2):
                if isinstance(g1, list) and isinstance(g2, list):
                    return [divide_gergens(sub1, sub2) for sub1, sub2 in zip(g1, g2)]
                else:
                    if g2 == 0:
                        raise ZeroDivisionError("Division by zero is not allowed.")
                    return g1 / g2
            result_data = divide_gergens(self.__veri, other.__veri)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            def scalar_divide(g, scalar):
                if isinstance(g, list):
                    return [scalar_divide(sub, scalar) for sub in g]
                else:
                    return g / scalar
            result_data = scalar_divide(self.__veri, other)
        else:
            raise TypeError("Operand must be gergen, int, or float.")

        return gergen(result_data)
    
    def __rtruediv__(self, other: Union[int, float]) -> 'gergen':
        return self.__truediv__(other)


    def __add__(self, other: Union['gergen', int, float]) -> 'gergen':
        if isinstance(other, gergen):
            if self.__boyut != other.__boyut:
                raise ValueError("Dimensions mismatch for element-wise addition.")
            def add_gergens(g1, g2):
                if isinstance(g1, list) and isinstance(g2, list):
                    return [add_gergens(sub1, sub2) for sub1, sub2 in zip(g1, g2)]
                else:
                    return g1 + g2
            result_data = add_gergens(self.__veri, other.__veri)
        elif isinstance(other, (int, float)):
            def scalar_add(g, scalar):
                if isinstance(g, list):
                    return [scalar_add(sub, scalar) for sub in g]
                else:
                    return g + scalar
            result_data = scalar_add(self.__veri, other)
        else:
            raise TypeError("Operand must be gergen, int, or float.")

        return gergen(result_data)
    
    def __radd__(self, other: Union[int, float]) -> 'gergen':
        return self.__add__(other)


    def __sub__(self, other: Union['gergen', int, float]) -> 'gergen':
        if isinstance(other, gergen):
            if self.__boyut != other.__boyut:
                raise ValueError("Dimensions mismatch for element-wise subtraction.")
            def subtract_gergens(g1, g2):
                if isinstance(g1, list) and isinstance(g2, list):
                    return [subtract_gergens(sub1, sub2) for sub1, sub2 in zip(g1, g2)]
                else:
                    return g1 - g2
            result_data = subtract_gergens(self.__veri, other.__veri)
        elif isinstance(other, (int, float)):
            def scalar_subtract(g, scalar):
                if isinstance(g, list):
                    return [scalar_subtract(sub, scalar) for sub in g]
                else:
                    return g - scalar
            result_data = scalar_subtract(self.__veri, other)
        else:
            raise TypeError("Operand must be gergen, int, or float.")

        return gergen(result_data)
    
    def __rsub__(self, other: Union[int, float]) -> 'gergen':
        return self.__sub__(other)


    def uzunluk(self):
        if self.__veri is None or not self.__veri:
            return 0
        elif len(self.__boyut) == 0:
            return 1
        else:
            total_elements = 1
            for dim in self.__boyut:
                total_elements *= dim
            return total_elements


    def boyut(self):
    # Returns the shape of the gergen
        return self.__boyut

    def devrik(self):
        if self.__veri is None:
            raise ValueError("Gergen is empty.")
        
        new_shape = self.__boyut[::-1]
        
        def init_empty_gergen(level=0):
            if level == len(new_shape) - 1:
                return [None] * new_shape[level]
            else:
                return [init_empty_gergen(level + 1) for _ in range(new_shape[level])]
        transposed_data = init_empty_gergen()

        def fill_transposed_gergen(source, dest, index, level=0):
            if level == len(self.boyut()):
                dest_index = index[::-1]
                for i in dest_index[:-1]:
                    dest = dest[i]
                dest[dest_index[-1]] = source
            else:
                for i, item in enumerate(source):
                    fill_transposed_gergen(item, dest, index + [i], level + 1)      

        fill_transposed_gergen(self.__veri, transposed_data, [])
        self.D = gergen(transposed_data)
        return gergen(transposed_data)


    def sin(self):
    #Calculates the sine of each element in the given `gergen`.
        def apply_sin(data):
            if isinstance(data, list):
                return [apply_sin(item) for item in data]
            else:
                return math.sin(data)
        sin_veri = apply_sin(self.__veri)
        
        return gergen(sin_veri)

    def cos(self):
    #Calculates the cosine of each element in the given `gergen`.
        def apply_cos(data):
            if isinstance(data, list):
                return [apply_cos(item) for item in data]
            else:
                return math.cos(data)
        cos_veri = apply_cos(self.__veri)
        
        return gergen(cos_veri)

    def tan(self):
    #Calculates the tangent of each element in the given `gergen`.
        def apply_tan(data):
            if isinstance(data, list):
                return [apply_tan(item) for item in data]
            else:
                return math.tan(data)
        tan_veri = apply_tan(self.__veri)
        
        return gergen(tan_veri)

    def us(self, n: int):
    #Raises each element of the gergen object to the power 'n'. This is an element-wise operation.
        if not isinstance(n, int) or n < 0:
            raise ValueError("Power must be a non-negative integer")
        else:
            def apply_power(data):
                if isinstance(data, list):
                    return [apply_power(item) for item in data]
                else:
                    return data**n
            power_veri = apply_power(self.__veri)
            return gergen(power_veri)

    def log(self):
    #Applies the logarithm function to each element of the gergen object, using the base 10.
        if isinstance(self.__veri, (int, float)):
            return gergen(math.log10(self.__veri))
        else:
            def apply_log10(data):
                if isinstance(data, list):
                    return [apply_log10(item) for item in data]
                else:
                    return math.log10(data)
            log10_veri = apply_log10(self.__veri)
            return gergen(log10_veri)

    def ln(self):
    #Applies the natural logarithm function to each element of the gergen object.
        if isinstance(self.__veri, (int, float)):
            return gergen(math.log(self.__veri))
        else:
            def apply_log(data):
                if isinstance(data, list):
                    return [apply_log(item) for item in data]
                else:
                    return math.log(data)
            log_veri = apply_log(self.__veri)
            return gergen(log_veri)

    def L1(self):
    # Calculates and returns the L1 norm
        return sum(self.duzlestir())

    def L2(self):
    # Calculates and returns the L2 norm
        duz = self.duzlestir()
        sum_=0
        for i in duz:
            sum_ += (i**2)
        return math.sqrt(sum_)


    def Lp(self, p):
    # Calculates and returns the Lp norm, where p should be positive integer
        if p <= 0:
            raise ValueError("p must be a positive integer")
        duz = self.duzlestir()
        sum_=0
        for i in duz:
            sum_ += (abs(i)**p)
        return (sum_)**(1/p)

    def listeye(self):
    #Converts the gergen object into a list or a nested list, depending on its dimensions.
        if self.__veri is None:
            return []
        return self.__veri


    def duzlestir(self):
    #Converts the gergen object's multi-dimensional structure into a 1D structure, effectively 'flattening' the object.
        def flatten(data):
            if isinstance(data, list):
                for element in data:
                    yield from flatten(element)
            else:
                yield data

        if isinstance(self.__veri, (int, float)):
            return gergen([self.__veri])
        
        flattened_data = list(flatten(self.__veri))
        
        return gergen(flattened_data)

    def boyutlandir(self, yeni_boyut):
    #Reshapes the gergen object to a new shape 'yeni_boyut', which is specified as a tuple.
        if not isinstance(yeni_boyut, tuple):
            raise TypeError("yeni_boyut must be a tuple")

        flattened_data = self.duzlestir().listeye()
        total_elements = self.uzunluk()

        new_total_elements = 1
        for dim in yeni_boyut:
            new_total_elements *= dim

        if new_total_elements != total_elements:
            raise ValueError("The product of the new dimensions does not match the total number of elements in the original gergen")

        def boyutlandir_helper(data, dims):
            if len(dims) == 0:
                return data.pop(0)
            size = dims[0]
            return [boyutlandir_helper(data, dims[1:]) for _ in range(size)]

        reshaped_data = boyutlandir_helper(flattened_data[:], list(yeni_boyut))
        return gergen(reshaped_data)

    def ic_carpim(self, other):
    #Calculates the inner (dot) product of this gergen object with another.
        if not isinstance(other, gergen):
            raise TypeError("Operands must be gergen objects")

        if len(self.__boyut) != len(other.__boyut):
            raise ValueError("Operands must have the same dimensionality")

        if self.__boyut[0] == 1:
            if self.__boyut[-1] != other.__boyut[-1]:
                raise ValueError("Operands must have compatible dimensions")
            sum_=0
            for i in range(0,self.__boyut[1]):
                sum_ += (self.__veri[0][i] * other.__veri[0][i])
            return sum_

        elif len(self.__boyut) == 2:
            if self.__boyut[1] != other.__boyut[0]:
                raise ValueError("Operands must have compatible inner dimensions")
            result = []
            for row_a in self.__veri:
                result_row = []
                for col_b in zip(*other.__veri):
                    dot_product = sum(a * b for a, b in zip(row_a, col_b))
                    result_row.append(dot_product)
                result.append(result_row)
            return gergen(result)

    def dis_carpim(self, other):
    #Calculates the outer product of this gergen object with another.
        if not isinstance(other, gergen):
            raise TypeError("Both operands must be gergen instances.")

        if not (self.__boyut[0] == 1 and other.__boyut[0] == 1):
            raise ValueError("Both operands must be 1-D arrays to compute the outer product.")

        result = []
        for i in range(self.uzunluk()):
            result_row = []
            for j in range(other.uzunluk()):
                result_row.append(self.__veri[0][i] * other.__veri[0][j])
            result.append(result_row)

        return gergen(result)

    def topla(self, eksen=None):
        def sum_elements(lst):
            if isinstance(lst[0], list):
                return [sum_elements(sublst) for sublst in zip(*lst)]
            else:
                return sum(lst)

        def sum_along_axis(data, axis):
            if axis == 0:
                return sum_elements(data)
            else:
                return [sum_along_axis(subdata, axis-1) for subdata in data]

        if eksen is None:
            return sum(self.duzlestir().listeye())

        if isinstance(eksen, int):
            if eksen < 0 or eksen >= len(self.__boyut):
                raise ValueError("Axis out of bounds for gergen's dimensionality")

            result = sum_along_axis(self.__veri, eksen)
            return gergen(result) if isinstance(result, list) else result
        else:
            raise TypeError("Eksen must be an integer or None")


    def ortalama(self, eksen=None):
    #Calculates the average of the elements of the gergen object, optionally along a specified axis 'eksen'.
        def mean_elements(lst):
            if isinstance(lst[0], list):
                return [mean_elements(sublst) for sublst in zip(*lst)]
            else:
                return sum(lst) / len(lst)

        def mean_along_axis(data, axis):
            if axis == 0:
                return mean_elements(data)
            else:
                return [mean_along_axis(subdata, axis-1) for subdata in data]

        if eksen is None:
            flattened_data = self.duzlestir().listeye()
            return sum(flattened_data) / len(flattened_data)

        if isinstance(eksen, int):
            if eksen < 0 or eksen >= len(self.__boyut):
                raise ValueError("Axis out of bounds for gergen's dimensionality")

            result = mean_along_axis(self.__veri, eksen)
            return gergen(result) if isinstance(result, list) else result
        else:
            raise TypeError("Eksen must be an integer or None")


"""## 2 Compare with NumPy"""

import numpy as np
import time

def example_1():
    # Example 1
    boyut = (64,64)
    a = rastgele_gercek(boyut)
    b = rastgele_gercek(boyut)

    # Gergen inner product
    start = time.time()
    result_gergen = a.ic_carpim(b)
    end = time.time()
    time_gergen = end - start

    # Numpy inner product
    start_np = time.time()
    a_np = np.array(a.listeye())
    b_np = np.array(b.listeye())
    result_np = np.dot(a_np, b_np)
    end_np = time.time()
    time_np = end_np - start_np

    # Compare results and time
    print("Gergen inner product result:", result_gergen)
    print("NumPy inner product result:", result_np)
    print("Time taken for gergen:", time_gergen)
    print("Time taken for numpy:", time_np)

def example_2():
    # Example 2
    boyut = (4, 16, 16, 16)
    a = rastgele_gercek(boyut)
    b = rastgele_gercek(boyut)
    c = rastgele_gercek(boyut)

    # Gergen calculation
    start = time.time()
    result_gergen = (a*b + a*c + b*c).ortalama()
    end = time.time()
    time_gergen = end - start

    # NumPy calculation
    a_np = np.array(a.listeye())
    b_np = np.array(b.listeye())
    c_np = np.array(c.listeye())
    start_np = time.time()
    result_np = (a_np * b_np + c_np * a_np + b_np * c_np).mean()
    end_np = time.time()
    time_np = end_np - start_np

    # Compare results and time
    print("Gergen calculation result:", result_gergen)
    print("NumPy calculation result:", result_np)
    print("Time taken for gergen:", time_gergen)
    print("Time taken for numpy:", time_np)

def example_3():
    # Example 3
    boyut = (3, 64, 64)
    a = rastgele_gercek(boyut)
    b = rastgele_gercek(boyut)

    # Gergen calculation
    start = time.time()
    result_gergen = (((a.sin() + b.cos()).us(2)).ln())/8
    end = time.time()
    time_gergen = end - start

    # NumPy calculation
    a_np = np.array(a.listeye())
    b_np = np.array(b.listeye())
    start_np = time.time()
    result_np = (np.log((np.sin(a_np) + np.cos(b_np)) ** 2) / 8)
    end_np = time.time()
    time_np = end_np - start_np

    # Compare results and time
    print("Gergen calculation result:", result_gergen)
    print("NumPy calculation result:", result_np)
    print("Time taken for gergen:", time_gergen)
    print("Time taken for numpy:", time_np)

example_1()
example_2()
example_3()

