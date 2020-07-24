# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/11/25 9:56
@Describe：
    super 以及多态的使用
"""
import abc


# 同一类事物:动物metaclass=abc.ABCMeta，用于实现多态
class Animal(metaclass=abc.ABCMeta):
    def __init__(self):
        self.name = 'this is animal class'
        print(self.name)
        print(self.__class__)

    # 指定子类必须要重写
    @abc.abstractmethod
    def eat(self, food):
        pass

    def run(self, animal):
        print(f'superclass {animal} is running ')

    def __str__(self):
        return f'{self.name} __str__ function'


class Dog(Animal):
    def __init__(self):
        # 继承父类，并且调用父类的构造方法   python3 super中可以省略类名
        super().__init__()
        self.name = 'this is dog class'

    def eat(self, food):
        print(f'subclass {self.name} eating  {food}')


class Cat(Animal):
    def __init__(self):
        # 继承父类，并且调用父类的构造方法  python2 super中必须加入类名参数
        super(Cat, self).__init__()
        self.name = 'this is cat class'

    def eat(self, food):
        print(f'subclass {self.name} eating  {food}')


class Pig(Animal):
    def __init__(self):
        # 继承父类，但是不会调用父类的构造方法，等价于super(Cat, self)
        super()
        self.name = 'this is pig class'
        print(self.__class__)

    def eat(self, food):
        pass


class Duck(Animal):
    def __init__(self):
        self.name = 'this is duck'
        self.__skill = 'swimming'

    def eat(self, food):
        print(f'subclass {self.name} eating  {food}')

    @property  # 等价于getter
    def skill(self):
        return self.__skill

    @skill.setter
    def skill(self, skill):
        self.__skill = skill


if __name__ == '__main__':
    dog = Dog()
    dog.run('dog')
    print('==============' * 4)
    cat = Cat()
    cat.eat('fish')
    print('==============' * 4)
    pig = Pig()
    pig.eat('noodle')
    print('==============' * 4)
    duck = Duck()
    duck.eat('bug')
    duck.run('duck')
    print(duck)
    print(duck.skill)
    duck.skill = 'gua gua gua'
    print(duck.skill)
