# -*- coding: utf-8 -*-

import math

def polysum(n, s):
    """
    n: integer - number of sides
    s: float - length
    return: Sum of the area and square of the perimeter of the regular polygon rounded to 4 decimal places
    """
    area = (0.25 * n * (s ** 2))/(math.tan(math.pi/n))
    perimiter = s * n
    return round(area + perimiter**2,4)