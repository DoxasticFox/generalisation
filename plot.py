import matplotlib.pyplot as plt
Xs = [x for x in range(0, 3510, 10)]
Ys = [
    49.8,
    51.2,
    50.5,
    48.0,
    51.9,
    45.3,
    41.7,
    37.0,
    31.7,
    31.5,
    26.0,
    24.2,
    24.9,
    28.1,
    25.8,
    23.6,
    24.4,
    23.0,
    24.9,
    27.3,
    24.2,
    26.7,
    26.7,
    24.0,
    24.3,
    24.8,
    24.7,
    27.6,
    26.2,
    24.6,
    27.1,
    23.8,
    27.7,
    26.7,
    26.1,
    23.4,
    25.0,
    27.6,
    23.0,
    25.5,
    27.5,
    26.6,
    28.5,
    25.3,
    23.6,
    24.3,
    26.0,
    23.5,
    24.5,
    22.2,
    25.6,
    45.8,
    47.7,
    44.6,
    46.0,
    41.1,
    32.6,
    18.1,
    13.5,
    11.9,
    10.2,
    10.1,
    9.5,
    10.6,
    10.2,
    8.7,
    8.1,
    9.8,
    9.6,
    8.5,
    8.2,
    9.0,
    10.5,
    9.7,
    9.6,
    9.9,
    8.8,
    10.9,
    10.2,
    9.5,
    9.0,
    10.0,
    9.6,
    10.3,
    8.9,
    9.6,
    9.6,
    9.0,
    10.5,
    7.9,
    8.0,
    9.7,
    9.0,
    9.6,
    9.5,
    8.7,
    9.9,
    11.1,
    8.8,
    8.7,
    10.1,
    43.7,
    42.4,
    24.0,
    14.1,
    8.2,
    6.9,
    5.9,
    5.5,
    6.2,
    6.5,
    5.8,
    5.4,
    5.2,
    7.0,
    6.3,
    5.6,
    7.3,
    7.4,
    5.7,
    8.0,
    6.2,
    6.4,
    5.1,
    7.4,
    6.5,
    6.1,
    7.9,
    6.8,
    7.1,
    7.9,
    6.7,
    6.5,
    7.9,
    6.1,
    9.5,
    6.4,
    8.2,
    5.2,
    7.1,
    7.2,
    5.5,
    9.0,
    8.1,
    8.5,
    6.6,
    6.9,
    7.7,
    9.0,
    8.3,
    6.7,
    36.9,
    16.4,
    5.2,
    3.4,
    2.2,
    2.4,
    3.4,
    2.4,
    3.2,
    1.5,
    2.3,
    2.4,
    2.6,
    2.5,
    1.7,
    3.1,
    3.0,
    1.7,
    3.0,
    2.7,
    2.6,
    2.5,
    3.0,
    1.3,
    3.3,
    3.1,
    2.8,
    2.7,
    2.2,
    3.7,
    1.9,
    3.3,
    3.2,
    2.6,
    2.0,
    2.5,
    2.5,
    2.9,
    2.9,
    3.5,
    2.9,
    2.4,
    2.1,
    2.8,
    2.3,
    4.1,
    2.2,
    3.4,
    2.5,
    3.6,
    10.3,
    5.6,
    7.0,
    3.8,
    1.8,
    1.3,
    1.0,
    1.7,
    1.5,
    1.5,
    1.4,
    0.9,
    1.8,
    2.0,
    1.6,
    1.6,
    2.5,
    1.9,
    2.3,
    1.7,
    1.4,
    2.2,
    1.3,
    1.0,
    2.0,
    1.3,
    2.4,
    2.1,
    2.1,
    0.5,
    2.8,
    1.9,
    2.4,
    2.1,
    1.8,
    1.4,
    2.3,
    2.2,
    2.3,
    1.1,
    0.7,
    0.9,
    1.1,
    1.7,
    3.0,
    1.9,
    1.4,
    1.7,
    1.2,
    1.2,
    23.1,
    9.0,
    7.9,
    6.7,
    4.0,
    2.1,
    1.5,
    2.0,
    1.8,
    1.3,
    1.4,
    0.8,
    0.8,
    0.5,
    1.7,
    1.3,
    1.0,
    1.8,
    1.7,
    2.9,
    1.1,
    1.4,
    1.2,
    1.9,
    0.9,
    1.5,
    1.1,
    1.6,
    0.8,
    1.5,
    1.6,
    2.1,
    0.6,
    2.1,
    1.7,
    0.8,
    0.8,
    2.5,
    1.1,
    0.5,
    0.8,
    2.7,
    1.3,
    1.3,
    0.9,
    2.3,
    1.2,
    0.6,
    0.7,
    1.5,
    30.4,
    18.4,
    8.5,
    3.6,
    2.0,
    1.5,
    1.3,
    1.1,
    0.7,
    1.5,
    0.4,
    0.5,
    0.5,
    0.9,
    1.0,
    0.4,
    0.9,
    0.7,
    0.6,
    0.7,
    0.3,
    1.1,
    1.2,
    0.4,
    1.2,
    0.7,
    1.0,
    1.0,
    0.9,
    0.9,
    1.0,
    0.4,
    0.8,
    0.9,
    0.8,
    0.5,
    1.1,
    0.8,
    0.9,
    0.5,
    1.0,
    1.0,
    0.5,
    0.9,
    0.9,
    0.6,
    1.2,
    1.4,
    1.0,
    0.8,
]

plt.xlabel('Batch Number')
plt.ylabel('Generalisation Error (%)')
plt.plot(Xs, Ys, 'k')
plt.show()
