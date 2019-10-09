************************************************************************
file with basedata            : md301_.bas
initial value random generator: 162442515
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  20
horizon                       :  148
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     18      0       27       15       27
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5  10
   3        3          3           7  14  18
   4        3          3           9  11  19
   5        3          2           6   8
   6        3          3           9  12  13
   7        3          3           8   9  13
   8        3          2          11  15
   9        3          1          17
  10        3          3          11  16  18
  11        3          1          17
  12        3          3          14  17  18
  13        3          1          15
  14        3          1          15
  15        3          1          16
  16        3          1          19
  17        3          1          20
  18        3          1          20
  19        3          1          20
  20        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     6       5    8    8    6
         2     8       2    5    6    6
         3    10       2    2    3    4
  3      1     6       9   10    3    7
         2     8       7   10    3    5
         3     9       7   10    3    3
  4      1     7       6   10    9    3
         2     8       6    7    7    2
         3     9       5    5    5    2
  5      1     2       5   10    6    8
         2     7       5    8    4    2
         3     7       4    7    4    5
  6      1     3       6    4    6    7
         2     9       3    3    5    4
         3     9       4    4    4    2
  7      1     7       2    5    7    9
         2     8       2    4    5    7
         3     9       1    4    4    7
  8      1     4       9    8    6    2
         2     5       7    8    6    2
         3    10       3    3    3    2
  9      1     1       6   10    7    1
         2     3       4    9    6    1
         3     8       2    8    5    1
 10      1     5      10    6    8    2
         2     7      10    5    8    1
         3     9       9    4    4    1
 11      1     6       5    3    9    4
         2     7       4    2    9    4
         3     9       2    2    9    3
 12      1     1       7    2    8    9
         2     4       6    2    4    6
         3     9       3    1    1    4
 13      1     1       8    6    8    9
         2     5       6    3    7    5
         3     6       6    1    4    5
 14      1     3       8    8    3   10
         2     5       8    8    2    9
         3     8       6    8    2    8
 15      1     6       5    5    9    8
         2     7       2    4    6    8
         3     9       1    3    3    8
 16      1     1       1    5    4    7
         2     1       1    4    6    7
         3     4       1    3    3    5
 17      1     3       9    9    9    9
         2     4       6    9    9    9
         3     7       4    7    9    9
 18      1     5       7   10    9    7
         2     6       3   10    7    7
         3     7       2   10    5    7
 19      1     3       4    9    4   10
         2     3       4   10    4    9
         3     9       4    2    3    1
 20      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   16   18  100   96
************************************************************************
