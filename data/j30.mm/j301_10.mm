************************************************************************
file with basedata            : mf1_.bas
initial value random generator: 111235251
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  32
horizon                       :  241
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     30      0       21       20       21
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           6   9  10
   3        3          3          11  18  20
   4        3          3           5  12  13
   5        3          2           6   8
   6        3          3           7  11  25
   7        3          2          20  24
   8        3          3          15  17  19
   9        3          1          21
  10        3          3          14  28  30
  11        3          1          28
  12        3          3          16  18  27
  13        3          2          15  21
  14        3          2          26  29
  15        3          2          20  24
  16        3          1          26
  17        3          2          22  30
  18        3          1          22
  19        3          2          23  25
  20        3          2          23  31
  21        3          1          23
  22        3          2          24  25
  23        3          2          27  28
  24        3          1          31
  25        3          2          26  29
  26        3          1          31
  27        3          1          30
  28        3          1          29
  29        3          1          32
  30        3          1          32
  31        3          1          32
  32        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     2       6    0    0   10
         2     3       5    0    7    0
         3     9       0   10    0   10
  3      1     4       7    0    0    3
         2     6       5    0    5    0
         3     9       0    7    3    0
  4      1     2       0    6    0    9
         2     7       8    0    7    0
         3    10       7    0    6    0
  5      1     1       5    0    0    7
         2     1       4    0    6    0
         3     2       4    0    3    0
  6      1     1       9    0    0    5
         2     5       3    0    5    0
         3     9       0    4    2    0
  7      1     8       0    9    0    9
         2     9       0    5    5    0
         3    10       0    4    0    8
  8      1     1       0    3    0    6
         2     3       7    0    7    0
         3     7       4    0    0    4
  9      1     3       5    0    0    9
         2     6       2    0    4    0
         3     9       2    0    0    8
 10      1     5       0    6    0    7
         2     8       9    0    1    0
         3     8       0    2    0    7
 11      1     2       0    8    8    0
         2     4       4    0    0    3
         3     5       0    6    5    0
 12      1     3       7    0    0    8
         2     6       0    6    0    6
         3     7       7    0    2    0
 13      1     4       0    5    0    8
         2     8       0    5    8    0
         3     9       6    0    8    0
 14      1     3       0    5    6    0
         2     3       0    2    0    5
         3     4       1    0    8    0
 15      1     2       8    0    0    7
         2     8       0    6    9    0
         3     9       0    5    9    0
 16      1     3       0    9    6    0
         2     6       0    7    0    5
         3    10       0    7    5    0
 17      1     3       0    8    7    0
         2     4       0    8    0    6
         3     9       7    0    5    0
 18      1     3       8    0    0    9
         2     6       7    0    0    6
         3     7       6    0    6    0
 19      1     4       5    0    0    2
         2     5       0    4    9    0
         3    10       2    0    9    0
 20      1     1       3    0    0    7
         2     1       3    0    9    0
         3     2       0    5    0    8
 21      1     2       0    7    0    6
         2     3       0    3    0    4
         3     7       6    0    0    4
 22      1     1       0    8    0    7
         2     3       0    4    3    0
         3     8       2    0    0    4
 23      1     2       7    0    0    2
         2     3       0    5    0    1
         3     8       0    3    5    0
 24      1     2       9    0    6    0
         2     2       7    0    0    5
         3    10       0    4    0    5
 25      1     4       7    0    0    8
         2     6       0    7    5    0
         3    10       0    6    4    0
 26      1     3       0    9    8    0
         2     4       0    8    3    0
         3    10       0    6    0    5
 27      1     2       3    0    6    0
         2     3       0    3    0    7
         3     7       0    2    0    7
 28      1     4       9    0    0    8
         2     6       0    2    0    3
         3     9       5    0    4    0
 29      1     2       8    0    5    0
         2     5       7    0    5    0
         3     7       1    0    0    5
 30      1     3       0    5   10    0
         2     6       5    0    0    4
         3    10       2    0    9    0
 31      1     2       0    5    0    8
         2     6       0    4    6    0
         3    10       5    0    0    6
 32      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   14   16   44   51
************************************************************************
