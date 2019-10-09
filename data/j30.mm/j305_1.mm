************************************************************************
file with basedata            : mf5_.bas
initial value random generator: 668
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  32
horizon                       :  247
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     30      0       31       22       31
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           6  10  12
   3        3          3          11  22  28
   4        3          3           5   7   9
   5        3          3           8  12  28
   6        3          3          15  17  18
   7        3          1          17
   8        3          2          10  11
   9        3          2          14  29
  10        3          2          30  31
  11        3          3          13  24  26
  12        3          2          16  19
  13        3          1          25
  14        3          3          15  16  21
  15        3          3          20  24  27
  16        3          1          30
  17        3          2          19  22
  18        3          2          21  23
  19        3          2          20  26
  20        3          1          23
  21        3          1          27
  22        3          1          26
  23        3          1          25
  24        3          1          25
  25        3          1          31
  26        3          3          27  29  30
  27        3          1          31
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
  2      1     4       7    5    6    0
         2     5       6    3    0    7
         3     8       6    1    0    5
  3      1     1       8    7    7    0
         2     1       7    9    0    4
         3    10       5    5    7    0
  4      1     2       7    3    2    0
         2     4       6    2    0    5
         3     9       3    2    0    4
  5      1     7       8    8    0   10
         2     8       7    7    0    8
         3     9       6    5    0    6
  6      1     3       4    7    9    0
         2     6       4    5    0    4
         3    10       4    1    0    3
  7      1     4      10    8    0    4
         2     9       8    8    0    4
         3     9       9    7    0    1
  8      1     2       9    7    6    0
         2     4       9    5    0    1
         3     8       8    3    3    0
  9      1     3       9    5    0    7
         2     3       7    4    0    9
         3     7       5    2    5    0
 10      1     1       5    6    5    0
         2     1       5    6    0    2
         3     4       4    6    0    2
 11      1     3       5    6    4    0
         2     3       5    6    0    6
         3     6       5    6    0    5
 12      1     2      10   10    0    1
         2     6       5    9    0    1
         3    10       3    8    0    1
 13      1     2       3    2    7    0
         2     7       2    2    0    7
         3     9       1    1    5    0
 14      1     3       7    4    0    6
         2     7       7    4    7    0
         3    10       5    4    5    0
 15      1     1       9    9    0    4
         2     4       7    5    6    0
         3     5       7    3    5    0
 16      1     1       8    5    6    0
         2     2       8    5    0    3
         3    10       8    3    0    2
 17      1     2       8    4    5    0
         2     3       4    4    5    0
         3     7       1    3    4    0
 18      1     1       6    8    0    9
         2     5       3    8    5    0
         3     7       2    7    0    8
 19      1     3       8    7    9    0
         2     5       8    7    8    0
         3     6       6    6    8    0
 20      1     5       6   10    8    0
         2     8       3    9    0    4
         3    10       3    9    5    0
 21      1     6       2    6    0    2
         2     9       2    4    4    0
         3    10       2    4    0    1
 22      1     6       9    6    0    8
         2     6       8    6   10    0
         3     7       7    5    0    7
 23      1     5       6    3   10    0
         2     7       5    2    7    0
         3     9       5    1    5    0
 24      1     2       9    5    6    0
         2     2       7    7    6    0
         3     3       5    2    6    0
 25      1     5       7    7    5    0
         2     6       6    6    0    3
         3    10       4    4    5    0
 26      1     2       7    2    9    0
         2     7       4    2    9    0
         3     9       4    1    0    9
 27      1     1       8    3    0    8
         2     1       8    4    0    5
         3    10       6    3    0    1
 28      1     1       7    7    0    5
         2     5       6    6    8    0
         3     9       5    6    6    0
 29      1     2       5    4   10    0
         2     5       5    3   10    0
         3     8       5    3    0    3
 30      1     2       3    8    7    0
         2     5       3    8    0    9
         3     8       3    5    3    0
 31      1     2       8    9    6    0
         2     8       7    8    6    0
         3    10       7    8    0    4
 32      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   20   24   60   41
************************************************************************