************************************************************************
file with basedata            : md279_.bas
initial value random generator: 1557578893
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  20
horizon                       :  157
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     18      0       22        5       22
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   8
   3        3          1          18
   4        3          3           6   7   9
   5        3          2           6   9
   6        3          3          12  13  19
   7        3          3           8  12  13
   8        3          2          17  19
   9        3          3          10  15  19
  10        3          3          11  13  16
  11        3          1          12
  12        3          1          14
  13        3          2          14  18
  14        3          1          17
  15        3          1          16
  16        3          2          17  18
  17        3          1          20
  18        3          1          20
  19        3          1          20
  20        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     2       3   10    9    0
         2     2       3    9    0    5
         3     9       2    7    8    0
  3      1     7       5    9    3    0
         2     9       3    8    0    5
         3    10       2    7    2    0
  4      1     4       4    8    6    0
         2     7       4    8    0    2
         3     8       3    7    3    0
  5      1     1       7    6    0    9
         2     4       4    6    5    0
         3    10       3    6    0    6
  6      1     1       6    5    7    0
         2     5       4    4    0    7
         3     7       4    4    4    0
  7      1     5       6    7    0    5
         2     8       5    7    0    4
         3     9       2    6    0    4
  8      1     4       7    8    0    5
         2     8       7    6    4    0
         3    10       7    5    0    3
  9      1     3       7    9    7    0
         2     4       7    9    0    8
         3     6       7    7    3    0
 10      1     2       2    2    0    4
         2     4       2    2    0    2
         3    10       1    1    9    0
 11      1     1       5    1    0   10
         2     5       4    1    6    0
         3     5       3    1    0    4
 12      1     2       5    5    8    0
         2     2       4    5    0   10
         3    10       4    5    9    0
 13      1     2       4    8    8    0
         2     7       3    7    8    0
         3    10       2    5    8    0
 14      1     2       7    3    0    8
         2     9       7    2    0    7
         3     9       7    3    3    0
 15      1     2       5    2    7    0
         2     3       5    2    6    0
         3     9       4    2    0    7
 16      1     2       4    8    0    6
         2     4       3    6    5    0
         3     7       3    5    3    0
 17      1     3       7    7    0    5
         2    10       4    5    0    3
         3    10       4    6    8    0
 18      1     7       7    5    8    0
         2     7       6    6    0    5
         3     8       4    4    8    0
 19      1     9       5    9    0    8
         2     9       5    6    4    0
         3    10       3    4    4    0
 20      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   20   24   83   83
************************************************************************
