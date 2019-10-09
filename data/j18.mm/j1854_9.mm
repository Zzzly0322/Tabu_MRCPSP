************************************************************************
file with basedata            : md310_.bas
initial value random generator: 387623449
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  20
horizon                       :  146
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     18      0       23        1       23
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   7  15
   3        3          3           6   8  14
   4        3          2           7  13
   5        3          1          17
   6        3          3          10  11  15
   7        3          2           9  11
   8        3          2          10  13
   9        3          2          10  12
  10        3          2          16  17
  11        3          2          18  19
  12        3          2          16  18
  13        3          2          16  18
  14        3          2          15  17
  15        3          1          19
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
  2      1     1       8    6    2    8
         2     8       8    6    2    7
         3    10       8    5    1    6
  3      1     5       6    7    3    8
         2     7       6    4    2    5
         3     8       6    3    2    4
  4      1     1       8    9    7    7
         2     5       6    8    5    7
         3     9       4    8    5    6
  5      1     4       5    9    6    6
         2     5       4    7    4    4
         3     7       2    5    3    3
  6      1     2       6    9    8    9
         2     2       9    8    6    9
         3     6       2    8    5    9
  7      1     5      10   10    6   10
         2     6       9    6    6    7
         3     7       9    2    5    5
  8      1     6       7    7   10    7
         2     7       6    6    5    6
         3    10       5    5    3    2
  9      1     5       5    5    6    6
         2     7       5    3    6    3
         3     9       5    1    6    2
 10      1     6       6    5    7    6
         2     8       4    3    7    5
         3     9       3    3    7    3
 11      1     1      10    6    5    3
         2     1       9    8    5    4
         3     7       9    5    5    2
 12      1     3       7    7    3    6
         2     6       5    6    3    4
         3     9       4    4    3    2
 13      1     1       7    5    4    8
         2     5       7    4    4    8
         3     6       2    2    2    7
 14      1     1       7    3    7    4
         2     4       6    2    6    3
         3     9       6    1    6    3
 15      1     6       9    7    5    6
         2     6       9    6    6    7
         3     8       9    1    3    3
 16      1     4       8    5    5   10
         2     4       7    7    4   10
         3    10       5    5    3    8
 17      1     5       6    7    7    7
         2     7       5    6    7    6
         3     9       5    3    6    5
 18      1     1       6    7    3    7
         2     4       5    7    3    7
         3     7       4    7    2    5
 19      1     2       5    6    7    7
         2     6       4    1    4    6
         3     6       3    2    5    5
 20      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   21   19   94  115
************************************************************************
