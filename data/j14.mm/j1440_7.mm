************************************************************************
file with basedata            : md168_.bas
initial value random generator: 1021309720
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  16
horizon                       :  103
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     14      0       19        0       19
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   7  12
   3        3          3           9  10  12
   4        3          2           8  10
   5        3          2           6  15
   6        3          1           9
   7        3          3           9  10  15
   8        3          2          11  12
   9        3          1          11
  10        3          2          13  14
  11        3          2          13  14
  12        3          2          14  15
  13        3          1          16
  14        3          1          16
  15        3          1          16
  16        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       7    6    9    4
         2     9       6    6    6    4
         3    10       4    6    2    4
  3      1     6       9    6    9    7
         2     6       7    9    9    8
         3    10       3    5    7    7
  4      1     2       9    8    7   10
         2     5       6    5    4    9
         3     6       4    4    4    8
  5      1     6       9    8    8    6
         2     7       5    6    7    5
         3     9       2    5    7    2
  6      1     1       4    9    5    7
         2     3       3    7    5    6
         3    10       1    2    5    4
  7      1     3       6    2   10    4
         2     7       1    2    9    2
         3     7       1    1   10    4
  8      1     3       1    3    7    3
         2     5       1    3    1    2
         3     5       1    2    4    3
  9      1     6       9    3    4    8
         2     7       6    2    4    3
         3     7       4    3    3    6
 10      1     1       7    5    5    7
         2     6       6    4    4    6
         3     7       6    2    3    4
 11      1     2       7    9    8    9
         2     8       5    5    8    9
         3     9       5    5    7    9
 12      1     6       1    3    8    4
         2     8       1    3    4    3
         3     8       1    2    6    3
 13      1     1       6    6    7    9
         2     2       5    4    7    6
         3     3       5    3    6    4
 14      1     3       8    8    6    7
         2     3       7    8    8    6
         3     7       6    8    2    5
 15      1     4       6    6    7    4
         2     5       2    4    4    2
         3     5       3    4    1    2
 16      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   33   27   71   67
************************************************************************
