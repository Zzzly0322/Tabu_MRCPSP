************************************************************************
file with basedata            : md83_.bas
initial value random generator: 259474187
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  14
horizon                       :  108
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     12      0       15        8       15
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           6   8  12
   3        3          3           9  11  13
   4        3          3           5   7   8
   5        3          3           6   9  10
   6        3          2          11  13
   7        3          3           9  10  11
   8        3          1          10
   9        3          1          12
  10        3          1          13
  11        3          1          14
  12        3          1          14
  13        3          1          14
  14        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       0    7    4    0
         2     5       0    6    2    0
         3     7       0    6    0    6
  3      1     3       6    0    6    0
         2     7       6    0    0    9
         3     9       0    4    0    6
  4      1     1       6    0    7    0
         2     7       0    2    5    0
         3     8       6    0    0    4
  5      1     2       0    5    0    6
         2     3       0    3    2    0
         3     9      10    0    0    5
  6      1     1       0    4    7    0
         2     2       4    0    7    0
         3    10       4    0    6    0
  7      1     2       0    8    0    6
         2     5       8    0    0    6
         3    10       0    8    0    3
  8      1     2       1    0    9    0
         2     2       0    3    0    4
         3     7       0    2    0    4
  9      1     1       0    6    0    8
         2     9       4    0    0    8
         3    10       3    0    0    8
 10      1     6       0    4    0    6
         2     8       3    0    8    0
         3    10       0    4    0    2
 11      1     4       7    0    3    0
         2     5       7    0    0    7
         3     9       0    8    0    7
 12      1     7       0    8    7    0
         2    10       0    5    0    4
         3    10       0    3    0    5
 13      1     4       0   10    8    0
         2     9       0    7    0    3
         3     9       0    7    4    0
 14      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   18   21   47   51
************************************************************************
