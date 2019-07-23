# Time series data
SKIP_ROWS = 7
TIMESTAMP_INDEX = 0
MUS1_INDEX = 1
MUS2_INDEX = 2
MUS3_INDEX = 3
MUS4_INDEX = 4
MUS5_INDEX = 5
MUS6_INDEX = 6

# Action label data
# Row mapper
CODE_INDEX = 0
KNOT_TYPE_INDEX = 1
TIME_DIFF_INDEX = 3
ROW_MAPPER_CUTOFF_INDEX = 4  # NON-inclusive

# Timestamps
START_TIME_INDEX = 4
END_TIME_INDEX = 5
ROW_MAPPER_INDICES = [0,1,2,3,4,20]  # the last two are start and end

# General
COLUMNS_USED = [0, 2, 3, 4, 5, 6, 7]

# Ratings
RATING_ROB_OR_LAP_INDEX = 1
RATING_INDEX = 3

MUS1 = "MUS1"
MUS2 = "MUS2"
MUS3 = "MUS3"
MUS4 = "MUS4"
MUS5 = "MUS5"
MUS6 = "MUS6"

TIME = "TIME"

ROB = "ROB"
LAP = "LAP"

# maps muscles to columns
MUS2COL = {MUS1:2, MUS2:3, MUS3:4, MUS4:5, MUS5:6, MUS6:7}
# maps time to column
T2COL = {TIME: 0}

DIVIDE_MARKER = "Interval="