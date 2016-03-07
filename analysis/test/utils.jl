include("test-helpers.jl")
src = getsrc("utils.jl")
include(src)

@assert is_roi_col("L123")
@assert !is_roi_col("L23a")
@assert !is_roi_col("a123")
@assert is_roi_col("R5")
