using Base.Test
using PyCall

include("test-helpers.jl")
@include_src "utils.jl"

@test is_roi_col("L123")
@test !is_roi_col("L23a")
@test !is_roi_col("a123")
@test is_roi_col("R5")

@assert is_roi_col("R5_normalized", normalized=true)


x = [1. 2]'

y1 = [2. 4]'
lr = linear_reg(x, y1)
y1_beta0 = lr[:beta0][1]
y1_beta1 = lr[:beta1][1]
@test_approx_eq_eps y1_beta0 0. 1e-10
@test_approx_eq y1_beta1 2

y2 = [3. 4]'
lr = linear_reg(x, y2)
y2_beta0 = lr[:beta0][1]
y2_beta1 = lr[:beta1][1]
@test_approx_eq y2_beta0 2
@test_approx_eq y2_beta1 1

y = hcat(y1, y2)
lr = linear_reg(x, y)
@test_approx_eq lr[:beta0][1] y1_beta0
@test_approx_eq lr[:beta1][1] y1_beta1
@test_approx_eq lr[:beta0][2] y2_beta0
@test_approx_eq lr[:beta1][2] y2_beta1

x = [1. 1]'
y1 = [1. 3]'
#ypred is [2, 2]
res = calc_residuals(x, y1)
@test res' == [-1. 1.]

y2 = [1. 4]'
#ypred is [2.5, 2.5]
res = calc_residuals(x, y2)
@test res' == [-1.5 1.5]

y = hcat(y1, y2)
res = calc_residuals(x, y)
@test res == [-1. -1.5; 1. 1.5]


x = DataFrame(a=[1, 2, 3], b=[4, 5, 6])
x[:a_1] = x[:a]
delete_dupes!(x, "_1")
@test !in(:a_1, names(x))
@test in(:a, names(x))
