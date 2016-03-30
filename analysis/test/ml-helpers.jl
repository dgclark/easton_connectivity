using Base.Test
using PyCall

include("test-helpers.jl")
@include_src "ml-helpers.jl"

py_r2_score = begin
  @pyimport sklearn.metrics as metrics
  metrics.r2_score
end

y_true = rand(5)
y_pred = rand(5)

@test_approx_eq r2_score(y_true, y_pred) py_r2_score(y_true, y_pred)
