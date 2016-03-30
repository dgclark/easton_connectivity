using PyCall


macro l2_from_true(x)
  :(sum( (y_true - $x).^2))
end


function r2_score(y_true::Vector{Float64}, y_pred::Vector{Float64})

  numerator::Float64 = @l2_from_true y_pred
  denominator::Float64 = @l2_from_true mean(y_true)

  1 - numerator/denominator
end


@pyimport sklearn.svm as svm
LinearSVR = svm.LinearSVR
