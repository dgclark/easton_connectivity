using DataFrames
using Memoize
using MLBase
using PyCall

include("ml-helpers.jl")
include("utils.jl")

pre_process = mk_pre_process_fn()

function diagnify(syms::Vector{Symbol})
  num_syms::Int64 = length(syms)

  ret::Vector{Symbol} = begin
    num_multis::Int64 = round(Int64, num_syms*(num_syms + 1) / 2)
    Array(Symbol, num_multis)
  end

  ix::Int64 = 1
  for (i, s) = enumerate(syms)
    for (i2, s2) in enumerate(syms[i:end])
      ret[ix] = symbol(s, "_", s2)
      ix = ix + 1
    end
  end

  ret
end


function transform_subjects(data::DataFrame= begin
                              df = all_data
                              pre_process(df)
                              df
                            end,
                            covars::Vector{Symbol}=default_covars)
  roi_cols::Vector{Symbol} = get_roi_cols(data)

  num_rois = length(roi_cols)
  num_subj = size(data, 1)
  num_covars = length(covars)

  syms_dia = diagnify(roi_cols)
  num_syms_dia = length(syms_dia)

  ret = DataFrame(repmat([Float64], num_syms_dia),
                  syms_dia,
                  num_subj)

  ut_mask = tril(ones(Bool, num_rois, num_rois))

  for s in 1:num_subj
    multis::Array{Float64, 1} = begin
      row = DataArray(data[s, roi_cols])[:]
      mat = row * row'
      @assert size(mat) == (num_rois, num_rois)
      mat[ut_mask][:]
    end

    #hack, cant set several cols simultaneously
    for i in 1:num_syms_dia
      ret[s, i] = multis[i]
    end
  end

  for c in covars
    ret[c] = data[c]
  end

  ret

end


function load_X()
  raw_data = all_data
  covars = pre_process(raw_data)
  transform_subjects(raw_data, covars)
end


function load_Y(target::Symbol, raw_data=nothing)
  if raw_data == nothing
    raw_data = all_data
  end
  raw_data[target]
end


function load_Xy_mat(target::Symbol)
  X = load_X()
  y = load_Y(target)
  Matrix(X), Vector(y)
end


function train_test_split(X::Matrix{Float64}, y::Vector{Float64}, ratio::Float64=.8)
  num_samples = length(y)
  all_samples = 1:num_samples

  train_samples = begin
    num_train_samples = round(Int64, ratio*num_samples)
    sample(all_samples, num_train_samples, replace=false)
  end
  test_samples = setdiff(all_samples, train_samples)

  X[train_samples, :], y[train_samples], X[test_samples, :], y[test_samples]
end


function simple_pipe(X::Matrix, y::Vector, svr::PyObject)
  X_train, y_train, X_test, truths = train_test_split(X, y)
  svr[:fit](X_train, y_train)

  test_score = r2_score(truths, svr[:predict](X_test))
  train_score = r2_score(y_train, svr[:predict](X_train))

  train_score, test_score
end


roi_corrs = begin
  rois = get_roi_cols(all_data)::Vector{Symbol}
  num_rois = length(rois)

  roi_cors_mat = begin
    roi_data = Matrix(all_data[:, rois])
    cor(roi_data, roi_data)
  end

  @assert size(roi_cors_mat) == (num_rois, num_rois)

  ret::DataFrame = begin
    num_edges = round(Int64, (num_rois^2 - num_rois)/2)

    left_rois = repmat([:a], num_edges)::Vector{Symbol}
    right_rois = repmat([:a], num_edges)::Vector{Symbol}
    cors = zeros(Float64, num_edges)

    ix = 0
    for i::Int64 in 1:num_rois
      for j::Int64 in (i+1):num_rois
        ix+=1
        left_rois[ix] = rois[i]
        right_rois[ix] = rois[j]
        cors[ix] = roi_cors_mat[i, j]
      end
    end

    DataFrame(left=left_rois, right=right_rois, cor=cors)

  end

  ret
end


function get_rois_above_cutoff(cutoff::Float64)
  gt_c::DataFrame = roi_corrs[roi_corrs[:cor] .> cutoff, :]
  collect(union(gt_c[:left], gt_c[:right]))
end


function explore_corrs(Cs::AbstractVector{Float64}, cutoffs::AbstractVector{Float64})
  svr = LinearSVR()
  y = Vector(all_data[:fluency])
  for C in Cs
    svr[:C] = C
    for cut in cutoffs
      rois::Vector{Symbol} = get_rois_above_cutoff(cut)
      X = Matrix(all_data[:, rois])
      train_score, test_score = simple_pipe(X, y, svr)
      println("C: $C, cutoff: $cut")
      println("train_score: $train_score")
      println("test_score: $test_score")
    end
  end
end
