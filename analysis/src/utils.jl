using DataFrames
using Distributions

function data_f(src_f)
  "../data/$src_f"
end


function linear_reg{xT, yT}(x::AbstractMatrix{xT}, y::AbstractMatrix{yT})
  # y = x * beta
  num_rows_x, num_features_x = size(x)
  num_rows, num_features_y = size(y)
  @assert num_rows_x == num_rows

  centered_x, centered_y = map((x,y)) do arr
    arr .- mean(arr, 1)
  end

  beta1 = \(centered_x, centered_y)
  @assert size(beta1) == (num_features_x, num_features_y)

  beta0 = begin
    scale_beta = mean(x, 1) * beta1
    @assert length(scale_beta) == num_features_y
    mean(y, 1) .- scale_beta
  end
  @assert length(beta0) == num_features_y

  res = calc_residuals(beta0, beta1, x, y)

  sd = sqrt(sum( res .^ 2 ))
  se = sd/sqrt(num_rows - 1)
  se_mean = sd * sqrt( sum( x.^2)/(length(x) * sum( (x - mean(x)).^2)))
  t_mean = beta0/se_mean
  t = beta1/se
  tdist = TDist(num_rows - 1)

  Dict(:beta0 => beta0, :beta1 => beta1, :se=>se, :se_mean=>se_mean, :t => t, :t_mean => t_mean)
end


function id_inner_join(a::DataFrame, b::DataFrame)
  join(a, b, kind=:inner, on=:id)
end

function calc_residuals{xT, yT}(beta0::AbstractMatrix{Float64},
                                beta1::AbstractMatrix{Float64},
                                x::AbstractMatrix{xT},
                                y::AbstractMatrix{yT})
  y_pred = begin
    @assert size(beta1) == (size(x, 2), size(y, 2))
    @assert size(beta0) == (1, size(y, 2))
    beta0 .+ x * beta1
  end
  y - y_pred
end

function calc_residuals{xT, yT}(x::AbstractMatrix{xT}, y::AbstractMatrix{yT})
  lr = linear_reg(x, y)
  beta0, beta1 = lr[:beta0], lr[:beta1]
  calc_residuals(beta0, beta1, x, y)
end

function calc_residuals(predictions::Vector{Symbol}, predictors::Vector{Symbol},
                        dataframe::DataFrame)
  num_subjects = size(dataframe, 1)
  num_predictions = length(predictions)

  predictor_mat = Matrix(dataframe[predictors])
  prediction_mat = Matrix(dataframe[predictions])

  residuals_mat = calc_residuals(predictor_mat, prediction_mat)
  @assert size(residuals_mat) == (num_subjects, num_predictions)

  residuals = DataFrame(residuals_mat)
  name_lookup = Dict{Symbol, Symbol}([v => predictions[i]
                                      for (i, v) in enumerate(names(residuals))])
  rename!(residuals, name_lookup)

  residuals
end


function is_roi_col(col_name; normalized::Bool=false)
  match_r = normalized ? r"[L|R][0-9]+_normalized$" : r"[L|R][0-9]+$"
  ismatch(match_r, col_name)
end

function get_roi_cols(df::DataFrame; normalized::Bool=false)
  function match(s::Symbol)
    is_roi_col(string(s), normalized=normalized)
  end
  filter(match, names(df))
end

default_covars = Symbol[:age, :sex, :edu, :total_gray]

function calc_total_gray!(df::DataFrame)
  roi_cols::Vector{Symbol} = get_roi_cols(df)
  row_sum = r -> sum([v for (s, v) in r])
  df[:total_gray] = Float64[row_sum(r) for r in eachrow(df[roi_cols])]
end

function load_all_data()
  meta_data = readtable(data_f("animal_scores.csv"))
  roi_data = readtable(data_f("ROI_matrix.txt"), separator='\t')
  handedness_data = readtable(data_f("handedness.csv"))

  num_cols = length(union(names(roi_data), names(meta_data), names(handedness_data)))

  all_data = begin
    ret = join(meta_data, roi_data, kind = :inner, on = :id)
    ret = join(handedness_data, ret, kind= :inner, on = :id)
    dupe_ending = "_1"
    for dupe in filter(c -> endswith(string(c), dupe_ending), names(ret))
      orig = symbol(replace(string(dupe), dupe_ending, ""))
      @assert all(ret[orig] == ret[dupe])
      delete!(ret, dupe)
    end
    ret[ret[:handedness] .== 2, :]
  end

  @assert size(all_data, 2) == num_cols

  return all_data
end


function correct_rois_for_covars(;output_f::ASCIIString="",
                                 covars::Vector{Symbol}= default_covars,
                                 covar_preprocess::Function = calc_total_gray!
                                 )

  function inner_data(df)
    Matrix(
      df[filter(c -> c != :id, names(df))]
      )
  end


  function with_id(cols::Vector{Symbol})
    push!(copy(cols), :id)
  end

  all_data = load_all_data()
  num_subjects = size(all_data, 1)

  roi_cols::Vector{Symbol} = filter(c-> is_roi_col(string(c)), names(all_data))

  normalized_rois = begin
    covar_preprocess(all_data)
    ret::DataFrame = calc_residuals(roi_cols, covars, all_data)
    rename!(c -> symbol(c, "_normalized"), ret)
    ret[:id] = all_data[:id]
    ret
  end
  @assert size(normalized_rois) == (num_subjects, length(roi_cols) + 1)

  normalized_and_orig_rois = id_inner_join(normalized_rois, all_data)

  @assert size(normalized_and_orig_rois, 1) == num_subjects

  if !isempty(output_f)
    writetable(output_f, normalized_and_orig_rois)
  end

  return normalized_and_orig_rois
end


function load_rois()
  readtable(data_f("ROI_matrix.txt"), separator='\t')
end


function calc_roi_corrs(rois)
  rois = isempty(rois) ? load_rois()[:, 3:end] : rois

  rois_corr = cor(rois[:, :], rois[:, :])

  num_rois = length(names(rois))
  @assert size(rois_corr) == (num_rois, num_rois)

  rois_corr
end


function apply_both(fn::Function, lows, highs)
  fn(lows), fn(highs)
end
