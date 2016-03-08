using DataFrames

function data_f(src_f)
  "../data/$src_f"
end


function id_inner_join(a::DataFrame, b::DataFrame)
  join(a, b, kind=:inner, on=:id)
end

function calc_residuals(predictions::Vector{Symbol}, predictors::Vector{Symbol},
                        dataframe::DataFrame)
  num_subjects = size(dataframe, 1)
  num_predictors = length(predictors)
  num_predictions = length(predictions)

  predictor_mat = Matrix(dataframe[predictors])
  prediction_mat = Matrix(dataframe[predictions])

  #prediction_mat = predictor_mat * coeffs
  coeffs::Matrix{Float64} = \(predictor_mat, prediction_mat)
  @assert size(coeffs) == (num_predictors, num_predictions)

  residuals_mat::Matrix{Float64} = prediction_mat - predictor_mat * coeffs
  @assert size(residuals_mat) == (num_subjects, num_predictions)

  residuals = DataFrame(residuals_mat)
  name_lookup = Dict{Symbol, Symbol}([v => predictions[i]
                                      for (i, v) in enumerate(names(residuals))])
  rename!(residuals, name_lookup)

  residuals
end


function is_roi_col(col_name)
  ismatch(r"[L|R][0-9]+$", col_name)
end


function correct_rois_for_covars(output_f::ASCIIString="", covars=[:age, :sex, :edu])
  meta_data = readtable(data_f("animal_scores.csv"))
  roi_data = readtable(data_f("ROI_matrix.txt"), separator='\t')

  num_subjects = min(size(roi_data, 1), size(meta_data, 1))
  num_cols = length(union(names(roi_data), names(meta_data)))

  all_data = begin
    ret = join(meta_data, roi_data, kind = :inner, on = :id)
    dupe_ending = "_1"
    for dupe in filter(c -> endswith(string(c), dupe_ending), names(ret))
      orig = symbol(replace(string(dupe), dupe_ending, ""))
      @assert all(ret[orig] == ret[dupe])
      delete!(ret, dupe)
    end
    ret
  end

  @assert size(all_data) == (num_subjects, num_cols)

  function inner_data(df)
    Matrix(
      df[filter(c -> c != :id, names(df))]
      )
  end


  function with_id(cols::Vector{Symbol})
    push!(copy(cols), :id)
  end

  roi_cols::Vector{Symbol} = filter(c-> is_roi_col(string(c)), names(all_data))
  rois_df::DataFrame = all_data[:, with_id(roi_cols)]
  @assert size(rois_df, 1) == num_subjects

  normalized_rois = begin
    ret::DataFrame = calc_residuals(roi_cols, covars, all_data)
    rename!(c -> symbol(c, "_normalized"), ret)
    ret[:id] = all_data[:id]
    ret
  end
  @assert size(normalized_rois) == size(rois_df)

  normalized_and_orig_rois = id_inner_join(normalized_rois,
                                           all_data[:, with_id(roi_cols)])

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
