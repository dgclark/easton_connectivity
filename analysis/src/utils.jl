using DataFrames

function data_f(src_f)
  "../data/$src_f"
end


function id_inner_join(a::DataFrame, b::DataFrame)
  join(a, b, kind=:inner, on=:id)
end

function calc_residuals(independent, dependent)
  num_ind_subjects, num_ind_features = size(independent)
  num_dep_subjects, num_dep_features = size(dependent)
  @assert num_ind_subjects == num_dep_subjects
  num_subjects = num_ind_subjects

  sln = \(independent, dependent)
  @assert size(sln) == (num_ind_features, num_dep_features)

  residuals = dependent - independent * sln
  @assert size(residuals) == (num_subjects, num_dep_features)

  residuals
end


function is_roi_col(col_name)
  ismatch(r"[L|R][0-9]+$", col_name)
end


function correct_rois_for_nuisance(output_f::ASCIIString="")
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

  function verify_id(df)
    @assert all(df[:id] == all_data[:id])
  end

  function inner_data(df)
    Matrix(
      df[filter(c -> c != :id, names(df))]
      )
  end

  roi_cols::Vector{Symbol} = filter(c-> is_roi_col(string(c)), names(all_data))
  rois_df::DataFrame = all_data[:, vcat(roi_cols, :id)]
  rois_data::Matrix{Float64} = inner_data(rois_df)
  @assert size(rois_df, 1) == num_subjects
  verify_id(rois_df)

  cols = [:edu, :age, :sex, :id]
  nuisance = all_data[cols]
  nuisance[:age_x_sex] = dot(nuisance[:age], nuisance[:sex])
  nuisance[:mean_roi] = mean(rois_data, 2)[:]
  @assert size(nuisance) == (num_subjects, 6)
  verify_id(nuisance)

  normalized_rois = begin
    norm_roi_data::Matrix{Float64} = calc_residuals(
      inner_data(nuisance),
      rois_data)
    ret = DataFrame(norm_roi_data)
    rename!(c -> symbol(c, "_normalized"), ret)
    ret[:id] = nuisance[:id]
    ret
  end
  @assert size(normalized_rois) == size(rois_df)
  verify_id(normalized_rois)

  normalized_and_orig_rois = begin
    tmp = id_inner_join(normalized_rois, rois_df)
    id_inner_join(nuisance, tmp)
  end

  @assert size(normalized_and_orig_rois, 1) == num_subjects
  verify_id(normalized_and_orig_rois)

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
